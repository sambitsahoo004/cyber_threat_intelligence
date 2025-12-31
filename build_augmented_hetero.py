#!/usr/bin/env python3
"""
Build Enhanced HeteroData for GNN Training
Merges:
1. Original HIN structure (from build_hin.py)
2. Node features (from build_node_features.py)
3. Meta-path similarity edges (from metapath.py)
"""

import json
import numpy as np
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from scipy.sparse import load_npz, csr_matrix

try:
    from torch_geometric.data import HeteroData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("❌ PyTorch Geometric not available!")
    exit(1)


class EnhancedHeteroDataBuilder:
    """Build enhanced HeteroData with all features and meta-paths"""
    
    def __init__(self, hin_dir: str, features_dir: str, metapath_dir: str, output_dir: str):
        self.hin_dir = Path(hin_dir)
        self.features_dir = Path(features_dir)
        self.metapath_dir = Path(metapath_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.node_meta = self._load_json(self.hin_dir / "node_meta.json")
        self.type_index_map = self._load_json(self.hin_dir / "nodes" / "type_index_map.json")
        self.relation_meta = self._load_json(self.hin_dir / "relations" / "relation_meta.json")
        
        # Initialize HeteroData
        self.data = HeteroData()
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file"""
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    
    def get_node_types(self) -> List[str]:
        """Get all node types"""
        return list(self.type_index_map.keys())
    
    # ============================================================
    # STEP 1: Load Node Features
    # ============================================================
    
    def load_node_features(self):
        """Load pre-computed node features"""
        print("\n📦 Loading node features...")
        
        for node_type in self.get_node_types():
            feat_file = self.features_dir / f"features_{node_type}.npy"
            
            if not feat_file.exists():
                print(f"   ⚠️  Features not found for {node_type}, using zeros")
                num_nodes = len(self.type_index_map.get(node_type, {}))
                features = torch.zeros((num_nodes, 1), dtype=torch.float)
            else:
                features = torch.from_numpy(np.load(feat_file)).float()
                print(f"   ✓ Loaded {node_type}: {features.shape}")
            
            self.data[node_type].x = features
            self.data[node_type].num_nodes = features.shape[0]
    
    # ============================================================
    # STEP 2: Load Original Edges (from HIN)
    # ============================================================
    
    def load_original_edges(self):
        """Load original relation edges from HIN"""
        print("\n🔗 Loading original edges from HIN...")
        
        for fname, meta in self.relation_meta.items():
            src_type = meta['src_type']
            dst_type = meta['dst_type']
            relation = meta['relation']
            
            # Load adjacency matrix
            adj_file = self.hin_dir / "relations" / meta['adjacency_file']
            if not adj_file.exists():
                continue
            
            adj = load_npz(adj_file)
            
            # Convert to edge_index
            edge_index, edge_weight = self._sparse_to_edge_index(adj)
            
            if edge_index.size(1) == 0:
                continue
            
            # Add to HeteroData
            self.data[src_type, relation, dst_type].edge_index = edge_index
            self.data[src_type, relation, dst_type].edge_attr = edge_weight.unsqueeze(1)
            
            print(f"   ✓ Added {src_type}-{relation}->{dst_type}: {edge_index.size(1)} edges")
    
    # ============================================================
    # STEP 3: Add Meta-Path Similarity Edges
    # ============================================================
    
    def add_metapath_edges(self, similarity_threshold: float = 0.1, top_k: int = 50):
        """
        Add meta-path similarity as new edge types
        
        For each node type with combined_similarity:
        - Create edge type: (node_type, 'metapath_sim', node_type)
        - Filter by threshold and/or top-k
        """
        print(f"\n🛤️  Adding meta-path similarity edges (threshold={similarity_threshold}, top_k={top_k})...")
        
        for node_type in self.get_node_types():
            combined_file = self.metapath_dir / node_type / "combined_similarity.npz"
            
            if not combined_file.exists():
                print(f"   ⚠️  No meta-path similarity for {node_type}")
                continue
            
            # Load similarity matrix
            S = load_npz(combined_file)
            
            # Apply threshold
            S = S.tocsr()
            S.data[S.data < similarity_threshold] = 0
            S.eliminate_zeros()
            
            # Keep only top-k per node (to control density)
            S = self._keep_topk_per_row(S, top_k)
            
            # Convert to edge_index
            edge_index, edge_weight = self._sparse_to_edge_index(S)
            
            if edge_index.size(1) == 0:
                print(f"   ⚠️  No edges after filtering for {node_type}")
                continue
            
            # Add as new edge type
            relation_name = 'metapath_sim'
            self.data[node_type, relation_name, node_type].edge_index = edge_index
            self.data[node_type, relation_name, node_type].edge_attr = edge_weight.unsqueeze(1)
            
            print(f"   ✓ Added {node_type}-{relation_name}->{node_type}: {edge_index.size(1)} edges")
    
    def _keep_topk_per_row(self, mat: csr_matrix, k: int) -> csr_matrix:
        """Keep only top-k values per row"""
        n = mat.shape[0]
        rows, cols, data = [], [], []
        
        for i in range(n):
            start, end = mat.indptr[i], mat.indptr[i+1]
            if end - start == 0:
                continue
            
            row_data = mat.data[start:end]
            row_cols = mat.indices[start:end]
            
            # Get top-k
            if len(row_data) > k:
                top_idx = np.argpartition(row_data, -k)[-k:]
                row_data = row_data[top_idx]
                row_cols = row_cols[top_idx]
            
            rows.extend([i] * len(row_data))
            cols.extend(row_cols.tolist())
            data.extend(row_data.tolist())
        
        return csr_matrix((data, (rows, cols)), shape=mat.shape)
    
    # ============================================================
    # STEP 4: Add Meta-Path Edge Attributes (Learned Weights)
    # ============================================================
    
    def add_metapath_weights_as_attributes(self):
        """
        Add learned meta-path weights as global graph attributes
        (for interpretability and weight-aware GNN layers)
        """
        print("\n⚖️  Loading meta-path weights...")
        
        weights_dict = {}
        
        for node_type in self.get_node_types():
            weights_file = self.metapath_dir / node_type / "metapath_weights.json"
            
            if not weights_file.exists():
                continue
            
            with open(weights_file, 'r') as f:
                weights = json.load(f)
            
            weights_dict[node_type] = weights
            print(f"   ✓ Loaded weights for {node_type}: {len(weights.get('weights', []))} meta-paths")
        
        # Store as graph-level attribute
        self.data.metapath_weights = weights_dict
    
    # ============================================================
    # HELPER: Sparse Matrix to Edge Index
    # ============================================================
    
    def _sparse_to_edge_index(self, mat: csr_matrix):
        """Convert sparse matrix to PyG edge_index format"""
        coo = mat.tocoo()
        
        edge_index = torch.tensor(
            [coo.row, coo.col],
            dtype=torch.long
        )
        
        edge_weight = torch.tensor(coo.data, dtype=torch.float)
        
        return edge_index, edge_weight
    
    # ============================================================
    # STEP 5: Add Training Masks (Optional)
    # ============================================================
    
    def add_training_masks(self, train_ratio: float = 0.6, val_ratio: float = 0.2):
        """
        Add train/val/test masks for node classification
        (useful if you have labels)
        """
        print(f"\n🎭 Adding training masks (train={train_ratio}, val={val_ratio})...")
        
        for node_type in self.get_node_types():
            num_nodes = self.data[node_type].num_nodes
            
            # Random split
            indices = torch.randperm(num_nodes)
            train_size = int(train_ratio * num_nodes)
            val_size = int(val_ratio * num_nodes)
            
            train_mask = torch.zeros(num_nodes, dtype=torch.bool)
            val_mask = torch.zeros(num_nodes, dtype=torch.bool)
            test_mask = torch.zeros(num_nodes, dtype=torch.bool)
            
            train_mask[indices[:train_size]] = True
            val_mask[indices[train_size:train_size+val_size]] = True
            test_mask[indices[train_size+val_size:]] = True
            
            self.data[node_type].train_mask = train_mask
            self.data[node_type].val_mask = val_mask
            self.data[node_type].test_mask = test_mask
            
            print(f"   ✓ {node_type}: train={train_mask.sum()}, val={val_mask.sum()}, test={test_mask.sum()}")
    
    # ============================================================
    # STEP 6: Add Node Labels (if available)
    # ============================================================
    
    def add_node_labels(self, labels_file: Optional[str] = None):
        """
        Add node labels for supervised learning
        Expected format: labels_<node_type>.json with {node_id: label}
        """
        if labels_file is None:
            print("\n⚠️  No labels file provided, skipping labels")
            return
        
        print(f"\n🏷️  Loading node labels from {labels_file}...")
        
        labels_path = Path(labels_file)
        if not labels_path.exists():
            print(f"   ⚠️  Labels file not found: {labels_file}")
            return
        
        with open(labels_path, 'r') as f:
            all_labels = json.load(f)
        
        for node_type, labels_dict in all_labels.items():
            if node_type not in self.data.node_types:
                continue
            
            num_nodes = self.data[node_type].num_nodes
            labels = torch.full((num_nodes,), -1, dtype=torch.long)  # -1 = unlabeled
            
            for node_id, label in labels_dict.items():
                try:
                    local_idx = self.type_index_map[node_type].get(str(node_id))
                    if local_idx is not None:
                        labels[local_idx] = int(label)
                except:
                    continue
            
            self.data[node_type].y = labels
            print(f"   ✓ Added labels for {node_type}: {(labels >= 0).sum()} labeled nodes")
    
    # ============================================================
    # MAIN BUILD PIPELINE
    # ============================================================
    
    def build(self, add_metapath: bool = True, 
              similarity_threshold: float = 0.1, 
              top_k: int = 50,
              add_masks: bool = True,
              labels_file: Optional[str] = None):
        """
        Complete pipeline to build enhanced HeteroData
        """
        print("\n" + "="*70)
        print("ENHANCED HETERODATA BUILDER")
        print("="*70)
        
        # Step 1: Load features
        self.load_node_features()
        
        # Step 2: Load original edges
        self.load_original_edges()
        
        # Step 3: Add meta-path similarity edges
        if add_metapath:
            self.add_metapath_edges(similarity_threshold, top_k)
            self.add_metapath_weights_as_attributes()
        
        # Step 4: Add training masks
        if add_masks:
            self.add_training_masks()
        
        # Step 5: Add labels (if provided)
        if labels_file:
            self.add_node_labels(labels_file)
        
        # Print summary
        self._print_summary()
        
        return self.data
    
    def _print_summary(self):
        """Print summary of HeteroData"""
        print("\n" + "="*70)
        print("HETERODATA SUMMARY")
        print("="*70)
        
        print(f"\n📊 Node Types: {len(self.data.node_types)}")
        for node_type in self.data.node_types:
            num_nodes = self.data[node_type].num_nodes
            feat_dim = self.data[node_type].x.shape[1] if hasattr(self.data[node_type], 'x') else 0
            has_labels = hasattr(self.data[node_type], 'y')
            print(f"   • {node_type}: {num_nodes} nodes, {feat_dim} features, labels={has_labels}")
        
        print(f"\n🔗 Edge Types: {len(self.data.edge_types)}")
        for edge_type in self.data.edge_types:
            src, rel, dst = edge_type
            num_edges = self.data[edge_type].edge_index.size(1)
            print(f"   • {src} -[{rel}]-> {dst}: {num_edges} edges")
        
        print(f"\n💾 Total memory: ~{self._estimate_memory_mb():.2f} MB")
    
    def _estimate_memory_mb(self) -> float:
        """Estimate memory usage"""
        total_bytes = 0
        
        # Node features
        for node_type in self.data.node_types:
            if hasattr(self.data[node_type], 'x'):
                total_bytes += self.data[node_type].x.nelement() * self.data[node_type].x.element_size()
        
        # Edges
        for edge_type in self.data.edge_types:
            total_bytes += self.data[edge_type].edge_index.nelement() * self.data[edge_type].edge_index.element_size()
            if hasattr(self.data[edge_type], 'edge_attr'):
                total_bytes += self.data[edge_type].edge_attr.nelement() * self.data[edge_type].edge_attr.element_size()
        
        return total_bytes / (1024 * 1024)
    
    # ============================================================
    # SAVE
    # ============================================================
    
    def save(self, filename: str = "enhanced_hetero_graph.pt"):
        """Save HeteroData to disk"""
        output_file = self.output_dir / filename
        torch.save(self.data, output_file)
        print(f"\n💾 Saved HeteroData to: {output_file}")
        
        # Save metadata
        meta = {
            'node_types': list(self.data.node_types),
            'edge_types': [f"{s}-{r}->{d}" for s, r, d in self.data.edge_types],
            'num_nodes': {nt: int(self.data[nt].num_nodes) for nt in self.data.node_types},
            'num_edges': {f"{s}-{r}->{d}": int(self.data[s,r,d].edge_index.size(1)) 
                         for s, r, d in self.data.edge_types},
            'has_metapath_weights': hasattr(self.data, 'metapath_weights')
        }
        
        meta_file = self.output_dir / "heterodata_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
        
        print(f"💾 Saved metadata to: {meta_file}")


# ============================================================
# CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Build enhanced HeteroData with meta-path augmentation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python build_enhanced_heterodata.py \\
    --hin-dir hin_out \\
    --features-dir node_features \\
    --metapath-dir metapath_out_per_ioc \\
    --output enhanced_graph \\
    --threshold 0.1 \\
    --top-k 50

This will create:
  enhanced_graph/
  ├── enhanced_hetero_graph.pt  (main graph file)
  └── heterodata_meta.json      (metadata)
        """
    )
    
    parser.add_argument(
        '--hin-dir',
        default='hin_out',
        help='HIN output directory (default: hin_out)'
    )
    
    parser.add_argument(
        '--features-dir',
        default='node_features',
        help='Node features directory (default: node_features)'
    )
    
    parser.add_argument(
        '--metapath-dir',
        default='metapath_out_per_ioc',
        help='Meta-path output directory (default: metapath_out_per_ioc)'
    )
    
    parser.add_argument(
        '--output',
        default='enhanced_graph',
        help='Output directory (default: enhanced_graph)'
    )
    
    parser.add_argument(
        '--no-metapath',
        action='store_true',
        help='Skip meta-path similarity edges'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Meta-path similarity threshold (default: 0.1)'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Keep top-k similar nodes per node (default: 50)'
    )
    
    parser.add_argument(
        '--no-masks',
        action='store_true',
        help='Skip train/val/test masks'
    )
    
    parser.add_argument(
        '--labels',
        type=str,
        default=None,
        help='Path to node labels JSON file (optional)'
    )
    
    args = parser.parse_args()
    
    # Build enhanced HeteroData
    builder = EnhancedHeteroDataBuilder(
        hin_dir=args.hin_dir,
        features_dir=args.features_dir,
        metapath_dir=args.metapath_dir,
        output_dir=args.output
    )
    
    data = builder.build(
        add_metapath=not args.no_metapath,
        similarity_threshold=args.threshold,
        top_k=args.top_k,
        add_masks=not args.no_masks,
        labels_file=args.labels
    )
    
    builder.save()
    
    print("\n✅ Enhanced HeteroData build complete!")
    print(f"\nNext steps:")
    print(f"  1. Train GNN model: python train_gnn.py --graph enhanced_graph/enhanced_hetero_graph.pt")
    print(f"  2. Compute embeddings: python compute_embeddings.py")
    print(f"  3. CTI analysis: python analyze_cti.py")


if __name__ == "__main__":
    main()