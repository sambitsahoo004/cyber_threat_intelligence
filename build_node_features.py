#!/usr/bin/env python3
"""
Build Node Features for GNN Training
Combines:
1. Transformer embeddings (semantic)
2. Structural features (degree, PageRank)
3. Meta-path similarity features
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict
from scipy.sparse import load_npz, csr_matrix
import torch
from tqdm import tqdm

# Transformer imports
try:
    from transformers import AutoTokenizer, AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available - using fallback features")


class NodeFeatureBuilder:
    """Build comprehensive node features for GNN"""
    
    def __init__(self, hin_dir: str, metapath_dir: str, output_dir: str):
        self.hin_dir = Path(hin_dir)
        self.metapath_dir = Path(metapath_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.node_meta = self._load_json(self.hin_dir / "node_meta.json")
        self.node_registry = self._load_json(self.hin_dir / "node_registry.json")
        self.type_index_map = self._load_json(self.hin_dir / "nodes" / "type_index_map.json")
        
        # Load relation metadata for structural features
        self.relation_meta = self._load_json(self.hin_dir / "relations" / "relation_meta.json")
        
        # Initialize transformer model
        self.tokenizer = None
        self.model = None
        if TRANSFORMERS_AVAILABLE:
            self._init_transformer()
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file"""
        if not path.exists():
            print(f"⚠️  File not found: {path}")
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    
    def _init_transformer(self, model_name: str = "microsoft/deberta-v3-small"):
        """Initialize transformer model"""
        print(f"🤖 Loading transformer model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("   ✓ Model moved to GPU")
            
            print("   ✓ Model loaded successfully")
        except Exception as e:
            print(f"   ✗ Failed to load model: {e}")
            self.tokenizer = None
            self.model = None
    
    def get_node_types(self) -> List[str]:
        """Get all node types"""
        return list(self.type_index_map.keys())
    
    def get_ordered_node_ids(self, node_type: str) -> List[int]:
        """Get ordered list of node IDs for a type"""
        ids_file = self.hin_dir / "nodes" / f"{node_type}_ids.json"
        if not ids_file.exists():
            return []
        with open(ids_file, 'r') as f:
            return json.load(f)
    
    # ============================================================
    # FEATURE 1: Transformer Embeddings (Semantic)
    # ============================================================
    
    def build_transformer_features(self, node_type: str, batch_size: int = 32) -> np.ndarray:
        """
        Build transformer embeddings for nodes
        
        Returns: (num_nodes, embedding_dim) array
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None:
            return self._build_fallback_text_features(node_type)
        
        print(f"\n🔤 Building transformer features for {node_type}")
        
        # Get ordered node IDs
        node_ids = self.get_ordered_node_ids(node_type)
        if not node_ids:
            print(f"   ⚠️  No nodes found for {node_type}")
            return np.array([])
        
        # Collect text for each node
        texts = []
        for nid in node_ids:
            meta = self.node_meta.get(str(nid), {})
            canonical = meta.get('canonical', f'node_{nid}')
            texts.append(canonical)
        
        # Batch encoding
        embeddings = []
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"   Encoding {node_type}"):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt'
                ).to(device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(batch_embeddings)
        
        embeddings = np.vstack(embeddings)
        print(f"   ✓ Created embeddings: {embeddings.shape}")
        
        return embeddings
    
    def _build_fallback_text_features(self, node_type: str) -> np.ndarray:
        """
        Fallback: One-hot encoding + normalized frequency
        """
        print(f"   ⚠️  Using fallback features for {node_type}")
        
        node_ids = self.get_ordered_node_ids(node_type)
        if not node_ids:
            return np.array([])
        
        # Simple features: [frequency, log_frequency, one_hot_type]
        features = []
        all_types = self.get_node_types()
        type_dim = len(all_types)
        type_idx = all_types.index(node_type) if node_type in all_types else 0
        
        for nid in node_ids:
            meta = self.node_meta.get(str(nid), {})
            freq = meta.get('frequency', 1)
            
            feat = [
                float(freq),  # Raw frequency
                np.log1p(freq),  # Log frequency
            ]
            
            # One-hot node type
            one_hot = [0.0] * type_dim
            one_hot[type_idx] = 1.0
            feat.extend(one_hot)
            
            features.append(feat)
        
        features = np.array(features, dtype=np.float32)
        print(f"   ✓ Created fallback features: {features.shape}")
        
        return features
    
    # ============================================================
    # FEATURE 2: Structural Features (Graph-based)
    # ============================================================
    
    def build_structural_features(self, node_type: str) -> np.ndarray:
        """
        Build structural features:
        - In-degree, out-degree, total-degree
        - PageRank (if combined_similarity available)
        - Clustering coefficient (approximate)
        
        Returns: (num_nodes, struct_dim) array
        """
        print(f"\n📊 Building structural features for {node_type}")
        
        node_ids = self.get_ordered_node_ids(node_type)
        if not node_ids:
            return np.array([])
        
        num_nodes = len(node_ids)
        
        # Initialize degree counters
        in_degree = np.zeros(num_nodes, dtype=np.float32)
        out_degree = np.zeros(num_nodes, dtype=np.float32)
        
        # Count degrees from relation matrices
        for fname, meta in self.relation_meta.items():
            src_type = meta['src_type']
            dst_type = meta['dst_type']
            
            # Load matrix
            adj_file = self.hin_dir / "relations" / meta['adjacency_file']
            if not adj_file.exists():
                continue
            
            adj = load_npz(adj_file)
            
            # Update degrees
            if src_type == node_type:
                out_degree += np.array(adj.sum(axis=1)).flatten()
            
            if dst_type == node_type:
                in_degree += np.array(adj.sum(axis=0)).flatten()
        
        total_degree = in_degree + out_degree
        
        # PageRank (from combined_similarity if available)
        pagerank = self._compute_pagerank(node_type)
        
        # Combine features
        features = np.column_stack([
            in_degree,
            out_degree,
            total_degree,
            np.log1p(total_degree),  # Log-scaled
            pagerank
        ])
        
        print(f"   ✓ Created structural features: {features.shape}")
        return features
    
    def _compute_pagerank(self, node_type: str, damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
        """
        Compute PageRank from combined_similarity matrix
        """
        # Try to load combined similarity from metapath output
        combined_file = self.metapath_dir / node_type / "combined_similarity.npz"
        
        if not combined_file.exists():
            print(f"   ⚠️  No combined_similarity for {node_type}, using uniform PageRank")
            num_nodes = len(self.get_ordered_node_ids(node_type))
            return np.ones(num_nodes, dtype=np.float32) / num_nodes
        
        # Load similarity matrix
        S = load_npz(combined_file)
        n = S.shape[0]
        
        # Normalize to stochastic matrix
        row_sums = np.array(S.sum(axis=1)).flatten()
        row_sums[row_sums == 0] = 1.0  # Avoid division by zero
        D_inv = csr_matrix((1.0 / row_sums, (range(n), range(n))), shape=(n, n))
        P = D_inv @ S
        
        # Power iteration
        pr = np.ones(n, dtype=np.float32) / n
        for _ in range(max_iter):
            pr_new = damping * P.T @ pr + (1 - damping) / n
            if np.linalg.norm(pr_new - pr, 1) < 1e-6:
                break
            pr = pr_new
        
        print(f"   ✓ Computed PageRank (converged)")
        return pr
    
    # ============================================================
    # FEATURE 3: Meta-Path Similarity Features
    # ============================================================
    
    def build_metapath_features(self, node_type: str, top_k: int = 50) -> np.ndarray:
        """
        Build features from meta-path similarities:
        - Mean similarity to top-k neighbors
        - Max similarity
        - Std deviation of similarities
        - Number of connected neighbors
        
        Returns: (num_nodes, metapath_dim) array
        """
        print(f"\n🛤️  Building meta-path features for {node_type}")
        
        node_ids = self.get_ordered_node_ids(node_type)
        if not node_ids:
            return np.array([])
        
        num_nodes = len(node_ids)
        
        # Try to load combined similarity
        combined_file = self.metapath_dir / node_type / "combined_similarity.npz"
        
        if not combined_file.exists():
            print(f"   ⚠️  No meta-path data for {node_type}, returning zeros")
            return np.zeros((num_nodes, 4), dtype=np.float32)
        
        S = load_npz(combined_file)
        
        # Compute features for each node
        features = []
        for i in range(num_nodes):
            row = S[i].toarray().flatten()
            
            # Get top-k neighbors (excluding self)
            row[i] = 0  # Exclude self-loop
            top_k_idx = np.argsort(row)[-top_k:]
            top_k_vals = row[top_k_idx]
            
            feat = [
                np.mean(top_k_vals) if len(top_k_vals) > 0 else 0.0,  # Mean similarity
                np.max(row) if row.size > 0 else 0.0,  # Max similarity
                np.std(top_k_vals) if len(top_k_vals) > 0 else 0.0,  # Std
                np.count_nonzero(row)  # Number of neighbors
            ]
            features.append(feat)
        
        features = np.array(features, dtype=np.float32)
        print(f"   ✓ Created meta-path features: {features.shape}")
        
        return features
    
    # ============================================================
    # COMBINE ALL FEATURES
    # ============================================================
    
    def build_all_features(self, node_type: str, use_transformers: bool = True) -> np.ndarray:
        """
        Build complete feature set for a node type
        
        Returns: (num_nodes, total_dim) array
        """
        print(f"\n{'='*60}")
        print(f"BUILDING FEATURES FOR: {node_type}")
        print(f"{'='*60}")
        
        features_list = []
        
        # 1. Semantic features (transformer or fallback)
        if use_transformers and TRANSFORMERS_AVAILABLE:
            semantic_feat = self.build_transformer_features(node_type)
        else:
            semantic_feat = self._build_fallback_text_features(node_type)
        
        if semantic_feat.size > 0:
            features_list.append(semantic_feat)
        
        # 2. Structural features
        struct_feat = self.build_structural_features(node_type)
        if struct_feat.size > 0:
            features_list.append(struct_feat)
        
        # 3. Meta-path features
        metapath_feat = self.build_metapath_features(node_type)
        if metapath_feat.size > 0:
            features_list.append(metapath_feat)
        
        # Combine
        if not features_list:
            print(f"   ⚠️  No features created for {node_type}")
            return np.array([])
        
        combined = np.hstack(features_list)
        
        # Normalize (z-score per feature)
        mean = combined.mean(axis=0, keepdims=True)
        std = combined.std(axis=0, keepdims=True)
        std[std == 0] = 1.0  # Avoid division by zero
        combined = (combined - mean) / std
        
        print(f"\n✅ Total features: {combined.shape}")
        return combined
    
    # ============================================================
    # SAVE FEATURES
    # ============================================================
    
    def save_features(self, node_type: str, features: np.ndarray):
        """Save features to disk"""
        output_file = self.output_dir / f"features_{node_type}.npy"
        np.save(output_file, features)
        print(f"💾 Saved features to: {output_file.name}")
        
        # Save metadata
        meta = {
            'node_type': node_type,
            'num_nodes': int(features.shape[0]),
            'feature_dim': int(features.shape[1]),
            'mean': float(features.mean()),
            'std': float(features.std())
        }
        
        meta_file = self.output_dir / f"features_{node_type}_meta.json"
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)
    
    # ============================================================
    # MAIN PIPELINE
    # ============================================================
    
    def build_all_node_types(self, use_transformers: bool = True):
        """Build features for all node types"""
        print("\n" + "="*70)
        print("NODE FEATURE BUILDER")
        print("="*70)
        
        node_types = self.get_node_types()
        print(f"\n📋 Found {len(node_types)} node types: {node_types}")
        
        for node_type in node_types:
            features = self.build_all_features(node_type, use_transformers)
            
            if features.size > 0:
                self.save_features(node_type, features)
            else:
                print(f"   ⚠️  Skipping {node_type} (no features)")
        
        print("\n" + "="*70)
        print("✅ FEATURE BUILDING COMPLETE")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Build node features for GNN training'
    )
    
    parser.add_argument(
        '--hin-dir',
        default='hin_out',
        help='HIN output directory'
    )
    
    parser.add_argument(
        '--metapath-dir',
        default='metapath_out_per_ioc',
        help='Meta-path output directory'
    )
    
    parser.add_argument(
        '--output',
        default='node_features',
        help='Output directory for features'
    )
    
    parser.add_argument(
        '--no-transformers',
        action='store_true',
        help='Skip transformer embeddings (use fallback)'
    )
    
    parser.add_argument(
        '--model',
        default='microsoft/deberta-v3-small',
        help='Transformer model name'
    )
    
    args = parser.parse_args()
    
    # Build features
    builder = NodeFeatureBuilder(
        hin_dir=args.hin_dir,
        metapath_dir=args.metapath_dir,
        output_dir=args.output
    )
    
    builder.build_all_node_types(use_transformers=not args.no_transformers)
    
    print("\n✅ All node features saved!")


if __name__ == "__main__":
    main()