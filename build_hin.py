#!/usr/bin/env python3
"""
Heterogeneous Information Network (HIN) Builder for Cyber Threat Intelligence
Inspired by HINTI framework - builds graph structures from IOC triples
"""

import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
import numpy as np
from scipy.sparse import csr_matrix, save_npz
import pickle

# Optional imports with fallbacks
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠ NetworkX not available - graph visualization disabled")

try:
    import torch
    from torch_geometric.data import HeteroData
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠ PyTorch Geometric not available - HeteroData export disabled")


# -------------------------
# Canonicalizers
# -------------------------
class EntityCanonicalizer:
    """Normalize entities to canonical forms for deduplication"""
    
    @staticmethod
    def normalize_cve(text: str) -> str:
        """CVE-YYYY-NNNN format"""
        text = text.upper().strip()
        match = re.match(r'CVE[- ]?(\d{4})[- ]?(\d+)', text, re.IGNORECASE)
        if match:
            return f"CVE-{match.group(1)}-{match.group(2)}"
        return text
    
    @staticmethod
    def normalize_hash(text: str) -> str:
        """Lowercase, no spaces"""
        return text.lower().strip().replace(" ", "")
    
    @staticmethod
    def normalize_ip(text: str) -> str:
        """Remove trailing dots"""
        return text.strip().rstrip('.')
    
    @staticmethod
    def normalize_domain(text: str) -> str:
        """Lowercase, remove trailing punctuation"""
        return text.lower().strip().rstrip('.,')
    
    @staticmethod
    def normalize_url(text: str) -> str:
        """Remove trailing commas"""
        return text.strip().rstrip(',')
    
    @staticmethod
    def normalize_threat_actor(text: str) -> str:
        """Lowercase, strip, normalize spaces"""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    @staticmethod
    def normalize(text: str, entity_type: str) -> str:
        """Main normalization dispatcher"""
        if not text or text == 'NULL':
            return text
        
        # Type-specific normalization
        if entity_type == "Vulnerability":
            return EntityCanonicalizer.normalize_cve(text)
        elif "Hash" in entity_type or entity_type == "File":
            return EntityCanonicalizer.normalize_hash(text)
        elif entity_type == "IP":
            return EntityCanonicalizer.normalize_ip(text)
        elif entity_type == "Domain":
            return EntityCanonicalizer.normalize_domain(text)
        elif entity_type == "URL":
            return EntityCanonicalizer.normalize_url(text)
        elif entity_type == "ThreatActor":
            return EntityCanonicalizer.normalize_threat_actor(text)
        else:
            # Default: strip whitespace
            return text.strip()


# -------------------------
# Node Registry
# -------------------------
@dataclass
class NodeInfo:
    """Information about a node in the HIN"""
    node_id: int
    canonical: str
    node_type: str
    original_texts: List[str]
    original_ids: List[str]
    sources: Set[str]
    frequency: int


class NodeRegistry:
    """Central registry for nodes with deduplication"""
    
    def __init__(self):
        self.canonicalizer = EntityCanonicalizer()
        self.node_counter = 0
        # canonical_key -> NodeInfo
        self.nodes: Dict[str, NodeInfo] = {}
        # node_id -> NodeInfo
        self.id_to_node: Dict[int, NodeInfo] = {}
        # node_type -> [node_ids]
        self.type_to_ids: Dict[str, List[int]] = defaultdict(list)
        
    def get_canonical_key(self, text: str, node_type: str) -> str:
        """Generate canonical key for deduplication"""
        canonical = self.canonicalizer.normalize(text, node_type)
        return f"{node_type}::{canonical}"
    
    def register_node(self, text: str, node_type: str, original_id: str, source: str) -> int:
        """Register a node and return its global node_id"""
        key = self.get_canonical_key(text, node_type)
        
        if key in self.nodes:
            # Update existing node
            node = self.nodes[key]
            if text not in node.original_texts:
                node.original_texts.append(text)
            if original_id not in node.original_ids:
                node.original_ids.append(original_id)
            node.sources.add(source)
            node.frequency += 1
            return node.node_id
        else:
            # Create new node
            node_id = self.node_counter
            self.node_counter += 1
            
            canonical = self.canonicalizer.normalize(text, node_type)
            node = NodeInfo(
                node_id=node_id,
                canonical=canonical,
                node_type=node_type,
                original_texts=[text],
                original_ids=[original_id],
                sources={source},
                frequency=1
            )
            
            self.nodes[key] = node
            self.id_to_node[node_id] = node
            self.type_to_ids[node_type].append(node_id)
            
            return node_id
    
    def get_node_by_id(self, node_id: int) -> Optional[NodeInfo]:
        """Retrieve node by its global ID"""
        return self.id_to_node.get(node_id)
    
    def get_statistics(self) -> Dict:
        """Get registry statistics"""
        return {
            'total_nodes': self.node_counter,
            'node_types': {
                node_type: len(ids) 
                for node_type, ids in self.type_to_ids.items()
            },
            'top_frequent_nodes': sorted(
                [(n.canonical, n.node_type, n.frequency) 
                 for n in self.nodes.values()],
                key=lambda x: x[2],
                reverse=True
            )[:20]
        }


# -------------------------
# Edge Registry
# -------------------------
class EdgeRegistry:
    """Registry for edges with aggregation by relation type"""
    
    def __init__(self):
        # relation_type -> list of (src_id, tgt_id, confidence, negation, source_doc)
        self.edges: Dict[Tuple[str,str,str], List[Tuple[int,int,float,bool,str]]] = defaultdict(list)
        
    def add_edge(self, src_id: int, src_type: str, tgt_id: int, tgt_type: str,
                 relation: str, confidence: float, negation: bool, source_doc: str):
        key = (src_type, relation, tgt_type)
        self.edges[key].append((src_id, tgt_id, confidence, negation, source_doc))
    
    def build_adjacency_matrices(self, node_registry: NodeRegistry) -> Dict:
        """
        Returns dict keyed by (src_type, relation, dst_type) -> matrix data dict.
        """
        matrices = {}
        # Build id -> local position maps for each type (deterministic ordering)
        type_to_ids = {t: list(ids) for t, ids in node_registry.type_to_ids.items()}
        # Ensure deterministic ordering (sort by node_id)
        for t in type_to_ids:
            type_to_ids[t] = sorted(type_to_ids[t])
        id_to_pos = {t: {nid: idx for idx, nid in enumerate(type_to_ids[t])} for t in type_to_ids}
        
        for (src_type, relation, dst_type), edge_list in self.edges.items():
            if not edge_list:
                continue
            edge_agg = defaultdict(lambda: {'weights': [], 'count': 0, 'negations': []})
            for src, tgt, conf, neg, _ in edge_list:
                edge_agg[(src, tgt)]['weights'].append(conf)
                edge_agg[(src, tgt)]['count'] += 1
                edge_agg[(src, tgt)]['negations'].append(neg)
            rows, cols, data, counts, negs = [], [], [], [], []
            for (src, tgt), info in edge_agg.items():
                # Map global id -> local position for src/dst (if absent, skip with warning)
                if src not in id_to_pos.get(src_type, {}) or tgt not in id_to_pos.get(dst_type, {}):
                    # skip invalid mapping
                    continue
                rows.append(id_to_pos[src_type][src])
                cols.append(id_to_pos[dst_type][tgt])
                data.append(float(np.mean(info['weights'])))
                counts.append(info['count'])
                negs.append(sum(info['negations']))
            nrows = len(type_to_ids[src_type])
            ncols = len(type_to_ids[dst_type])
            adj = csr_matrix((data, (rows, cols)), shape=(nrows, ncols), dtype=np.float32)
            matrices[(src_type, relation, dst_type)] = {
                'adjacency': adj,
                'rows': np.array(rows, dtype=np.int32),
                'cols': np.array(cols, dtype=np.int32),
                'weights': np.array(data, dtype=np.float32),
                'counts': np.array(counts, dtype=np.int32),
                'negations': np.array(negs, dtype=np.int32),
                'src_type': src_type,
                'dst_type': dst_type,
                'relation': relation,
                'shape': (nrows, ncols),
                'num_edges': len(rows),
                'density': len(rows) / (nrows * ncols) if nrows*ncols>0 else 0.0
            }
        return matrices

    
    def get_statistics(self) -> Dict:
        stats = {}
        for relation_key, edges in self.edges.items():
            confidences = [e[2] for e in edges]
            negations = sum(1 for e in edges if e[3])
            # human-readable key
            if isinstance(relation_key, tuple):
                k = f"{relation_key[0]}::{relation_key[1]}::{relation_key[2]}"
            else:
                k = str(relation_key)
            stats[k] = {
                'count': int(len(edges)),
                'avg_confidence': float(np.mean(confidences)) if confidences else 0.0,
                'std_confidence': float(np.std(confidences)) if confidences else 0.0,
                'negation_count': int(negations),
                'negation_ratio': float(negations / len(edges)) if edges else 0.0
            }
        return stats



# -------------------------
# HIN Builder
# -------------------------
class HINBuilder:
    """Main HIN construction class"""
    
    def __init__(self, output_dir: str = "hin_out"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.node_registry = NodeRegistry()
        self.edge_registry = EdgeRegistry()
        
        self.triples_loaded = 0
        self.triples_skipped = 0
        
    def load_triples(self, triples_file: str) -> None:
        """Load triples from JSONL file"""
        print(f"Loading triples from {triples_file}...")
        
        with open(triples_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    triple = json.loads(line.strip())
                    
                    # Skip invalid triples
                    if not triple.get('head_text') or not triple.get('tail_text'):
                        self.triples_skipped += 1
                        continue
                    
                    # Register nodes
                    src_id = self.node_registry.register_node(
                        text=triple['head_text'],
                        node_type=triple.get('head_type', 'Unknown'),
                        original_id=triple.get('head_id', f"H{line_num}"),
                        source=triple.get('source', 'unknown')
                    )
                    
                    tgt_id = self.node_registry.register_node(
                        text=triple['tail_text'],
                        node_type=triple.get('tail_type', 'Unknown'),
                        original_id=triple.get('tail_id', f"T{line_num}"),
                        source=triple.get('source', 'unknown')
                    )
                    
                    # Register edge
                    # in load_triples when calling add_edge:
                    self.edge_registry.add_edge(
                        src_id=src_id,
                        src_type=triple.get('head_type', 'Unknown'),
                        tgt_id=tgt_id,
                        tgt_type=triple.get('tail_type', 'Unknown'),
                        relation=triple.get('relation', 'related_to'),
                        confidence=float(triple.get('confidence', 0.8)),
                        negation=bool(triple.get('negation', False)),
                        source_doc=triple.get('source_id', 'unknown')
                    )
                   
                    self.triples_loaded += 1
                    
                    if self.triples_loaded % 1000 == 0:
                        print(f"  Loaded {self.triples_loaded} triples...")
                        
                except json.JSONDecodeError as e:
                    print(f"⚠ Line {line_num}: JSON decode error - {e}")
                    self.triples_skipped += 1
                except Exception as e:
                    print(f"⚠ Line {line_num}: Error processing triple - {e}")
                    self.triples_skipped += 1
        
        print(f"✓ Loaded {self.triples_loaded} triples ({self.triples_skipped} skipped)")
    
    def build_matrices(self) -> Dict[str, Dict]:
        """Build adjacency matrices"""
        print("\nBuilding adjacency matrices...")
        num_nodes = self.node_registry.node_counter
        matrices = self.edge_registry.build_adjacency_matrices(self.node_registry)
        
        for relation, data in matrices.items():
            print(f"  {relation}: {data['num_edges']} edges, "
                  f"density={data['density']:.6f}")
        
        return matrices
    
    def save_matrices(self, matrices: Dict) -> None:
        print("\nSaving adjacency matrices...")

        relations_dir = self.output_dir / "relations"
        relations_dir.mkdir(exist_ok=True)

        relation_meta = {}

        for key, data in matrices.items():
            # Determine src_type, relation, dst_type and a stable filename
            if isinstance(key, tuple) and len(key) == 3:
                src_type, relation, dst_type = key
                safe_relation_name = relation
                fname_base = f"relation__{src_type}__{safe_relation_name}__{dst_type}"
            else:
                # fallback to original string-key behavior (less preferred)
                relation = str(key)
                # attempt to infer types from provided matrix metadata (if present)
                src_type = data.get('src_type', 'unknown_src')
                dst_type = data.get('dst_type', 'unknown_dst')
                fname_base = f"relation__{src_type}__{relation}__{dst_type}"

            # adjacency and meta arrays
            adj: csr_matrix = data['adjacency']
            rows = data.get('rows', np.array(adj.nonzero()[0], dtype=np.int32))
            cols = data.get('cols', np.array(adj.nonzero()[1], dtype=np.int32))
            weights = data.get('weights', np.array(adj.data, dtype=np.float32))
            counts = data.get('counts', np.array([1]*len(rows), dtype=np.int32))
            negations = data.get('negations', np.array([0]*len(rows), dtype=np.int32))

            # Save adjacency CSR
            adj_file = relations_dir / f"{fname_base}.npz"
            save_npz(adj_file, adj)
            # Save edge metadata as .npz for compactness
            meta_file = relations_dir / f"{fname_base}_meta.npz"
            np.savez(meta_file, rows=rows, cols=cols, weights=weights, counts=counts, negations=negations)

            # Add to relation_meta summary
            relation_meta[adj_file.name] = {
                'src_type': src_type,
                'dst_type': dst_type,
                'relation': relation,
                'adjacency_file': str(adj_file.name),
                'meta_file': str(meta_file.name),
                'shape': adj.shape,
                'num_edges': int(data.get('num_edges', int(rows.shape[0]))),
                'density': float(data.get('density', 0.0))
            }

            # keep your debug print
            print(f"  ✓ Saved {relation}: {adj_file.name}, {meta_file.name}")

        # Save consolidated relation_meta.json
        relation_meta_file = relations_dir / "relation_meta.json"
        with open(relation_meta_file, 'w', encoding='utf-8') as f:
            json.dump(relation_meta, f, indent=2, ensure_ascii=False)
        print(f"\n  ✓ Saved relation metadata to {relation_meta_file.name}")

    
    def save_node_registry(self) -> None:
        print("\nSaving node registry...")

        # Ensure nodes/ directory
        nodes_dir = self.output_dir / "nodes"
        nodes_dir.mkdir(exist_ok=True)

        # Node registry (canonical -> info) - preserve existing behavior
        registry_data = {
            key: {
                'node_id': node.node_id,
                'canonical': node.canonical,
                'node_type': node.node_type,
                'original_texts': node.original_texts,
                'original_ids': node.original_ids,
                'sources': list(node.sources),
                'frequency': node.frequency
            }
            for key, node in self.node_registry.nodes.items()
        }

        registry_file = self.output_dir / "node_registry.json"
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(registry_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {registry_file.name}")

        # Node metadata (id -> info) - preserve existing behavior
        meta_data = {
            str(node_id): {
                'canonical': node.canonical,
                'node_type': node.node_type,
                'frequency': node.frequency,
                'num_sources': len(node.sources)
            }
            for node_id, node in self.node_registry.id_to_node.items()
        }

        meta_file = self.output_dir / "node_meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {meta_file.name}")

        # Prepare deterministic ordering for each node type
        # Sort by node_id to ensure reproducibility (could be changed to first_seen if available)
        ordered_type_to_ids = {
            node_type: sorted(list(ids))
            for node_type, ids in self.node_registry.type_to_ids.items()
        }

        # Save per-type ordered id lists
        for node_type, ordered_ids in ordered_type_to_ids.items():
            type_file = nodes_dir / f"{node_type}_ids.json"
            with open(type_file, 'w', encoding='utf-8') as f:
                json.dump(ordered_ids, f, indent=2, ensure_ascii=False)
            print(f"  ✓ Saved {type_file.name} (count={len(ordered_ids)})")

        # Save legacy combined mapping as well (for backward compatibility)
        type_file_legacy = self.output_dir / "node_type_to_ids.json"
        with open(type_file_legacy, 'w', encoding='utf-8') as f:
            json.dump({k: v for k, v in ordered_type_to_ids.items()}, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {type_file_legacy.name}")

        # Build and save type_index_map: node_id -> position per type
        type_index_map = {
            node_type: {str(nid): int(pos) for pos, nid in enumerate(ordered_ids)}
            for node_type, ordered_ids in ordered_type_to_ids.items()
        }
        type_index_map_file = nodes_dir / "type_index_map.json"
        with open(type_index_map_file, 'w', encoding='utf-8') as f:
            json.dump(type_index_map, f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved {type_index_map_file.name}")

    
    def build_networkx_graph(self, matrices: Dict[str, Dict]) -> Optional[object]:
        """Build NetworkX MultiDiGraph for visualization"""
        if not NETWORKX_AVAILABLE:
            return None
        
        print("\nBuilding NetworkX graph...")
        G = nx.MultiDiGraph()
        
        # Add nodes
        for node_id, node in self.node_registry.id_to_node.items():
            G.add_node(
                node_id,
                canonical=node.canonical,
                node_type=node.node_type,
                frequency=node.frequency,
                sources=len(node.sources)
            )
        
        # Add edges
        for relation, data in matrices.items():
            for i, (src, tgt, weight) in enumerate(zip(data['rows'], data['cols'], data['weights'])):
                G.add_edge(
                    src, tgt,
                    relation=relation,
                    weight=float(weight),
                    count=int(data['counts'][i]),
                    negations=int(data['negations'][i])
                )
        
        print(f"  ✓ Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Save
        graph_file = self.output_dir / "graph_multidigraph.gpickle"
        with open(graph_file, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  ✓ Saved {graph_file.name}")
        
        return G
    
    def build_pytorch_heterodata(self, matrices: Dict[str, Dict]) -> Optional[object]:
        if not TORCH_AVAILABLE:
            return None

        print("\nBuilding PyTorch Geometric HeteroData...")
        data = HeteroData()

        # --- Prepare deterministic ordering and global->local maps ---
        # Ensure deterministic ordering: sort node ids for each type
        ordered_type_to_ids = {
            node_type: sorted(list(ids))
            for node_type, ids in self.node_registry.type_to_ids.items()
        }

        # global id -> local pos maps
        global_to_local = {
            node_type: {gid: idx for idx, gid in enumerate(ordered_ids)}
            for node_type, ordered_ids in ordered_type_to_ids.items()
        }

        # Create basic node features (frequency) using deterministic ordering
        for node_type, ordered_ids in ordered_type_to_ids.items():
            num_nodes = len(ordered_ids)
            features = torch.zeros((num_nodes, 1), dtype=torch.float)
            for local_idx, global_id in enumerate(ordered_ids):
                node = self.node_registry.id_to_node.get(global_id)
                if node is not None:
                    features[local_idx, 0] = float(node.frequency)
                else:
                    # fallback 0 and warn (shouldn't normally happen)
                    features[local_idx, 0] = 0.0
            # attach to HeteroData
            data[node_type].x = features
            data[node_type].num_nodes = num_nodes

        # --- Add edge types ---
        for key, mat_data in matrices.items():
            # Determine src_type, relation, dst_type
            if isinstance(key, tuple) and len(key) == 3:
                src_type, relation, dst_type = key
            else:
                # legacy: try to read types from mat_data, otherwise assign 'unknown'
                relation = str(key)
                src_type = mat_data.get('src_type', 'unknown_src')
                dst_type = mat_data.get('dst_type', 'unknown_dst')

            rows = mat_data.get('rows')
            cols = mat_data.get('cols')
            weights = mat_data.get('weights', None)

            # If rows/cols not provided, infer from adjacency matrix
            if rows is None or cols is None:
                adj: csr_matrix = mat_data.get('adjacency')
                if adj is None:
                    print(f"  ⚠ Skipping edge type {relation} ({src_type} -> {dst_type}): no rows/cols or adjacency.")
                    continue
                rr, cc = adj.nonzero()
                rows = np.array(rr, dtype=np.int32)
                cols = np.array(cc, dtype=np.int32)
                if weights is None:
                    weights = adj.data if hasattr(adj, 'data') else np.ones(len(rows), dtype=np.float32)
            else:
                # ensure numpy arrays
                rows = np.asarray(rows, dtype=np.int64)
                cols = np.asarray(cols, dtype=np.int64)
                if weights is None:
                    weights = np.ones(len(rows), dtype=np.float32)
                else:
                    weights = np.asarray(weights, dtype=np.float32)

            # Determine if rows/cols are already local indexes (i.e., less than the type sizes)
            src_n = len(ordered_type_to_ids.get(src_type, []))
            dst_n = len(ordered_type_to_ids.get(dst_type, []))

            # Helper: check local vs global heuristics
            rows_are_local = rows.size == 0 or (rows.max() < src_n if src_n > 0 else False)
            cols_are_local = cols.size == 0 or (cols.max() < dst_n if dst_n > 0 else False)

            # If rows/cols look like global ids (not local), map to local positions
            if not rows_are_local or not cols_are_local:
                # We expect rows/cols contain global node ids; map them with global_to_local
                mapped_rows = []
                mapped_cols = []
                mapped_weights = []
                src_map = global_to_local.get(src_type, {})
                dst_map = global_to_local.get(dst_type, {})
                for r, c, w in zip(rows, cols, weights):
                    if (r in src_map) and (c in dst_map):
                        mapped_rows.append(src_map[r])
                        mapped_cols.append(dst_map[c])
                        mapped_weights.append(float(w))
                    else:
                        # skip edges that cannot be mapped (log occasionally)
                        continue
                if not mapped_rows:
                    print(f"  ⚠ No mappable edges for {src_type} - {relation} -> {dst_type}, skipping.")
                    continue
                rows_local = np.array(mapped_rows, dtype=np.int64)
                cols_local = np.array(mapped_cols, dtype=np.int64)
                weights_local = np.array(mapped_weights, dtype=np.float32)
            else:
                # Already local
                rows_local = rows.astype(np.int64)
                cols_local = cols.astype(np.int64)
                weights_local = weights.astype(np.float32)

            # Convert to torch tensors and set edge data
            edge_index = torch.tensor([rows_local, cols_local], dtype=torch.long)
            edge_attr = torch.tensor(weights_local, dtype=torch.float).unsqueeze(1)

            # Assign into HeteroData under explicit triplet
            try:
                data[src_type, relation, dst_type].edge_index = edge_index
                data[src_type, relation, dst_type].edge_attr = edge_attr
            except Exception as e:
                # In case PyG doesn't like creation on-the-fly, create the container first
                data[src_type, relation, dst_type].edge_index = edge_index
                data[src_type, relation, dst_type].edge_attr = edge_attr

        # Final prints (preserve original debug outputs)
        print(f"  ✓ HeteroData: {data.num_nodes} total nodes, {data.num_edges} total edges")
        print(f"  ✓ Node types: {list(data.node_types)}")
        print(f"  ✓ Edge types: {list(data.edge_types)}")

        # Save
        hetero_file = self.output_dir / "hetero_graph.pt"
        torch.save(data, hetero_file)
        print(f"  ✓ Saved {hetero_file.name}")

        return data

    
    def generate_report(self, matrices: Dict[str, Dict]) -> None:
        """Generate summary report"""
        print("\n" + "="*60)
        print("HIN BUILD SUMMARY")
        print("="*60)
        
        # Node statistics
        node_stats = self.node_registry.get_statistics()
        print(f"\nNodes: {node_stats['total_nodes']}")
        print("Node types:")
        for node_type, count in sorted(node_stats['node_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {node_type:20s}: {count:6d}")
        
        # Edge statistics
        # Edge statistics
        edge_stats = self.edge_registry.get_statistics()
        total_edges = sum(s['count'] for s in edge_stats.values())
        print(f"\nEdges: {total_edges}")
        print("\nRelation types:")
        for relation, stats in sorted(edge_stats.items(), key=lambda x: x[1]['count'], reverse=True):
            rel_name = (
                f"{relation[0]}::{relation[1]}::{relation[2]}"
                if isinstance(relation, tuple) else relation
            )
            print(f"  {rel_name:40s}: {stats['count']:6d} "
                f"(avg_conf={stats['avg_confidence']:.3f}, "
                f"neg={stats['negation_ratio']:.2%})")

       
        # Top frequent nodes
        print("\nTop 10 most frequent nodes:")
        for canonical, node_type, freq in node_stats['top_frequent_nodes'][:10]:
            print(f"  {canonical[:50]:50s} ({node_type:15s}): {freq:4d}")
        
        # Save report
        report = {
            'summary': {
                'triples_loaded': self.triples_loaded,
                'triples_skipped': self.triples_skipped,
                'total_nodes': node_stats['total_nodes'],
                'total_edges': total_edges
            },
            'node_statistics': node_stats,
            'edge_statistics': edge_stats
        }
        
        report_file = self.output_dir / "hin_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Report saved to {report_file.name}")
        print("="*60)
    
    def build(self, triples_file: str, build_networkx: bool = True, build_pytorch: bool = True) -> None:
        """Main build pipeline"""
        print("="*60)
        print("HETEROGENEOUS INFORMATION NETWORK BUILDER")
        print("="*60)

        # Load triples and build registries
        self.load_triples(triples_file)

        # Build matrices (forward relations only at first)
        matrices = self.build_matrices()

        # --- ADD BELOW: Auto-generate reverse relations ---
        def _make_reverse_name(rel_name: str) -> str:
            mapping = {
                'uses': 'used_by',
                'exploits': 'exploited_by',
                'targets': 'targeted_by',
                'contains': 'contained_in',
                'runs': 'runs_on',
                'connects_to': 'connected_from',
                'affects': 'affected_by',
                'has_type': 'type_of',
                'delivers': 'delivered_by',
                'belongs_to': 'has_member',
                'includes': 'included_in',
                'evolves_from': 'evolves_to',
                # symmetric relations: keep same name
                'communicates_with': 'communicates_with',
                'related_to': 'related_to'
            }
            return mapping.get(rel_name, f"{rel_name}_rev")

        reverse_entries = {}
        for (src_type, relation, dst_type), data in list(matrices.items()):
            adj = data.get("adjacency")
            if adj is None:
                continue

            rev_rel = _make_reverse_name(relation)
            rev_key = (dst_type, rev_rel, src_type)
            if rev_key in matrices or rev_key in reverse_entries:
                continue

            rev_adj = adj.T.tocsr()

            rows = data.get("cols")
            cols = data.get("rows")
            weights = data.get("weights")
            counts = data.get("counts")
            negations = data.get("negations")

            if rows is None or cols is None:
                rr, cc = rev_adj.nonzero()
                rows, cols = rr, cc
            if weights is None:
                weights = rev_adj.data
            if counts is None:
                counts = np.ones(len(weights), dtype=np.int32)
            if negations is None:
                negations = np.zeros(len(weights), dtype=np.int32)

            reverse_entries[rev_key] = {
                "adjacency": rev_adj,
                "rows": np.array(rows, dtype=np.int32),
                "cols": np.array(cols, dtype=np.int32),
                "weights": np.array(weights, dtype=np.float32),
                "counts": np.array(counts, dtype=np.int32),
                "negations": np.array(negations, dtype=np.int32),
                "src_type": dst_type,
                "dst_type": src_type,
                "relation": rev_rel,
                "shape": rev_adj.shape,
                "num_edges": int(rev_adj.nnz),
                "density": float(rev_adj.nnz) / (rev_adj.shape[0] * rev_adj.shape[1]) if rev_adj.shape[0]*rev_adj.shape[1] > 0 else 0.0
            }

        if reverse_entries:
            matrices.update(reverse_entries)
            print(f"  ✓ Auto-added {len(reverse_entries)} reverse relations.")
        # --- END REVERSE GENERATION BLOCK ---

        # Continue as usual
        self.save_matrices(matrices)
        self.save_node_registry()

        if build_networkx:
            self.build_networkx_graph(matrices)
        if build_pytorch:
            self.build_pytorch_heterodata(matrices)

        self.generate_report(matrices)
        print(f"\n✓ All outputs saved to: {self.output_dir.absolute()}")



# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Build Heterogeneous Information Network from IOC triples',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python build_hin.py --input triples_output.jsonl --output hin_out
  
Output files:
  - adj_<relation>.npz: Sparse adjacency matrices
  - edges_meta_<relation>.npz: Edge metadata (weights, counts, negations)
  - node_registry.json: Full node information
  - node_meta.json: Node metadata by ID
  - node_type_to_ids.json: Node type mappings
  - graph_multidigraph.gpickle: NetworkX graph (optional)
  - hetero_graph.pt: PyTorch Geometric HeteroData (optional)
  - hin_report.json: Statistics and summary
        """
    )
    
    parser.add_argument(
        '--input', '-i',
        default='triples_output.jsonl',
        help='Input triples JSONL file (default: triples_output.jsonl)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='hin_out',
        help='Output directory (default: hin_out)'
    )
    
    parser.add_argument(
        '--no-networkx',
        action='store_true',
        help='Skip NetworkX graph generation'
    )
    
    parser.add_argument(
        '--no-pytorch',
        action='store_true',
        help='Skip PyTorch Geometric HeteroData generation'
    )
    
    args = parser.parse_args()
    
    # Build HIN
    builder = HINBuilder(output_dir=args.output)
    builder.build(
        triples_file=args.input,
        build_networkx=not args.no_networkx,
        build_pytorch=not args.no_pytorch
    )
    
    print("\n✓ HIN build complete!")


if __name__ == "__main__":
    main()