#!/usr/bin/env python3
"""
Generate Predictions for LLM Interpretation
Creates structured JSON predictions for:
1. Link prediction (new relations)
2. Node classification (IOC categorization)
3. Cluster summaries (behavioral patterns)
"""

import json
import torch
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.data.storage import BaseStorage, GlobalStorage, NodeStorage, EdgeStorage
    from sklearn.metrics.pairwise import cosine_similarity
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    print("❌ Dependencies missing!")
    exit(1)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)
class PredictionGenerator:
    """Generate structured predictions for LLM"""
    
    def __init__(self, 
                 model_path: str,
                 graph_path: str,
                 embeddings_dir: str,
                 hin_dir: str,
                 metapath_dir: str,
                 output_dir: str):
        
        self.model_path = Path(model_path)
        self.graph_path = Path(graph_path)
        self.embeddings_dir = Path(embeddings_dir)
        self.hin_dir = Path(hin_dir)
        self.metapath_dir = Path(metapath_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.node_meta = self._load_json(self.hin_dir / "node_meta.json")
        self.type_index_map = self._load_json(self.hin_dir / "nodes" / "type_index_map.json")
        self.relation_meta = self._load_json(self.hin_dir / "relations" / "relation_meta.json")
        
        # Load embeddings
        self.embeddings = {}
        self.node_ids = {}
        self._load_all_embeddings()
        
        # Load graph with safe globals for PyTorch Geometric
        print("📥 Loading graph...")
        torch.serialization.add_safe_globals([
            BaseStorage, 
            GlobalStorage, 
            NodeStorage, 
            EdgeStorage,
            HeteroData
        ])
        self.data = torch.load(self.graph_path, weights_only=False)
        
        # Load metapath weights
        self.metapath_weights = {}
        self._load_metapath_weights()
    
    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_all_embeddings(self):
        """Load all available embeddings"""
        for node_type in self.type_index_map.keys():
            emb_file = self.embeddings_dir / f"embeddings_{node_type}.npy"
            if emb_file.exists():
                self.embeddings[node_type] = np.load(emb_file)
                
                # Load node IDs
                ids_file = self.hin_dir / "nodes" / f"{node_type}_ids.json"
                if ids_file.exists():
                    with open(ids_file, 'r') as f:
                        self.node_ids[node_type] = json.load(f)
                
                print(f"✓ Loaded {node_type}: {self.embeddings[node_type].shape}")
    
    def _load_metapath_weights(self):
        """Load learned metapath weights - handle both flat and nested directory structures"""
        # Try nested structure first (node_type/metapath_weights.json)
        for node_type in self.type_index_map.keys():
            weights_file = self.metapath_dir / node_type / "metapath_weights.json"
            if weights_file.exists():
                with open(weights_file, 'r') as f:
                    self.metapath_weights[node_type] = json.load(f)
                print(f"   ✓ Loaded meta-path weights for {node_type} (nested)")
                continue
            
            # Fallback: Try flat structure (metapath_weights_<node_type>.json)
            weights_file_flat = self.metapath_dir / f"metapath_weights_{node_type}.json"
            if weights_file_flat.exists():
                with open(weights_file_flat, 'r') as f:
                    self.metapath_weights[node_type] = json.load(f)
                print(f"   ✓ Loaded meta-path weights for {node_type} (flat)")
                continue
            
            # Try generic file
            weights_file_generic = self.metapath_dir / "metapath_weights.json"
            if weights_file_generic.exists() and not self.metapath_weights:
                with open(weights_file_generic, 'r') as f:
                    generic_weights = json.load(f)
                    # Try to extract node-specific weights
                    if node_type in generic_weights:
                        self.metapath_weights[node_type] = generic_weights[node_type]
                        print(f"   ✓ Loaded meta-path weights for {node_type} (generic)")
        
        if not self.metapath_weights:
            print(f"   ⚠️  No meta-path weights found in {self.metapath_dir}")
    
    def get_node_info(self, node_type: str, node_idx: int) -> Dict:
        """Get node metadata"""
        node_id = self.node_ids.get(node_type, [])[node_idx] if node_idx < len(self.node_ids.get(node_type, [])) else None
        
        if node_id is None:
            return {'type': node_type, 'id': None, 'canonical': f'unknown_{node_idx}'}
        
        meta = self.node_meta.get(str(node_id), {})
        
        return {
            'type': node_type,
            'id': int(node_id),
            'canonical': meta.get('canonical', f'node_{node_id}'),
            'frequency': meta.get('frequency', 0),
            'first_seen': meta.get('first_seen', ''),
            'last_seen': meta.get('last_seen', '')
        }
    
    def predict_links(self, 
                     edge_type: Tuple[str, str, str],
                     top_k: int = 100,
                     threshold: float = 0.5) -> List[Dict]:
        """
        Predict new links for a given edge type
        
        Returns list of predictions in LLM format
        """
        src_type, rel, dst_type = edge_type
        
        if src_type not in self.embeddings or dst_type not in self.embeddings:
            print(f"⚠️  Embeddings not available for {edge_type}")
            return []
        
        print(f"\n🔮 Predicting links: {src_type} -[{rel}]-> {dst_type}")
        
        src_emb = self.embeddings[src_type]
        dst_emb = self.embeddings[dst_type]
        
        # Compute all pairwise similarities
        similarities = cosine_similarity(src_emb, dst_emb)
        
        # Get existing edges
        existing_edges = set()
        if edge_type in self.data.edge_types:
            edge_index = self.data[edge_type].edge_index.cpu().numpy()
            existing_edges = set(zip(edge_index[0], edge_index[1]))
        
        # Find top predictions (excluding existing)
        predictions = []
        
        for src_idx in range(similarities.shape[0]):
            # Get top-k destinations for this source
            scores = similarities[src_idx]
            top_dst_indices = np.argsort(scores)[::-1]
            
            for dst_idx in top_dst_indices[:top_k]:
                if (src_idx, dst_idx) in existing_edges:
                    continue
                
                score = float(scores[dst_idx])
                
                if score < threshold:
                    break
                
                # Get metapath support
                metapath_support = self._get_metapath_support(
                    src_type, dst_type, src_idx, dst_idx
                )
                
                prediction = {
                    'prediction_type': 'link_prediction',
                    'subject': self.get_node_info(src_type, src_idx),
                    'object': self.get_node_info(dst_type, dst_idx),
                    'predicted_relation': rel,
                    'confidence': score,
                    'embedding_similarity': score,
                    'neighborhood_overlap': self._compute_neighborhood_overlap(
                        src_type, dst_type, src_idx, dst_idx
                    ),
                    'meta_path_support': metapath_support,
                    'evidence': [],  # will be replaced by the evidence extractor
                    'computed_on': datetime.now().isoformat()
                }

                # --- fill evidence (place right after creating the prediction) ---
                prediction['evidence'] = self._find_evidence_for_pair(
                    prediction['subject']['canonical'],
                    prediction['object']['canonical']
                )

                predictions.append(prediction)

                
                if len(predictions) >= top_k:
                    break
            
            if len(predictions) >= top_k:
                break
        
        print(f"   ✓ Generated {len(predictions)} predictions")
        return predictions
    
    def _get_metapath_support(self,
                              src_type: str,
                              dst_type: str,
                              src_idx: int,
                              dst_idx: int,
                              top_k: int = 10,
                              min_weight: float = 1e-12) -> List[Dict]:
        """
        Robust metapath support builder.
        - Chooses a node key to look up learned weights (prefer src_type, then dst_type).
        - Supports fallback from 'metapath_ids' -> readable 'metapaths' using metapath_report.json.
        - Returns list of dicts: {"metapath_id":..., "metapath":..., "weight": ...}
        """
        # pick which node_type's metapath weights to consult (prefer src_type)
        node_key = src_type if src_type in self.metapath_weights else dst_type if dst_type in self.metapath_weights else None
        if node_key is None:
            return []

        weights_data = self.metapath_weights.get(node_key, {})
        weights = weights_data.get("weights", []) or []
        metapaths = weights_data.get("metapaths") or []
        metapath_ids = weights_data.get("metapath_ids") or []

        # If readable labels missing, try to load metapath_report.json to map ids->path
        if not metapaths and metapath_ids:
            # try node-specific folder then top-level metapath_dir
            report_candidates = [
                self.metapath_dir / node_key / "metapath_report.json",
                self.metapath_dir / "metapath_report.json",
                self.metapath_dir.parent / "metapath_report.json",
            ]
            id2path = {}
            for rc in report_candidates:
                if rc.exists():
                    try:
                        r = json.loads(rc.read_text())
                        for mp in r.get("metapaths", []):
                            if "id" in mp and "path" in mp:
                                id2path[mp["id"]] = mp["path"]
                    except Exception:
                        pass
                    break
            metapaths = [ id2path.get(mid, mid) for mid in metapath_ids ]

        # Final fallback: synthetic labels if weights exist
        if not metapaths and weights:
            metapaths = [f"metapath_{i}" for i in range(len(weights))]

        # Align lengths and build candidate pairs (keep non-trivial weights)
        L = min(len(weights), len(metapaths))
        pairs = []
        for i in range(L):
            try:
                w = float(weights[i])
            except Exception:
                continue
            if w is None or w <= min_weight:
                continue
            mpid = metapath_ids[i] if i < len(metapath_ids) else None
            pairs.append((i, metapaths[i], mpid, w))

        # sort & take top_k
        pairs.sort(key=lambda x: x[3], reverse=True)
        pairs = pairs[:top_k]

        support = []
        for idx, label, mpid, w in pairs:
            if isinstance(label, list):
                label_str = " -> ".join(label)
            else:
                label_str = str(label)
            support.append({
                "metapath_id": mpid,
                "metapath": label_str,
                "weight": float(w)
            })

        return support
    
    def _describe_metapath(self, metapath: List[str]) -> str:
        """Generate human-readable meta-path description"""
        # Extract pattern from node types
        return f"Connection through: {' → '.join(metapath)}"
    
    def _compute_neighborhood_overlap(self,
                                     src_type: str,
                                     dst_type: str,
                                     src_idx: int,
                                     dst_idx: int) -> float:
        """Compute Jaccard similarity of 2-hop neighborhoods"""
        # Get 1-hop neighbors from graph
        src_neighbors = self._get_neighbors(src_type, src_idx)
        dst_neighbors = self._get_neighbors(dst_type, dst_idx)
        
        if len(src_neighbors) == 0 and len(dst_neighbors) == 0:
            return 0.0
        
        # Jaccard similarity
        intersection = len(src_neighbors & dst_neighbors)
        union = len(src_neighbors | dst_neighbors)
        
        return intersection / union if union > 0 else 0.0
    
    def _get_neighbors(self, node_type: str, node_idx: int) -> set:
        """Get all neighbors of a node"""
        neighbors = set()
        
        # Check all edge types
        for edge_type in self.data.edge_types:
            src_t, rel, dst_t = edge_type
            
            # Outgoing edges
            if src_t == node_type:
                edge_index = self.data[edge_type].edge_index.cpu().numpy()
                mask = edge_index[0] == node_idx
                neighbors.update(edge_index[1][mask].tolist())
            
            # Incoming edges
            if dst_t == node_type:
                edge_index = self.data[edge_type].edge_index.cpu().numpy()
                mask = edge_index[1] == node_idx
                neighbors.update(edge_index[0][mask].tolist())
        
        return neighbors
    def _find_evidence_for_pair(self,
                                subject_canonical: str,
                                object_canonical: str,
                                ioc_dataset_path: Optional[Path] = None,
                                max_hits: int = 5) -> List[Dict]:
        """
        Simple evidence extractor as instance method.
        - If ioc_dataset_path is None, tries common fallbacks in self.hin_dir.
        - Supports: single JSON list file, JSON-lines file, or a directory with json/jsonl files.
        - Caches loaded docs on self._cached_ioc_docs for performance.
        """
        evidences = []

        # Resolve dataset path fallbacks if not supplied
        if ioc_dataset_path is None:
            candidates = [
                self.hin_dir / "iocs.json",
                self.hin_dir / "ioc_extracted.json",
                self.hin_dir / "balanced_ioc_dataset.json",
                self.hin_dir / "iocs"  # folder with jsonlines perhaps
            ]
            chosen = None
            for c in candidates:
                if c.exists():
                    chosen = c
                    break
            if chosen is None:
                return evidences
            ioc_dataset_path = chosen

        # Use cached docs if available
        if not hasattr(self, "_cached_ioc_docs"):
            self._cached_ioc_docs = []

            # If a directory: load all .json / .jsonl files inside
            try:
                if ioc_dataset_path.is_dir():
                    for child in sorted(ioc_dataset_path.iterdir()):
                        if child.is_file() and child.suffix.lower() in {".json", ".jsonl"}:
                            try:
                                txt = child.read_text()
                                try:
                                    parsed = json.loads(txt)
                                    if isinstance(parsed, list):
                                        self._cached_ioc_docs.extend(parsed)
                                    else:
                                        self._cached_ioc_docs.append(parsed)
                                except Exception:
                                    # fallback to jsonlines
                                    for ln in txt.splitlines():
                                        ln = ln.strip()
                                        if not ln:
                                            continue
                                        try:
                                            self._cached_ioc_docs.append(json.loads(ln))
                                        except Exception:
                                            continue
                            except Exception:
                                continue
                else:
                    txt = ioc_dataset_path.read_text()
                    try:
                        parsed = json.loads(txt)
                        if isinstance(parsed, list):
                            self._cached_ioc_docs = parsed
                        else:
                            self._cached_ioc_docs = [parsed]
                    except Exception:
                        # try jsonlines
                        docs = []
                        for ln in txt.splitlines():
                            ln = ln.strip()
                            if not ln:
                                continue
                            try:
                                docs.append(json.loads(ln))
                            except Exception:
                                continue
                        self._cached_ioc_docs = docs
            except Exception:
                self._cached_ioc_docs = []

        docs = self._cached_ioc_docs

        s_low = subject_canonical.lower()
        o_low = object_canonical.lower()
        for doc in docs:
            text = (doc.get("text") or doc.get("content") or "").lower()
            if not text:
                continue
            if s_low in text and o_low in text:
                start = max(0, text.find(s_low) - 120)
                snippet = (text[start: start + 400]).strip()
                evidences.append({
                    "source": doc.get("source", ""),
                    "doc_id": doc.get("id", doc.get("source_id", "")),
                    "snippet": snippet
                })
            if len(evidences) >= max_hits:
                break

        return evidences

    
    def predict_node_classes(self, node_type: str, top_k: int = 50) -> List[Dict]:
        """
        Produce (A) evaluation predictions on labeled test set (if available)
                (B) predictions for unlabeled nodes (nearest labeled neighbor)
        Returns combined list (tagged). Top_k limits unlabeled predictions.
        """
        if not hasattr(self.data[node_type], 'y'):
            print(f"⚠️  No labels available for {node_type}")
            return []

        print(f"\n🏷️  Predicting node classes for {node_type}")
        labels = self.data[node_type].y.cpu().numpy()

        # Safely get masks (defaults)
        train_mask = self.data[node_type].train_mask.cpu().numpy() if hasattr(self.data[node_type], 'train_mask') else np.zeros_like(labels, dtype=bool)
        val_mask = self.data[node_type].val_mask.cpu().numpy() if hasattr(self.data[node_type], 'val_mask') else np.zeros_like(labels, dtype=bool)
        test_mask = self.data[node_type].test_mask.cpu().numpy() if hasattr(self.data[node_type], 'test_mask') else np.zeros_like(labels, dtype=bool)

        embeddings = self.embeddings.get(node_type)
        if embeddings is None:
            print("   ⚠️  No embeddings for", node_type)
            return []

        # Labeled train set
        train_indices = np.where(train_mask & (labels >= 0))[0]
        if len(train_indices) == 0:
            print("   ⚠️  No training nodes for reference")
            return []

        train_embeddings = embeddings[train_indices]
        train_labels = labels[train_indices]

        predictions = []

        # A) Evaluation on test set (if labeled test nodes exist)
        test_indices = np.where(test_mask & (labels >= 0))[0]
        if len(test_indices) > 0:
            print(f"   • Evaluating {len(test_indices)} labeled test nodes")
            for idx in test_indices:
                query_emb = embeddings[idx].reshape(1, -1)
                sims = cosine_similarity(query_emb, train_embeddings)[0]
                nearest_idx = np.argmax(sims)
                predicted_class = int(train_labels[nearest_idx])
                true_class = int(labels[idx])
                confidence = float(sims[nearest_idx])
                predictions.append({
                    'prediction_type': 'node_classification_eval',
                    'subject': self.get_node_info(node_type, idx),
                    'predicted_class': predicted_class,
                    'true_class': true_class,
                    'correct': predicted_class == true_class,
                    'confidence': confidence,
                    'nearest_labeled_node': self.get_node_info(node_type, int(train_indices[nearest_idx])),
                    'computed_on': datetime.now().isoformat()
                })
            acc = sum(p['correct'] for p in predictions if p['prediction_type']=='node_classification_eval') / max(1, len([p for p in predictions if p['prediction_type']=='node_classification_eval']))
            print(f"   ✓ Eval accuracy: {acc:.2%}")

        # B) Predict unlabeled nodes (nearest labeled neighbor)
        unlabeled_mask = (labels < 0)
        candidate_mask = unlabeled_mask & (~train_mask)  # avoid predicting on train
        unlabeled_indices = np.where(candidate_mask)[0]
        if len(unlabeled_indices) == 0:
            print("   ⚠️  No unlabeled nodes to predict")
        else:
            print(f"   • Predicting {min(len(unlabeled_indices), top_k)} unlabeled nodes (nearest-neighbor)")
            # limit to top_k
            sample_indices = unlabeled_indices[:top_k] if len(unlabeled_indices) > top_k else unlabeled_indices
            for idx in sample_indices:
                query_emb = embeddings[idx].reshape(1, -1)
                sims = cosine_similarity(query_emb, train_embeddings)[0]
                nearest_idx = np.argmax(sims)
                predicted_class = int(train_labels[nearest_idx])
                confidence = float(sims[nearest_idx])
                predictions.append({
                    'prediction_type': 'node_classification_unlabeled',
                    'subject': self.get_node_info(node_type, idx),
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'nearest_labeled_node': self.get_node_info(node_type, int(train_indices[nearest_idx])),
                    'computed_on': datetime.now().isoformat()
                })

        print(f"   ✓ Generated {len(predictions)} total node-class predictions for {node_type}")
        return predictions
    
    def generate_cluster_summaries(self, node_type: str) -> List[Dict]:
        """
        Generate cluster summaries for LLM interpretation
        """
        cluster_file = self.output_dir.parent / 'embedding_analysis' / f'cluster_summaries_{node_type}.json'
        
        if not cluster_file.exists():
            print(f"⚠️  Cluster summaries not found: {cluster_file}")
            return []
        
        print(f"\n📊 Generating cluster summaries for {node_type}")
        
        with open(cluster_file, 'r') as f:
            cluster_data = json.load(f)
        
        summaries = []
        
        for cluster in cluster_data:
            # Get representative members (highest frequency)
            members = sorted(cluster['members'], key=lambda x: x.get('frequency', 0), reverse=True)
            
            # Get common relations
            common_relations = self._get_common_relations(node_type, [m['node_id'] for m in members[:10]])
            
            # Get temporal patterns
            temporal_info = self._analyze_temporal_patterns([m for m in members if m.get('first_seen')])
            
            summary = {
                'prediction_type': 'cluster_summary',
                'cluster_id': cluster['cluster_id'],
                'node_type': node_type,
                'cluster_statistics': {
                    'size': cluster['size'],
                    'cohesion': cluster['cohesion'],
                    'member_count': cluster['member_count']
                },
                'representative_members': members[:10],  # Top 10 by frequency
                'common_relations': common_relations,
                'temporal_patterns': temporal_info,
                'meta_path_support': self._get_cluster_metapath_support(node_type, cluster['cluster_id']),
                'computed_on': datetime.now().isoformat()
            }
            
            summaries.append(summary)
        
        print(f"   ✓ Generated {len(summaries)} cluster summaries")
        return summaries
    
    def _get_common_relations(self, node_type: str, node_ids: List[int]) -> List[Dict]:
        """Find most common relations for a set of nodes"""
        relation_counts = defaultdict(int)
        
        # Convert node_ids to local indices
        id_to_idx = {nid: idx for idx, nid in enumerate(self.node_ids.get(node_type, []))}
        local_indices = [id_to_idx[nid] for nid in node_ids if nid in id_to_idx]
        
        if not local_indices:
            return []
        
        # Count relations
        for edge_type in self.data.edge_types:
            src_t, rel, dst_t = edge_type
            
            if src_t == node_type:
                edge_index = self.data[edge_type].edge_index.cpu().numpy()
                for idx in local_indices:
                    count = int(np.sum(edge_index[0] == idx))  # Convert to Python int
                    if count > 0:
                        relation_counts[f"{rel}->{dst_t}"] += count
            
            if dst_t == node_type:
                edge_index = self.data[edge_type].edge_index.cpu().numpy()
                for idx in local_indices:
                    count = int(np.sum(edge_index[1] == idx))  # Convert to Python int
                    if count > 0:
                        relation_counts[f"{src_t}->{rel}"] += count
        
        # Return top 5 as list of dicts (JSON-serializable)
        top_relations = sorted(relation_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return [{'relation': rel, 'count': int(cnt)} for rel, cnt in top_relations]
    
    def _analyze_temporal_patterns(self, members: List[Dict]) -> Dict:
        """Analyze temporal patterns in cluster members"""
        if not members:
            return {}
        
        first_seen_dates = [m['first_seen'] for m in members if m.get('first_seen')]
        last_seen_dates = [m['last_seen'] for m in members if m.get('last_seen')]
        
        return {
            'earliest_activity': min(first_seen_dates) if first_seen_dates else None,
            'latest_activity': max(last_seen_dates) if last_seen_dates else None,
            'activity_span_days': self._compute_date_span(first_seen_dates, last_seen_dates),
            'num_active_members': len([m for m in members if m.get('frequency', 0) > 0])
        }
    
    def _compute_date_span(self, first_dates: List[str], last_dates: List[str]) -> Optional[int]:
        """Compute span in days between earliest and latest dates"""
        if not first_dates or not last_dates:
            return None
        
        try:
            from datetime import datetime
            earliest = min([datetime.fromisoformat(d.replace('Z', '+00:00')) for d in first_dates])
            latest = max([datetime.fromisoformat(d.replace('Z', '+00:00')) for d in last_dates])
            return (latest - earliest).days
        except:
            return None
    
    def _get_cluster_metapath_support(self, node_type: str, cluster_id: int) -> List[Dict]:
        """Get meta-path support for clusters with safe fallbacks for metapath_ids -> labels"""
        weights_data = self.metapath_weights.get(node_type, {})
        weights = weights_data.get('weights', []) or []
        metapaths = weights_data.get('metapaths') or []
        metapath_ids = weights_data.get('metapath_ids') or []

        # fallback: try loading metapath_report.json for node_type
        if not metapaths and metapath_ids:
            report_candidates = [
                self.metapath_dir / node_type / "metapath_report.json",
                self.metapath_dir / "metapath_report.json",
                self.metapath_dir.parent / "metapath_report.json",
            ]
            id2path = {}
            for rc in report_candidates:
                if rc.exists():
                    try:
                        r = json.loads(rc.read_text())
                        for mp in r.get("metapaths", []):
                            if "id" in mp and "path" in mp:
                                id2path[mp["id"]] = mp["path"]
                    except Exception:
                        pass
                    break
            metapaths = [ id2path.get(mid, mid) for mid in metapath_ids ]

        if not metapaths and weights:
            metapaths = [f"metapath_{i}" for i in range(len(weights))]

        support = []
        L = min(len(weights), len(metapaths))
        for i in range(L):
            try:
                w = float(weights[i])
            except Exception:
                continue
            label = metapaths[i]
            if isinstance(label, list):
                path_pattern = " -> ".join(label)
                desc = self._describe_metapath(label)
            else:
                path_pattern = str(label)
                desc = self._describe_metapath([path_pattern])
            support.append({
                'mp_id': metapath_ids[i] if i < len(metapath_ids) else f"P{i}",
                'path_pattern': path_pattern,
                'weight': float(w),
                'description': desc
            })

        support.sort(key=lambda x: x['weight'], reverse=True)
        return support[:5]
    
    def generate_all_predictions(self, 
                                link_top_k: int = 100,
                                link_threshold: float = 0.5,
                                node_top_k: int = 50):
        """
        Generate all predictions and save to JSON
        """
        print("\n" + "="*70)
        print("PREDICTION GENERATOR")
        print("="*70)
        
        all_predictions = {
            'link_predictions': {},
            'node_classifications': {},
            'cluster_summaries': {},
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'model_path': str(self.model_path),
                'graph_path': str(self.graph_path)
            }
        }
        
        # 1. Link predictions (for meta-path edge types)
        print("\n📍 1. LINK PREDICTIONS")
        print("-" * 70)
        
        for edge_type in self.data.edge_types:
            src_type, rel, dst_type = edge_type
            
            # Skip meta-path edges (we want to predict semantic edges)
            if 'metapath_sim' in rel:
                continue
            
            predictions = self.predict_links(
                edge_type, 
                top_k=link_top_k,
                threshold=link_threshold
            )
            
            if predictions:
                key = f"{src_type}-{rel}-{dst_type}"
                all_predictions['link_predictions'][key] = predictions
        
        # 2. Node classifications
        print("\n📍 2. NODE CLASSIFICATIONS")
        print("-" * 70)
        
        for node_type in self.embeddings.keys():
            predictions = self.predict_node_classes(node_type, top_k=node_top_k)
            
            if predictions:
                all_predictions['node_classifications'][node_type] = predictions
        
        # 3. Cluster summaries
        print("\n📍 3. CLUSTER SUMMARIES")
        print("-" * 70)
        
        for node_type in self.embeddings.keys():
            summaries = self.generate_cluster_summaries(node_type)
            
            if summaries:
                all_predictions['cluster_summaries'][node_type] = summaries
        
        # Save all predictions
        output_file = self.output_dir / 'predictions_for_llm.json'
        with open(output_file, 'w') as f:
            json.dump(all_predictions, f, indent=2, cls=NumpyEncoder)
        
        print("\n" + "="*70)
        print(f"✅ PREDICTIONS SAVED: {output_file}")
        print("="*70)
        
        # Print summary
        print("\n📊 Summary:")
        print(f"   • Link predictions: {sum(len(v) for v in all_predictions['link_predictions'].values())} total")
        print(f"   • Node classifications: {sum(len(v) for v in all_predictions['node_classifications'].values())} total")
        print(f"   • Cluster summaries: {sum(len(v) for v in all_predictions['cluster_summaries'].values())} total")
        
        return all_predictions


def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions for LLM interpretation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python generate_predictions.py \\
    --model gnn_output/best_model.pt \\
    --graph enhanced_graph/enhanced_hetero_graph.pt \\
    --embeddings gnn_output/embeddings \\
    --output predictions

Output:
  predictions/
  └── predictions_for_llm.json  (all predictions in LLM format)
        """
    )
    
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--graph', required=True, help='Path to enhanced graph')
    parser.add_argument('--embeddings', required=True, help='Embeddings directory')
    parser.add_argument('--hin-dir', default='hin_out', help='HIN directory')
    parser.add_argument('--metapath-dir', default='metapath_out_per_ioc', help='Meta-path directory')
    parser.add_argument('--output', default='predictions', help='Output directory')
    
    parser.add_argument('--link-top-k', type=int, default=100, help='Top-k link predictions')
    parser.add_argument('--link-threshold', type=float, default=0.5, help='Link confidence threshold')
    parser.add_argument('--node-top-k', type=int, default=50, help='Top-k node predictions')
    
    args = parser.parse_args()
    
    generator = PredictionGenerator(
        model_path=args.model,
        graph_path=args.graph,
        embeddings_dir=args.embeddings,
        hin_dir=args.hin_dir,
        metapath_dir=args.metapath_dir,
        output_dir=args.output
    )
    
    generator.generate_all_predictions(
        link_top_k=args.link_top_k,
        link_threshold=args.link_threshold,
        node_top_k=args.node_top_k
    )
    
    print("\n🎯 Next step:")
    print(f"   Send predictions to LLM: python interpret_with_llm.py --predictions {args.output}/predictions_for_llm.json")


if __name__ == "__main__":
    main()