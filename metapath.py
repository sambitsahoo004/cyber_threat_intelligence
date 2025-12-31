#!/usr/bin/env python3
"""
Meta-Path Analysis Pipeline for HIN
Implements HINTI-style meta-path computation, PathSim similarity, and weight learning
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.sparse import csr_matrix, load_npz, save_npz, eye
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Optional: PyTorch for weight learning
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️  PyTorch not available - weight learning disabled")

# -------------------------
# Helpers: Reverse Relation & Pruning
# -------------------------
def reverse_relation_name(rel: str) -> str:
    """Return the reverse of a relation (same mapping used in build_hin)."""
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
        'communicates_with': 'communicates_with',
        'related_to': 'related_to'
    }
    return mapping.get(rel, f"{rel}_rev")


def prune_sparse_csr(mat, tol=1e-9, topk=100):
    """Drop tiny entries or keep top-k per row to control sparsity."""
    from scipy.sparse import csr_matrix
    if mat is None:
        return None
    mat = mat.tocsr().copy()
    mat.data[mat.data < tol] = 0
    mat.eliminate_zeros()

    if topk is None:
        return mat
    rows, cols, data = [], [], []
    for i in range(mat.shape[0]):
        start, end = mat.indptr[i], mat.indptr[i+1]
        if end - start == 0:
            continue
        idx = mat.indices[start:end]
        val = mat.data[start:end]
        if len(val) > topk:
            top_idx = np.argpartition(val, -topk)[-topk:]
            idx, val = idx[top_idx], val[top_idx]
        rows.extend([i]*len(idx))
        cols.extend(idx.tolist())
        data.extend(val.tolist())
    return csr_matrix((np.array(data), (np.array(rows), np.array(cols))), shape=mat.shape)

# -------------------------
# Meta-Path Definitions
# -------------------------
@dataclass
class MetaPath:
    """Meta-path definition"""
    id: str
    name: str
    path: List[Tuple[str, str, str]]  # [(src_type, relation, dst_type), ...]
    description: str = ""
    
    def __repr__(self):
        path_str = " → ".join([f"{s}-{r}->{d}" for s, r, d in self.path])
        return f"MetaPath({self.id}: {path_str})"
    
    def is_symmetric(self) -> bool:
        """Check if meta-path is symmetric (starts and ends with same type)"""
        if not self.path:
            return False
        return self.path[0][0] == self.path[-1][2]


# Default meta-paths based on HINTI framework
DEFAULT_METAPATHS = [
    # Basic symmetric paths (length 2)
    MetaPath("P1", "Attacker-Attacker", 
             [("ThreatActor", "uses", "Malware"), 
              ("Malware", "used_by", "ThreatActor")],
             "Attackers connected via shared malware"),
    
    MetaPath("P2", "Device-Device",
             [("Device", "connects_to", "Domain"),
              ("Domain", "connected_from", "Device")],
             "Devices connected via shared domains"),
    
    MetaPath("P3", "Vulnerability-Vulnerability",
             [("Vulnerability", "exploited_by", "Malware"),
              ("Malware", "exploits", "Vulnerability")],
             "Vulnerabilities connected via shared malware"),
    
    MetaPath("P4", "Attacker-Vulnerability-Attacker",
             [("ThreatActor", "exploits", "Vulnerability"),
              ("Vulnerability", "exploited_by", "ThreatActor")],
             "Attackers targeting same vulnerabilities"),
    
    MetaPath("P5", "Attacker-Device-Attacker",
             [("ThreatActor", "targets", "Device"),
              ("Device", "targeted_by", "ThreatActor")],
             "Attackers targeting same devices"),
    
    # Medium paths (length 3)
    MetaPath("P6", "Device-File-Device",
             [("Device", "contains", "File"),
              ("File", "contained_in", "Device")],
             "Devices sharing files"),
    
    MetaPath("P7", "Device-Platform-Device",
             [("Device", "runs", "Platform"),
              ("Platform", "runs_on", "Device")],
             "Devices sharing platforms"),
    
    MetaPath("P8", "Vulnerability-File-Vulnerability",
             [("Vulnerability", "affects", "File"),
              ("File", "affected_by", "Vulnerability")],
             "Vulnerabilities affecting same files"),
    
    MetaPath("P9", "Vulnerability-Type-Vulnerability",
             [("Vulnerability", "has_type", "VulnType"),
              ("VulnType", "type_of", "Vulnerability")],
             "Vulnerabilities of same type"),
    
    # Longer paths (length 4+)
    MetaPath("P10", "Vulnerability-Device-Vulnerability",
             [("Vulnerability", "affects", "Device"),
              ("Device", "affected_by", "Vulnerability")],
             "Vulnerabilities on same device"),
    
    MetaPath("P12", "Attacker-Device-Platform-Device-Attacker",
             [("ThreatActor", "targets", "Device"),
              ("Device", "runs", "Platform"),
              ("Platform", "runs_on", "Device"),
              ("Device", "targeted_by", "ThreatActor")],
             "Complex attacker-device-platform path"),
]


# -------------------------
# Matrix Loader
# -------------------------
class MatrixLoader:
    """Load and manage adjacency matrices from HIN output"""
    
    def __init__(self, hin_dir: str):
        self.hin_dir = Path(hin_dir)
        self.relations_dir = self.hin_dir / "relations"
        self.nodes_dir = self.hin_dir / "nodes"
        
        # Load metadata
        self.relation_meta = self._load_relation_meta()
        self.node_meta = self._load_node_meta()
        self.type_index_map = self._load_type_index_map()
        
        # Cache for loaded matrices
        self.matrix_cache: Dict[Tuple[str, str, str], csr_matrix] = {}
        
    def _load_relation_meta(self) -> Dict:
        """Load relation metadata"""
        meta_file = self.relations_dir / "relation_meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Relation metadata not found: {meta_file}")
        with open(meta_file, 'r') as f:
            return json.load(f)
    
    def _load_node_meta(self) -> Dict:
        """Load node metadata"""
        meta_file = self.hin_dir / "node_meta.json"
        if not meta_file.exists():
            raise FileNotFoundError(f"Node metadata not found: {meta_file}")
        with open(meta_file, 'r') as f:
            return json.load(f)
    
    def _load_type_index_map(self) -> Dict:
        """Load type index mapping"""
        map_file = self.nodes_dir / "type_index_map.json"
        if not map_file.exists():
            print(f"⚠️  Type index map not found: {map_file}")
            return {}
        with open(map_file, 'r') as f:
            return json.load(f)
    
    def get_matrix(self, src_type: str, relation: str, dst_type: str) -> Optional[csr_matrix]:
        """Load adjacency matrix for given relation"""
        key = (src_type, relation, dst_type)
        
        # Check cache
        if key in self.matrix_cache:
            return self.matrix_cache[key]
        
        # Find matching file
        pattern = f"relation__{src_type}__{relation}__{dst_type}.npz"
        matrix_file = self.relations_dir / pattern
        
        if not matrix_file.exists():
            print(f"⚠️  Matrix not found: {pattern}")
            return None
        
        try:
            matrix = load_npz(matrix_file)
            self.matrix_cache[key] = matrix
            return matrix
        except Exception as e:
            print(f"❌ Error loading matrix {pattern}: {e}")
            return None
    
    def get_available_relations(self) -> List[Tuple[str, str, str]]:
        """Get list of available relation triplets"""
        relations = []
        for fname, meta in self.relation_meta.items():
            relations.append((meta['src_type'], meta['relation'], meta['dst_type']))
        return relations
    
    def get_node_count(self, node_type: str) -> int:
        """Get number of nodes of given type"""
        type_file = self.nodes_dir / f"{node_type}_ids.json"
        if not type_file.exists():
            return 0
        with open(type_file, 'r') as f:
            ids = json.load(f)
        return len(ids)

def enumerate_meta_paths(loader, start_type: str, max_mid: int = 2):
    """
    Unified enumerator for symmetric meta-paths starting and ending with `start_type`.

    - If max_mid >= 1: yields A–B–A patterns (one middle type).
    - If max_mid >= 2: additionally yields A–B–C–B–A patterns (two middle types).
    - Each returned object is a MetaPath(id, name, path, description) (same as your existing MetaPath).
    - Performs existence & non-empty checks on each adjacency matrix before yielding.
    """
    relations = loader.get_available_relations()  # list of (src, rel, dst)
    # Build quick adjacency index: src -> set(dst) and store relation names per edge
    out_edges = {}   # src -> list of (rel, dst)
    in_edges = {}    # dst -> list of (rel, src)
    for src, rel, dst in relations:
        out_edges.setdefault(src, []).append((rel, dst))
        in_edges.setdefault(dst, []).append((rel, src))

    metapaths = []
    mp_counter = 0

    # Helper to check all matrices in a path exist and non-empty
    def _valid_path(path):
        for s, rel, d in path:
            mat = loader.get_matrix(s, rel, d)
            if mat is None or getattr(mat, "nnz", 0) == 0:
                return False
        return True

    # ---------- A–B–A (one middle) ----------
    if max_mid >= 1:
        # find candidate middle types: neighbors of start_type (via out or in)
        candidate_mid_types = {dst for (_, dst) in out_edges.get(start_type, [])} | \
                              {src for (_, src) in in_edges.get(start_type, [])}
        for mid in sorted(candidate_mid_types):
            # find edges start->mid and mid->start (could be different relation names)
            forward = [(r, dst) for (r, dst) in out_edges.get(start_type, []) if dst == mid]
            backward = [(r, src) for (r, src) in in_edges.get(start_type, []) if src == mid]
            for r1, _ in forward:
                for r2, _ in backward:
                    path = [(start_type, r1, mid), (mid, r2, start_type)]
                    if not _valid_path(path):
                        continue
                    mp_id = f"{start_type}_{r1}_{mid}_{r2}_{start_type}_{mp_counter}"
                    mp_name = f"{start_type}-{r1}->{mid}-{r2}->{start_type}"
                    metapaths.append(MetaPath(mp_id, mp_name, path, "auto A–B–A"))
                    mp_counter += 1

    # ---------- A–B–C–B–A (two middle types) ----------
    if max_mid >= 2:
        # for every mid (B) that is neighbor of start, look for neighbors of B (C)
        candidate_mid_types = {dst for (_, dst) in out_edges.get(start_type, [])} | \
                              {src for (_, src) in in_edges.get(start_type, [])}
        for mid in sorted(candidate_mid_types):
            # neighbors of mid (excluding start_type to avoid degenerate A–B–A duplication)
            mid_neighbors = {dst for (r, dst) in out_edges.get(mid, []) if dst != start_type} | \
                            {src for (r, src) in in_edges.get(mid, []) if src != start_type}
            for mid2 in sorted(mid_neighbors):
                # Need relations: start->mid (r1), mid->mid2 (r2), mid2->mid (r3), mid->start (r4)
                f_ab = [(r, dst) for (r, dst) in out_edges.get(start_type, []) if dst == mid]
                f_bc = [(r, dst) for (r, dst) in out_edges.get(mid, []) if dst == mid2]
                b_cb = [(r, src) for (r, src) in in_edges.get(mid, []) if src == mid2]  # mid2->mid
                b_ba = [(r, src) for (r, src) in in_edges.get(start_type, []) if src == mid]

                # combine relation choices
                for r1, _ in f_ab:
                    for r2, _ in f_bc:
                        for r3, _ in b_cb:
                            for r4, _ in b_ba:
                                path = [
                                    (start_type, r1, mid),
                                    (mid, r2, mid2),
                                    (mid2, r3, mid),
                                    (mid, r4, start_type)
                                ]
                                if not _valid_path(path):
                                    continue
                                mp_id = f"{start_type}_{r1}_{mid}_{r2}_{mid2}_{r3}_{mid}_{r4}_{start_type}_{mp_counter}"
                                mp_name = f"{start_type}-{r1}->{mid}-{r2}->{mid2}-{r3}->{mid}-{r4}->{start_type}"
                                metapaths.append(MetaPath(mp_id, mp_name, path, "auto A–B–C–B–A"))
                                mp_counter += 1

    return metapaths

# -------------------------
# Meta-Path Computer
# -------------------------
class MetaPathComputer:
    """Compute commuting matrices for meta-paths"""
    
    def __init__(self, loader: MatrixLoader):
        self.loader = loader
        self.commuting_cache: Dict[str, csr_matrix] = {}
    
    def compute_commuting_matrix(self, metapath: MetaPath, 
                                 verbose: bool = True) -> Optional[csr_matrix]:
        """
        Compute commuting matrix for meta-path by multiplying adjacency matrices
        
        Returns: Sparse CSR matrix C where C[i,j] = number of paths from i to j
        """
        if metapath.id in self.commuting_cache:
            return self.commuting_cache[metapath.id]
        
        if verbose:
            print(f"\n📊 Computing commuting matrix for {metapath.id}: {metapath.name}")
        
        # Load matrices for each step
        matrices = []
        for src_type, relation, dst_type in metapath.path:
            mat = self.loader.get_matrix(src_type, relation, dst_type)
            if mat is None:
                print(f"  ❌ Missing matrix: {src_type}-{relation}->{dst_type}")
                return None
            matrices.append(mat)
            if verbose:
                print(f"  ✓ Loaded {src_type}-{relation}->{dst_type}: {mat.shape}")
        
        # Multiply matrices in sequence
        result = matrices[0].tocsr()
        for i, mat in enumerate(matrices[1:], 1):
            if verbose:
                print(f"  🔄 Step {i}/{len(matrices)-1}: {result.shape} @ {mat.shape}")
            result = result @ mat
            result = prune_sparse_csr(result, tol=1e-9, topk=100)

        
        if verbose:
            print(f"  ✓ Final commuting matrix: {result.shape}, nnz={result.nnz}, density={result.nnz/np.prod(result.shape):.6f}")
        
        # Cache result
        self.commuting_cache[metapath.id] = result
        return result
    
    def compute_all_commuting_matrices(self, metapaths: List[MetaPath]) -> Dict[str, csr_matrix]:
        """Compute commuting matrices for all meta-paths"""
        print("\n" + "="*60)
        print("COMPUTING COMMUTING MATRICES")
        print("="*60)
        
        results = {}
        for mp in metapaths:
            C = self.compute_commuting_matrix(mp, verbose=True)
            if C is not None:
                results[mp.id] = C
        
        print(f"\n✓ Computed {len(results)}/{len(metapaths)} commuting matrices")
        return results


# -------------------------
# PathSim Similarity
# -------------------------
class PathSimComputer:
    """Compute PathSim similarity from commuting matrices"""
    
    @staticmethod
    def compute_pathsim(C: csr_matrix, epsilon: float = 1e-10) -> csr_matrix:
        """
        Compute PathSim similarity: S[i,j] = 2*C[i,j] / (C[i,i] + C[j,j])
        Only computes for non-zero entries in C to maintain sparsity.
        Handles empty commuting matrices safely.
        """
        print(f"  🧮 Computing PathSim from commuting matrix {C.shape}")

        # Extract diagonal (C[i,i])
        diag = np.array(C.diagonal()).flatten()

        # For sparse computation, only process non-zero entries
        C_coo = C.tocoo()
        rows = C_coo.row
        cols = C_coo.col
        data = C_coo.data

        if data.size == 0:
            # empty commuting matrix -> return empty PathSim CSR
            S = csr_matrix(C.shape)
            print(f"  ✓ PathSim matrix: {S.shape}, nnz=0, density=0.000000")
            return S

        # Compute PathSim for each non-zero entry
        pathsim_data = []
        for i, j, c_ij in zip(rows, cols, data):
            denom = diag[i] + diag[j]
            if denom > epsilon:
                s_ij = 2.0 * c_ij / denom
            else:
                s_ij = 0.0
            pathsim_data.append(s_ij)

        # Build sparse matrix
        S = csr_matrix((pathsim_data, (rows, cols)), shape=C.shape)

        # Safe stats printing
        if S.data.size > 0:
            print(f"  ✓ PathSim matrix: {S.shape}, nnz={S.nnz}, density={S.nnz/np.prod(S.shape):.6f}")
            print(f"    Stats: min={S.data.min():.4f}, max={S.data.max():.4f}, mean={S.data.mean():.4f}")
        else:
            print(f"  ✓ PathSim matrix: {S.shape}, nnz=0, density=0.000000")

        return S

    
    @staticmethod
    def compute_all_pathsim(commuting_matrices: Dict[str, csr_matrix]) -> Dict[str, csr_matrix]:
        """Compute PathSim for all commuting matrices"""
        print("\n" + "="*60)
        print("COMPUTING PATHSIM SIMILARITIES")
        print("="*60)
        
        similarities = {}
        for mp_id, C in commuting_matrices.items():
            print(f"\n📐 PathSim for {mp_id}")
            S = PathSimComputer.compute_pathsim(C)
            if S.nnz > 0:
                similarities[mp_id] = S
            else:
                print(f"  ⚠️  Skipping empty PathSim for {mp_id}")

        
        print(f"\n✓ Computed {len(similarities)} PathSim matrices")
        return similarities


# -------------------------
# Training Data Preparation
# -------------------------
class TrainingDataPreparator:
    """Prepare positive/negative pairs for weight learning"""
    
    def __init__(self, loader: MatrixLoader, seed: int = 42):
        self.loader = loader
        self.rng = np.random.RandomState(seed)
    
    def create_proxy_labels(self, node_type: str, 
                           num_positives: int = 1000,
                           num_negatives: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create proxy training labels from existing edges
        
        Returns:
            positives: (N, 2) array of node pairs with similarity 1.0
            negatives: (M, 2) array of node pairs with similarity 0.0
        """
        print(f"\n🏷️  Creating proxy labels for node type: {node_type}")
        
        # Collect all edges for this node type as positives
        all_edges = []
        relations = self.loader.get_available_relations()
        
        for src_type, relation, dst_type in relations:
            if src_type == node_type and dst_type == node_type:
                mat = self.loader.get_matrix(src_type, relation, dst_type)
                if mat is not None:
                    edges_coo = mat.tocoo()
                    edges = list(zip(edges_coo.row, edges_coo.col))
                    all_edges.extend(edges)
        
        if not all_edges:
            print(f"  ⚠️  No edges found for {node_type}")
            return np.array([]), np.array([])
        
        # Sample positives
        all_edges = list(set(all_edges))  # Remove duplicates
        num_available = len(all_edges)
        num_positives = min(num_positives, num_available)
        
        sampled_idx = self.rng.choice(num_available, size=num_positives, replace=False)
        positives = np.array([all_edges[i] for i in sampled_idx])
        
        # Create negatives (random pairs not in positives)
        n_nodes = self.loader.get_node_count(node_type)
        edge_set = set(all_edges)
        
        negatives = []
        attempts = 0
        max_attempts = num_negatives * 10
        
        while len(negatives) < num_negatives and attempts < max_attempts:
            i = self.rng.randint(0, n_nodes)
            j = self.rng.randint(0, n_nodes)
            if i != j and (i, j) not in edge_set:
                negatives.append((i, j))
                edge_set.add((i, j))  # Avoid duplicates
            attempts += 1
        
        negatives = np.array(negatives)
        
        print(f"  ✓ Created {len(positives)} positive pairs, {len(negatives)} negative pairs")
        
        return positives, negatives


# -------------------------
# Weight Learning
# -------------------------
class MetaPathWeightLearner:
    """Learn optimal weights for meta-paths using BPR loss"""
    
    def __init__(self, num_metapaths: int):
        self.num_metapaths = num_metapaths
        self.weights_logits = None
        
        if TORCH_AVAILABLE:
            self.weights_logits = nn.Parameter(torch.zeros(num_metapaths))
            self.optimizer = None
    
    def get_weights(self) -> np.ndarray:
        """Get current softmax weights"""
        if not TORCH_AVAILABLE or self.weights_logits is None:
            return np.ones(self.num_metapaths) / self.num_metapaths
        
        with torch.no_grad():
            w = torch.softmax(self.weights_logits, dim=0).cpu().numpy()
        return w
    
    def compute_combined_similarity(self, similarities: List[csr_matrix], 
                                    weights: Optional[np.ndarray] = None) -> csr_matrix:
        """Compute weighted combination of similarities"""
        if weights is None:
            weights = self.get_weights()
        
        # Ensure all matrices have same shape
        shape = similarities[0].shape
        combined = csr_matrix(shape, dtype=np.float32)
        
        for w, S in zip(weights, similarities):
            if S.shape == shape:
                combined = combined + w * S
        
        return combined
    
    def train(self, similarities: Dict[str, csr_matrix],
             positives: np.ndarray, negatives: np.ndarray,
             epochs: int = 100, lr: float = 0.01,
             verbose: bool = True) -> Dict:
        """
        Train meta-path weights using BPR loss
        
        BPR Loss: -log(sigmoid(S(i,j+) - S(i,j-)))
        where j+ is positive, j- is negative
        """
        if not TORCH_AVAILABLE:
            print("⚠️  PyTorch not available - returning uniform weights")
            weights = np.ones(self.num_metapaths) / self.num_metapaths
            return {'weights': weights.tolist(), 'loss_history': []}
        
        print(f"\n🎓 Training meta-path weights ({epochs} epochs, lr={lr})")
        
        # Setup optimizer
        self.optimizer = optim.Adam([self.weights_logits], lr=lr)
        
        # --- FIXED TRAINING LOOP (PyTorch gradient-safe) ---

        # Convert similarities dict to ordered list
        mp_ids = sorted(similarities.keys())
        S_list = [similarities[mp_id].tocsr() for mp_id in mp_ids]

        device = torch.device('cpu')
        self.weights_logits = nn.Parameter(torch.zeros(self.num_metapaths)).to(device)
        self.optimizer = optim.Adam([self.weights_logits], lr=lr)

        loss_history = []
        M = len(S_list)
        n_pos = len(positives)
        n_neg = len(negatives)
        batch_size = min(100, n_pos) if n_pos > 0 else 0

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            w = torch.softmax(self.weights_logits, dim=0)

            if batch_size == 0:
                break

            # Random mini-batch
            pos_idx = np.random.choice(n_pos, batch_size, replace=False)
            neg_idx = np.random.choice(n_neg, batch_size, replace=False)
            pos_pairs = positives[pos_idx]
            neg_pairs = negatives[neg_idx]

            # Gather per-metapath scores for each pair
            S_pos_vals = []
            S_neg_vals = []
            for S in S_list:
                pos_vals = np.array([S[i, j] for (i, j) in pos_pairs], dtype=np.float32)
                neg_vals = np.array([S[i, j] for (i, j) in neg_pairs], dtype=np.float32)
                S_pos_vals.append(pos_vals)
                S_neg_vals.append(neg_vals)

            # Convert to tensors (M, batch)
            S_pos_tensor = torch.from_numpy(np.stack(S_pos_vals, axis=0)).to(device)
            S_neg_tensor = torch.from_numpy(np.stack(S_neg_vals, axis=0)).to(device)

            # Weighted similarity per pair
            s_pos = (w.unsqueeze(1) * S_pos_tensor).sum(dim=0)
            s_neg = (w.unsqueeze(1) * S_neg_tensor).sum(dim=0)

            # BPR loss
            loss_tensor = -torch.log(torch.sigmoid(s_pos - s_neg + 1e-10) + 1e-12).mean()

            # Backprop
            loss_tensor.backward()
            self.optimizer.step()

            loss_history.append(float(loss_tensor.item()))
            if verbose and (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{epochs}: loss={loss_tensor.item():.4f}, weights={w.detach().cpu().numpy().round(3)}")

        # Final weights
        final_weights = torch.softmax(self.weights_logits, dim=0).detach().cpu().numpy()
        return {
            'weights': final_weights.tolist(),
            'metapath_ids': mp_ids,
            'loss_history': loss_history,
            'num_epochs': epochs
        }

# -------------------------
# Pipeline
# -------------------------
class MetaPathPipeline:
    """End-to-end meta-path analysis pipeline"""
    
    def __init__(self, hin_dir: str, output_dir: str):
        self.hin_dir = Path(hin_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Components
        self.loader = MatrixLoader(hin_dir)
        self.computer = MetaPathComputer(self.loader)
        self.pathsim = PathSimComputer()
        self.preparator = TrainingDataPreparator(self.loader)
        
        # Results
        self.metapaths: List[MetaPath] = []
        self.commuting_matrices: Dict[str, csr_matrix] = {}
        self.similarities: Dict[str, csr_matrix] = {}
        self.weights: Dict = {}
        self.combined_similarity: Optional[csr_matrix] = None
    
    def run(self, metapaths: Optional[List[MetaPath]] = None,
        target_node_type: str = "ThreatActor",
        mid_types: Optional[List[str]] = None,
        num_positives: int = 500,
        num_negatives: int = 1000,
        train_epochs: int = 100,
        learning_rate: float = 0.01):
        """Run complete pipeline (auto-enumerate A-R-B-R'-A meta-paths, compute, learn)."""

        print("\n" + "="*70)
        print("META-PATH ANALYSIS PIPELINE")
        print("="*70)

                # Step A: Define meta-paths (support ALL start types)
        if metapaths is None:
            print("🔎 Auto-generating meta-paths of form A–R–B–R'–A and A–B–C–B–A ...")

            all_relations = self.loader.get_available_relations()

            # Determine start types:
            # If user explicitly set target_node_type="ALL" (case-insensitive), enumerate all start types found in relations.
            # Otherwise keep the existing behaviour (single start type).
            if isinstance(target_node_type, str) and target_node_type.lower() == "all":
                start_types = sorted({src for (src, _, _) in all_relations} | {dst for (_, _, dst) in all_relations})
                print(f"  ✓ Running auto-enumeration for ALL start types: {start_types}")
            else:
                start_types = [target_node_type]

            meta_list = []
            for start in start_types:
                # Use the unified enumerator which auto-discovers middle types
                cand = enumerate_meta_paths(self.loader, start_type=start, max_mid=2)

                # Filter out any meta-paths that reference missing/empty matrices (enumerator already checks,
                # but keep this defensive filter to be safe)
                filtered = []
                for mp in cand:
                    ok = True
                    for (src, rel, dst) in mp.path:
                        mat = self.loader.get_matrix(src, rel, dst)
                        if mat is None or getattr(mat, 'nnz', 0) == 0:
                            ok = False
                            break
                    if ok:
                        filtered.append(mp)

                if filtered:
                    print(f"  ✓ {len(filtered)} meta-paths added for start-type '{start}'")
                else:
                    print(f"  ⚠️  No valid meta-paths found for start-type '{start}'")

                meta_list.extend(filtered)

            self.metapaths = meta_list
        else:
            self.metapaths = metapaths

        if not self.metapaths:
            print("❌ No meta-paths available after filtering. Exiting pipeline.")
            return

        print(f"\n📋 Using {len(self.metapaths)} meta-paths:")
        for mp in self.metapaths:
            print(f"  {mp}")

        # Step B: Compute commuting matrices
        self.commuting_matrices = self.computer.compute_all_commuting_matrices(self.metapaths)
        # Save commuting matrices
        self._save_commuting_matrices()

        # Step C: Compute PathSim similarities
        self.similarities = self.pathsim.compute_all_pathsim(self.commuting_matrices)
        # Save similarities
        self._save_similarities()

        # Step D: Prepare training data
        positives, negatives = self.preparator.create_proxy_labels(
            target_node_type, num_positives, num_negatives
        )

        # Fallback: if explicit positives absent, derive from commuting matrices
        def collect_pairs_from_commuting(comms: Dict[str, csr_matrix], max_pos: int = 500):
            pairs = set()
            for C in comms.values():
                coo = C.tocoo()
                for i, j, v in zip(coo.row, coo.col, coo.data):
                    if i != j and v > 0:
                        pairs.add((int(i), int(j)))
                        if len(pairs) >= max_pos:
                            return np.array(list(pairs))
            return np.array(list(pairs))

        if len(positives) == 0:
            print("🔁 No explicit same-type positive edges found — deriving positives from commuting matrices.")
            derived = collect_pairs_from_commuting(self.commuting_matrices, max_pos=num_positives)
            if len(derived) > 0:
                positives = derived
                # build negatives by random sampling avoiding positives
                n_nodes = self.loader.get_node_count(target_node_type)
                if n_nodes == 0:
                    print("❌ No nodes found for target node type; skipping training and using uniform weights.")
                    positives = np.array([])
                else:
                    negs = []
                    pos_set = set(map(tuple, positives))
                    rng = np.random.RandomState(42)
                    attempts = 0
                    while len(negs) < num_negatives and attempts < num_negatives * 20:
                        i = int(rng.randint(0, n_nodes))
                        j = int(rng.randint(0, n_nodes))
                        if i != j and (i, j) not in pos_set:
                            negs.append((i, j))
                        attempts += 1
                    negatives = np.array(negs)

        # If still no positives, fall back to uniform weights
        if len(positives) == 0:
            print("⚠️  No training data - skipping weight learning")
            self.weights = {
                'weights': [1.0/len(self.metapaths)] * len(self.metapaths),
                'metapath_ids': [mp.id for mp in self.metapaths]
            }
            # build combined similarity by averaging (uniform)
            S_list = [self.similarities[mp.id] for mp in self.metapaths if mp.id in self.similarities]
            if S_list:
                avg = sum(S_list) / len(S_list)
                self.combined_similarity = avg
            else:
                self.combined_similarity = None
        else:
            # Step E: Learn weights
            learner = MetaPathWeightLearner(len(self.similarities))
            self.weights = learner.train(
                self.similarities, positives, negatives,
                epochs=train_epochs, lr=learning_rate
            )

            # Step F: Build combined similarity
            weights_array = np.array(self.weights['weights'])
            S_list = [self.similarities[mp_id] for mp_id in self.weights['metapath_ids']]
            self.combined_similarity = learner.compute_combined_similarity(S_list, weights_array)
            # Optional: prune combined similarity to keep sparse
            if self.combined_similarity is not None:
                self.combined_similarity = prune_sparse_csr(self.combined_similarity, tol=1e-9, topk=200)

        # Save results
        self._save_results()

        print("\n" + "="*70)
        print("✅ PIPELINE COMPLETE")
        print("="*70)
        print(f"\nOutputs saved to: {self.output_dir.absolute()}")

    
    def _save_commuting_matrices(self):
        """Save commuting matrices"""
        comm_dir = self.output_dir / "commuting_matrices"
        comm_dir.mkdir(exist_ok=True)
        
        for mp_id, C in self.commuting_matrices.items():
            save_npz(comm_dir / f"{mp_id}_commuting.npz", C)
        
        print(f"\n✓ Saved {len(self.commuting_matrices)} commuting matrices to {comm_dir.name}/")
    
    def _save_similarities(self):
        """Save PathSim similarity matrices"""
        sim_dir = self.output_dir / "similarities"
        sim_dir.mkdir(exist_ok=True)
        
        for mp_id, S in self.similarities.items():
            save_npz(sim_dir / f"{mp_id}_pathsim.npz", S)
        
        print(f"\n✓ Saved {len(self.similarities)} similarity matrices to {sim_dir.name}/")
    
    def _save_results(self):
        """Save final results and metadata"""
        
        # Save weights
        weights_file = self.output_dir / "metapath_weights.json"
        with open(weights_file, 'w') as f:
            json.dump(self.weights, f, indent=2)
        print(f"\n✓ Saved weights to {weights_file.name}")
        
        # Save combined similarity
        if self.combined_similarity is not None:
            combined_file = self.output_dir / "combined_similarity.npz"
            save_npz(combined_file, self.combined_similarity)
            print(f"✓ Saved combined similarity to {combined_file.name}")
        
        # Save summary report
        report = {
            'metapaths': [
                {
                    'id': mp.id,
                    'name': mp.name,
                    'path': [f"{s}-{r}->{d}" for s, r, d in mp.path],
                    'is_symmetric': mp.is_symmetric(),
                    'commuting_shape': self.commuting_matrices[mp.id].shape if mp.id in self.commuting_matrices else None,
                    'commuting_nnz': int(self.commuting_matrices[mp.id].nnz) if mp.id in self.commuting_matrices else 0,
                    'pathsim_nnz': int(self.similarities[mp.id].nnz) if mp.id in self.similarities else 0
                }
                for mp in self.metapaths if mp.id in self.similarities
            ],
            'weights': self.weights,
            'combined_similarity_nnz': int(self.combined_similarity.nnz) if self.combined_similarity is not None else 0
        }
        
        report_file = self.output_dir / "metapath_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"✓ Saved report to {report_file.name}")


# -------------------------
# CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Meta-Path Analysis Pipeline for HIN',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--hin-dir', '-i',
        default='hin_out',
        help='HIN output directory (default: hin_out)'
    )
    
    parser.add_argument(
        '--output', '-o',
        default='metapath_out',
        help='Output directory (default: metapath_out)'
    )
    
    parser.add_argument(
        '--target-type', '-t',
        default='ThreatActor',
        help='Target node type for training (default: ThreatActor)'
    )
    
    parser.add_argument(
        '--num-positives',
        type=int,
        default=500,
        help='Number of positive training pairs (default: 500)'
    )
    
    parser.add_argument(
        '--num-negatives',
        type=int,
        default=1000,
        help='Number of negative training pairs (default: 1000)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Training epochs (default: 100)'
    )
    
    parser.add_argument(
        '--lr',
        type=float,
        default=0.01,
        help='Learning rate (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    pipeline = MetaPathPipeline(args.hin_dir, args.output)
    pipeline.run(
        target_node_type=args.target_type,
        num_positives=args.num_positives,
        num_negatives=args.num_negatives,
        train_epochs=args.epochs,
        learning_rate=args.lr
    )
    
    print("\n✅ Meta-path analysis complete!")


if __name__ == "__main__":
    main()