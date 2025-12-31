#!/usr/bin/env python3
"""
Analyze and Visualize Node Embeddings
Performs:
1. Clustering (DBSCAN, KMeans)
2. Dimensionality reduction (t-SNE, UMAP)
3. Cluster quality metrics
4. Export cluster summaries for LLM
"""

import json
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score
    from sklearn.manifold import TSNE
    import umap
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  sklearn/umap not available - limited functionality")


class EmbeddingAnalyzer:
    """Analyze learned node embeddings"""
    
    def __init__(self, embeddings_dir: str, hin_dir: str, output_dir: str):
        self.embeddings_dir = Path(embeddings_dir)
        self.hin_dir = Path(hin_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load metadata
        self.node_meta = self._load_json(self.hin_dir / "node_meta.json")
        self.node_registry = self._load_json(self.hin_dir / "node_registry.json")
        self.type_index_map = self._load_json(self.hin_dir / "nodes" / "type_index_map.json")
        
        # Store embeddings and clusters
        self.embeddings = {}
        self.clusters = {}
    
    def _load_json(self, path: Path) -> Dict:
        if not path.exists():
            return {}
        with open(path, 'r') as f:
            return json.load(f)
    
    def load_embeddings(self, node_type: str) -> np.ndarray:
        """Load embeddings for a node type"""
        emb_file = self.embeddings_dir / f"embeddings_{node_type}.npy"
        if not emb_file.exists():
            print(f"⚠️  Embeddings not found: {emb_file}")
            return np.array([])
        
        embeddings = np.load(emb_file)
        self.embeddings[node_type] = embeddings
        print(f"✓ Loaded {node_type}: {embeddings.shape}")
        return embeddings
    
    def get_node_ids(self, node_type: str) -> List[int]:
        """Get ordered node IDs"""
        ids_file = self.hin_dir / "nodes" / f"{node_type}_ids.json"
        if not ids_file.exists():
            return []
        with open(ids_file, 'r') as f:
            return json.load(f)
    
    def cluster_dbscan(self, node_type: str, eps: float = 0.5, min_samples: int = 3):
        """Cluster embeddings using DBSCAN"""
        print(f"\n🔍 Clustering {node_type} with DBSCAN (eps={eps}, min_samples={min_samples})")
        
        embeddings = self.embeddings.get(node_type)
        if embeddings is None or embeddings.size == 0:
            print(f"⚠️  No embeddings for {node_type}")
            return
        
        # Normalize embeddings
        from sklearn.preprocessing import normalize
        embeddings_norm = normalize(embeddings, norm='l2')
        
        # Cluster
        clusterer = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clusterer.fit_predict(embeddings_norm)
        
        # Statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"   • Clusters found: {n_clusters}")
        print(f"   • Noise points: {n_noise}")
        
        # Compute silhouette score (excluding noise)
        if n_clusters > 1:
            mask = labels != -1
            if mask.sum() > 0:
                score = silhouette_score(embeddings_norm[mask], labels[mask], metric='cosine')
                print(f"   • Silhouette score: {score:.4f}")
        
        self.clusters[node_type] = labels
        return labels
    
    def cluster_kmeans(self, node_type: str, n_clusters: int = 10):
        """Cluster embeddings using KMeans"""
        print(f"\n🎯 Clustering {node_type} with KMeans (k={n_clusters})")
        
        embeddings = self.embeddings.get(node_type)
        if embeddings is None or embeddings.size == 0:
            print(f"⚠️  No embeddings for {node_type}")
            return
        
        from sklearn.preprocessing import normalize
        embeddings_norm = normalize(embeddings, norm='l2')
        
        # Cluster
        clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = clusterer.fit_predict(embeddings_norm)
        
        # Statistics
        print(f"   • Clusters: {n_clusters}")
        
        # Cluster sizes
        cluster_sizes = Counter(labels)
        print(f"   • Cluster sizes: {dict(cluster_sizes)}")
        
        # Silhouette score
        score = silhouette_score(embeddings_norm, labels, metric='cosine')
        print(f"   • Silhouette score: {score:.4f}")
        
        # Calinski-Harabasz score
        ch_score = calinski_harabasz_score(embeddings_norm, labels)
        print(f"   • Calinski-Harabasz score: {ch_score:.2f}")
        
        self.clusters[node_type] = labels
        return labels
    
    def visualize_embeddings(self, node_type: str, method: str = 'tsne', perplexity: int = 30):
        """Visualize embeddings in 2D"""
        print(f"\n📊 Visualizing {node_type} with {method.upper()}")
        
        embeddings = self.embeddings.get(node_type)
        labels = self.clusters.get(node_type)
        
        if embeddings is None or embeddings.size == 0:
            print(f"⚠️  No embeddings for {node_type}")
            return
        
        # Reduce dimensionality
        if method == 'tsne':
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        elif method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        else:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        if labels is not None:
            # Color by cluster
            scatter = plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                c=labels,
                cmap='tab20',
                alpha=0.6,
                s=20
            )
            plt.colorbar(scatter, label='Cluster')
        else:
            plt.scatter(
                embeddings_2d[:, 0],
                embeddings_2d[:, 1],
                alpha=0.6,
                s=20
            )
        
        plt.title(f'{node_type} Embeddings ({method.upper()})')
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.tight_layout()
        
        # Save
        viz_file = self.output_dir / f'{node_type}_viz_{method}.png'
        plt.savefig(viz_file, dpi=150)
        print(f"   ✓ Saved visualization: {viz_file.name}")
        plt.close()
    
    def export_cluster_summaries(self, node_type: str):
        """Export cluster summaries for LLM interpretation"""
        print(f"\n📋 Exporting cluster summaries for {node_type}")
        
        labels = self.clusters.get(node_type)
        if labels is None:
            print(f"⚠️  No clusters for {node_type}")
            return
        
        node_ids = self.get_node_ids(node_type)
        embeddings = self.embeddings[node_type]
        
        # Group by cluster
        clusters_dict = defaultdict(list)
        for idx, (node_id, cluster_label) in enumerate(zip(node_ids, labels)):
            if cluster_label == -1:
                continue  # Skip noise
            
            clusters_dict[int(cluster_label)].append({
                'node_id': int(node_id),
                'local_idx': int(idx),  # CRITICAL: Save local index!
                'embedding': embeddings[idx].tolist()
            })
        
        # Compute cluster statistics
        cluster_summaries = []
        
        for cluster_id, members in clusters_dict.items():
            # Get member info
            member_info = []
            for m in members:
                node_id = m['node_id']
                local_idx = m['local_idx']
                meta = self.node_meta.get(str(node_id), {})
                
                member_info.append({
                    'node_id': node_id,
                    'local_idx': local_idx,  # CRITICAL: Include in member info!
                    'canonical': meta.get('canonical', f'node_{node_id}'),
                    'frequency': meta.get('frequency', 0),
                    'first_seen': meta.get('first_seen', ''),
                    'last_seen': meta.get('last_seen', '')
                })
            
            # Compute cohesion (avg pairwise similarity)
            if len(members) > 1:
                embeddings_cluster = np.array([m['embedding'] for m in members])
                from sklearn.metrics.pairwise import cosine_similarity
                sim_matrix = cosine_similarity(embeddings_cluster)
                cohesion = (sim_matrix.sum() - len(members)) / (len(members) * (len(members) - 1))
            else:
                cohesion = 1.0
            
            # Compute centroid
            centroid = np.mean([m['embedding'] for m in members], axis=0)
            
            summary = {
                'cluster_id': cluster_id,
                'node_type': node_type,
                'size': len(members),
                'cohesion': float(cohesion),
                'centroid': centroid.tolist(),
                'members': member_info[:50],  # Limit to 50 for LLM context
                'member_count': len(member_info)
            }
            
            cluster_summaries.append(summary)
        
        # Save
        output_file = self.output_dir / f'cluster_summaries_{node_type}.json'
        with open(output_file, 'w') as f:
            json.dump(cluster_summaries, f, indent=2)
        
        print(f"   ✓ Saved {len(cluster_summaries)} cluster summaries: {output_file.name}")
        
        return cluster_summaries


def main():
    parser = argparse.ArgumentParser(description='Analyze node embeddings')
    
    parser.add_argument('--embeddings', required=True, help='Embeddings directory')
    parser.add_argument('--hin-dir', default='hin_out', help='HIN directory')
    parser.add_argument('--output', default='embedding_analysis', help='Output directory')
    
    parser.add_argument('--method', choices=['dbscan', 'kmeans', 'both'], default='both')
    parser.add_argument('--eps', type=float, default=0.5, help='DBSCAN eps')
    parser.add_argument('--min-samples', type=int, default=3, help='DBSCAN min_samples')
    parser.add_argument('--k', type=int, default=10, help='KMeans k')
    
    parser.add_argument('--visualize', action='store_true', help='Create visualizations')
    parser.add_argument('--viz-method', choices=['tsne', 'umap', 'pca'], default='tsne')
    
    args = parser.parse_args()
    
    if not SKLEARN_AVAILABLE:
        print("❌ sklearn required: pip install scikit-learn umap-learn")
        return
    
    print("\n" + "="*70)
    print("EMBEDDING ANALYZER")
    print("="*70)
    
    analyzer = EmbeddingAnalyzer(args.embeddings, args.hin_dir, args.output)
    
    # Get all node types
    node_types = list(analyzer.type_index_map.keys())
    print(f"\nNode types: {node_types}")
    
    for node_type in node_types:
        # Load embeddings
        embeddings = analyzer.load_embeddings(node_type)
        if embeddings.size == 0:
            continue
        
        # Clustering
        if args.method in ['dbscan', 'both']:
            analyzer.cluster_dbscan(node_type, eps=args.eps, min_samples=args.min_samples)
        
        if args.method in ['kmeans', 'both']:
            analyzer.cluster_kmeans(node_type, n_clusters=args.k)
        
        # Visualization
        if args.visualize:
            analyzer.visualize_embeddings(node_type, method=args.viz_method)
        
        # Export summaries
        analyzer.export_cluster_summaries(node_type)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()