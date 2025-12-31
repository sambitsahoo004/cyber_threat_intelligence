#!/usr/bin/env python3
"""
Create Node Labels from Cluster Assignments
Uses embedding clusters as pseudo-labels for semi-supervised learning
"""

import json
import torch
import numpy as np
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split
from collections import Counter

def create_labels_from_clusters(
    graph_path: str,
    cluster_dir: str,
    hin_dir: str,
    output_path: str,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    min_cluster_size: int = 3
):
    """
    Add cluster-based labels to graph
    
    Args:
        graph_path: Path to enhanced_hetero_graph.pt
        cluster_dir: Directory with cluster_summaries_*.json files
        hin_dir: HIN directory with node ID mappings
        output_path: Where to save labeled graph
        train_ratio: Fraction of nodes for training
        val_ratio: Fraction of nodes for validation
        min_cluster_size: Minimum cluster size to include (default: 3 for stratified split)
    """
    
    print("\n" + "="*70)
    print("CREATING NODE LABELS FROM CLUSTERS")
    print("="*70)
    
    # Load graph
    print(f"\n📦 Loading graph: {graph_path}")
    data = torch.load(graph_path, weights_only=False)
    
    cluster_dir = Path(cluster_dir)
    hin_dir = Path(hin_dir)
    
    total_labeled = 0
    total_filtered = 0
    
    # Process each node type
    for node_type in data.node_types:
        cluster_file = cluster_dir / f'cluster_summaries_{node_type}.json'
        
        if not cluster_file.exists():
            print(f"⚠️  No clusters for {node_type}, skipping...")
            continue
        
        print(f"\n🏷️  Processing {node_type}...")
        
        # Load cluster assignments
        with open(cluster_file, 'r') as f:
            clusters = json.load(f)
        
        if len(clusters) == 0:
            print(f"   ⚠️  Empty cluster file, skipping...")
            continue
        
        # Load node ID mapping (node_id -> local_index)
        ids_file = hin_dir / "nodes" / f"{node_type}_ids.json"
        if not ids_file.exists():
            print(f"   ⚠️  No ID mapping file: {ids_file}, skipping...")
            continue
        
        with open(ids_file, 'r') as f:
            node_ids = json.load(f)  # Ordered list: [node_id_0, node_id_1, ...]
        
        # Create mapping: node_id -> local_index
        id_to_idx = {int(node_id): idx for idx, node_id in enumerate(node_ids)}
        
        # Create label mapping: local_index -> cluster_id
        num_nodes = data[node_type].num_nodes
        labels = np.full(num_nodes, -1, dtype=np.int64)  # -1 = unlabeled
        
        num_members_found = 0
        num_members_missing = 0
        
        # First pass: assign all labels
        for cluster in clusters:
            cluster_id = cluster['cluster_id']
            members = cluster.get('members', [])
            
            for member in members:
                # Try to get node_id
                node_id = member.get('node_id')
                
                if node_id is None:
                    num_members_missing += 1
                    continue
                
                # Convert to int if needed
                node_id = int(node_id)
                
                # Look up local index
                local_idx = id_to_idx.get(node_id)
                
                if local_idx is None:
                    # Try using local_idx directly if available
                    local_idx = member.get('local_idx')
                    if local_idx is None:
                        num_members_missing += 1
                        continue
                
                if local_idx >= num_nodes:
                    print(f"   ⚠️  Invalid index {local_idx} >= {num_nodes}")
                    continue
                
                # Assign label
                labels[local_idx] = cluster_id
                num_members_found += 1
        
        # Debug info
        print(f"   • Total nodes in graph: {num_nodes}")
        print(f"   • Node IDs in mapping: {len(node_ids)}")
        print(f"   • Clusters: {len(clusters)}")
        print(f"   • Members found: {num_members_found}")
        if num_members_missing > 0:
            print(f"   • Members missing: {num_members_missing}")
        
        # Count labeled nodes
        labeled_mask = labels >= 0
        num_labeled_raw = labeled_mask.sum()
        
        if num_labeled_raw == 0:
            print(f"   ⚠️  No labeled nodes after mapping, skipping...")
            continue
        
        # Check cluster sizes and filter small clusters
        label_counts = Counter(labels[labeled_mask])
        print(f"\n   Cluster sizes before filtering:")
        for cluster_id, count in sorted(label_counts.items()):
            print(f"      Cluster {cluster_id}: {count} nodes")
        
        # Filter out clusters that are too small for stratified split
        small_clusters = [cid for cid, count in label_counts.items() if count < min_cluster_size]
        
        if small_clusters:
            print(f"\n   ⚠️  Filtering {len(small_clusters)} clusters with < {min_cluster_size} samples:")
            for cid in small_clusters:
                print(f"      Cluster {cid}: {label_counts[cid]} samples")
                # Set these labels to -1 (unlabeled)
                labels[labels == cid] = -1
                total_filtered += label_counts[cid]
        
        # Recount after filtering
        labeled_mask = labels >= 0
        num_labeled = labeled_mask.sum()
        num_clusters = len(set(labels[labeled_mask]))
        
        print(f"\n   After filtering:")
        print(f"   • Labeled nodes: {num_labeled} ({num_labeled/num_nodes*100:.1f}%)")
        print(f"   • Valid clusters: {num_clusters}")
        print(f"   • Filtered out: {num_labeled_raw - num_labeled} nodes")
        
        if num_labeled == 0:
            print(f"   ⚠️  No labeled nodes remaining after filtering, skipping...")
            continue
        
        if num_clusters < 2:
            print(f"   ⚠️  Need at least 2 clusters for classification, skipping...")
            continue
        
        # Remap cluster IDs to be contiguous (0, 1, 2, ...)
        unique_labels = sorted(set(labels[labeled_mask]))
        label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        
        # Apply remapping
        remapped_labels = labels.copy()
        for old_label, new_label in label_map.items():
            remapped_labels[labels == old_label] = new_label
        
        labels = remapped_labels
        
        print(f"   • Remapped cluster IDs: 0 to {len(unique_labels)-1}")
        
        # Create train/val/test splits
        labeled_indices = np.where(labeled_mask)[0]
        
        # Check if we have enough samples per class for stratification
        label_counts_final = Counter(labels[labeled_indices])
        min_samples_per_class = min(label_counts_final.values())
        
        print(f"\n   Final cluster size range: {min_samples_per_class} to {max(label_counts_final.values())}")
        
        # Use stratified split
        try:
            train_indices, temp_indices = train_test_split(
                labeled_indices,
                train_size=train_ratio,
                random_state=42,
                stratify=labels[labeled_indices]
            )
            
            val_size = val_ratio / (1 - train_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_size,
                random_state=42,
                stratify=labels[temp_indices]
            )
            
            print(f"   ✓ Using stratified split")
            
        except ValueError as e:
            print(f"   ⚠️  Stratified split failed: {e}")
            print(f"   ℹ️  Using random split instead")
            
            # Fallback to random split
            train_indices, temp_indices = train_test_split(
                labeled_indices,
                train_size=train_ratio,
                random_state=42
            )
            
            val_size = val_ratio / (1 - train_ratio)
            val_indices, test_indices = train_test_split(
                temp_indices,
                train_size=val_size,
                random_state=42
            )
        
        # Create masks
        train_mask = np.zeros(num_nodes, dtype=bool)
        val_mask = np.zeros(num_nodes, dtype=bool)
        test_mask = np.zeros(num_nodes, dtype=bool)
        
        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True
        
        # Add to graph
        data[node_type].y = torch.from_numpy(labels).long()
        data[node_type].train_mask = torch.from_numpy(train_mask)
        data[node_type].val_mask = torch.from_numpy(val_mask)
        data[node_type].test_mask = torch.from_numpy(test_mask)
        
        print(f"\n   ✓ Split created:")
        print(f"      - Train: {train_mask.sum()} ({train_mask.sum()/num_labeled*100:.1f}%)")
        print(f"      - Val: {val_mask.sum()} ({val_mask.sum()/num_labeled*100:.1f}%)")
        print(f"      - Test: {test_mask.sum()} ({test_mask.sum()/num_labeled*100:.1f}%)")
        
        total_labeled += num_labeled
    
    if total_labeled == 0:
        print("\n❌ No labels created for any node type!")
        print("\nPossible issues:")
        print("   1. All clusters are too small (< 3 samples)")
        print("   2. Node ID mappings don't match between cluster files and graph")
        print("   3. Check that analyze_embeddings.py ran successfully with local_idx")
        return None
    
    # Save labeled graph
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    torch.save(data, output_path)
    print(f"\n✅ Saved labeled graph: {output_path}")
    print(f"   • Total labeled nodes: {total_labeled}")
    if total_filtered > 0:
        print(f"   • Filtered small clusters: {total_filtered} nodes")
    
    return data


def inspect_cluster_file(cluster_file: str):
    """Debug helper to inspect cluster file structure"""
    print(f"\n🔍 Inspecting: {cluster_file}")
    
    with open(cluster_file, 'r') as f:
        clusters = json.load(f)
    
    if len(clusters) == 0:
        print("   ❌ Empty cluster list")
        return
    
    print(f"   • Number of clusters: {len(clusters)}")
    
    # Inspect first cluster
    first_cluster = clusters[0]
    print(f"\n   First cluster structure:")
    print(f"      Keys: {list(first_cluster.keys())}")
    
    if 'members' in first_cluster:
        members = first_cluster['members']
        print(f"      Number of members: {len(members)}")
        
        if len(members) > 0:
            first_member = members[0]
            print(f"      First member keys: {list(first_member.keys())}")
            print(f"      First member: {first_member}")
    
    # Check cluster sizes
    cluster_sizes = [len(c.get('members', [])) for c in clusters]
    print(f"\n   Cluster size statistics:")
    print(f"      Min: {min(cluster_sizes)}")
    print(f"      Max: {max(cluster_sizes)}")
    print(f"      Mean: {np.mean(cluster_sizes):.1f}")
    print(f"      Total members: {sum(cluster_sizes)}")
    
    # Count small clusters
    small_clusters = sum(1 for size in cluster_sizes if size < 3)
    if small_clusters > 0:
        print(f"\n   ⚠️  Warning: {small_clusters} clusters have < 3 samples")
        print(f"      These will be filtered out during label creation")


def main():
    parser = argparse.ArgumentParser(description='Create node labels from clusters')
    
    parser.add_argument('--graph', required=True, help='Path to enhanced_hetero_graph.pt')
    parser.add_argument('--clusters', required=True, help='Cluster summaries directory')
    parser.add_argument('--hin-dir', default='hin_out', help='HIN directory with ID mappings')
    parser.add_argument('--output', default='enhanced_graph/labeled_hetero_graph.pt',
                       help='Output path')
    parser.add_argument('--train-ratio', type=float, default=0.6,
                       help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.2,
                       help='Validation set ratio')
    parser.add_argument('--min-cluster-size', type=int, default=3,
                       help='Minimum cluster size to include (default: 3)')
    parser.add_argument('--inspect', action='store_true',
                       help='Inspect cluster files for debugging')
    
    args = parser.parse_args()
    
    # Debug mode
    if args.inspect:
        cluster_dir = Path(args.clusters)
        print("\n" + "="*70)
        print("CLUSTER FILE INSPECTION")
        print("="*70)
        
        for cluster_file in sorted(cluster_dir.glob('cluster_summaries_*.json')):
            inspect_cluster_file(cluster_file)
        
        return
    
    # Normal mode
    result = create_labels_from_clusters(
        graph_path=args.graph,
        cluster_dir=args.clusters,
        hin_dir=args.hin_dir,
        output_path=args.output,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        min_cluster_size=args.min_cluster_size
    )
    
    if result is None:
        print("\n💡 Debug suggestions:")
        print(f"   1. Inspect cluster files:")
        print(f"      python create_node_labels.py --inspect --clusters {args.clusters}")
        print(f"   2. Try reducing minimum cluster size:")
        print(f"      python create_node_labels.py --min-cluster-size 2 ...")
        print(f"   3. Increase k in clustering:")
        print(f"      python analyze_embeddings.py --method kmeans --k 5 ...")
        return
    
    print("\n🎯 Next steps:")
    print(f"   1. Train GNN with node classification:")
    print(f"      python train_gnn.py --graph {args.output} --task both --alpha 0.7")
    print(f"   2. Generate predictions:")
    print(f"      python generate_predictions.py --graph {args.output}")


if __name__ == '__main__':
    main()