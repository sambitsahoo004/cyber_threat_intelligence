#!/usr/bin/env python3
"""
Train GNN on Enhanced Heterogeneous Graph
Supports:
1. Unsupervised pre-training (link prediction with meta-path supervision)
2. Semi-supervised node classification (if labels available)
3. Multi-task learning (both objectives)
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
from collections import defaultdict

try:
    from torch_geometric.data import HeteroData
    from torch_geometric.nn import HeteroConv, SAGEConv, GATConv, Linear
    from torch_geometric.loader import NeighborLoader
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("❌ PyTorch Geometric not available!")
    exit(1)


# ============================================================
# GNN MODELS
# ============================================================

class CTI_HeteroGNN(nn.Module):
    """
    Heterogeneous GNN for CTI with separate handling of:
    1. Original semantic edges (e.g., Malware-delivers->Domain)
    2. Meta-path similarity edges (e.g., Malware-metapath_sim->Malware)
    """
    
    def __init__(self, 
                 metadata,
                 hidden_channels: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 use_attention: bool = True):
        super().__init__()
        
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Separate edge types: original vs meta-path
        self.original_edge_types = [
            et for et in metadata[1] if 'metapath_sim' not in et[1]
        ]
        self.metapath_edge_types = [
            et for et in metadata[1] if 'metapath_sim' in et[1]
        ]
        
        node_types = metadata[0]
        
        # Input projection (handle different feature dimensions)
        self.input_projs = nn.ModuleDict()
        for node_type in node_types:
            # Will be set dynamically based on input features
            self.input_projs[node_type] = None
        
        # Original structure convolutions
        self.original_convs = nn.ModuleList()
        for i in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv(
                    hidden_channels, 
                    hidden_channels,
                    normalize=True
                )
                for edge_type in self.original_edge_types
            }, aggr='mean')
            self.original_convs.append(conv)
        
        # Meta-path convolutions (with attention if enabled)
        self.metapath_convs = nn.ModuleList()
        for i in range(num_layers):
            if use_attention:
                conv = HeteroConv({
                    edge_type: GATConv(
                        hidden_channels,
                        hidden_channels // 4,  # 4 attention heads
                        heads=4,
                        concat=True,
                        dropout=dropout,
                        add_self_loops=False
                    )
                    for edge_type in self.metapath_edge_types
                }, aggr='mean')
            else:
                conv = HeteroConv({
                    edge_type: SAGEConv(
                        hidden_channels,
                        hidden_channels,
                        normalize=True
                    )
                    for edge_type in self.metapath_edge_types
                }, aggr='mean')
            self.metapath_convs.append(conv)
        
        # Fusion layers (combine original + meta-path)
        self.fusion_layers = nn.ModuleDict({
            node_type: nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )
            for node_type in node_types
        })
        
        # Layer normalization
        self.norms = nn.ModuleList([
            nn.ModuleDict({
                node_type: nn.LayerNorm(hidden_channels)
                for node_type in node_types
            })
            for _ in range(num_layers)
        ])
    
    def _init_input_projs(self, x_dict):
        """Initialize input projections based on actual feature dimensions"""
        for node_type, x in x_dict.items():
            if self.input_projs[node_type] is None:
                in_channels = x.size(-1)
                self.input_projs[node_type] = Linear(in_channels, self.hidden_channels)
                # Move to same device as input
                self.input_projs[node_type] = self.input_projs[node_type].to(x.device)
    
    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass with dual-path architecture
        
        Args:
            x_dict: Dict[node_type, Tensor] - node features
            edge_index_dict: Dict[edge_type, Tensor] - edge indices
        
        Returns:
            Dict[node_type, Tensor] - node embeddings
        """
        # Initialize input projections if needed
        self._init_input_projs(x_dict)
        
        # Project input features
        h_dict = {
            node_type: self.input_projs[node_type](x)
            for node_type, x in x_dict.items()
        }
        
        # Store for residual connections
        h_residual = {k: v.clone() for k, v in h_dict.items()}
        
        # Layer-wise propagation
        for i in range(self.num_layers):
            # Path 1: Original semantic edges
            h_original = self.original_convs[i](h_dict, edge_index_dict)
            h_original = {k: F.relu(v) for k, v in h_original.items() if v is not None}
            
            # Path 2: Meta-path similarity edges
            h_metapath = self.metapath_convs[i](h_dict, edge_index_dict)
            h_metapath = {k: F.relu(v) for k, v in h_metapath.items() if v is not None}
            
            # Fuse both paths
            h_fused = {}
            for node_type in h_dict.keys():
                if node_type in h_original and node_type in h_metapath:
                    # Both paths available
                    combined = torch.cat([h_original[node_type], h_metapath[node_type]], dim=-1)
                    h_fused[node_type] = self.fusion_layers[node_type](combined)
                elif node_type in h_original:
                    # Only original path
                    h_fused[node_type] = h_original[node_type]
                elif node_type in h_metapath:
                    # Only meta-path
                    h_fused[node_type] = h_metapath[node_type]
                else:
                    # No updates (shouldn't happen)
                    h_fused[node_type] = h_dict[node_type]
            
            # Layer normalization
            h_dict = {
                node_type: self.norms[i][node_type](h)
                for node_type, h in h_fused.items()
            }
            
            # Residual connection (except first layer)
            if i > 0:
                h_dict = {
                    node_type: h + h_residual[node_type]
                    for node_type, h in h_dict.items()
                }
            
            # Dropout
            h_dict = {
                node_type: F.dropout(h, p=self.dropout, training=self.training)
                for node_type, h in h_dict.items()
            }
        
        return h_dict


class LinkPredictor(nn.Module):
    """Link prediction head"""
    
    def __init__(self, in_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1)
        )
    
    def forward(self, h_src, h_dst):
        """
        Predict link probability between source and destination nodes
        
        Args:
            h_src: [num_edges, hidden_dim]
            h_dst: [num_edges, hidden_dim]
        
        Returns:
            [num_edges] - link probabilities
        """
        h = torch.cat([h_src, h_dst], dim=-1)
        return self.mlp(h).squeeze(-1)


class NodeClassifier(nn.Module):
    """Node classification head"""
    
    def __init__(self, in_channels: int, num_classes: int, hidden_channels: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, num_classes)
        )
    
    def forward(self, h):
        return self.mlp(h)


# ============================================================
# TRAINER
# ============================================================

class CTI_GNN_Trainer:
    """Trainer for CTI heterogeneous GNN"""
    
    def __init__(self,
                 data: HeteroData,
                 output_dir: str,
                 hidden_channels: int = 128,
                 num_layers: int = 2,
                 learning_rate: float = 0.001,
                 weight_decay: float = 5e-4,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.data = data.to(device)
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize model
        self.model = CTI_HeteroGNN(
            metadata=data.metadata(),
            hidden_channels=hidden_channels,
            num_layers=num_layers
        ).to(device)
        
        # Link predictor
        self.link_predictor = LinkPredictor(hidden_channels).to(device)
        
        # Node classifiers (one per node type with labels)
        self.node_classifiers = nn.ModuleDict()
        for node_type in data.node_types:
            if hasattr(data[node_type], 'y'):
                num_classes = int(data[node_type].y.max().item()) + 1
                self.node_classifiers[node_type] = NodeClassifier(
                    hidden_channels, num_classes
                ).to(device)
        
        # Optimizer
        params = list(self.model.parameters()) + list(self.link_predictor.parameters())
        for classifier in self.node_classifiers.values():
            params += list(classifier.parameters())
        
        self.optimizer = torch.optim.Adam(
            params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10
        )
        
        # Training history
        self.history = defaultdict(list)
        
        print(f"\n🏗️  Model initialized on {device}")
        print(f"   • Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   • Hidden channels: {hidden_channels}")
        print(f"   • Layers: {num_layers}")
    
    def _sample_negative_edges(self, 
                               edge_index: torch.Tensor, 
                               num_nodes_src: int,
                               num_nodes_dst: int,
                               num_neg: int) -> torch.Tensor:
        """Sample negative edges (non-existing edges)"""
        
        # Create set of positive edges for fast lookup
        pos_edges = set(map(tuple, edge_index.t().cpu().numpy()))
        
        neg_edges = []
        max_attempts = num_neg * 10
        attempts = 0
        
        while len(neg_edges) < num_neg and attempts < max_attempts:
            src = np.random.randint(0, num_nodes_src, size=num_neg - len(neg_edges))
            dst = np.random.randint(0, num_nodes_dst, size=num_neg - len(neg_edges))
            
            for s, d in zip(src, dst):
                if (s, d) not in pos_edges:
                    neg_edges.append([s, d])
            
            attempts += 1
        
        if len(neg_edges) == 0:
            # Fallback: return random edges
            neg_edges = torch.randint(0, min(num_nodes_src, num_nodes_dst), (num_neg, 2))
        else:
            neg_edges = torch.tensor(neg_edges[:num_neg], dtype=torch.long).t()
        
        return neg_edges.to(self.device)
    
    def compute_link_prediction_loss(self, h_dict, edge_type_subset=None):
        """
        Compute link prediction loss using meta-path similarity edges as supervision
        
        Strategy: High-similarity meta-path edges = positive, random pairs = negative
        """
        total_loss = 0.0
        num_edge_types = 0
        
        # Use meta-path edges as positive examples
        edge_types_to_use = edge_type_subset or [
            et for et in self.data.edge_types if 'metapath_sim' in et[1]
        ]
        
        for edge_type in edge_types_to_use:
            src_type, rel, dst_type = edge_type
            
            if src_type not in h_dict or dst_type not in h_dict:
                continue
            
            edge_index = self.data[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue
            
            # Get edge weights (similarity scores)
            if hasattr(self.data[edge_type], 'edge_attr'):
                edge_weights = self.data[edge_type].edge_attr.squeeze()
            else:
                edge_weights = torch.ones(edge_index.size(1), device=self.device)
            
            # Sample subset if too many edges
            max_edges = 10000
            if edge_index.size(1) > max_edges:
                perm = torch.randperm(edge_index.size(1))[:max_edges]
                edge_index = edge_index[:, perm]
                edge_weights = edge_weights[perm]
            
            # Positive edges
            h_src = h_dict[src_type][edge_index[0]]
            h_dst = h_dict[dst_type][edge_index[1]]
            pos_score = self.link_predictor(h_src, h_dst)
            
            # Negative edges (random sampling)
            num_neg = edge_index.size(1)
            neg_edge_index = self._sample_negative_edges(
                edge_index,
                h_dict[src_type].size(0),
                h_dict[dst_type].size(0),
                num_neg
            )
            
            h_src_neg = h_dict[src_type][neg_edge_index[0]]
            h_dst_neg = h_dict[dst_type][neg_edge_index[1]]
            neg_score = self.link_predictor(h_src_neg, h_dst_neg)
            
            # Binary cross-entropy with edge weights
            pos_loss = F.binary_cross_entropy_with_logits(
                pos_score,
                torch.ones_like(pos_score),
                weight=edge_weights,
                reduction='mean'
            )
            
            neg_loss = F.binary_cross_entropy_with_logits(
                neg_score,
                torch.zeros_like(neg_score),
                reduction='mean'
            )
            
            total_loss += (pos_loss + neg_loss) / 2
            num_edge_types += 1
        
        return total_loss / max(num_edge_types, 1)
    
    def compute_node_classification_loss(self, h_dict):
        """Compute node classification loss for labeled nodes"""
        total_loss = 0.0
        num_node_types = 0
        
        for node_type, classifier in self.node_classifiers.items():
            if not hasattr(self.data[node_type], 'y'):
                continue
            
            if not hasattr(self.data[node_type], 'train_mask'):
                continue
            
            h = h_dict[node_type]
            logits = classifier(h)
            
            labels = self.data[node_type].y
            mask = self.data[node_type].train_mask
            
            # Filter out unlabeled nodes
            valid_mask = mask & (labels >= 0)
            
            if valid_mask.sum() > 0:
                loss = F.cross_entropy(logits[valid_mask], labels[valid_mask])
                total_loss += loss
                num_node_types += 1
        
        return total_loss / max(num_node_types, 1) if num_node_types > 0 else torch.tensor(0.0).to(self.device)
    
    def train_epoch(self, task='link_prediction', alpha=0.5):
        """
        Train one epoch
        
        Args:
            task: 'link_prediction', 'node_classification', or 'both'
            alpha: weight for multi-task learning (link_loss * alpha + node_loss * (1-alpha))
        """
        self.model.train()
        self.link_predictor.train()
        for classifier in self.node_classifiers.values():
            classifier.train()
        
        # Forward pass
        h_dict = self.model(self.data.x_dict, self.data.edge_index_dict)
        
        # Compute losses
        link_loss = torch.tensor(0.0).to(self.device)
        node_loss = torch.tensor(0.0).to(self.device)
        
        if task in ['link_prediction', 'both']:
            link_loss = self.compute_link_prediction_loss(h_dict)
        
        if task in ['node_classification', 'both'] and len(self.node_classifiers) > 0:
            node_loss = self.compute_node_classification_loss(h_dict)
        
        # Combined loss
        if task == 'both':
            loss = alpha * link_loss + (1 - alpha) * node_loss
        elif task == 'link_prediction':
            loss = link_loss
        else:
            loss = node_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'link_loss': link_loss.item(),
            'node_loss': node_loss.item()
        }
    
    @torch.no_grad()
    def evaluate(self, task='link_prediction'):
        """Evaluate model"""
        self.model.eval()
        self.link_predictor.eval()
        for classifier in self.node_classifiers.values():
            classifier.eval()
        
        # Forward pass
        h_dict = self.model(self.data.x_dict, self.data.edge_index_dict)
        
        metrics = {}
        
        # Link prediction metrics
        if task in ['link_prediction', 'both']:
            link_metrics = self._evaluate_link_prediction(h_dict)
            metrics.update(link_metrics)
        
        # Node classification metrics
        if task in ['node_classification', 'both'] and len(self.node_classifiers) > 0:
            node_metrics = self._evaluate_node_classification(h_dict)
            metrics.update(node_metrics)
        
        return metrics
    
    def _evaluate_link_prediction(self, h_dict):
        """Evaluate link prediction on validation set"""
        all_pos_scores = []
        all_neg_scores = []
        
        # Evaluate on meta-path edges
        for edge_type in self.data.edge_types:
            if 'metapath_sim' not in edge_type[1]:
                continue
            
            src_type, rel, dst_type = edge_type
            
            if src_type not in h_dict or dst_type not in h_dict:
                continue
            
            edge_index = self.data[edge_type].edge_index
            if edge_index.size(1) == 0:
                continue
            
            # Sample validation edges (use a subset)
            num_val = min(1000, edge_index.size(1))
            perm = torch.randperm(edge_index.size(1))[:num_val]
            val_edge_index = edge_index[:, perm]
            
            # Positive scores
            h_src = h_dict[src_type][val_edge_index[0]]
            h_dst = h_dict[dst_type][val_edge_index[1]]
            pos_scores = torch.sigmoid(self.link_predictor(h_src, h_dst))
            
            # Negative scores
            neg_edge_index = self._sample_negative_edges(
                val_edge_index,
                h_dict[src_type].size(0),
                h_dict[dst_type].size(0),
                num_val
            )
            
            h_src_neg = h_dict[src_type][neg_edge_index[0]]
            h_dst_neg = h_dict[dst_type][neg_edge_index[1]]
            neg_scores = torch.sigmoid(self.link_predictor(h_src_neg, h_dst_neg))
            
            all_pos_scores.append(pos_scores.cpu().numpy())
            all_neg_scores.append(neg_scores.cpu().numpy())
        
        if len(all_pos_scores) == 0:
            return {'link_auc': 0.0, 'link_ap': 0.0}
        
        # Compute metrics
        all_pos_scores = np.concatenate(all_pos_scores)
        all_neg_scores = np.concatenate(all_neg_scores)
        
        y_true = np.concatenate([np.ones_like(all_pos_scores), np.zeros_like(all_neg_scores)])
        y_score = np.concatenate([all_pos_scores, all_neg_scores])
        
        auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        
        return {
            'link_auc': auc,
            'link_ap': ap
        }
    
    def _evaluate_node_classification(self, h_dict):
        """Evaluate node classification on validation set"""
        metrics = {}
        
        for node_type, classifier in self.node_classifiers.items():
            if not hasattr(self.data[node_type], 'y'):
                continue
            
            if not hasattr(self.data[node_type], 'val_mask'):
                continue
            
            h = h_dict[node_type]
            logits = classifier(h)
            preds = logits.argmax(dim=-1)
            
            labels = self.data[node_type].y
            mask = self.data[node_type].val_mask & (labels >= 0)
            
            if mask.sum() > 0:
                acc = (preds[mask] == labels[mask]).float().mean().item()
                f1 = f1_score(
                    labels[mask].cpu().numpy(),
                    preds[mask].cpu().numpy(),
                    average='weighted',
                    zero_division=0
                )
                
                metrics[f'{node_type}_acc'] = acc
                metrics[f'{node_type}_f1'] = f1
        
        return metrics
    
    def train(self, 
              num_epochs: int = 100,
              task: str = 'link_prediction',
              alpha: float = 0.5,
              eval_every: int = 5,
              save_every: int = 20,
              early_stopping_patience: int = 20):
        """
        Main training loop
        
        Args:
            num_epochs: Number of training epochs
            task: 'link_prediction', 'node_classification', or 'both'
            alpha: Multi-task weight (only used if task='both')
            eval_every: Evaluate every N epochs
            save_every: Save checkpoint every N epochs
            early_stopping_patience: Stop if no improvement for N epochs
        """
        print(f"\n🚀 Starting training: task={task}, epochs={num_epochs}")
        print("="*70)
        
        best_metric = -float('inf')
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            start_time = time.time()
            
            # Train
            train_metrics = self.train_epoch(task=task, alpha=alpha)
            
            # Log
            self.history['epoch'].append(epoch)
            for k, v in train_metrics.items():
                self.history[k].append(v)
            
            # Evaluate
            if epoch % eval_every == 0:
                val_metrics = self.evaluate(task=task)
                
                for k, v in val_metrics.items():
                    self.history[k].append(v)
                
                # Primary metric for early stopping
                if task == 'link_prediction':
                    primary_metric = val_metrics.get('link_auc', 0.0)
                elif task == 'node_classification':
                    primary_metric = np.mean([v for k, v in val_metrics.items() if '_acc' in k])
                else:
                    primary_metric = val_metrics.get('link_auc', 0.0)
                
                # Early stopping
                if primary_metric > best_metric:
                    best_metric = primary_metric
                    patience_counter = 0
                    self.save_checkpoint('best_model.pt')
                else:
                    patience_counter += 1
                
                # Learning rate scheduling
                self.scheduler.step(train_metrics['total_loss'])
                
                # Print progress
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch:03d} | Loss: {train_metrics['total_loss']:.4f} | "
                      f"Link AUC: {val_metrics.get('link_auc', 0.0):.4f} | "
                      f"Time: {epoch_time:.2f}s")
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    print(f"\n⚠️  Early stopping triggered (patience={early_stopping_patience})")
                    break
            else:
                # Just print training loss
                if epoch % 1 == 0:
                    print(f"Epoch {epoch:03d} | Loss: {train_metrics['total_loss']:.4f}")
            
            # Save checkpoint
            if epoch % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt')
        
        print("\n" + "="*70)
        print(f"✅ Training complete! Best metric: {best_metric:.4f}")
        
        # Load best model
        self.load_checkpoint('best_model.pt')
        
        # Save final results
        self.save_history()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint (history converted to plain Python types to avoid numpy scalars)."""
        # Convert history lists to Python floats (avoid numpy types)
        safe_history = {}
        for k, v in self.history.items():
            # ensure v is iterable (list of numbers)
            safe_history[k] = [float(x) for x in v]

        checkpoint = {
            'model_state': self.model.state_dict(),
            'link_predictor_state': self.link_predictor.state_dict(),
            'node_classifiers_state': {k: v.state_dict() for k, v in self.node_classifiers.items()},
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': safe_history
        }

        torch.save(checkpoint, self.output_dir / filename)
        print(f"💾 Saved checkpoint: {filename}")

    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint with a safe fallback for older checkpoints containing numpy scalars."""
        checkpoint_path = self.output_dir / filename
        if not checkpoint_path.exists():
            print(f"⚠️  Checkpoint not found: {filename}")
            return

        try:
            # Prefer default load (PyTorch >=2.6 uses safe weights-only behavior)
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            # If we get a WeightsUnpicklingError related to numpy scalars, try a guarded allowlist load.
            # This is potentially unsafe if the checkpoint is from an untrusted source.
            print(f"⚠️  torch.load failed with: {e}")
            print("    Attempting fallback load with numpy scalar allowlist (only if you trust this file).")
            try:
                import numpy as _np
                from torch.serialization import add_safe_globals

                # Allowlist numpy scalar global required by some saved checkpoints
                try:
                    add_safe_globals([_np.core.multiarray.scalar])
                except Exception:
                    # Some numpy builds may expose scalar differently; also try numpy._core
                    try:
                        add_safe_globals([_np._core.multiarray.scalar])
                    except Exception:
                        # If allowlisting fails, re-raise to avoid silent failure
                        raise

                # Retry loading (weights_only=False so arbitrary objects can be unpickled)
                checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            except Exception as e2:
                print("❌ Fallback load also failed:", e2)
                raise

        # Load states into model/optim/etc.
        self.model.load_state_dict(checkpoint['model_state'])
        self.link_predictor.load_state_dict(checkpoint['link_predictor_state'])

        for k, v in checkpoint.get('node_classifiers_state', {}).items():
            if k in self.node_classifiers:
                self.node_classifiers[k].load_state_dict(v)

        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.history = defaultdict(list, checkpoint.get('history', {}))

        print(f"✅ Loaded checkpoint: {filename}")

    
    def save_history(self):
        """Save training history"""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        print(f"💾 Saved training history: {history_file}")
    
    @torch.no_grad()
    def compute_embeddings(self):
        """Compute and save node embeddings"""
        print("\n🧮 Computing final embeddings...")
        
        self.model.eval()
        h_dict = self.model(self.data.x_dict, self.data.edge_index_dict)
        
        # Save embeddings per node type
        embeddings_dir = self.output_dir / 'embeddings'
        embeddings_dir.mkdir(exist_ok=True)
        
        for node_type, h in h_dict.items():
            embeddings = h.cpu().numpy()
            
            # Save embeddings
            np.save(embeddings_dir / f'embeddings_{node_type}.npy', embeddings)
            
            # Save metadata
            meta = {
                'node_type': node_type,
                'num_nodes': embeddings.shape[0],
                'embedding_dim': embeddings.shape[1],
                'mean': float(embeddings.mean()),
                'std': float(embeddings.std())
            }
            
            with open(embeddings_dir / f'embeddings_{node_type}_meta.json', 'w') as f:
                json.dump(meta, f, indent=2)
            
            print(f"   ✓ Saved {node_type}: {embeddings.shape}")
        
        print(f"💾 Embeddings saved to: {embeddings_dir}")

        return h_dict

#============================================================
#CLI
#============================================================
def main():
    parser = argparse.ArgumentParser(
    description='Train GNN on enhanced heterogeneous CTI graph',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
    Example Usage:

    Unsupervised link prediction (recommended for pre-training):
    python train_gnn.py \
    --graph enhanced_graph/enhanced_hetero_graph.pt \
    --output gnn_output \
    --task link_prediction \
    --epochs 100
    Semi-supervised node classification (if labels available):
    python train_gnn.py \
    --graph enhanced_graph/enhanced_hetero_graph.pt \
    --output gnn_output \
    --task node_classification \
    --epochs 100
    Multi-task learning:
    python train_gnn.py \
    --graph enhanced_graph/enhanced_hetero_graph.pt \
    --output gnn_output \
    --task both \
    --alpha 0.7 \
    --epochs 150

    Output:
    gnn_output/
    ├── best_model.pt              (best checkpoint)
    ├── checkpoint_epoch_*.pt      (periodic checkpoints)
    ├── training_history.json      (loss curves)
    └── embeddings/                (final node embeddings)
    ├── embeddings_Malware.npy
    ├── embeddings_Domain.npy
    └── ...
    """
    )
    # Data arguments
    parser.add_argument(
        '--graph',
        required=True,
        help='Path to enhanced_hetero_graph.pt'
    )

    parser.add_argument(
        '--output',
        default='gnn_output',
        help='Output directory (default: gnn_output)'
    )

    # Model arguments
    parser.add_argument(
        '--hidden-channels',
        type=int,
        default=128,
        help='Hidden embedding dimension (default: 128)'
    )

    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of GNN layers (default: 2)'
    )

    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (default: 0.3)'
    )

    # Training arguments
    parser.add_argument(
        '--task',
        choices=['link_prediction', 'node_classification', 'both'],
        default='link_prediction',
        help='Training task (default: link_prediction)'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs (default: 100)'
    )

    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate (default: 0.001)'
    )

    parser.add_argument(
        '--weight-decay',
        type=float,
        default=5e-4,
        help='Weight decay (L2 regularization) (default: 5e-4)'
    )

    parser.add_argument(
        '--alpha',
        type=float,
        default=0.5,
        help='Multi-task weight: alpha*link + (1-alpha)*node (default: 0.5)'
    )

    parser.add_argument(
        '--eval-every',
        type=int,
        default=5,
        help='Evaluate every N epochs (default: 5)'
    )

    parser.add_argument(
        '--save-every',
        type=int,
        default=20,
        help='Save checkpoint every N epochs (default: 20)'
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )

    parser.add_argument(
        '--device',
        default='auto',
        help='Device: cuda, cpu, or auto (default: auto)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed (default: 42)'
    )

    # Post-training arguments
    parser.add_argument(
        '--compute-embeddings',
        action='store_true',
        help='Compute and save node embeddings after training'
    )

    parser.add_argument(
        '--no-train',
        action='store_true',
        help='Skip training (only load model and compute embeddings)'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print("\n" + "="*70)
    print("CTI HETEROGENEOUS GNN TRAINER")
    print("="*70)
    print(f"\n📊 Configuration:")
    print(f"   • Graph: {args.graph}")
    print(f"   • Task: {args.task}")
    print(f"   • Device: {device}")
    print(f"   • Hidden channels: {args.hidden_channels}")
    print(f"   • Layers: {args.num_layers}")
    print(f"   • Epochs: {args.epochs}")
    print(f"   • Learning rate: {args.lr}")
    print(f"   • Seed: {args.seed}")

    # Load graph
    print(f"\n📦 Loading graph from: {args.graph}")
    graph_path = Path(args.graph)

    if not graph_path.exists():
        print(f"❌ Error: Graph file not found: {args.graph}")
        return

    data = torch.load(graph_path, weights_only=False)

    print(f"   ✓ Loaded HeteroData:")
    print(f"      - Node types: {len(data.node_types)}")
    print(f"      - Edge types: {len(data.edge_types)}")

    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        feat_dim = data[node_type].x.shape[1] if hasattr(data[node_type], 'x') else 0
        print(f"      - {node_type}: {num_nodes} nodes, {feat_dim} features")

    # Initialize trainer
    trainer = CTI_GNN_Trainer(
        data=data,
        output_dir=args.output,
        hidden_channels=args.hidden_channels,
        num_layers=args.num_layers,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        device=device
    )

    # Training
    if not args.no_train:
        trainer.train(
            num_epochs=args.epochs,
            task=args.task,
            alpha=args.alpha,
            eval_every=args.eval_every,
            save_every=args.save_every,
            early_stopping_patience=args.patience
        )
    else:
        print("\n⚠️  Skipping training (--no-train flag)")
        # Try to load best model
        trainer.load_checkpoint('best_model.pt')

    # Compute embeddings
    if args.compute_embeddings or args.no_train:
        embeddings_dict = trainer.compute_embeddings()
        
        # Print embedding statistics
        print("\n📊 Embedding Statistics:")
        for node_type, emb in embeddings_dict.items():
            emb_np = emb.cpu().numpy()
            print(f"   • {node_type}:")
            print(f"      - Shape: {emb_np.shape}")
            print(f"      - Mean: {emb_np.mean():.4f}")
            print(f"      - Std: {emb_np.std():.4f}")
            print(f"      - Min: {emb_np.min():.4f}")
            print(f"      - Max: {emb_np.max():.4f}")

    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)

    print(f"\n📁 Output saved to: {args.output}/")
    print(f"   • Model: best_model.pt")
    print(f"   • History: training_history.json")
    if args.compute_embeddings:
        print(f"   • Embeddings: embeddings/")

    print("\n🔮 Next steps:")
    print("   1. Analyze embeddings:")
    print(f"      python analyze_embeddings.py --embeddings {args.output}/embeddings")
    print("   2. Cluster and interpret:")
    print(f"      python cluster_and_interpret.py --embeddings {args.output}/embeddings")
    print("   3. Generate predictions for LLM:")
    print(f"      python generate_predictions.py --model {args.output}/best_model.pt")

if __name__ == '__main__':
    main()