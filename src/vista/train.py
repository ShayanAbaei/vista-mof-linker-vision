import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import time
import json
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
)


class MultiLabelMetrics:
    """
    Compute and track per-label metrics for multi-label classification.

    Handles:
    - Per-label precision, recall, F1, accuracy, balanced accuracy, specificity
    - Per-label AUROC and AUPRC (average precision) from probabilities
    - Per-label positive/negative support
    - Macro/micro averaged metrics (including macro/micro AUPRC)
    - Subset accuracy (exact match) and Hamming loss

    Threshold-dependent metrics use ``threshold`` (a scalar or a per-label
    array). Probability-based metrics (AUROC/AUPRC) are threshold-free.
    """

    def __init__(
        self,
        label_names: List[str],
        threshold: Union[float, np.ndarray] = 0.5,
    ):
        self.label_names = label_names
        self.num_labels = len(label_names)
        self.threshold = threshold

    def _binarize(self, predictions: np.ndarray) -> np.ndarray:
        thr = self.threshold
        if np.isscalar(thr):
            return (predictions > thr).astype(int)
        thr = np.asarray(thr).reshape(1, -1)
        return (predictions > thr).astype(int)

    @staticmethod
    def _specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    @staticmethod
    def _safe_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        # AUROC undefined if only one class present.
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_score))

    @staticmethod
    def _safe_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
        if y_true.sum() == 0:
            return float("nan")
        return float(average_precision_score(y_true, y_score))

    def compute_metrics(
        self,
        predictions: np.ndarray,
        targets: np.ndarray,
    ) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Parameters
        ----------
        predictions : np.ndarray
            Predicted probabilities, shape (N, num_labels).
        targets : np.ndarray
            Ground-truth binary labels, shape (N, num_labels).
        """
        pred_binary = self._binarize(predictions)

        per_label_metrics: Dict[str, Dict[str, float]] = {}
        per_label_ba: List[float] = []
        per_label_spec: List[float] = []
        per_label_auprc: List[float] = []
        per_label_auroc: List[float] = []

        for i, label_name in enumerate(self.label_names):
            y_true = targets[:, i]
            y_pred = pred_binary[:, i]
            y_score = predictions[:, i]

            balanced_accuracy = balanced_accuracy_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float("nan")
            specificity = self._specificity(y_true, y_pred)
            auprc = self._safe_auprc(y_true, y_score)
            auroc = self._safe_auroc(y_true, y_score)
            pos_support = int(y_true.sum())
            neg_support = int(len(y_true) - pos_support)

            per_label_ba.append(balanced_accuracy)
            per_label_spec.append(specificity)
            per_label_auprc.append(auprc)
            per_label_auroc.append(auroc)

            per_label_metrics[label_name] = {
                'precision': float(precision_score(y_true, y_pred, zero_division=0)),
                'recall': float(recall_score(y_true, y_pred, zero_division=0)),
                'f1': float(f1_score(y_true, y_pred, zero_division=0)),
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'balanced_accuracy': balanced_accuracy,
                'specificity': specificity,
                'auprc': auprc,
                'auroc': auroc,
                'pos_support': pos_support,
                'neg_support': neg_support,
            }

        targets_flat = targets.flatten()
        pred_binary_flat = pred_binary.flatten()

        overall_metrics = {
            # Macro averages (mean over labels; nan-safe for AUPRC/AUROC/BA)
            'macro_precision': float(np.mean([m['precision'] for m in per_label_metrics.values()])),
            'macro_recall': float(np.mean([m['recall'] for m in per_label_metrics.values()])),
            'macro_f1': float(np.mean([m['f1'] for m in per_label_metrics.values()])),
            'macro_accuracy': float(np.mean([m['accuracy'] for m in per_label_metrics.values()])),
            'macro_balanced_accuracy': float(np.nanmean(per_label_ba)),
            'macro_specificity': float(np.mean(per_label_spec)),
            'macro_auprc': float(np.nanmean(per_label_auprc)),
            'macro_auroc': float(np.nanmean(per_label_auroc)),

            # Micro averages
            'micro_precision': float(precision_score(targets, pred_binary, average='micro', zero_division=0)),
            'micro_recall': float(recall_score(targets, pred_binary, average='micro', zero_division=0)),
            'micro_f1': float(f1_score(targets, pred_binary, average='micro', zero_division=0)),
            'micro_auprc': float(average_precision_score(targets, predictions, average='micro')),
            'micro_balanced_accuracy': float(
                balanced_accuracy_score(targets_flat, pred_binary_flat)
                if len(np.unique(targets_flat)) > 1 else float("nan")
            ),
            'micro_specificity': self._specificity(targets_flat, pred_binary_flat),

            'subset_accuracy': float(accuracy_score(targets, pred_binary)),
            'hamming_loss': float(np.mean(targets != pred_binary)),
        }

        return {'per_label': per_label_metrics, 'overall': overall_metrics}


class MultiLabelTrainer:
    """
    Trainer for multi-label classification with comprehensive metric tracking.

    Notes
    -----
    * The wrapped model returns ``(logits, attention)``; this trainer correctly
      unpacks ``logits, _ = self.model(images)``.
    * The criterion is moved to ``device`` so frequency-adapted gamma buffers
      live on the right device (portable; no hard-coded ``.to('cuda')``).
    * Default checkpoint selection is **validation macro-AUPRC** (maximise) with
      validation loss as a tie-breaker. Set ``selection_metric='val_loss'`` to
      reproduce the original validation-loss selection.
    """

    def __init__(
            self,
            model: nn.Module,
            train_loader: DataLoader,
            val_loader: DataLoader,
            criterion: nn.Module,
            label_names: List[str],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler = None,
            device: str = 'cuda',
            log_dir: str = 'runs',
            checkpoint_dir: str = 'checkpoints',
            experiment_name: Optional[str] = None,
            selection_metric: str = 'macro_auprc',
            experiment_config: Optional[Dict] = None,
            seeds: Optional[Dict] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Move the whole criterion to the device so adaptive-loss buffers follow.
        self.criterion = criterion.to(device) if isinstance(criterion, nn.Module) else criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.label_names = label_names
        self.device = device
        self.selection_metric = selection_metric
        self.experiment_config = experiment_config or {}
        self.seeds = seeds or {}

        self.metrics = MultiLabelMetrics(label_names)

        if experiment_name is None:
            experiment_name = f"multilabel_{time.strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name

        self.log_dir = Path(log_dir) / experiment_name
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(self.log_dir)

        # Training state
        self.epoch = 0
        self.patience_counter = 0
        # best_score: higher-is-better selection value; best_val_loss: tie-break.
        self.best_score = -float('inf')
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.history: List[Dict] = []

    # ----------------------------------------------------------------- epochs
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        running_loss = 0.0
        all_predictions, all_targets = [], []

        pbar = tqdm(self.train_loader, desc=f'Train Epoch {self.epoch}')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Model returns (logits, attention); unpack logits.
            logits, _ = self.model(images)
            loss = self.criterion(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            with torch.no_grad():
                all_predictions.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(targets.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        metrics['loss'] = running_loss / len(self.train_loader)
        return metrics

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        self.model.eval()
        running_loss = 0.0
        all_predictions, all_targets = [], []

        pbar = tqdm(self.val_loader, desc=f'Val Epoch {self.epoch}')
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            logits, _ = self.model(images)
            loss = self.criterion(logits, targets)

            running_loss += loss.item()
            all_predictions.append(torch.sigmoid(logits).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        metrics['loss'] = running_loss / len(self.val_loader)
        return metrics

    # --------------------------------------------------------------- logging
    def log_metrics(self, metrics: Dict, phase: str):
        for metric_name, value in metrics['overall'].items():
            self.writer.add_scalar(f'metrics/{phase}_{metric_name}', value, self.epoch)
        self.writer.add_scalar(f'metrics/{phase}_loss', metrics['loss'], self.epoch)

        for label_name in self.label_names:
            for metric_name, value in metrics['per_label'][label_name].items():
                self.writer.add_scalar(
                    f'per_label_{metric_name}/{phase}_{label_name}', value, self.epoch
                )

    def print_metrics(self, train_metrics: Dict, val_metrics: Dict):
        print(f"\nEpoch {self.epoch} Summary:")
        print("=" * 80)
        print(f"{'Metric':<22} {'Train':<15} {'Val':<15}")
        print("-" * 52)
        print(f"{'Loss':<22} {train_metrics['loss']:<15.4f} {val_metrics['loss']:<15.4f}")
        for key, name in [('macro_auprc', 'Macro AUPRC'),
                          ('macro_auroc', 'Macro AUROC'),
                          ('macro_f1', 'Macro F1'),
                          ('macro_balanced_accuracy', 'Macro BA'),
                          ('subset_accuracy', 'Subset Acc')]:
            print(f"{name:<22} {train_metrics['overall'][key]:<15.4f} {val_metrics['overall'][key]:<15.4f}")

        print("\nPer-Label (Val):")
        print("-" * 92)
        print(f"{'Label':<25} {'AUPRC':<9} {'AUROC':<9} {'F1':<9} {'Recall':<9} {'Prec':<9} {'BA':<9}")
        print("-" * 92)
        for label in self.label_names:
            m = val_metrics['per_label'][label]
            print(f"{label:<25} {m['auprc']:<9.4f} {m['auroc']:<9.4f} {m['f1']:<9.4f} "
                  f"{m['recall']:<9.4f} {m['precision']:<9.4f} {m['balanced_accuracy']:<9.4f}")
        print("=" * 92, "\n")

    # ------------------------------------------------------------ checkpoints
    def _model_config(self) -> Dict:
        """Best-effort extraction of model/mask config for the checkpoint."""
        cfg: Dict = dict(self.experiment_config)
        inner = getattr(self.model, 'model', self.model)
        mask_cfg = getattr(inner, 'mask_config', None)
        if mask_cfg is not None and hasattr(mask_cfg, 'to_dict'):
            cfg.setdefault('mask_config', mask_cfg.to_dict())
        for attr in ('n_labels', 'n_encoder_layers', 'n_patches', 'seq_len'):
            if hasattr(inner, attr):
                cfg.setdefault(attr, getattr(inner, attr))
        return cfg

    def _loss_config(self) -> Dict:
        c = self.criterion
        out = {'class': type(c).__name__}
        for attr in ('clip', 'eps', 'freq_gamma_schedule'):
            if hasattr(c, attr):
                v = getattr(c, attr)
                out[attr] = v if not isinstance(v, torch.Tensor) else v.detach().cpu().tolist()
        for attr in ('gamma_pos', 'gamma_neg', 'label_frequencies'):
            if hasattr(c, attr):
                v = getattr(c, attr)
                if isinstance(v, torch.Tensor):
                    out[attr] = v.detach().cpu().tolist()
                else:
                    out[attr] = v
        return out

    def save_checkpoint(self, val_metrics: Dict, is_best: bool = False,
                        thresholds: Optional[Dict] = None):
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': (self.scheduler.state_dict()
                                     if self.scheduler is not None else None),
            'label_names': self.label_names,
            'model_config': self._model_config(),
            'mask_config': self._model_config().get('mask_config'),
            'loss_config': self._loss_config(),
            'thresholds': thresholds,
            'seeds': self.seeds,
            'selection_metric': self.selection_metric,
            'best_score': self.best_score,
            'best_val_loss': self.best_val_loss,
            'best_val': self.best_val_loss,  # back-compat alias (loss-based)
            'val_metrics': val_metrics['overall'],
        }
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{self.epoch}.pth')
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / 'best_model.pth')
            print(f"Saved best model | {self.selection_metric}={self.best_score:.4f} "
                  f"val_loss={self.best_val_loss:.4f}\n")

    # ------------------------------------------------------------- optimizer
    def update_optimizer(self, new_lr_backbone: float):
        """Add the now-trainable backbone parameters to the existing optimizer."""
        current_lr = self.optimizer.param_groups[0]['lr']
        print(f"\nUnfreezing backbone at epoch {self.epoch}")
        print(f"Head LR: {current_lr:.2e}, Backbone LR: {new_lr_backbone:.2e}")

        existing = {id(p) for g in self.optimizer.param_groups for p in g['params']}
        backbone_params = [
            p for name, p in self.model.named_parameters()
            if 'backbone' in name and p.requires_grad and id(p) not in existing
        ]
        if backbone_params:
            self.optimizer.add_param_group({'params': backbone_params, 'lr': new_lr_backbone})
            print(f"Added {len(backbone_params)} backbone parameters to optimizer")
        self.writer.add_text('training/events',
                             f'Backbone unfrozen at epoch {self.epoch}', self.epoch)

    # ----------------------------------------------------------- selection
    def _selection_score(self, val_metrics: Dict) -> float:
        """Higher-is-better selection score for the configured metric."""
        if self.selection_metric == 'val_loss':
            return -val_metrics['loss']
        return val_metrics['overall'].get(self.selection_metric, float('nan'))

    # ---------------------------------------------------------------- train
    def train(
            self,
            num_epochs: int,
            patience: int = 10,
            unfreeze_epoch: int = 10,
            unfreeze_lr_ratio: float = 0.1,
    ):
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Tracking {len(self.label_names)} labels: {', '.join(self.label_names)}")
        print(f"Selection metric: {self.selection_metric} (val-loss tie-break)")
        print(f"Logs: {self.log_dir}\nCheckpoints: {self.checkpoint_dir}")
        print("=" * 80)

        for epoch in range(1, num_epochs + 1):
            self.epoch = epoch

            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            self.log_metrics(train_metrics, 'train')
            self.log_metrics(val_metrics, 'val')
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('metrics/learning_rate', lr, epoch)
            self.print_metrics(train_metrics, val_metrics)

            self.history.append({
                'epoch': epoch,
                'train': train_metrics['overall'], 'train_loss': train_metrics['loss'],
                'val': val_metrics['overall'], 'val_loss': val_metrics['loss'],
                'lr': lr,
            })

            # Selection: primary score, then val-loss tie-break.
            score = self._selection_score(val_metrics)
            val_loss = val_metrics['loss']
            improved = (score > self.best_score) or (
                np.isclose(score, self.best_score) and val_loss < self.best_val_loss
            )
            if improved:
                self.best_score = score
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(val_metrics, is_best=True)
            else:
                self.patience_counter += 1

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['overall'].get(self.selection_metric, val_loss))
                else:
                    self.scheduler.step()

            if self.patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

            # Unfreeze backbone and ACTUALLY add its params to the optimizer.
            if self.epoch == unfreeze_epoch:
                self.model.unfreeze_backbone()
                backbone_lr = self.optimizer.param_groups[0]['lr'] * unfreeze_lr_ratio
                self.update_optimizer(backbone_lr)

        # Persist training history.
        with open(self.checkpoint_dir / 'history.json', 'w') as f:
            json.dump(self.history, f, indent=2, default=float)

        print("\nTraining completed!")
        print(f"Best epoch {self.best_epoch} | {self.selection_metric} score "
              f"{self.best_score:.4f} | val_loss {self.best_val_loss:.4f}")
