"""The multi-task neural network and its pretraining -- the transfer-learning arm.

What this is for
----------------
A shared MLP trunk with one classification head per endpoint. Compounds measured on only
one endpoint still train that endpoint's head, through a masked loss, which is the whole
point of the multi-task formulation: on the PSMA set a single-task pKi model discards 229
pKi-only compounds, a third of the data.

The pretraining ("transfer") step fits the trunk on a broad potency corpus with a
single-output regression head -- the pretext labels are real numbers, so regression is the
right task -- then transfers the trunk and fine-tunes with an optional freeze schedule.

**Read docs/RESULTS.md before using it.** Two ablations in opposite directions found the
pretraining bought nothing on this dataset: +0.062 R2 on random regression splits and
nothing on grouped ones, then -0.0066 AUC on classification. The network itself was also
the joint-worst model on pKi (0.688, tied with nearest-neighbour lookup). The code is kept
because the ablation is worth being able to re-run on a new target, not because it is
recommended here. ``torch`` is an optional dependency for exactly that reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .._optional import require, tqdm

__all__ = [
    "MultiTaskClassifier",
    "masked_multitask_cross_entropy",
    "class_weights_per_task",
    "pretrain_trunk",
    "finetune",
    "predict_proba_multitask",
    "predict_with_mc_dropout",
    "pick_device",
    "PretrainingAblation",
    "run_pretraining_ablation",
]


def pick_device(prefer: Optional[str] = None):
    """CUDA, then Apple MPS, then CPU. ``prefer`` forces a choice."""
    torch = require("torch")
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(getattr(torch, "backends", None), "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _build_module(n_in: int, hidden_dims: Sequence[int], dropout: float,
                  n_tasks: int, n_classes: int):
    """Construct the module class lazily, so importing this file needs no torch."""
    torch = require("torch")
    nn = torch.nn

    class _MultiTaskClassifier(nn.Module):
        """Shared MLP trunk, one head per endpoint, plus a pretraining head."""

        def __init__(self) -> None:
            super().__init__()
            layers: List = []
            d = n_in
            for h in hidden_dims:
                layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(dropout)]
                d = int(h)
            self.trunk = nn.Sequential(*layers)
            self.trunk_out_dim = d
            self.n_tasks = n_tasks
            self.n_classes = n_classes
            # Single-output regression head, used only while pretraining. With
            # pretraining off it is never touched and the trunk starts random.
            self.pretrain_head = nn.Linear(d, 1)
            self.task_heads = nn.ModuleList(
                [nn.Linear(d, n_classes) for _ in range(n_tasks)]
            )

        def forward_pretrain(self, x):
            return self.pretrain_head(self.trunk(x)).squeeze(-1)

        def forward_logits(self, x):
            """``(batch, n_tasks, n_classes)``"""
            z = self.trunk(x)
            return torch.stack([head(z) for head in self.task_heads], dim=1)

        forward = forward_logits

    return _MultiTaskClassifier()


def MultiTaskClassifier(n_in: int, hidden_dims: Sequence[int] = (512, 256),
                        dropout: float = 0.25, n_tasks: int = 1, n_classes: int = 2):
    """Factory for the multi-task module. Needs ``torch``."""
    return _build_module(n_in, hidden_dims, dropout, n_tasks, n_classes)


def masked_multitask_cross_entropy(logits, targets, class_weights=None,
                                   sample_weights=None):
    """Cross-entropy averaged only over ``(row, task)`` pairs that carry a label.

    This is what lets a compound with one measurement contribute: it trains that
    endpoint's head and leaves the others alone. ``targets`` is ``(batch, n_tasks)`` with
    NaN for "no label".
    """
    torch = require("torch")
    nn = torch.nn
    n_tasks = logits.shape[1]
    total = torch.tensor(0.0, device=logits.device)
    count = torch.tensor(0.0, device=logits.device)
    for t in range(n_tasks):
        target = targets[:, t]
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            continue
        weight = None if class_weights is None else class_weights[t]
        per_row = nn.functional.cross_entropy(
            logits[mask, t, :], target[mask].long(), weight=weight, reduction="none"
        )
        if sample_weights is not None:
            sw = sample_weights[mask]
            total = total + (per_row * sw).sum()
            count = count + sw.sum()
        else:
            total = total + per_row.sum()
            count = count + mask.sum()
    if float(count) == 0.0:
        return torch.tensor(0.0, requires_grad=True, device=logits.device)
    return total / count


def class_weights_per_task(Y: np.ndarray, *, n_classes: int = 2, device=None,
                           enabled: bool = True):
    """Balanced class weights per task, computed on that task's labelled rows only."""
    if not enabled:
        return None
    torch = require("torch")
    out = []
    for t in range(Y.shape[1]):
        col = Y[:, t]
        labelled = col[np.isfinite(col)].astype(int)
        if len(labelled) == 0:
            out.append(torch.ones(n_classes, dtype=torch.float32, device=device))
            continue
        counts = np.bincount(labelled, minlength=n_classes).astype(np.float64)
        counts[counts == 0] = 1.0
        weights = len(labelled) / (n_classes * counts)
        out.append(torch.tensor(weights, dtype=torch.float32, device=device))
    return out


def pretrain_trunk(
    X: np.ndarray,
    y: np.ndarray,
    *,
    hidden_dims: Sequence[int] = (512, 256),
    dropout: float = 0.25,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    epochs: int = 40,
    batch_size: int = 128,
    n_tasks: int = 1,
    n_classes: int = 2,
    seed: int = 0,
    device=None,
    progress: bool = True,
    desc: str = "Pretraining trunk",
):
    """Fit the trunk on a broad potency corpus with a regression pretext task."""
    torch = require("torch")
    device = device or pick_device()
    torch.manual_seed(seed)
    model = _build_module(X.shape[1], hidden_dims, dropout, n_tasks, n_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    yt = torch.tensor(np.asarray(y, dtype=np.float32), device=device)
    n = len(X)
    it = range(epochs)
    if progress:
        it = tqdm(it, desc=desc, unit="epoch")
    model.train()
    for _ in it:
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model.forward_pretrain(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
    return model


def finetune(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    trunk_state=None,
    hidden_dims: Sequence[int] = (512, 256),
    dropout: float = 0.25,
    lr: float = 5e-4,
    weight_decay: float = 1e-5,
    total_epochs: int = 60,
    freeze_epochs: int = 10,
    batch_size: int = 32,
    sample_weight: Optional[np.ndarray] = None,
    balanced: bool = True,
    n_classes: int = 2,
    seed: int = 0,
    device=None,
    progress: bool = True,
    desc: str = "Fine-tuning",
):
    """Train the multi-task classifier, optionally from a pretrained trunk.

    ``trunk_state=None`` means random initialisation, which is both the default and the
    ablation's control arm. With no pretrained trunk there is nothing to thaw, so the
    freeze schedule is skipped entirely rather than wasting the first epochs.
    """
    torch = require("torch")
    device = device or pick_device()
    torch.manual_seed(seed)
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    model = _build_module(X.shape[1], hidden_dims, dropout, Y.shape[1], n_classes)
    transferring = trunk_state is not None
    if transferring:
        model.trunk.load_state_dict(trunk_state)
    model = model.to(device)

    Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
    Yt = torch.tensor(Y.astype(np.float32), device=device)
    swt = (
        None
        if sample_weight is None
        else torch.tensor(np.asarray(sample_weight, dtype=np.float32), device=device)
    )
    weights = class_weights_per_task(
        Y, n_classes=n_classes, device=device, enabled=balanced
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    freeze_for = int(freeze_epochs) if transferring else 0
    if freeze_for:
        for p in model.trunk.parameters():
            p.requires_grad = False

    n = len(X)
    it = range(total_epochs)
    if progress:
        it = tqdm(it, desc=desc, unit="epoch", leave=False)
    model.train()
    for epoch in it:
        if freeze_for and epoch == freeze_for:
            for p in model.trunk.parameters():
                p.requires_grad = True
            opt = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            logits = model.forward_logits(Xt[idx])
            loss = masked_multitask_cross_entropy(
                logits, Yt[idx], class_weights=weights,
                sample_weights=None if swt is None else swt[idx],
            )
            loss.backward()
            opt.step()
    model.eval()
    return model


def predict_proba_multitask(model, X: np.ndarray, *, device=None) -> np.ndarray:
    """``(n, n_tasks, n_classes)`` probabilities."""
    torch = require("torch")
    device = device or pick_device()
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        logits = model.forward_logits(Xt)
        return torch.softmax(logits, dim=-1).cpu().numpy()


def predict_with_mc_dropout(
    model, X: np.ndarray, n_passes: int = 20, *, device=None
) -> Tuple[np.ndarray, np.ndarray]:
    """Mean and standard deviation over ``n_passes`` stochastic forward passes.

    The spread is *model* disagreement under dropout, not predictive error: a confidently
    wrong prediction has a small spread. The calibrated probability is the uncertainty
    that matters.
    """
    torch = require("torch")
    device = device or pick_device()
    model.train()  # keep dropout active
    outs = []
    with torch.no_grad():
        Xt = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        for _ in range(n_passes):
            outs.append(torch.softmax(model.forward_logits(Xt), dim=-1).cpu().numpy())
    model.eval()
    stack = np.stack(outs, axis=0)
    return stack.mean(axis=0), stack.std(axis=0)


# --------------------------------------------------------------------------- the ablation


@dataclass
class PretrainingAblation:
    """Paired comparison: the same architecture with and without a pretrained trunk."""

    with_pretraining: Dict[str, float] = field(default_factory=dict)
    without_pretraining: Dict[str, float] = field(default_factory=dict)
    per_fold: List[Dict[str, object]] = field(default_factory=list)

    @property
    def delta(self) -> Dict[str, float]:
        return {
            k: self.with_pretraining.get(k, float("nan"))
            - self.without_pretraining.get(k, float("nan"))
            for k in self.with_pretraining
        }

    def summary(self) -> str:
        lines = ["Pretraining ablation (paired, same folds, same architecture):"]
        for k, v in self.delta.items():
            lines.append(
                f"  {k:<12s} with {self.with_pretraining.get(k, float('nan')):.4f}  "
                f"without {self.without_pretraining.get(k, float('nan')):.4f}  "
                f"delta {v:+.4f}"
            )
        lines.append(
            "\nInterpret a small delta as what it is. On this dataset two ablations in "
            "opposite\ndirections both came back empty, which is why transfer is off by "
            "default."
        )
        return "\n".join(lines)


def run_pretraining_ablation(
    X: np.ndarray,
    Y: np.ndarray,
    folds,
    *,
    pretrain_X: np.ndarray,
    pretrain_y: np.ndarray,
    hidden_dims: Sequence[int] = (512, 256),
    dropout: float = 0.25,
    lr: float = 5e-4,
    pretrain_epochs: int = 40,
    total_epochs: int = 60,
    freeze_epochs: int = 10,
    batch_size: int = 32,
    seed: int = 0,
    device=None,
    verbose: bool = True,
) -> PretrainingAblation:
    """Measure what pretraining buys, on the same folds, paired.

    Both arms share the folds, the architecture and the seed, so the only difference is
    the trunk's starting point. Anything less controlled than that cannot answer the
    question -- which is why the result here is trustworthy even though it is negative.
    """
    from ..evaluation.metrics import safe_roc_auc

    device = device or pick_device()
    Y = np.asarray(Y, dtype=float)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

    trunk = pretrain_trunk(
        pretrain_X, pretrain_y, hidden_dims=hidden_dims, dropout=dropout,
        epochs=pretrain_epochs, n_tasks=Y.shape[1], seed=seed, device=device,
        progress=verbose, desc="Ablation: pretraining a trunk",
    )
    trunk_state = {k: v.detach().cpu() for k, v in trunk.trunk.state_dict().items()}

    rows: List[Dict[str, object]] = []
    for fold in folds:
        test = fold.mask
        train = ~fold.mask
        for arm, state in (("with", trunk_state), ("without", None)):
            model = finetune(
                X[train], Y[train], trunk_state=state, hidden_dims=hidden_dims,
                dropout=dropout, lr=lr, total_epochs=total_epochs,
                freeze_epochs=freeze_epochs, batch_size=batch_size, seed=seed,
                device=device, progress=False,
            )
            proba = predict_proba_multitask(model, X[test], device=device)
            for t in range(Y.shape[1]):
                col = Y[test][:, t]
                keep = np.isfinite(col)
                if keep.sum() == 0 or len(np.unique(col[keep])) < 2:
                    continue
                rows.append(
                    {
                        "fold": fold.name,
                        "arm": arm,
                        "task": t,
                        "roc_auc": safe_roc_auc(col[keep].astype(int), proba[keep, t, 1]),
                    }
                )

    result = PretrainingAblation(per_fold=rows)
    for arm, target in (("with", result.with_pretraining), ("without", result.without_pretraining)):
        vals = [r["roc_auc"] for r in rows if r["arm"] == arm and np.isfinite(r["roc_auc"])]
        target["roc_auc"] = float(np.mean(vals)) if vals else float("nan")
    if verbose:
        print(result.summary())
    return result
