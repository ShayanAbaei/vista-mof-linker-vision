import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, List, Sequence, Tuple


# Executed frequency -> (gamma_pos, gamma_neg) schedule, as ascending
# (freq_upper_threshold, gamma_pos, gamma_neg) rows. A label with frequency f is
# assigned the first row whose ``f < freq_upper_threshold``. These reproduce the
# trained checkpoints; override via AsymmetricLossOptimized(freq_gamma_schedule=...).
DEFAULT_FREQ_GAMMA_SCHEDULE: Tuple[Tuple[float, float, float], ...] = (
    (0.05, 1.8, 4.0),   # very rare
    (0.20, 1.5, 3.5),   # rare
    (0.70, 2.5, 2.5),   # balanced
    (0.80, 3.0, 2.0),   # common
    (float("inf"), 3.5, 1.5),  # very common (catch-all)
)


def _gamma_for_frequency(
    freq: float,
    schedule: Sequence[Tuple[float, float, float]],
) -> Tuple[float, float]:
    """Return (gamma_pos, gamma_neg) for ``freq`` from an ascending schedule."""
    for threshold, gamma_pos, gamma_neg in schedule:
        if freq < threshold:
            return float(gamma_pos), float(gamma_neg)
    # Fallback to the last row if no catch-all threshold was provided.
    _, gamma_pos, gamma_neg = schedule[-1]
    return float(gamma_pos), float(gamma_neg)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric loss for multi-label Classification

    Reference:
        Ben-Baruch et al., 2020. "Asymmetric Loss For Multi-Label Classification"
        arXiv:2009.14119 [cs.CV]
        https://arxiv.org/abs/2009.14119

    This loss function addresses the positive-negative imbalance inherent in multi-label
    classification by applying asymmetric focusing on positive and negative samples.
    It's particularly effective for datasets where different labels have vastly different
    frequencies (e.g., some labels appear in 2% of samples while others appear in 80%).


    Loss Formulation:
        - Positive samples (y=1): L+ = -(1-p)^γ+ * log(p)
        - Negative samples (y=0): L- = -(p_m)^γ- * log(1-p_m)
          where p_m = max(p - m, 0) and m is the probability margin

    Parameters:
    -----------
    gamma_neg: float, default=4.0
        Focusing parameter for negative samples. Higher values put more focus on
        hard (uncertain) negative samples. Typical values: 2-6.
        - Higher γ- → more focus on hard negatives
        - Lower γ- → more uniform weighting of negatives

    gamma_pos: float, default=1.0
        Focusing parameter for positive samples. Lower values reduce focus on
        hard positives, which protects rare positive samples from being ignored.
        - γ+ = 0: No focusing, all positives weighted equally (best for rare classes)
        - γ+ = 1: Mild focusing on hard positives
        - γ+ = 2: Standard focusing (only use when positives are common)

    clip : float, default=0.05
        Probability margin for asymmetric clipping. Negative samples with
        probability p < clip are considered "easy" and their gradients are disabled.
        This prevents the numerous easy negatives from dominating the loss.
        - clip = 0.05: Ignore negatives with < 5% probability (recommended)
        - clip = 0.0: No clipping, standard focal loss behavior

    eps : float, default=1e-8
        Small epsilon for numerical stability in logarithm calculations.
        Prevents log(0) which would result in infinite loss.


    Examples:
    ---------
    Standard ASL for multi-label classification
    criteria = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    logits = model(images) # Raw predictions (before sigmoid)
    loss = criterion(logits, targets) # targets are binary {0,1}

    For very imbalanced dataset with rare positives
    criteria = AsymmetricLoss(gamma_neg=5, gamma_pos=0, clip=0.05)

    For relatively balanced dataset
    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.01)
    """
    def __init__(self,
                 gamma_neg: float = 4.0,
                 gamma_pos: float = 1.0,
                 clip: float = 0.05,
                 eps=1e-8
                 ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate the asymmetric loss.

        Parameters:
        -----------
        logits : torch.Tensor
            Raw predictions from the model (before sigmoid).
            Shape: (batch_size, num_labels)

        targets : torch.Tensor
            Binary ground truth labels.
            Shape: (batch_size, num_labels)
            Values should be 0 or 1 for each label.

        Returns:
        --------
        torch.Tensor
            Computed loss value.
        """
        # Convert logits to probabilities using sigmoid activation
        # Sigmoid maps (-∞, +∞) → (0, 1)
        probs = torch.sigmoid(logits)

        # Calculate probability margin for negative samples
        # p_m = max(p - margin, 0) implements hard thresholding
        # When p < clip, p_m = 0, completely disabling gradient
        probs_m = torch.clamp(probs - self.clip, min=0)

        # Positive loss component: -(1-p)^γ+ * log(p)
        # - targets acts as a mask, selecting only positive samples
        # - (1-probs)^γ+ down-weights easy positives (where p ≈ 1)
        # - log(p) is the standard cross-entropy for positive samples
        # - clamp prevents log(0) when p = 0
        loss_pos = targets * torch.pow(1 - probs, self.gamma_pos) * torch.log(probs.clamp(min=self.eps))

        # Negative loss component: -(p_m)^γ- * log(1-p_m)
        # - (1-targets) acts as a mask, selecting only negative samples
        # - (p_m)^γ- down-weights easy negatives (where p_m ≈ 0)
        # - log(1-p_m) is the standard cross-entropy for negative samples
        # - When p < clip, p_m = 0, so the loss contribution is -0^γ- * log(1) = 0
        loss_neg = (1 - targets) * torch.pow(probs_m, self.gamma_neg) * torch.log((1 - probs_m).clamp(min=self.eps))

        # Combine positive and negative losses,
        # The negative sign makes this a loss to minimize
        loss = -(loss_pos + loss_neg)

        # Return mean loss over all samples and labels
        # This makes the loss magnitude independent of batch size
        return loss.mean()



class AsymmetricLossOptimized(nn.Module):
    """
    Asymmetric Loss with automatic per-label parameter adaptation based on label frequencies.

    This optimized version automatically adjusts the focusing parameters (gamma_pos, gamma_neg)
    for each label based on its frequency in the dataset. This is particularly useful for
    multi-label datasets where different labels have vastly different occurrence rates.

    Adaptation Strategy (DEFAULT_FREQ_GAMMA_SCHEDULE):
    --------------------------------------------------
    These are the frequency buckets and (gamma_pos, gamma_neg) values that were
    actually used in the executed training runs. They are exposed as the
    configurable ``freq_gamma_schedule`` argument so experiments can override
    them, but the defaults below reproduce the trained checkpoints.

    - Very rare labels   (freq < 0.05) : γ+ = 1.8, γ- = 4.0
    - Rare labels        (freq < 0.20) : γ+ = 1.5, γ- = 3.5
    - Balanced labels    (0.20–0.70)   : γ+ = 2.5, γ- = 2.5
    - Common labels      (0.70–0.80)   : γ+ = 3.0, γ- = 2.0
    - Very common labels (freq > 0.80) : γ+ = 3.5, γ- = 1.5

    Bucket boundaries are evaluated as ``freq < threshold`` against the ascending
    thresholds (0.05, 0.20, 0.70, 0.80); a frequency at an exact boundary falls
    into the higher bucket (e.g. freq == 0.70 -> the 0.70–0.80 bucket).

    Parameters:
    -----------
    label_frequencies: List[float]
        Frequency of positive samples for each label. Values should be in [0, 1].
        For example, [0.02, 0.95] means label 0 appears in 2% of samples,
        label 1 appears in 95% of samples.

    gamma_neg : float or List[float], default=4.0
        Default negative focusing parameter. Used only if label_frequencies are None.
        It Can be a single value (applied to all labels) or per-label values.

    gamma_pos : float or List[float], default=1.0
        Default positive focusing parameter. Used only if label_frequencies are None.
        It Can be a single value (applied to all labels) or per-label values.

    clip : float, default=0.005
        Probability margin for asymmetric clipping. Matches the executed runs.
        Same role as in base AsymmetricLoss.

    eps : float, default=1e-8
        Numerical stability constant. Same as in base AsymmetricLoss.

    freq_gamma_schedule : Optional[Sequence[Tuple[float, float, float]]]
        Ascending list of ``(freq_upper_threshold, gamma_pos, gamma_neg)`` rows.
        A label with frequency ``f`` gets the first row whose
        ``f < freq_upper_threshold``. Defaults to ``DEFAULT_FREQ_GAMMA_SCHEDULE``
        (the executed values above). Use the final row's threshold >= 1.0 so all
        frequencies are covered.

    Device note
    -----------
    ``gamma_pos`` / ``gamma_neg`` / ``label_frequencies`` are registered as
    buffers, so calling ``criterion.to(device)`` moves them with the module.
    There is intentionally **no** hard-coded ``.to("cuda")`` so the loss is
    portable across CPU/GPU.
    """

    def __init__(
        self,
        label_frequencies: Optional[Union[List[float], torch.Tensor]] = None,
        gamma_neg: Union[float, List[float]] = 4.0,
        gamma_pos: Union[float, List[float]] = 1.0,
        clip: float = 5e-3,
        eps: float = 1e-8,
        freq_gamma_schedule: Optional[Sequence[Tuple[float, float, float]]] = None,
        label_weights: Optional[Union[List[float], torch.Tensor]] = None,
    ):
        super(AsymmetricLossOptimized, self).__init__()

        self.clip = clip
        self.eps = eps
        self.freq_gamma_schedule = (
            list(DEFAULT_FREQ_GAMMA_SCHEDULE)
            if freq_gamma_schedule is None
            else list(freq_gamma_schedule)
        )
        # Per-label loss weights (default all ones). Used by the dominant-label
        # supervision ablation (spec 9B): set a label's weight to 0 to remove its
        # supervision while keeping the architecture and all tokens unchanged.
        self._num_labels = None if label_frequencies is None else len(label_frequencies)
        if label_weights is None:
            self.register_buffer('label_weights', None, persistent=False)
        else:
            self.register_buffer(
                'label_weights', torch.as_tensor(label_weights, dtype=torch.float))

        if label_frequencies is not None:
            # Automatic adaptation based on label frequencies
            if not isinstance(label_frequencies, torch.Tensor):
                label_frequencies = torch.tensor(label_frequencies, dtype=torch.float)
            label_frequencies = label_frequencies.float()

            num_labels = len(label_frequencies)

            # Initialize gamma arrays
            gamma_pos_values = torch.zeros(num_labels)
            gamma_neg_values = torch.zeros(num_labels)

            # Assign gamma values based on label frequency using the configurable
            # ascending schedule (first bucket where freq < threshold wins).
            for i, freq in enumerate(label_frequencies):
                gp, gn = _gamma_for_frequency(float(freq), self.freq_gamma_schedule)
                gamma_pos_values[i] = gp
                gamma_neg_values[i] = gn

            # Register as buffers for automatic device placement. Moving the
            # whole criterion via ``.to(device)`` is the portable way to place
            # these on GPU/CPU (no hard-coded ``.to("cuda")``).
            self.register_buffer('gamma_pos', gamma_pos_values)
            self.register_buffer('gamma_neg', gamma_neg_values)
            self.register_buffer('label_frequencies', label_frequencies)


        else:
            # Manual gamma values (no adaptation)
            if isinstance(gamma_pos, (int, float)):
                self.gamma_pos = float(gamma_pos)
            else:
                self.register_buffer('gamma_pos', torch.tensor(gamma_pos, dtype=torch.float))

            if isinstance(gamma_neg, (int, float)):
                self.gamma_neg = float(gamma_neg)
            else:
                self.register_buffer('gamma_neg', torch.tensor(gamma_neg, dtype=torch.float))


    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate ASL with per-label adapted parameters.

        This forward pass is identical to the base AsymmetricLoss, except that
        gamma values can vary per label based on the initialization strategy.

        Parameters:
        -----------
        logits : torch.Tensor
            Raw model predictions, shape (batch_size, num_labels)

        targets : torch.Tensor
            Binary ground truth labels, shape (batch_size, num_labels)

        Returns:
        --------
        torch.Tensor
            Scalar loss value (mean reduction)
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Calculate probability margin for negative samples
        probs_m = torch.clamp(probs - self.clip, min=0)

        # Positive loss with per-label gamma_pos
        # Broadcasting handles both scalar and tensor gamma values automatically
        loss_pos = targets * torch.pow(1 - probs, self.gamma_pos) * torch.log(probs.clamp(min=self.eps))

        # Negative loss with per-label gamma_neg
        loss_neg = (1 - targets) * torch.pow(probs_m, self.gamma_neg) * torch.log((1 - probs_m).clamp(min=self.eps))

        # Combine; (B, L) before reduction.
        loss = -(loss_pos + loss_neg)

        # Optional per-label supervision weighting (e.g. zero out a dominant
        # label for the supervision ablation).
        if self.label_weights is not None:
            loss = loss * self.label_weights.to(loss.device)

        return loss.mean()

    def set_label_weights(self, label_weights: Optional[Union[List[float], "torch.Tensor"]]):
        """Set/replace per-label loss weights (None == all ones)."""
        if label_weights is None:
            self.label_weights = None
        else:
            w = torch.as_tensor(label_weights, dtype=torch.float)
            if isinstance(self.gamma_pos, torch.Tensor):
                w = w.to(self.gamma_pos.device)
            self.label_weights = w


class BCELoss(nn.Module):
    """
    Binary Cross-Entropy Loss for Multi-Label Classification.

    This is a standard BCE loss implementation that serves as a baseline for comparison
    with Asymmetric Loss.

    Loss Formulation:
        L = -[y * log(p) + (1-y) * log(1-p)]

    Where:
        - y is the target (0 or 1)
        - p is the predicted probability (after sigmoid)

    Parameters:
    -----------
    pos_weight : torch.Tensor or float, optional
        Weight for positive samples. It Can be used to address class imbalance.
        If tensor, should have shape (num_labels)
        ) for per-label weights.
        Values > 1 increase recall, values < 1 increase precision.


    eps : float, default=1e-8
        Small constant for numerical stability in logarithm calculations.

    """

    def __init__(
            self,
            pos_weight: Optional[Union[float, torch.Tensor]] = None,
            eps: float = 1e-8
    ):
        super(BCELoss, self).__init__()

        self.eps = eps

        if pos_weight is not None:
            if isinstance(pos_weight, (int, float)):
                self.pos_weight = float(pos_weight)
            else:
                self.register_buffer('pos_weight', torch.tensor(pos_weight, dtype=torch.float))
        else:
            self.pos_weight = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate Binary Cross-Entropy loss.

        Parameters:
        -----------
        logits : torch.Tensor
            Raw model predictions (before sigmoid).
            Shape: (batch_size, num_labels)

        targets : torch.Tensor
            Binary ground truth labels.
            Shape: (batch_size, num_labels)
            Values should be 0 or 1 for each label.

        Returns:
        --------
        torch.Tensor
            Computed loss value.
        """
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Clamp probabilities for numerical stability
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)

        # Calculate BCE: -[y * log(p) + (1-y) * log(1-p)]
        bce = -(targets * torch.log(probs) + (1 - targets) * torch.log(1 - probs))

        # Apply positive class weights if provided
        if self.pos_weight is not None:
            # pos_weight only applies to positive samples
            # Formula becomes: -[pos_weight * y * log(p) + (1-y) * log(1-p)]
            weight = targets * self.pos_weight + (1 - targets)
            bce = bce * weight

        return bce.mean()


