from typing import Tuple, Any, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from masking import (
    AttentionMaskConfig,
    build_attention_block_mask,
    apply_block_mask_to_logits,
)


class FeatureEmbedding(nn.Module):
    """
    Modified PatchEmbedding that takes features instead of raw images.
    Adds multiple class tokens (one per label) and position embeddings.

    Parameters:
    -----------
    feature_size : tuple
        The spatial size of the input features (H, W).

    feature channels : int
        The number of the channels in the input features (e.g., 2048 for ResNet-50)

    embed_dim : int
        The embedding dimension.

    number_labels:int
        The number of the labels for the multi-label classification.

    dropout: float, optional
        The dropout probability for the embedding. Default is 0.0.

    Attributes:
    -----------
    feature_size : tuple
        Spatial size of the input features.

    feature_channels : int
        Number of input feature channels.

    embed_dim : int
        Dimensionality of the embedding.

    number_labels : int
        Number of labels and class tokens.

    n_patches : int
        The total number of spatial positions in the feature map.

    projection: nn.Linear
        Linear layer to project features to embedding space.

    cls_tokens: nn.Parameter
        learnable parameters representing classification tokens for each label.
        Shape: (1, number_labels, embedding_dim).

    pos_embedding: nn.Parameter
        learnable positional embedding added to the embeddings.
        Shape: (1, n_patches + number_labels, embedding_dim).

    dropout: nn.Dropout
        Dropout layer applied to the final embeddings.
    """

    def __init__(self, feature_size:tuple[int, int], feature_channels:int, embed_dim:int, n_labels:int, dropout=0.):
        super(FeatureEmbedding, self).__init__()

        self.feature_size = feature_size
        self.feature_channels = feature_channels
        self.embed_dim = embed_dim
        self.n_labels = n_labels
        self.n_patches = self.feature_size[0] * self.feature_size[1]

        # Linear Projection
        self.projection = nn.Linear(self.feature_channels, self.embed_dim)

        # Multiple learnable classification tokens (one per label)
        self.cls_tokens = nn.Parameter(torch.randn(1, self.n_labels, self.embed_dim), requires_grad=True)

        # Learnable positional embeddings (for spatial positions + all cls_tokens)
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + n_labels, self.embed_dim),
            requires_grad=True
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the FeatureEmbedding layer.

        Parameters:
        -----------
        x: torch.Tensor
            Input feature tensor of shape (B, C, H, W), where B is batch size,
            C is the number of channels, H is the height, and W is the width.

        Returns:
        ________
        torch.Tensor
            Output tensor of embedded features including the class tokens,
            with shape (B, S, E), where S is sequence length, E is embedding dim.
        """

        batch_size = x.size(0)

        # Flatten spatial dimension and transpose: (B, C, H, W) -> (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)

        # Project features to embedding dimension: (B, H*W, C) -> (B, H*W, E)
        x = self.projection(x)

        # Prepend the classification tokens to the spatial embeddings
        # cls_tokens shape: (1, number_labels, E) -> repeat for batch -> (B, number_labels, E)
        cls_tokens = self.cls_tokens.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # # (B, number_labels + n_patches, E)

        # Add positional embeddings
        x = x + self.pos_embedding

        x = self.dropout(x)

        return x



class MultiLabelClassifier(nn.Module):
    """
    Multi-label classification head for the vision Transformer.
    Takes embeddings of multiple CLS token and produces a prediction for each label.

    Parameters:
    ___________
    embed_dim : int
        Dimensionality of the input ClS token embeddings.

    number_labels : int
        Number of labels for the multi-label classification.

    Attributes:
    ___________
        heads: nn.Module
        List of linear layers, one for each label.
    """

    def __init__(self, embed_dim:int, n_labels:int):
        super(MultiLabelClassifier, self).__init__()

        self.n_labels = n_labels
        # Create a separate head for each label
        self.heads = nn.ModuleList([
            nn.Linear(in_features=embed_dim, out_features=1)
            for _ in range(n_labels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the MultiLabelClassifier layer.

        Parameters:
        ___________
        x: torch.Tensor
            Input tensor representing the output of the transformer encoder,
            shape (B, S, E). The first number_labels tokens are expected to be [CLS] tokens.

        Returns:
        ________
        torch.Tensor
            Output tensor of logits for all labels, shape (B, number_labels).
        """
        # Extract all CLS token embeddings (first number_labels tokens)
        cls_embed = x[:, :self.n_labels] # Shape: (B, number_labels, E)

        # Apply each head to its corresponding cls token
        logits = []
        for i, head in enumerate(self.heads):
            label_logit = head(cls_embed[:, i])  # (B, 1)
            logits.append(label_logit)

        # Concatenate all logits
        logits = torch.cat(logits, dim=1)  # Shape: (B, number_labels)

        return logits


class SelfAttention(nn.Module):

    """
    Multi-Head Self-Attention Mechanism.

    Parameters:
    -----------
    embed_dim : int
        Dimensionality of the input and output embeddings.

    n_heads : int
        Number of parallel attention heads.

    attention_prob_dropout : float, optional
        Dropout probability for attention probabilities (after softmax). Default is 0.0.

    projection_dropout : float, optional
        Dropout probability for the output projection layer. Default is 0.0.

    Attributes:
    ___________
    embed_dim : int
        Dimensionality of the input embeddings.

    n_heads : int
        Number of attention heads.

    head_dim: int
        Dimensionality of each attention head's query, key, and value.
        Calculated as `embedding_dim / n_attention_heads`.

    scaling_factor: float
        Scaling factor for the dot products in attention, typically `1 / sqrt(head_dim)`.

    qkv_layer: nn.Linear
        Linear layer to project input `x` into combined Query, Key, and Value (QKV) tensors.
        This is an optimization to compute Q, K, V in one go.

    attention_probs_dropout: nn.Dropout
        Dropout layer applied to the attention probabilities.

    projection_layer: nn.Linear
        Linear layer to project the concatenated head outputs back to `embedding_dim`.

    projection_dropout: nn.Dropout
        Dropout layer applied to the output of the projection_layer.
    """
    def __init__(self, embed_dim:int, n_heads:int, attention_prob_dropout=0., projection_dropout=0.):
        super(SelfAttention, self).__init__()

        if embed_dim % n_heads != 0:
            raise ValueError("Embedding dimension must be divisible by the number of attention heads.")

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scaling_factor = self.head_dim ** -0.5

        # Combined QKV projection layer
        self.qkv_layer = nn.Linear(self.embed_dim, self.embed_dim * 3)

        self.attention_probs_dropout = nn.Dropout(attention_prob_dropout)

        # Output projection
        self.projection_layer = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection_dropout = nn.Dropout(projection_dropout)

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
        attn_block_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass of the SelfAttention layer.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (B, S, E), where S is sequence length, E is embedding dim.

        return_attention_weights: bool
            If True, the post-softmax attention probabilities (before attention
            dropout) are returned alongside the output. Default is False.

        attn_block_mask: Optional[torch.Tensor]
            Boolean tensor of shape (S, S) (broadcastable over (B, NH, S, S))
            where ``True`` marks an attention entry ``A[i, j]`` to block. Blocked
            logits are set to ``-inf`` *before* softmax so the corresponding
            probabilities are exactly zero. ``None`` disables masking. Adds no
            learned parameters. See ``masking`` for the directed-edge convention.

        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor | None]
            - Output tensor of shape (B, S, E).
            - Attention probabilities tensor of shape (B, NH, S, S) if 'return_attention_weights' is True,
              otherwise None.
        """
        batch_size, seq_len, _  =x.shape
        attention_probs_saved = None # Reset at the beginning of each forward pass

        qkv = self.qkv_layer(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2),
                      qkv
                      ) # # Each (B, NH, S, HD) where NH is n_attention_heads, HD is head_dim

        # Calculate attention scores: (B, NH, S, HD) @ (B, NH, HD, S) -> (B, NH, S, S)
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling_factor

        # Apply the (optional) attention mask to the logits *before* softmax.
        # Blocked entries become -inf -> exactly 0 probability after softmax.
        attention_scores = apply_block_mask_to_logits(attention_scores, attn_block_mask)

        # Apply softmax to get attention probabilities
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # Save attention AFTER softmax but BEFORE attention dropout (analysis uses this).
        if return_attention_weights:
            attention_probs_saved = attention_probs

        # Apply dropout to attention probabilities
        attention_probs = self.attention_probs_dropout(attention_probs)

        # Calculate the weighted sum of values: (B, NH, S, S) @ (B, NH, S, HD) -> (B, NH, S, HD)
        attn_output = torch.matmul(attention_probs, v)

        # Concatenate heads and reshape: (B, NH, S, HD) -> (B, S, NH, HD) -> (B, S, E)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        # Final linear projection
        x = self.projection_layer(attn_output)
        x = self.projection_dropout(x)

        if return_attention_weights:
            return x, attention_probs_saved
        else:
            return x, None



class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feedforward Network) for Transformer block.

    Parameters:
    -----------
    embed_dim : int
        Dimensionality of the input and output embeddings.

    hidden_dim_mult : int
        Multiplier for 'embedding dim' to determine the size of the hidden layer.
        'hidden_dim' = hidden_dim_mult * embed_dim

    dropout_prob : float, optional
        Dropout probability foe the second linear layer. Default is 0.0.

    activation_fn: nn.Module, optional
        Activation function to use. Default is nn.GELU.


    Attributes:
    -----------
    fc1 = nn.Linear
        First linear layer, expands 'embedding dim' to 'hidden_dim'

    activation_fn: nn.Module
        Activation function (e.g., GELU, ReLU)

    fc2: nn.Linear
        Second linear layer, projects `hidden_dim` back to `embedding_dim`.

    dropout: nn.Dropout
        Dropout layer applied after the second linear layer.
    """

    def __init__(self,embed_dim: int, hidden_dim_mult:int = 4,
                 dropout_prob:float=0., activation_fn:nn.Module = nn.GELU()):
        super(MLP, self).__init__()

        hidden_dim = int(hidden_dim_mult * embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.activation_fn = activation_fn
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Multi-Layer Perceptron.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (B, S, E).

        Returns:
        --------
        torch.Tensor
            Output tensor of shape (B, S, E).
        """
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)

        return x

class DropPath(nn.Module):
    """
    True per-sample stochastic depth (DropPath).

    Unlike ``nn.Dropout`` (which zeros individual elements), this drops the
    *entire* residual branch for a random subset of samples in the batch and
    rescales the survivors by ``1/(1-p)``. Identity when ``p == 0`` or in eval
    mode. Disabled by default so it does not affect existing checkpoints. Has no
    learned parameters.
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        if not (0.0 <= drop_prob < 1.0):
            raise ValueError("drop_prob must be in [0, 1).")
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Per-sample mask broadcast over all non-batch dims.
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(shape, dtype=x.dtype, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask


class EncoderBlock(nn.Module):
    """
    A single block of the Vision Transformer encoder.
    Consists of Multi-Head Self-Attention and an MLP, with LayerNorm and residual connections.

    Parameters:
    ___________
    embed_dim : int
        Dimensionality of the input embeddings.

    n_heads : int
        Number of attention heads for the SelfAttention module.

    hidden_dim_mult : int
        Multiplier for `embedding_dim` to set MLP hidden layer size. Default is 4.

    attention_prob_dropout : float, optional
        Dropout probability for attention probabilities in SelfAttention. Default is 0.0.

    attention_output_dropout : float, optional
        Dropout probability for the output projection in SelfAttention. Default is 0.0.

    mlp_dropout : float, optional
        Dropout probability for the MLP module. Default is 0.0.

    mlp_activation_fn: nn.Module, optional
        Activation function for the MLP. Default is `nn.GELU()`.

    path_dropout: float, optional
        Probability for the *residual-path elementwise dropout* applied to the
        attention and MLP sub-block outputs before they are added to the
        residual. NOTE: despite the historical name this is plain ``nn.Dropout``
        (independent Bernoulli noise per element), **not** true stochastic depth
        (which would drop the whole residual branch per-sample). The name and
        ``nn.Dropout`` behaviour are preserved for checkpoint compatibility; for
        genuine stochastic depth set ``stochastic_depth > 0`` instead (disabled
        by default). Default is 0.0.

    stochastic_depth: float, optional
        Probability for *true* per-sample stochastic depth (DropPath) on the
        residual branches. Disabled by default (0.0) so completed-checkpoint
        behaviour is unchanged. When > 0 it is applied in addition to / instead
        of ``path_dropout`` on the residual branch outputs.


    Attributes:
    ___________

    norm1: nn.LayerNorm
        Layer normalization applied before the self-attention module.

    attention: SelfAttention
        The multi-head self-attention module.

    path_dropout1: nn.Dropout
        Residual-path elementwise dropout applied after the attention module
        (before adding the residual). Plain ``nn.Dropout``; see note above.

    norm2: nn.LayerNorm
        Layer normalization applied before the MLP module.

    mlp: MLP
        The multi-layer perceptron modul

    path_dropout2: nn.Dropout
        Residual-path elementwise dropout applied after the MLP module
        (before adding the residual). Plain ``nn.Dropout``; see note above.
    """
    def __init__(self, embed_dim:int, n_heads:int,
                 hidden_dim_mult:int = 4,
                 attention_prob_dropout:float=0.,
                 attention_output_dropout:float=0.,
                 mlp_dropout:float=0.,
                 mlp_activation_fn:nn.Module = nn.GELU(),
                 path_dropout:float=0.,
                 stochastic_depth:float=0.):
        super(EncoderBlock, self).__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = SelfAttention(
            embed_dim = embed_dim,
            n_heads = n_heads,
            attention_prob_dropout = attention_prob_dropout,
            projection_dropout = attention_output_dropout,
        )

        # Residual-path elementwise dropout (plain nn.Dropout; NOT stochastic
        # depth). Name kept for checkpoint compatibility -- see class docstring.
        self.path_dropout1 = nn.Dropout(path_dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(
            embed_dim = embed_dim,
            hidden_dim_mult = hidden_dim_mult,
            dropout_prob = mlp_dropout,
            activation_fn = mlp_activation_fn,
        )
        self.path_dropout2 = nn.Dropout(path_dropout)

        # Optional *true* stochastic depth, disabled by default. Identity unless
        # stochastic_depth > 0, so existing checkpoints are unaffected.
        self.drop_path1 = DropPath(stochastic_depth)
        self.drop_path2 = DropPath(stochastic_depth)


    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
        attn_block_mask: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor | None]:
        """
        Forward pass of the EncoderBlock.

        Parameters:
        -----------
        x: torch.Tensor
            Input tensor of shape (B, S, E).

        return_attention_weights: bool, optional
            If True, attention probabilities from the SelfAttention module are returned.
            Default is False.

        attn_block_mask: Optional[torch.Tensor], optional
            Boolean (S, S) block mask forwarded to the attention module. ``None``
            disables masking at this layer.


        Returns:
        --------
        tuple[torch.Tensor, torch.Tensor | None]
            - Output tensor of shape (B, S, E).
            - Attention probabilities tensor of shape (B, NH, S, S) if `return_attention_weights` is True,
              otherwise None.
        """
        # Self-Attention sub-block. Residual connection is unchanged by masking.
        residual = x
        x = self.norm1(x)
        x, attention_probs = self.attention(
            x,
            return_attention_weights=return_attention_weights,
            attn_block_mask=attn_block_mask,
        )
        x = residual + self.drop_path1(self.path_dropout1(x))

        # MLP sub-block
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + self.drop_path2(self.path_dropout2(x))

        return x, attention_probs


class MultiLabelVisionTransformer(nn.Module):

    """
    Vision Transformer (ViT) model adapted for multi-label classification

    Parameters:
    -----------
    feature_size: tuple[int]
        Spatial size of the input features (H, W).

    feature_channels:
        Number of channels in input features (e.g., 2048 for ResNet).

    embed_dim:
        Dimensionality of the embeddings throughout the transformer.

    n_encoder_layers:
        Number of the encoder blocks in the Transformer.

    n_heads:
        Number of attention heads in the SelfAttention module.

    number_labels:
        Number of labels for multi-label classification.

    mlp_hidden_dim_mult: int, optional
        Multiplier for `embedding_dim` to set MLP hidden layer size. Default is 4.

    dropout_embed: float, optional
        Dropout probability for the FeatureEmbedding layer. Default is 0.0.

    attention_prob_dropout: float, optional
        Dropout for attention probabilities in SelfAttention. Default is 0.0.

    attention_output_dropout: float, optional
        Dropout for the output projection in SelfAttention. Default is 0.0.

    dropout_mlp: float, optional
        Dropout for the MLP module in each encoder block. Default is 0.0.

    path_dropout: float, optional
        Path dropout probability for encoder blocks. Default is 0.0.

    mlp_activation_fn: nn.Module, optional
        Activation function for the MLP. Default is `nn.GELU()`.

    Attributes:
    -----------
    feature_embed: FeatureEmbedding
        The feature embedding layer.

    encoder_blocks: nn.ModuleList
        A list of encoder blocks.

    norm_final: nn.LayerNorm
        Layer normalization applied after the final encoder block,
        before the classifier.

    classifier: MultiLabelClassifier
        The multi-label classification head.
    """
    def __init__(self,
                 feature_size:tuple[int, int],
                 feature_channels:int,
                 embed_dim:int,
                 n_encoder_layers:int,
                 n_heads:int,
                 n_labels:int,
                 mlp_hidden_dim_mult:int = 4,
                 dropout_embed:float=0.,
                 attention_prob_dropout:float = 0.,
                 attention_output_dropout:float = 0.,
                 dropout_mlp:float = 0.,
                 path_dropout:float = 0.,
                 mlp_activation_fn:nn.Module = nn.GELU(),
                 stochastic_depth:float = 0.,
                 # --- attention masking (no learned parameters added) ---------
                 attention_mask_mode: str = "none",
                 masked_layers: Optional[Sequence[int]] = None,
                 blocked_label_edges: Optional[Sequence[Tuple[int, int]]] = None,
                 n_label_tokens: Optional[int] = None,
                 mask_config: Optional[AttentionMaskConfig] = None,
                 ):
        super(MultiLabelVisionTransformer, self).__init__()

        self.feature_embed = FeatureEmbedding(
            feature_size = feature_size,
            feature_channels = feature_channels,
            embed_dim = embed_dim,
            n_labels = n_labels,
            dropout= dropout_embed,
        )

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                embed_dim = embed_dim,
                n_heads = n_heads,
                hidden_dim_mult = mlp_hidden_dim_mult,
                attention_prob_dropout = attention_prob_dropout,
                attention_output_dropout = attention_output_dropout,
                mlp_dropout = dropout_mlp,
                path_dropout = path_dropout,
                stochastic_depth = stochastic_depth,
                mlp_activation_fn = mlp_activation_fn,
            ) for _ in range(n_encoder_layers)
        ])

        self.norm_final = nn.LayerNorm(embed_dim)
        self.classifier = MultiLabelClassifier(embed_dim, n_labels)

        # Bookkeeping used by attention analysis and masking.
        self.n_labels = n_labels
        self.n_encoder_layers = n_encoder_layers
        self.n_patches = self.feature_embed.n_patches
        self.seq_len = self.n_labels + self.n_patches

        # Resolve the mask configuration. An explicit ``mask_config`` wins;
        # otherwise build one from the keyword arguments. Default == no masking,
        # which keeps unmasked checkpoints behaving identically.
        if mask_config is None:
            mask_config = AttentionMaskConfig(
                n_label_tokens=(n_labels if n_label_tokens is None else n_label_tokens),
                mode=attention_mask_mode,
                masked_layers=masked_layers,
                blocked_label_edges=blocked_label_edges,
            )
        self.mask_config = mask_config
        # Concrete set of layers the mask applies to (validated against depth).
        self._masked_layer_set = set(mask_config.resolve_masked_layers(n_encoder_layers))

    def forward(self, x: torch.Tensor, return_attention_weights: bool = False) -> tuple[Tensor, list[Tensor] | None]:
        """
        Forward pass of the MultiLabelVisionTransformer.

        Parameters:
        ___________
        x: torch.Tensor
            Input feature tensor of shape (B, C, H, W).

        return_attention_weights: bool, optional
            If True, a list of attention probabilities from each encoder block is returned.
            Each element in the list is a tensor of shape (B, NH, S, S).
            Default is False.

        Returns:
        --------
        tuple[torch.Tensor, list[torch.Tensor] | None]
            - Output tensor of multi-label logits, shape (B, number_labels).
            - List of attention probabilities (one per encoder block) if
              ``return_attention_weights`` is True, otherwise None.
        """

        all_attention_weights = [] if return_attention_weights else None

        x = self.feature_embed(x) # (B, S, E) where S = number_labels + n_patches

        # Build the boolean block mask once, on the right device, and reuse it
        # for every masked layer. ``None`` when masking is disabled.
        block_mask = build_attention_block_mask(
            seq_len=x.size(1),
            config=self.mask_config,
            device=x.device,
        )

        for layer_idx, block in enumerate(self.encoder_blocks):
            layer_mask = block_mask if layer_idx in self._masked_layer_set else None
            x, attention_weights = block(
                x,
                return_attention_weights=return_attention_weights,
                attn_block_mask=layer_mask,
            )
            if return_attention_weights and attention_weights is not None:
                all_attention_weights.append(attention_weights)

        x = self.norm_final(x)
        logits = self.classifier(x)

        if return_attention_weights:
            return logits, all_attention_weights

        else:
            return logits, None
