import torch
import torch.nn as nn
import torchvision.models as models
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union


class BaseBackbone(nn.Module, ABC):
    """
    Abstract base class for backbone architectures.

    All backbones must implement:
    - feature_dim: Number of output channels
    - _build_features(): Method to construct feature extractor
    """

    def __init__(self, weights: Optional[str] = "default", spatial: bool = True):
        """
        Initialize backbone.

        Args:
            weights: Pretrained weights name, "default", or None
            spatial: If True, return spatial features. If False, return pooled features
        """
        super().__init__()

        self.spatial = spatial
        self._frozen = False

        # Build feature extractor
        self._build_features(weights)

        # Add pooling if needed
        if not spatial:
            self.pool = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten()
            )
        else:
            self.pool = nn.Identity()

    @property
    @abstractmethod
    def feature_dim(self) -> int:
        """Number of output channels."""
        pass

    @abstractmethod
    def _build_features(self, weights: Optional[str]) -> None:
        """Build the feature extraction layers."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W)

        Returns:
            Features of shape (B, C, H, W) if spatial, else (B, C)
        """
        features = self.features(x)
        return self.pool(features)

    def freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self._frozen = True

    def unfreeze(self) -> None:
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
        self._frozen = False

    @property
    def is_frozen(self) -> bool:
        """Check if the backbone is frozen."""
        return self._frozen

    def output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get output shape for given input shape.

        Args:
            input_shape: Input tensor shape (B, C, H, W)

        Returns:
            Output shape tuple
        """
        x = torch.zeros(input_shape)
        with torch.no_grad():
            output = self.forward(x)
        return tuple(output.shape)


class ResNet50(BaseBackbone):
    """
    ResNet-50 backbone.
    Input: 224x224, Output: 2048 channels, Spatial: 7x7
    """

    @property
    def feature_dim(self) -> int:
        return 2048

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V2"

        base = models.resnet50(weights=weights)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(base.children())[:-2])


class ResNet18(BaseBackbone):
    """
    ResNet-18 backbone.
    Input: 224x224, Output: 512 channels, Spatial: 7x7
    """
    @property
    def feature_dim(self) -> int:
        return 512

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V2"

        base = models.resnet18(weights=weights)
        # # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(base.children())[:-2])


class ResNet34(BaseBackbone):
    """
    ResNet-34 backbone.
    Input: 224x224, Output: 512 channels, Spatial: 7x7
    """

    @property
    def feature_dim(self) -> int:
        return 512

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V2"

        base = models.resnet34(weights=weights)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(base.children())[:-2])


class ConvNeXtBase(BaseBackbone):
    """
    ConvNeXt-Base backbone.
    Input: 224x224, Output: 1024 channels, Spatial: 7x7
    """

    @property
    def feature_dim(self) -> int:
        return 1024

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = models.convnext_base(weights=weights)
        self.features = base.features


class ConvNextSmall(BaseBackbone):
    """
    ConvNext-Small backbone.
    Input: 224x224, Output: 1024 channels, Spatial: 7x7
    """
    @property
    def feature_dim(self) -> int:
        return 1024

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = models.convnext_small(weights=weights)
        self.features = base.features


class ConvNextTiny(BaseBackbone):
    """
    ConvNext-Tiny backbone.
    Input: 224x224, Output: 768 channels, Spatial: 7x7
    """
    @property
    def feature_dim(self) -> int:
        return 768

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = models.convnext_tiny(weights=weights)
        self.features = base.features


class InceptionV3(BaseBackbone):
    """
    InceptionV3 backbone.
    Input: 299x299, Output: 2048 channels, Spatial: 8x8
    """

    @property
    def feature_dim(self) -> int:
        return 2048

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = models.inception_v3(weights=weights, aux_logits=True)
        base.aux_logits = False
        base.AuxLogits = None
        # Extract layers up to Mixed_7c
        layers = []
        for name, module in base.named_children():
            layers.append(module)
            if name == 'Mixed_7c':
                break

        self.features = nn.Sequential(*layers)


class SwinTransformerB(BaseBackbone):
    """
    Swin Transformer-Base backbone.
    Input: 224x224, Output: 1024 channels, Spatial: 7x7
    """

    @property
    def feature_dim(self) -> int:
        return 1024

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = models.swin_b(weights=weights)

        # Extract the parts we need
        self.features = nn.Sequential(
            base.features,
            base.norm,
            base.permute  # This already outputs (B, 7, 7, 1024)
        )


class SwinTransformerS(BaseBackbone):
    """
    Swin-Transformer-Small backbone.
    Input: 224x224, Output: 1024 channels, Spatial: 7x7
    """

    @property
    def feature_dim(self) -> int:
        return 1024

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = models.swin_s(weights=weights)

        # Extract the parts we need
        self.features = nn.Sequential(
            base.features,
            base.norm,
            base.permute  # This already outputs (B, 7, 7, 1024)
        )


class SwinTransformerT(BaseBackbone):
    """
    Swin-Transformer-Tiny backbone.
    Input: 224x224, Output: 768 channels, Spatial: 7x7
    """

    @property
    def feature_dim(self) -> int:
        return 768

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = models.swin_t(weights=weights)

        # Extract the parts we need
        self.features = nn.Sequential(
            base.features,
            base.norm,
            base.permute  # This already outputs (B, 7, 7, 1024)
        )


class ResNeXt50(BaseBackbone):
    """
    ResNeXt-50 backbone.
    Input: 224x224, Output: 2048 channels, Spatial: 7x7
    """
    @property
    def feature_dim(self) -> int:
        return 2048

    def _build_features(self, weights: Optional[str]) -> None:
        if weights == "default":
            weights = "IMAGENET1K_V1"

        base = base = models.resnext50_32x4d(weights=weights)
        # Remove avgpool and fc layers
        self.features = nn.Sequential(*list(base.children())[:-2])

# Backbone registry
BACKBONES = {
    "resnet18": ResNet18,
    "resnet34": ResNet34,
    "resnet50": ResNet50,
    "convnext_b": ConvNeXtBase,
    "convnext_s": ConvNextSmall,
    "convnext_t": ConvNextTiny,
    "inception_v3": InceptionV3,
    "swin_b": SwinTransformerB,
    "swin_s": SwinTransformerS,
    "swin_t": SwinTransformerT,
    "resnext50": ResNeXt50
}


def create_backbone(
        name: str,
        weights: Optional[str] = "default",
        spatial: bool = True,
        freeze: bool = False
) -> BaseBackbone:
    """
    Create a backbone instance.

    parameter:
    __________
    name: Backbone name ('resnet50', 'convnext_base', 'inception_v3', 'swin_b')
    weights: Pretrained weights - "default", specific name, or None
    spatial: Return spatial features if True, pooled if False
    freeze: Freeze backbone if True

    Returns:
    ________
    Backbone instance

    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone: {name}. Available: {list(BACKBONES.keys())}")

    backbone_class = BACKBONES[name]
    backbone = backbone_class(weights=weights, spatial=spatial)

    if freeze:
        backbone.freeze()

    return backbone



class BackboneWithModel(nn.Module):
    """
    Combine a backbone with a downstream model.
    """

    def __init__(
            self,
            backbone: Union[BaseBackbone, str],
            model: nn.Module,
            freeze_backbone: bool = True
    ):
        """
        Initialize combined model.

        Parameters:
        ___________
        backbone: Backbone instance or name string
        model: Downstream model that expects features
        freeze_backbone: Whether to freeze backbone initially
        """
        super().__init__()

        # Create backbone if string
        if isinstance(backbone, str):
            backbone = create_backbone(backbone, spatial=True)

        self.backbone = backbone
        self.model = model

        if freeze_backbone:
            self.backbone.freeze()

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Forward pass through backbone and downstream model.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (B, 3, H, W).
        return_attention_weights : bool, optional
            If True, the downstream model's per-layer attention probabilities are
            returned. The wrapped model is expected to accept this keyword and
            return ``(logits, attention)``; ``attention`` is ``None`` when not
            requested. This lets callers obtain attention *through the wrapper*
            without bypassing it, e.g.::

                logits, attention = model(images, return_attention_weights=True)

        Returns
        -------
        Tuple[torch.Tensor, Optional[list]]
            ``(logits, attention)`` where ``logits`` has shape (B, n_labels) and
            ``attention`` is a list of per-layer ``(B, n_heads, S, S)`` tensors
            or ``None``.
        """
        features = self.backbone(x)
        return self.model(features, return_attention_weights=return_attention_weights)

    def freeze_backbone(self):
        """Freeze backbone parameters."""
        self.backbone.freeze()

    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        self.backbone.unfreeze()

    @property
    def is_backbone_frozen(self) -> bool:
        """Check if the backbone is frozen."""
        return self.backbone.is_frozen