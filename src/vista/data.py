import torch
import numpy as np
import pandas as pd
from PIL import Image
from typing import Optional, Dict, Tuple, List, Union
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset


def image_augmentation(
        image_size: int = 224,
        n_channels: int = 3,
        p_grayscale: float = 0.2,
        normalize: bool = True,
        aug_params: Optional[Dict] = None,
        chemistry_safe: bool = False,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Generates image transformation pipelines for training and validation/testing.

    These transformations prepare images for deep learning models, including resizing,
    data augmentation for training robustness, proper ImageNet normalization, and
    ensuring a consistent number of channels.

    Args:
        image_size (int): The target height and width for resizing images. Defaults to 224.
        n_channels (int): The desired number of output channels (e.g., 3 for RGB).
                          Handles conversion from 1-channel (grayscale) or >n-channel
                          (e.g., RGBA) inputs. Default to 3.
        p_grayscale (float): The probability of applying the RandomGrayscale transform
                             during training. Must be between 0 and 1. Defaults to 0.2.
                             Ignored when ``chemistry_safe=True``.
        normalize (bool): Whether to apply ImageNet normalization after converting to tensor.
        aug_params (dict): Dictionary with augmentation parameters to override defaults.
                          Available keys: 'rotation', 'affine_degrees', 'translate', 'scale',
                          'blur_prob', 'color_jitter'.
        chemistry_safe (bool): If True, build a *chemistry-preserving* training
                          pipeline that keeps geometry augmentations (rotation,
                          horizontal/vertical flips, mild affine translate/scale)
                          and mild brightness/contrast, but applies **no hue
                          distortion and no random grayscale** — both of which can
                          corrupt chemically meaningful atom colors in molecular
                          renderings. Leave False to reproduce previously completed
                          training (which used color jitter + random grayscale).

    Returns:
        tuple: A tuple containing two torchvision.transforms.Compose objects:
               - train_transforms: Augmentations suitable for training data.
               - val_test_transforms: Deterministic, non-augmented transforms for
                 validation/test data. Use these (on every split, including train)
                 whenever extracting attention so maps are never produced from
                 randomly augmented images.
    """
    # Validate input probability for grayscale
    if not (0 <= p_grayscale <= 1):
        raise ValueError("p_grayscale must be between 0 and 1")

    # --- Default Augmentation Parameters ---
    # These parameters provide moderate augmentation that works well for most molecule datasets.
    # Users can override any of these by providing their own aug_params dictionary
    default_aug_params = {
        'rotation': 10,  # Random rotation in degrees
        'affine_degrees': 5,  # Affine transformation rotation range
        'translate': (0.05, 0.05),  # Maximum translation as a fraction of image size
        'scale': (0.95, 1.05),  # Scale factor range for random scaling
        'blur_prob': 0.5,  # Probability of applying Gaussian blur
        'color_jitter': 0.15  # Strength of color jittering (brightness, contrast, etc.)
    }

    # Update defaults with user-provided parameters
    if aug_params is not None:
        default_aug_params.update(aug_params)

    aug_params = default_aug_params

    # --- Base Transformations Applied to Both Training and Validation ---
    # These transformations ensure images are in the correct format for the model
    base_transforms = [
        # Resize the image to a fixed size. This is often required by model architectures.
        # Using antialias=True for better quality when resizing
        transforms.Resize((image_size, image_size), antialias=True),

        # Convert PIL Image to PyTorch Tensor and scale pixel values to [0, 1] range
        transforms.ToTensor(),
    ]

    # --- Training Data Augmentations ---
    # These transformations are applied to introduce variability and make the model
    # more robust to variations in the input data during training.
    #
    # Geometry augmentations are chemistry-safe (they do not change atom colors),
    # so they are shared by both pipelines.
    geometry_augs = [
        # Apply random rotation within the specified degree range
        transforms.RandomRotation(degrees=aug_params['rotation']),

        # Apply random affine transformations: translation, scaling, shearing
        # translate: maximum absolute fraction for horizontal and vertical translations
        # scale: scaling factor interval
        transforms.RandomAffine(
            degrees=aug_params['affine_degrees'],
            translate=aug_params['translate'],
            scale=aug_params['scale']
        ),

        # Horizontally flip the image with 50% probability
        transforms.RandomHorizontalFlip(p=0.5),

        # Vertically flip the image with 50% probability
        transforms.RandomVerticalFlip(p=0.5),

        # Apply Gaussian blur with specified probability
        # Can help the model be less sensitive to noise and fine details
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))
        ], p=aug_params['blur_prob']),
    ]

    if chemistry_safe:
        # Chemistry-preserving photometric augmentation: mild brightness/contrast
        # only, NO hue shift and NO random grayscale (both can corrupt chemically
        # meaningful atom colors in molecular renderings).
        color_augs = [
            transforms.ColorJitter(
                brightness=aug_params['color_jitter'],
                contrast=aug_params['color_jitter'],
                saturation=0.0,
                hue=0.0,
            ),
        ]
    else:
        # Original pipeline (reproduces completed training): hue jitter + grayscale.
        color_augs = [
            # Apply random changes to brightness, contrast, saturation, and hue
            transforms.ColorJitter(
                brightness=aug_params['color_jitter'],
                contrast=aug_params['color_jitter'],
                saturation=aug_params['color_jitter'],
                hue=aug_params['color_jitter']
            ),

            # Convert the image to grayscale with the specified probability
            transforms.RandomGrayscale(p=p_grayscale),
        ]

    train_augs = [*geometry_augs, *color_augs]

    # --- Channel Adjustment ---
    # Custom transform to ensure the tensor has the desired number of channels (n_channels)
    # This handles cases where the input might be grayscale (1 channel),
    # have an alpha channel (4 channels), or already be RGB (3 channels)
    channel_transform = transforms.Lambda(lambda x: x.repeat(n_channels, 1, 1) if x.size(0) == 1
                          else (x[:n_channels, :, :] if x.size(0) > n_channels else x))

    # --- ImageNet Normalization ---
    # Apply standard ImageNet normalization if requested
    # This normalizes the [0, 1] tensor values using ImageNet mean and std
    # Important: This must come AFTER ToTensor() which scales to [0, 1]
    normalization = []
    if normalize:
        normalization.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet channel means
                std=[0.229, 0.224, 0.225]  # ImageNet channel standard deviations
            )
        )

    # --- Combine All Transformations ---
    # Training pipeline: includes all augmentations for robustness
    train_transforms = transforms.Compose([
        *base_transforms,
        *train_augs,
        channel_transform,
        *normalization
    ])

    # Validation/Test pipeline: minimal transformations for consistent evaluation
    val_transforms = transforms.Compose([
        *base_transforms,
        channel_transform,
        *normalization
    ])

    return train_transforms, val_transforms


class MultiLabelMoleculeDataset(Dataset):
    """
    Dataset for molecular images with multi-label binary classification.

    This dataset expects a CSV file where:
    - One column contains image filenames
    - Multiple columns contain binary labels (0 or 1) for each property

    Example CSV format:
        filename,aromatic, polar,charged, cyclic
        mol001.png,1,0,0,1
        mol002.png,0,1,1,0

    The dataset ensures all images are loaded as RGB and properly transformed.
    """

    def __init__(
            self,
            image_dir: Union[str, Path],
            annotations_csv: Union[str, Path],
            label_columns: List[str],
            transform: Optional[transforms.Compose] = None,
            filename_column: str = 'filename',
            cache_images: bool = False
    ):
        """
        Initialize the multi-label molecular dataset.

        Args:
            image_dir: Directory containing the molecular PNG images
            annotations_csv: Path to CSV file with image filenames and labels
            label_columns: List of column names containing binary labels
            transform: Optional image transformations to apply
            filename_column: Name of the column containing image filenames
            cache_images: If True, cache images in memory (faster but uses more RAM)
        """
        self.image_dir = Path(image_dir)
        self.annotations_csv = Path(annotations_csv)
        self.label_columns = label_columns
        self.transform = transform
        self.filename_column = filename_column
        self.cache_images = cache_images

        # Cache for storing loaded images
        self._image_cache = {} if cache_images else None

        # Load and validate annotations
        self._load_annotations()

        # Print dataset summary
        self._print_loading_summary()

    def _load_annotations(self):
        """
        Load annotations from CSV and validate the data.

        This method reads the CSV file, extracts filenames and labels,
        and performs validation to ensure data integrity.
        """
        # Read CSV file
        if not self.annotations_csv.exists():
            raise FileNotFoundError(f"Annotations file not found: {self.annotations_csv}")

        df = pd.read_csv(self.annotations_csv)

        # Validate required columns exist
        required_columns = [self.filename_column] + self.label_columns
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in CSV: {missing_columns}\n"
                f"Available columns: {list(df.columns)}"
            )

        # Filter for PNG files only
        mask = df[self.filename_column].str.lower().str.endswith('.png')
        df_filtered = df[mask].copy()

        if len(df_filtered) == 0:
            raise ValueError("No PNG files found in annotations file")

        if len(df_filtered) < len(df):
            print(f"Filtered out {len(df) - len(df_filtered)} non-PNG files")

        # Extract filenames
        self.filenames = df_filtered[self.filename_column].tolist()

        # Extract and validate labels
        label_data = df_filtered[self.label_columns].values

        # Check for invalid values (should be 0 or 1)
        unique_values = np.unique(label_data)
        invalid_values = [v for v in unique_values if v not in [0, 1]]

        if invalid_values:
            raise ValueError(
                f"Invalid label values found: {invalid_values}. "
                f"Labels must be binary (0 or 1)."
            )

        # Convert to float32 for compatibility with loss functions
        self.labels = label_data.astype(np.float32)

        # Validate that images exist
        self._validate_image_files()

    def _validate_image_files(self):
        """
        Check that all image files exist and filter out missing ones.

        This method ensures we only keep samples where the image file
        actually exists in the specified directory.
        """
        valid_indices = []
        missing_files = []

        for idx, filename in enumerate(self.filenames):
            filepath = self.image_dir / filename
            if filepath.exists():
                valid_indices.append(idx)
            else:
                missing_files.append(filename)

        if missing_files:
            print(f"\nWarning: {len(missing_files)} image files not found")
            if len(missing_files) <= 5:
                print(f"Missing files: {missing_files}")
            else:
                print(f"Missing files (first 5): {missing_files[:5]} ...")

            # Keep only valid samples
            self.filenames = [self.filenames[i] for i in valid_indices]
            self.labels = self.labels[valid_indices]

            print(f"Continuing with {len(self.filenames)} valid images")

    def _print_loading_summary(self):
        """Print a summary of the loaded dataset."""
        print("\nDataset loaded successfully!")
        print(f"Number of images: {len(self)}")
        print(f"Number of labels: {len(self.label_columns)}")
        print(f"Labels: {', '.join(self.label_columns)}")
        print(f"Image directory: {self.image_dir}")

        # Calculate basic statistics
        labels_per_image = self.labels.sum(axis=1)
        print(f"\nAverage labels per image: {labels_per_image.mean():.2f}")
        print(f"Min labels per image: {int(labels_per_image.min())}")
        print(f"Max labels per image: {int(labels_per_image.max())}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.filenames)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Index of the sample to retrieve

        Returns:
            Tuple of (image_tensor, label_tensor) where:
            - image_tensor has shape (3, H, W)
            - label_tensor has shape (num_labels, ) with binary values
        """
        if not 0 <= idx < len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        # Load image
        filename = self.filenames[idx]
        filepath = self.image_dir / filename

        # Check cache first if caching is enabled
        if self.cache_images and filename in self._image_cache:
            image = self._image_cache[filename].copy()
        else:
            try:
                # Load image and convert to RGB
                image = Image.open(filepath).convert('RGB')

                # Cache if enabled
                if self.cache_images:
                    self._image_cache[filename] = image.copy()

            except Exception as e:
                raise RuntimeError(f"Error loading image {filepath}: {e}")

        # Apply transformations
        if self.transform:
            image = self.transform(image)
        else:
            # If no transform provided, at least convert to tensor
            image = transforms.ToTensor()(image)

        # Get labels for this sample
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)

        return image, labels

    def get_label_statistics(self) -> pd.DataFrame:
        """
        Calculate detailed statistics for each label.

        Returns:
            DataFrame with statistics for each label including
            - Positive count
            - Negative count
            - Frequency (percentage of positive samples)
            - Positive weight (for loss function weighting)
        """
        stats = []

        for i, label_name in enumerate(self.label_columns):
            label_data = self.labels[:, i]
            positive_count = int(label_data.sum())
            total_count = len(label_data)
            negative_count = total_count - positive_count
            frequency = positive_count / total_count

            # Calculate positive weight for handling imbalance
            # This is useful for weighted loss functions
            pos_weight = negative_count / max(positive_count, 1)

            stats.append({
                'label': label_name,
                'positive_count': positive_count,
                'negative_count': negative_count,
                'total_count': total_count,
                'frequency': frequency,
                'pos_weight': pos_weight
            })

        return pd.DataFrame(stats)

    def get_label_frequencies(self) -> List[float]:
        """
        Get the frequency of positive samples for each label.

        This is useful for initializing loss functions like ASL
        that adapt to label frequencies.

        Returns:
            List of frequencies (0-1) for each label
        """
        return [self.labels[:, i].mean() for i in range(len(self.label_columns))]
