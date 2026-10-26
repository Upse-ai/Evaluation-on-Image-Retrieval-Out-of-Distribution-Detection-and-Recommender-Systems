"""
Feature Extraction using WideResNet-50-2

Two configurations as required by the project:
- Conf1: ImageNet-pretrained WideResNet-50-2 (IMAGENET1K_V2 weights)
- Conf2: Contrastively Fine-tuned WideResNet-50-2 (ConWideResNet50.pt weights)

Both configurations extract from the SECOND-LAST (penultimate) layer,
producing 2048-dimensional feature vectors.

Class-confidence scores from the final classification layer are also saved,
as they are needed for the NN-Guide OoD detection method (Task 2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import Wide_ResNet50_2_Weights
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional, Union


# ── Conf2 model architecture (from NeuralNetworks/Defining_LoadingTheSupConTrained.py) ──
class SupConClassifier(nn.Module):
    """
    Contrastively trained WideResNet-50-2 for Conf2.
    Returns raw_feat (2048-dim) and logits from the FC layer.
    """
    def __init__(self, num_classes):
        super().__init__()
        m = models.wide_resnet50_2()
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        raw_feat = self.backbone(x).flatten(1)   # 2048-dim features
        z = F.normalize(raw_feat, dim=1)          # L2 normalized (for contrastive model)
        logits = self.fc(z)
        return raw_feat, logits


def load_conf2_model(ckpt_path, device):
    """Load Conf2 model from wrn50_supcon.pth checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    num_classes = len(ckpt['classes'])
    model = SupConClassifier(num_classes).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt['classes']


class WideResNetFeatureExtractor:
    """
    Feature extractor using WideResNet-50-2.

    Conf1: ImageNet-pretrained weights (IMAGENET1K_V2)
    Conf2: Contrastively fine-tuned weights from ConWideResNet50.pt

    Both extract 2048-dim feature vectors from the penultimate layer.
    Class-confidence scores from the final layer are also captured.
    """

    def __init__(self, config: str = 'conf1',
                 conf2_weights_path: Optional[Union[str, Path]] = None,
                 device: Optional[str] = None):
        """
        Initialize the feature extractor.

        Args:
            config: 'conf1' (ImageNet weights) or 'conf2' (ConWideResNet50.pt)
            conf2_weights_path: Path to ConWideResNet50.pt (required for conf2)
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if config not in ('conf1', 'conf2'):
            raise ValueError("config must be 'conf1' or 'conf2'")

        self.config = config

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        if config == 'conf1':
            # Conf1: Standard ImageNet pretrained weights
            self.model = models.wide_resnet50_2(weights=Wide_ResNet50_2_Weights.IMAGENET1K_V2)
            print("Conf1: Loaded IMAGENET1K_V2 pretrained weights")

        else:
            # Conf2: Contrastively fine-tuned WideResNet-50-2 (SupConClassifier)
            # Weights file: NeuralNetworks/wrn50_supcon.pth
            if conf2_weights_path is None:
                raise ValueError(
                    "conf2_weights_path is required for Conf2.\n"
                    "Point to: NeuralNetworks/wrn50_supcon.pth"
                )
            conf2_weights_path = Path(conf2_weights_path)
            if not conf2_weights_path.exists():
                raise FileNotFoundError(
                    f"wrn50_supcon.pth not found at: {conf2_weights_path}"
                )

            # Load using the SupConClassifier architecture defined above
            self.model, self._conf2_classes = load_conf2_model(conf2_weights_path, self.device)
            print(f"Conf2: Loaded contrastive weights from {conf2_weights_path}")
            print(f"       Classes: {len(self._conf2_classes)}")

        self.model = self.model.to(self.device)
        self.model.eval()

        # Hooks to capture penultimate layer features and final logits
        self._features = None
        self._logits = None
        self._setup_hooks()

        # ImageNet normalization + resize to 224x224 (required by WideResNet-50-2)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        print(f"Device: {self.device}")
        print(f"Feature dimension: 2048 (penultimate layer)")

    def _setup_hooks(self):
        """Register forward hooks for Conf1. Conf2 uses direct forward() output."""

        if self.config == 'conf2':
            # SupConClassifier.forward() already returns (raw_feat, logits) directly
            # No hooks needed
            return

        def penultimate_hook(module, input, output):
            self._features = output  # shape: (batch, 2048, 1, 1)

        def final_hook(module, input, output):
            self._logits = output    # shape: (batch, 1000)

        # Conf1: standard WideResNet-50-2 hooks
        self.model.avgpool.register_forward_hook(penultimate_hook)
        self.model.fc.register_forward_hook(final_hook)

    def extract_single(self, image: Union[str, Path, Image.Image]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and confidence scores from a single image.

        Args:
            image: Path to image or PIL Image

        Returns:
            (feature_vector [2048], confidence_scores [1000])
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        img_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if self.config == 'conf2':
                raw_feat, logits = self.model(img_tensor)
                features = raw_feat.squeeze().cpu().numpy().astype(np.float32)
                logits = logits.squeeze().cpu().numpy().astype(np.float32)
            else:
                _ = self.model(img_tensor)
                features = self._features.squeeze().cpu().numpy().astype(np.float32)
                logits = self._logits.squeeze().cpu().numpy().astype(np.float32)

        return features, logits

    def extract_batch(self, images: List[Union[str, Path, Image.Image]],
                      batch_size: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and confidence scores from multiple images.

        Args:
            images: List of image paths or PIL Images
            batch_size: Number of images per forward pass

        Returns:
            (feature_matrix [N, 2048], logits_matrix [N, 1000])
        """
        all_features = []
        all_logits = []

        for i in tqdm(range(0, len(images), batch_size), desc=f"Extracting ({self.config})"):
            batch = images[i:i + batch_size]
            tensors = []

            for img in batch:
                if isinstance(img, (str, Path)):
                    img = Image.open(img).convert('RGB')
                tensors.append(self.transform(img))

            batch_tensor = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                if self.config == 'conf2':
                    raw_feat, logits_t = self.model(batch_tensor)
                    feats = raw_feat.cpu().numpy()
                    logits = logits_t.cpu().numpy()
                else:
                    _ = self.model(batch_tensor)
                    # avgpool output: (B, 2048, 1, 1) -> squeeze to (B, 2048)
                    feats = self._features.squeeze(-1).squeeze(-1).cpu().numpy()
                    logits = self._logits.cpu().numpy()

            all_features.append(feats)
            all_logits.append(logits)

        return (
            np.vstack(all_features).astype(np.float32),
            np.vstack(all_logits).astype(np.float32)
        )

    def extract_from_directory(
        self,
        directory: Union[str, Path],
        extensions: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.JPEG', '.JPG', '.PNG'),
        batch_size: int = 64,
        recursive: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Extract features from all images in a directory.

        The ImageNet100 dataset has subdirectories per class.
        Use recursive=True to find images in subdirectories.

        Args:
            directory: Path to image directory
            extensions: Valid image extensions
            batch_size: Images per forward pass
            recursive: Whether to search subdirectories (True for ImageNet100)

        Returns:
            (features [N, 2048], logits [N, 1000], relative_image_paths)
        """
        directory = Path(directory)
        image_paths = []

        if recursive:
            for ext in extensions:
                image_paths.extend(directory.rglob(f'*{ext}'))
        else:
            for ext in extensions:
                image_paths.extend(directory.glob(f'*{ext}'))

        image_paths = sorted(image_paths)

        if not image_paths:
            raise ValueError(f"No images found in {directory} (recursive={recursive})")

        print(f"Found {len(image_paths):,} images in {directory}")

        features, logits = self.extract_batch(image_paths, batch_size)

        # Store relative paths as identifiers
        rel_paths = [str(p.relative_to(directory)) for p in image_paths]

        return features, logits, rel_paths


def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits."""
    exp_shifted = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def save_features(features: np.ndarray,
                  logits: np.ndarray,
                  image_ids: List[str],
                  output_dir: Union[str, Path],
                  split_name: str,
                  config: str):
    """
    Save extracted features, logits, and image IDs to disk.

    Args:
        features: Feature matrix [N, 2048]
        logits: Raw logits matrix [N, 1000]
        image_ids: List of image identifiers (relative paths)
        output_dir: Output directory
        split_name: Name of the split ('InDSearch', 'InDQuery', 'Places365')
        config: Configuration name ('conf1' or 'conf2')
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{split_name}_{config}"

    np.save(output_dir / f"{prefix}_features.npy", features)
    np.save(output_dir / f"{prefix}_logits.npy", logits)
    np.save(output_dir / f"{prefix}_image_ids.npy", np.array(image_ids))

    print(f"Saved {split_name} ({config}): {features.shape} features, {logits.shape} logits")


def load_features(input_dir: Union[str, Path],
                  split_name: str,
                  config: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load saved features, logits, and image IDs.

    Returns:
        (features [N, 2048], logits [N, 1000], image_ids [N])
    """
    input_dir = Path(input_dir)
    prefix = f"{split_name}_{config}"

    features = np.load(input_dir / f"{prefix}_features.npy")
    logits = np.load(input_dir / f"{prefix}_logits.npy")
    image_ids = np.load(input_dir / f"{prefix}_image_ids.npy")

    return features, logits, image_ids


def normalize_l2(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a matrix (required before OoD tasks)."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return vectors / norms


if __name__ == "__main__":
    print("WideResNet-50-2 Feature Extraction Module")
    print("=" * 50)
    print("\nConf1: IMAGENET1K_V2 pretrained weights")
    print("Conf2: ConWideResNet50.pt contrastive weights")
    print("\nBoth extract 2048-dim features from the penultimate (avgpool) layer.")
    print("Class confidence logits from the fc layer are also saved.")
    print("\nDatasets to process:")
    print("  - InDSearch  (~95,000 images from ImageNet100)")
    print("  - InDQuery   (~40,000 images from ImageNet100)")
    print("  - Places365  (~36,500 images)")
