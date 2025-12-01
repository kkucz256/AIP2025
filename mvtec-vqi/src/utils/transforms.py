from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_transform(image_size, normalize=True, augment=False):
    """
    Create an image transform.

    normalize=True keeps ImageNet statistics (required for feature extractors like ResNet).
    normalize=False leaves the tensor in [0, 1] which is better suited for reconstruction models.
    augment=True applies light augmentations for training CAE (jitter/flip).
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    ops = []
    if augment:
        ops.extend(
            [
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )
    ops.append(transforms.Resize((image_size, image_size)))
    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(ops)


def build_mask_transform(image_size):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ]
    )
