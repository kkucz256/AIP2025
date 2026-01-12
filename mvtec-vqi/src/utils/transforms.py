from torchvision import transforms
from torchvision.transforms import InterpolationMode


def build_transform(image_size, normalize=True, augment=False, rotation=0):
    """
    Create an image transform.

    normalize=True keeps ImageNet statistics (required for feature extractors like ResNet).
    normalize=False leaves the tensor in [0, 1] which is better suited for reconstruction models.
    augment=True applies light augmentations for training CAE (jitter/flip).
    rotation (int): Degrees for RandomRotation.
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    ops = []
    if augment:
        ops.extend(
            [
                # Symulacja zmian oswietlenia (balans bieli, jasnosc) - agresywniejsza dla kamer
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                # Symulacja niedokladnego pozycjonowania obiektu na tasmie
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                # Symulacja nieostrosci (motion blur / zly focus)
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3.0))], p=0.3),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
            ]
        )
        if rotation > 0:
             ops.append(transforms.RandomRotation(degrees=rotation))

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
