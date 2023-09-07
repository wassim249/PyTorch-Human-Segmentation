from torchvision import transforms
import json
import torch
import cv2


def get_train_augs(image_size=128) -> transforms.Compose:
    """
    Augmentations for training.

    :param int image_size: size of the image
    :return: Compose object with augmentations
    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
    ])


def get_valid_augs(image_size=128) -> transforms.Compose:
    """
    Augmentations for validation.

    :param int image_size: size of the image
    :return: Compose object with augmentations

    """
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def config_parser(config_path: str) -> dict:
    """
    Parse config file.

    :param str config_path: path to config file
    :return: dict with config
    """
    config = json.load(open(config_path))
    return {
        'img_size': config['data_loader']['args']['image_size'],
        'batch_size': config['data_loader']['args']['batch_size'],
        'shuffle': config['data_loader']['args']['shuffle'],
        'validation_split': config['data_loader']['args']['validation_split'],
        'num_workers': config['data_loader']['args']['num_workers'],
        'data_path': config['data_loader']['args']['data_dir'],
        'learning_rate': config['optimizer']['args']['lr'],
        'weight_decay': config['optimizer']['args']['weight_decay'],
        'ams_grad': config['optimizer']['args']['amsgrad'],
        'optimizer': config['optimizer']['type'],
        'epochs': config['trainer']['epochs'],
        'save_dir': config['trainer']['save_dir'],
        'tensorboard': config['trainer']['tensorboard']
    }


def save_tensor_image(image: torch.Tensor, path: str) -> None:
    """
    Save tensor image to path.

    :param tensor image:
    :param str path:
    :return: None
    """
    image = image.cpu().detach().numpy()
    result = cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(path, result)
