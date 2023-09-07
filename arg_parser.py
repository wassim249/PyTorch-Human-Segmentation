"""
This file contains the class Config which is used to parse the arguments from the config file.
"""

import argparse
import json


class Config:
    def __init__(self, config_path: str):
        """
        Initialize the Config class.

        :param str config_path: path to config file
        """
        self.config = json.load(open(config_path))
        self.parser = argparse.ArgumentParser("Human Segmentation training")
        self.parser.add_argument("--img_size", type=int, default=self.config['data_loader']['args']['image_size'],
                                 help="size of each image dimension")
        self.parser.add_argument("--data_dir", type=str, default=self.config['data_loader']['args']['data_dir'],
                                 help="path to dataset")

        self.parser.add_argument("--batch_size", type=int, default=self.config['data_loader']['args']['batch_size'],
                                 help="size of the batches")
        self.parser.add_argument("--shuffle", type=bool, default=self.config['data_loader']['args']['shuffle'],
                                 help="shuffle dataset")
        self.parser.add_argument("--validation_split", type=float,
                                 default=self.config['data_loader']['args']['validation_split'],
                                 help="size of the validation split")
        self.parser.add_argument("--num_workers", type=int, default=self.config['data_loader']['args']['num_workers'],
                                 help="number of cpu threads to use during batch generation")
        self.parser.add_argument("--data_path", type=str, default=self.config['data_loader']['args']['data_dir'],
                                 help="path to dataset")
        self.parser.add_argument("--learning_rate", type=float, default=self.config['optimizer']['args']['lr'],
                                 help="learning rate")
        self.parser.add_argument("--weight_decay", type=float, default=self.config['optimizer']['args']['weight_decay'],
                                 help="weight decay")
        self.parser.add_argument("--ams_grad", type=bool, default=self.config['optimizer']['args']['amsgrad'],
                                 help="amsgrad")
        self.parser.add_argument("--epochs", type=int, default=self.config['trainer']['epochs'],
                                 help="number of epochs")
        self.parser.add_argument("--save_dir", type=str, default=self.config['trainer']['save_dir'],
                                 help="save directory")
        self.parser.add_argument("--tensorboard", type=bool, default=self.config['trainer']['tensorboard'],
                                 help="tensorboard")
        self.parser.add_argument("--checkpoint_path", type=str, default=None,
                                 help="checkpoint path")

    def __str__(self) -> str:
        return json.dumps(self.config, indent=4)

    def __repr__(self) -> str:
        return json.dumps(self.config, indent=4)

    @property
    def args(self) -> argparse.Namespace:
        return self.parser.parse_args()
