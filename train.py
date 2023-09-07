
"""


"""
import torch
from model.model import SegmentationModel
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
from dataset import SegmentationDataset
from utils.util import get_train_augs, get_valid_augs
from Trainer.trainer import Trainer
from arg_parser import Config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    df = pd.read_csv(f'{args.data_dir}/train.csv')
    train_df, val_df = train_test_split(df, test_size=args.validation_split)
    train_set = SegmentationDataset(train_df, augmentations=get_train_augs(args.img_size), data_dir=args.data_path)
    val_set = SegmentationDataset(val_df, augmentations=get_valid_augs(args.img_size), data_dir=args.data_path)

    print(train_set[0][0].shape)
    train_loader = DataLoader(train_set, batch_size=args.batch_size
                              , shuffle=args.shuffle, num_workers=args.num_workers)

    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SegmentationModel()
    model.to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, amsgrad=args.ams_grad)
    trainer = Trainer(
        device=DEVICE,
        train_loader=train_loader,
        test_loader=val_loader,
        optimizer=optimizer,
        checkpoint_path=None,)

    trainer.train(epochs=args.epochs
                  , save_path=args.save_dir,
                  tensorboard=args.tensorboard)


if __name__ == '__main__':
    args = Config('./config.json').args
    main(args)
