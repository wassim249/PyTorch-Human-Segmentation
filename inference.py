import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from dataset import SegmentationDataset
from utils import get_valid_augs
from Trainer.trainer import Trainer
from utils import save_tensor_image
from arg_parser import Config

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    """
    Main function.
    Test the model on the validation set with the first batch of assets.
    :param args:
    :return:
    """
    df = pd.read_csv(f'{args.data_dir}/train.csv')
    _, val_df = train_test_split(df, test_size=args.validation_split)
    val_set = SegmentationDataset(val_df, augmentations=get_valid_augs(args.img_size), data_dir=args.data_path)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


    trainer = Trainer(
        train_loader=None,
        test_loader=None,
        optimizer=None,
        device=DEVICE,
        checkpoint_path=args.checkpoint_path,
    )

    images = next(iter(val_loader))[0]
    logits_mask = trainer.predict(images)

    print(f"Saving assets to {args.save_dir}/assets")
    print(f"Total assets: {logits_mask.shape[0]}")
    for idx in range(logits_mask.shape[0]):
        pred_mask = torch.sigmoid(logits_mask[idx])
        pred_mask = ((pred_mask > 0.5) * 1.0).squeeze(0).squeeze(0)
        save_tensor_image(pred_mask, f'{args.save_dir}/assets/mask_{idx}.png')
        img = images[idx].squeeze(0).permute(1, 2, 0)
        save_tensor_image(img, f'{args.save_dir}/assets/img_{idx}.png')
        print(f'*** Saved image {idx} to {args.save_dir}/assets')

if __name__ == '__main__':
    args = Config('./config.json').args
    main(args)
