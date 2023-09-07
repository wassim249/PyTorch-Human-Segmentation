import torch
from tqdm.auto import tqdm
from model import SegmentationModel


class Trainer:
    """
    This class is used to train and validate the model.
    """

    def __init__(self, device: torch.device,
                 test_loader: torch.utils.data.DataLoader = None,
                 train_loader: torch.utils.data.DataLoader = None,
                 optimizer: torch.optim.Optimizer = None,
                 checkpoint_path: str = None
                 ) -> None:
        """
        Initialize the trainer class.

        :param str device: device to train on
        :param DataLoader test_loader: test data loader
        :param DataLoader train_loader: train data loader
        :param Optimizer optimizer: optimizer
        :param str checkpoint_path: path to checkpoint
        """
        self.test_loader = test_loader
        self.train_loader = train_loader
        self.model = SegmentationModel()
        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)
        self.optimizer = optimizer
        self.device = device
        self.model.to(self.device)

    def train(self,epochs:int,
              save_path:str,
              tensorboard=False) -> None:
        """
        Train the model.

        :param int epochs: number of epochs
        :param str save_path: path to save the model
        :param bool tensorboard: whether to log to tensorboard

        :return: None
        """
        print("Starting training on ", self.device)
        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter()
                print("Logging to tensorboard")
            except ImportError:
                print("Tensorboard not available, not logging to tensorboard")
                tensorboard = False

        best_loss = float('inf')

        for epoch in tqdm(range(epochs)):
            print(f'Epoch {epoch + 1}/{epochs}')

            train_loss = self._train_epoch()
            val_loss = self._val_epoch()

            if val_loss < best_loss:
                print(f'** New best model with val loss: {val_loss}')
                best_loss = val_loss
                self._save_checkpoint(save_path + f'epoch_{epoch}_val_loss_{val_loss:.2f}.pth')
                print("Model saved")

            if tensorboard:
                writer.add_scalar('Loss/train', train_loss / len(self.train_loader), epoch)
                writer.add_scalar('Loss/val', val_loss / len(self.train_loader), epoch)
                writer.flush()

            print()

    def _train_epoch(self):
        """
        Train one epoch.
        :return: float train loss
        """
        self.model.train()
        train_loss = 0
        for images, masks in self.train_loader:
            images = images.to(self.device)
            masks = masks.to(self.device)
            self.optimizer.zero_grad()
            logits, loss = self.model(images, masks)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            print(f'Batch loss: {loss.item()}', end='\r')
        print(f'Train loss: {train_loss / len(self.train_loader)}')

        return train_loss / len(self.train_loader)

    def _val_epoch(self):
        """
        Validate one epoch.
        :return: float val loss
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, masks in self.train_loader:
                images = images.to(self.device)
                masks = masks.to(self.device)
                logits, loss = self.model(images, masks)
                val_loss += loss.item()
        print(f'Val loss: {val_loss / len(self.train_loader)}')
        return val_loss / len(self.train_loader)

    def _save_checkpoint(self, save_path: str):
        """
        Save the model.

        :param str save_path: path to save the model
        :return: None
        """
        torch.save(self.model.state_dict(), save_path)

    def load_checkpoint(self, load_path: str):
        """
        Load the model.

        :param str load_path: path to load the model from
        :return: None
        """
        self.model.load_state_dict(torch.load(load_path))
        self.model.eval()
        print(f'Model loaded from {load_path}')

    def predict(self, images: torch.Tensor) -> torch.Tensor:
        """
        Predict the mask for the given image.

        :param torch.Tensor images: image to predict the mask for
        :return: predicted mask
        """
        self.model.eval()
        with torch.inference_mode():
            images = images.to(self.device)
            logits = self.model(images)
            return logits
