from torch import nn
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss

ENCODER = 'timm-efficientnet-b0'
WEIGHTS='imagenet'

class SegmentationModel(nn.Module):
  """
    This class is used to define the model.
  """
  def __init__(self, *args, **kwargs) -> None:
    """
    Initialize the model.

    :param args:
    :param kwargs:
    """
    super(SegmentationModel,self).__init__(*args, **kwargs)

    self.arc = smp.Unet(
        encoder_name = ENCODER,
        encoder_weights = WEIGHTS,
        in_channels=3,
        classes= 1,
        activation = None
    )

  def forward(self,images : torch.Tensor,
              masks: torch.Tensor = None):
    """
    Forward pass of the model.

    :param Tensor images:
    :param Tensor masks:
    :return: Tensor logits:
    """
    logits = self.arc(images)

    if masks != None:
      loss1 = DiceLoss(mode='binary')(logits,masks)
      loss2 = nn.BCEWithLogitsLoss()(logits,masks)
      return logits , loss1 + loss2

    return logits