import torch
import torch.nn as nn
from torchvision.models import resnet
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from typing import Optional, Callable, List

def resnet_fpn_backbone(
    *,
    backbone_name: str,
    weights: Optional[resnet.WeightsEnum],
    norm_layer: Callable[..., nn.Module] = misc_nn_ops.FrozenBatchNorm2d,
    trainable_layers: int = 3,
    returned_layers: Optional[List[int]] = None,
    extra_blocks: Optional[resnet.ExtraFPNBlock] = None,
) -> BackboneWithFPN:
    """
    Constructs a specified ResNet backbone with FPN on top. Freezes the specified number of layers in the backbone.

    Examples::

        >>> import torch
        >>> from torchvision.models import ResNet50_Weights
        >>> backbone = resnet_fpn_backbone(backbone_name='resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=3)
        >>> # get some dummy image
        >>> x = torch.rand(1,3,64,64)
        >>> # compute the output
        >>> output = backbone(x)
        >>> print([(k, v.shape) for k, v in output.items()])
        >>> # returns
        >>>   [('0', torch.Size([1, 256, 16, 16])),
        >>>    ('1', torch.Size([1, 256, 8, 8])),
        >>>    ('2', torch.Size([1, 256, 4, 4])),
        >>>    ('3', torch.Size([1, 256, 2, 2])),
        >>>    ('pool', torch.Size([1, 256, 1, 1]))]

    Args:
        backbone_name (string): resnet architecture. Possible values are 'resnet18', 'resnet34', 'resnet50',
             'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2'
        weights (WeightsEnum, optional): The pretrained weights for the model
        norm_layer (callable): it is recommended to use the default value. For details visit:
            (https://github.com/facebookresearch/maskrcnn-benchmark/issues/267)
        trainable_layers (int): number of trainable (not frozen) layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable.
        returned_layers (list of int): The layers of the network to return. Each entry must be in ``[1, 4]``.
            By default, all layers are returned.
        extra_blocks (ExtraFPNBlock or None): if provided, extra operations will
            be performed. It is expected to take the fpn features, the original
            features and the names of the original features as input, and returns
            a new list of feature maps and their corresponding names. By
            default, a ``LastLevelMaxPool`` is used.
    """
    # Load the specified ResNet model
    backbone = resnet.__dict__[backbone_name](
        weights=weights,
        norm_layer=norm_layer
    )

    # Freeze the specified number of layers
    assert 0 <= trainable_layers <= 5
    layers_to_train = ['layer4', 'layer3', 'layer2', 'layer1', 'conv1']
    if trainable_layers < 5:
        for layer in layers_to_train[trainable_layers:]:
            for param in getattr(backbone, layer).parameters():
                param.requires_grad = False

    # If returned_layers is None, return all layers
    if returned_layers is None:
        returned_layers = [1, 2, 3, 4]

    # Check that returned_layers is a subset of [1, 2, 3, 4]
    assert all(layer in [1, 2, 3, 4] for layer in returned_layers)

    # Create the FPN
    in_channels_stage2 = backbone.inplanes // 8
    in_channels_list = [
        in_channels_stage2 * 2 ** (i - 1) for i in returned_layers
    ]
    out_channels = 256
    return_layers = {f'layer{i}': str(i - 1) for i in returned_layers}
    if extra_blocks is None:
        extra_blocks = resnet.LastLevelMaxPool()

    return BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels, extra_blocks)