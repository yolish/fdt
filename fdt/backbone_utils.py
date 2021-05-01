"""
Code for the backbone of FDT
Backbone code is based on https://github.com/facebookresearch/detr/tree/master/models with the following modifications:
- use efficient-net as backbone and extract different activation maps from different reduction maps
- change learned encoding to have a learned token for the pose
"""
import torch.nn.functional as F
from torch import nn
from .pos_encoding_utils import build_position_encoding, NestedTensor
from typing import Dict, List
import torch

class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, reduction):
        super().__init__()
        self.body = backbone
        self.reduction = reduction
        self.reduction_map = {"reduction_3": 40, "reduction_4": 112}
        self.num_channels = self.reduction_map[reduction]

    def forward(self, tensors):
        xs = self.body.extract_endpoints(tensors)
        out: Dict[str, NestedTensor] = {}
        x = xs[self.reduction]
        mask = torch.zeros((x.shape[0], *x.shape[-2:])).to(torch.bool).to(x.device)# No masking for now
        out[self.reduction] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    def __init__(self, backbone_model_path: str, reduction):
        backbone = torch.load(backbone_model_path)
        super().__init__(backbone, reduction)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            ret = self[1](x)
            if isinstance(ret, tuple):
                p_emb, m_emb = ret
                pos.append([p_emb.to(x.tensors.dtype), m_emb.to(x.tensors.dtype)])
            else:
                pos.append(ret.to(x.tensors.dtype))

        return out, pos

def build_backbone(config):
    position_embedding = build_position_encoding(config)
    backbone = Backbone(config.get("backbone"), config.get("reduction"))
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
