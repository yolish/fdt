"""
The FDT model
"""
import torch
import torch.nn.functional as F
from torch import nn
from .transformer_utils import TransformerEncoderWithGloablToken, TransformerDecoder
from .backbone_utils import build_backbone
from utils import box_ops


class FDT(nn.Module):

    def __init__(self, config):
        """
        config: (dict) configuration of the model
        pretrained_path: (str) path to the pretrained backbone
        """
        super().__init__()

        # CNN backbone
        config["learn_with_global_token"] = True
        self.backbone = build_backbone(config)

        # Transformer encoder
        self.transformer_encoder = TransformerEncoderWithGloablToken(config)

        transformer_dim = self.transformer_encoder.d_model

        # The learned global token
        self.global_token_embed = nn.Parameter(torch.zeros((1, transformer_dim)), requires_grad=True)

        # The projection of the activation map before going into the Transformer's encoder
        self.input_proj = nn.Conv2d(self.backbone.num_channels, transformer_dim, kernel_size=1)

        # Trasnformer Decoder
        self.transformer_decoder = TransformerDecoder(config)

        # Query embedding
        self.query_embed = nn.Embedding(config.get("num_faces"), transformer_dim)

        # Box specific
        self.box_cls = nn.Linear(transformer_dim, 1)
        self.bbox_embed = MLP(transformer_dim, transformer_dim, 4, 3)

        # Global/Local Classifcation / Regression heads
        self.global_cls_heads = {}
        if "global_cls_heads" in config:
            for name, my_config in config.get("global_cls_heads").items():
                self.global_cls_heads[name] = ClassifierHead(transformer_dim, 0, my_config)

        self.global_regr_heads = {}
        if "global_regr_heads" in config:
            for name, out_dim in config.get("global_regr_heads").items():
                self.global_regr_heads[name] = RegressorHead(transformer_dim, 0, my_config)

        self.local_cls_heads = {}
        if "local_cls_heads" in config:
            for name, my_config in config.get("local_cls_heads").items():
                self.local_regr_heads[name] = ClassifierHead(transformer_dim, 1, my_config)

        self.local_regr_heads = {}
        if "local_regr_heads" in config:
            for name, my_config in config.get("local_regr_heads").items():
                self.local_regr_heads[name] = RegressorHead(transformer_dim, 1, my_config)

        self.heads = [self.global_cls_heads, self.global_regr_heads, self.global_regr_heads, self.local_regr_heads]

    def put_heads_on_device(self, device):
        for h in self.heads:
            for name, model in h.items():
                h[name] = model.to(device)

    def forward(self, imgs):
        """
        return a dictionary with keys as names of local and global regressors/classifiers and outputs as their outputs
        """
        outputs = {}
        for h in self.heads:
            for name, model in h.items():
                outputs[name] = []

        outputs['box_coords'] = []
        outputs['box_logits'] = []

        all_local_descs = []
        all_global_descs = []
        # Images can be of difference sizes so the initial processing is done sequentially
        for img in imgs:

            # Extract the features and the position embedding from the visual backbone
            features, pos = self.backbone(img)

            src, mask = features[0].decompose()
            descs = self.transformer_encoder(self.input_proj(src), mask, pos[0], self.global_token_embed)

            # Take the global desc from the pose token
            global_desc = descs[:, 0, :]

            # Take the local descs from the remaining outputs
            local_descs = descs[:, 1:, :]
            local_descs = self.transformer_decoder(local_descs, mask, self.query_embed.weight)[0]

            all_local_descs.append(local_descs)
            all_global_descs.append(global_desc)

        all_global_descs = torch.stack(all_global_descs).to(imgs[0].device).squeeze(1)
        all_local_descs = torch.stack(all_local_descs).to(imgs[0].device).squeeze(1)

        # Handle box detection
        boxes_cxcywh = torch.sigmoid(self.bbox_embed(all_local_descs))
        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(boxes_cxcywh)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = img.shape[2:]
        boxes[:, :, 0] *= img_w
        boxes[:, :, 1] *= img_h
        boxes[:, :, 2] *= img_w
        boxes[:, :, 3] *= img_h
        outputs['box_coords'] =boxes
        outputs['box_logits'] = (self.box_cls(all_local_descs))

        descs = [all_global_descs, all_local_descs]
        for h in self.heads:
            for name, model in h.items():
                outputs[name] = model(descs)
        return outputs


        # Batch
        ''''
        outputs['box_coords'] = torch.stack(outputs.get('box_coords')).to(descs[0].device)
        outputs['box_logits'] = torch.stack(outputs.get('box_logits')).to(descs[0].device)
        for h in self.heads:
            for name, model in h.items():
                outputs[name] = torch.stack(outputs.get(name)).to(descs[0].device)
    
            return outputs
    '''


class ClassifierHead(nn.Module):
    def __init__(self, input_dim, desc_id, config):
        super().__init__()
        self.classifier = nn.Linear(input_dim, config.get("num_classes"))
        self.desc_id = desc_id

    def forward(self, descs):
        return self.classifier(descs[self.desc_id])


class RegressorHead(nn.Module):
    def __init__(self, input_dim, desc_id, config):
        super().__init__()
        self.final_activation = None
        self.desc_id = desc_id
        input_dim = config.get("input_dim")
        hidden_dim = config.get("hidden_dim")
        output_dim = config.get("output_dim")
        num_layers = config.get("num_layers")
        if config.get("activation"):
            self.final_activation = nn.Sigmoid()
        self.regressor = MLP(input_dim, hidden_dim, output_dim, num_layers)

    def forward(self, descs):
        output = self.regressor(descs[self.desc_id])
        if self.final_activation:
            output = self.final_activation(output)
        return output


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




