import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from utils import box_ops
from models.HungarianMatcher import HungarianMatcher


class FDTCriterion(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.global_cls_losses = {}
        if "global_cls_heads" in config:
            self.global_cls_losses = config.get("global_cls_heads")

        self.global_regr_losses = {}
        if "global_regr_heads" in config:
            self.global_regr_losses = config.get("global_regr_heads")

        self.local_cls_losses = {}
        if "local_cls_heads" in config:
            self.global_cls_losses = config.get("local_cls_heads")

        self.local_regr_losses = {}
        if "local_regr_heads" in config:
            self.local_regr_losses = config.get("local_regr_heads")

        #self.matcher = HungarianMatcher(cost_class=config.get("set_cost_class"), cost_bbox=config.get("set_cost_bbox"),
        #                            cost_giou=config.get("set_cost_giou"))

        # Default weighting
        self.matcher = HungarianMatcher()

    def forward(self, outputs, targets):
        # Global losses
        losses = {}
        for name in self.global_cls_losses:
            losses.update(self.global_classifier_loss(name, outputs, targets))

        for name in self.global_regr_losses:
            losses.update(self.global_regressor_loss(name, outputs, targets))

        # Find boxes and compute associate loss
        predicted_box_coords = outputs['box_coords']
        predicted_box_logits = outputs['box_logits']

        indices = self.matcher(predicted_box_coords, predicted_box_logits, targets)

        num_boxes = torch.as_tensor(np.sum([t["boxes"].shape[-1] for t in targets]), dtype=torch.float,
                                    device=targets[0]["boxes"].device)

        # GIOU, boxes, box classification (yes/no)
        losses.update(self.boxes_loss(predicted_box_coords, predicted_box_logits, targets, indices, num_boxes))

        # Compute all other local losses
        for name in self.local_cls_losses:
            losses.update(self.local_classifier_loss(name, outputs, targets, indices, num_boxes))
        for name in self.local_regr_losses:
            losses.update(self.local_regressor_loss(name, outputs, targets, indices, num_boxes))

        return losses

    def get_src_permutation_idx(self, indices):

        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def global_classifier_loss(self, name, outputs, targets):
        return {name: F.l1_loss(outputs[name], targets[name])}

    def global_regressor_loss(self, name, outputs, targets):
        return {name: F.mse_loss(outputs[name], targets[name])}

    def local_classifier_loss(self, name, outputs, targets, indices, ignore_indices):
        assert name in outputs
        src_logits = outputs[name]

        idx = self.get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[name][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 2,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        return {name: F.l1_loss(src_logits, target_classes, reduction='mean', ignore_indices=ignore_indices)}

    def local_regressor_loss(self, name, outputs, targets, indices, ignore_indices):
        assert name in outputs
        src_logits = outputs[name]

        idx = self.get_src_permutation_idx(indices)

        target_classes_o = torch.cat([t[name][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], 2,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        return {name: F.l1_loss(src_logits, target_classes, reduction='mean', ignore_indices=ignore_indices)}

    def boxes_loss(self, predicted_box_coords, predicted_box_logits, targets, indices, num_boxes):
        idx = self.get_src_permutation_idx(indices)
        src_boxes = predicted_box_coords[idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            target_boxes))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        losses['loss_cls'] = -torch.sum(predicted_box_logits[idx])
        return losses
