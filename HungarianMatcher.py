# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.

Modification made:
- minor modifications to interface
- modified loss - removed object classification loss (will be handled outside by each local loss)
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

from utils.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.weight_class = cost_class
        self.weight_bbox = cost_bbox
        self.weight_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, predicted_box_coords, predicted_box_logits, targets):
        """ Performs the matching

        Params:
            predicted_box_coords: Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            predicted_box_logits: Tensor of dim [batch_size, num_queries, num_classes] with the classification logits


            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = predicted_box_coords.shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = predicted_box_coords.flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Add the classification box cost (yes/no)
        # but approximate it in 1 - prob box .
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        out_prob = predicted_box_logits.flatten(0, 1)  # [batch_size * num_queries, num_classes]

        # Final cost matrix
        C = self.weight_bbox * cost_bbox + self.weight_class * -out_prob + self.weight_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

