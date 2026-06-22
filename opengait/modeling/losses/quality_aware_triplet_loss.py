
import torch
from .base import BaseLoss
from .base import BaseLoss, gather_and_scale_wrapper
import torch.nn.functional as F
import torch.nn as nn
class QualityAwareTripletLoss(BaseLoss):
    """
    Quality-Aware Loss (L_qa) used in ReCoGait.

    The implementation preserves the original distance function, triplet
    construction, ranking-loss formulation, and reduction strategy.
    """
    def __init__(self, margin=0.3, loss_term_weight = 0.5):
        super(QualityAwareTripletLoss, self).__init__(loss_term_weight)
        self.base_margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=0.0) 
        
    @gather_and_scale_wrapper
    def forward(self, embeddings, labels, quality_scores):
        # embeddings: [n, c, p], labels: [n], quality_scores: [n]
        embeddings = embeddings.permute(2, 0, 1).contiguous().float()  # [p, n, c]
        
        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)  # [p, n, n]
        mean_dist = dist.mean((1, 2))  # [p]
    
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)  # [p, n, num_pos, 1], [p, n, 1, num_neg]
        
        dynamic_margin = self.ComputeQualityAwareMargin(labels, ref_label, quality_scores)  # [n, num_pos, 1]
        dynamic_margin = dynamic_margin.unsqueeze(0)  # [1, n, num_pos, 1]
        
        target = torch.ones_like(an_dist)
        loss = self.ranking_loss(an_dist, ap_dist + dynamic_margin, target)
        
        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def ComputeQualityAwareMargin(self, row_labels, clo_label, quality_scores):
        matches = (row_labels.unsqueeze(1) == clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        
        q_anchor = quality_scores.unsqueeze(1)  # [n_r, 1]
        q_positive = quality_scores.unsqueeze(0)  # [1, n_c]
        
        dynamic_margin_matrix = self.base_margin * (q_anchor * q_positive)  # [n_r, n_c]
        
        dynamic_margin = dynamic_margin_matrix[matches].view(row_labels.size(0), -1, 1)  # [n_r, num_pos, 1]
        
        return dynamic_margin

    def AvgNonZeroReducer(self, loss):
        
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
       
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)  # [p, n_x, 1]
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)  # [p, 1, n_y]
        inner = x.matmul(y.transpose(1, 2))  # [p, n_x, n_y]
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))  # [p, n_x, n_y]
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        
        matches = (row_labels.unsqueeze(1) == clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
        diffenc = torch.logical_not(matches)  # [n_r, n_c]
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist