
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ThanFocalLoss(nn.Module):
    def __init__(self, gamma=2, theta=0.5, alpha=None, reduction='mean'):
        """
        Shifted Tanh Focal Loss
        f(A) = scale * (tanh(k * (A - theta)) - base)
        """
        super(ThanFocalLoss, self).__init__()
        self.gamma = gamma
        self.theta = theta
        self.alpha = alpha
        self.reduction = reduction
        self.eps = 1e-7

        # 1. A=0 (Base Offset) -> tanh(-k * theta)
        self.base = math.tanh(-self.gamma * self.theta)
        # 2. tanh(k * (1 - theta))
        max_val = math.tanh(self.gamma * (1.0 - self.theta))
        # 3. Scale -> 1 / (Max - Base)
        self.scale = 1.0 / (max_val - self.base)

    def forward(self, inputs, targets):
        """
        inputs: Logits
        targets: 0 or 1 label
        """
        targets = targets.float()
        
        # BCE
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Error (A = 1 - Pt)
        probs = torch.sigmoid(inputs)
        probs = torch.clamp(probs, self.eps, 1 - self.eps) 
        p_t = probs * targets + (1 - probs) * (1 - targets)
        error = 1 - p_t
        
        tanh_term = torch.tanh(self.gamma * (error - self.theta))
        focal_weight = self.scale * (tanh_term - self.base)
        # -------------------------------

        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                 alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            else:
                 alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
