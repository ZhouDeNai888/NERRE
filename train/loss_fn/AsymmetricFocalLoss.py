import torch
import torch.nn as nn
import torch.nn.functional as F

class AsymmetricFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma_pos=1.0, gamma_neg=4.0, reduction='mean'):
        """
        Args:
            alpha: น้ำหนักให้ Entity (Positive) แนะนำ 0.75-0.9
            gamma_pos: ตัวกด Entity (ใช้น้อยๆ 1.0 เพื่อให้โมเดลสนใจ Entity นานๆ)
            gamma_neg: ตัวกดคลาส O (ใช้เยอะๆ 4.0 เพื่อให้โมเดลเลิกสนใจคลาส O ทันทีที่เริ่มทายถูก)
        """
        super(AsymmetricFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch, num_spans, num_classes]
        # targets: [batch, num_spans, num_classes] (One-hot)
        
        probs = torch.sigmoid(inputs)
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # แยกชิ้นส่วน Focal Term
        # สำหรับ Positive (Entity)
        pos_focal = self.alpha * (1 - probs).pow(self.gamma_pos) * targets
        
        # สำหรับ Negative (คลาส O)
        neg_focal = (1 - self.alpha) * probs.pow(self.gamma_neg) * (1 - targets)
        
        loss = (pos_focal + neg_focal) * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()