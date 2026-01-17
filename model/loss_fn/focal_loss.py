import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Define Focal Loss Class (พระเอกของเรา) ---
class SigmoidFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Args:
            alpha (float): น้ำหนักสำหรับ Positive class (0.25 คือมาตรฐานสำหรับลดบทบาท Background class)
            gamma (float): ตัวยกกำลังเพื่อลดความสำคัญของเคสที่ทายถูกง่ายๆ (Easy examples)
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super(SigmoidFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. คำนวณ BCE Loss แบบปกติก่อน (ใช้ Logits โดยตรงเพื่อความเสถียร)
        # reduction='none' เพื่อให้เราเอามาคูณ factor ต่อได้
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # 2. แปลง Logits เป็น Probabilities (p_t) ของ Class ที่ถูกต้อง
        # สูตร: pt = exp(-bce_loss) 
        # เหตุผล: ถ้าทายถูก loss ต่ำ -> exp(-0) = 1 (มั่นใจมาก)
        #        ถ้าทายผิด loss สูง -> exp(-high) = 0 (ไม่มั่นใจ)
        pt = torch.exp(-bce_loss)
        
        # 3. คำนวณ Focal Term: (1 - pt)^gamma
        # ยิ่งทายถูก (pt เข้าใกล้ 1) -> Term นี้จะเข้าใกล้ 0 -> Loss หายไป (ไม่สนใจข้อที่ง่าย)
        focal_term = (1.0 - pt).pow(self.gamma)
        
        # 4. Alpha Balancing (จัดการเรื่อง Imbalance Positive/Negative)
        if self.alpha is not None:
            # ถ้า target=1 ใช้ alpha, ถ้า target=0 ใช้ (1-alpha)
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * focal_term * bce_loss
        else:
            loss = focal_term * bce_loss

        # 5. Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss