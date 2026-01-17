import os
import sys
# 1. หา path ของโฟลเดอร์แม่ (NERRE)
# (__file__ คือไฟล์ปัจจุบัน -> dirname ครั้งที่ 1 ได้ 'train/' -> dirname ครั้งที่ 2 ได้ 'NERRE/')
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. เพิ่ม path นั้นเข้าไปในระบบ
sys.path.append(parent_dir)

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Imports ของเรา ---
from model.loss_fn.focal_loss import SigmoidFocalLoss 
from model.model import ZeroShotJointModel 
import train_config as config

# นำเข้า Dataset ที่คุณสร้างไว้
from data.ZeroShotDataset import ZeroShotDataset, collate_fn


from data.hf_dataloader import generate_merged_dataset # import ฟังก์ชันที่เราเขียนเมื่อกี้

# เช็คว่ามีไฟล์ train_data.json หรือยัง
if not os.path.exists(config.TRAIN_FILE):
    print("📢 Train file not found. Generating from Hugging Face datasets...")
    generate_merged_dataset(output_file=config.TRAIN_FILE)
else:
    print(f"✅ Found existing train file: {config.TRAIN_FILE}")



# ==========================================
# Helper Function: จัดการ Label ที่ขนาดไม่เท่ากัน
# ==========================================
def prepare_batch_inputs(batch, tokenizer, device):
    """
    แปลง List of Lists of Strings (Labels) ให้เป็น Tensor พร้อม Padding
    เพื่อให้ส่งเข้า Model ได้
    """
    # 1. จัดการ Entity Labels
    # ent_labels_text เป็น List[List[str]] เช่น [['Per', 'Org'], ['Loc']]
    # เราต้องทำให้เป็น Tensor ขนาด [Batch, Max_Num_Labels, Seq_Len]
    
    batch_ent_labels = batch['ent_labels_text']
    
    # หาจำนวน Label ที่เยอะที่สุดใน Batch นี้
    max_ent_labels = max(len(labels) for labels in batch_ent_labels)
    
    # Flatten เพื่อ Tokenize ทีเดียว (เร็ววกว่า loop)
    # แต่ต้องจำไว้ว่าแต่ละ sample มีกี่ label
    flat_ent_labels = []
    for labels in batch_ent_labels:
        flat_ent_labels.extend(labels)
        # เติม Dummy label ให้ครบ Max (Padding logic)
        flat_ent_labels.extend(["O"] * (max_ent_labels - len(labels)))

    # Tokenize
    ent_inputs = tokenizer(flat_ent_labels, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Reshape กลับเป็น [Batch, Max_Labels, Seq_Len]
    # เพื่อให้ Model เข้าใจว่าเป็น Label ของ Sample ไหน
    b = len(batch_ent_labels)
    seq_len = ent_inputs['input_ids'].shape[1]
    
    ent_label_ids = ent_inputs['input_ids'].view(b, max_ent_labels, seq_len)
    ent_label_mask = ent_inputs['attention_mask'].view(b, max_ent_labels, seq_len)
    
    # 2. จัดการ Entity Targets (Padding)
    # ent_targets เดิมเป็น List of Tensors [Spans, Num_Labels]
    # เราต้อง Pad ให้ Num_Labels เท่ากับ max_ent_labels
    padded_ent_targets = []
    for i, t in enumerate(batch['ent_targets']):
        # t shape: [Num_Spans, Num_Actual_Labels]
        # เราต้องการ: [Num_Spans, Max_Ent_Labels]
        num_spans, num_actual = t.shape
        pad_size = max_ent_labels - num_actual
        
        if pad_size > 0:
            # สร้างแผ่น Zero มาต่อท้าย
            padding = torch.zeros((num_spans, pad_size))
            t_padded = torch.cat([t, padding], dim=1)
        else:
            t_padded = t
        padded_ent_targets.append(t_padded.to(device))

    # --- ทำแบบเดียวกันกับ Relation (ถ้ามี) ---
    # (ในตัวอย่างนี้ขอละไว้ ใช้ Logic เดียวกับ Entity)
    # สมมติ Relation Labels เหมือนกันทั้ง Batch ไปก่อนเพื่อความง่าย
    rel_inputs = tokenizer(batch['rel_labels_text'][0], return_tensors="pt", padding=True, truncation=True).to(device)
    rel_label_ids = rel_inputs['input_ids'].unsqueeze(0).repeat(b, 1, 1) # Repeat ให้เท่า Batch
    rel_label_mask = rel_inputs['attention_mask'].unsqueeze(0).repeat(b, 1, 1)
    
    # Pad Relation Targets (คล้าย Entity)
    padded_rel_targets = [t.to(device) for t in batch['rel_targets']]

    return (ent_label_ids, ent_label_mask), padded_ent_targets, \
           (rel_label_ids, rel_label_mask), padded_rel_targets

if __name__ == "__main__":
    # ==========================================
    # Main Training Script
    # ==========================================

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # 1. Setup Data & Model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # สร้าง Dataset ของจริง
    train_dataset = ZeroShotDataset(
        json_file=config.TRAIN_FILE, # แก้เป็น path json ของคุณ
        tokenizer=tokenizer,
        max_len=512,
        neg_sample_ratio=0.5
    )

    # สร้าง DataLoader (สำคัญ: ต้องใส่ collate_fn)
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn, # <--- เรียกใช้จาก Class หรือไฟล์ที่ import มา
        num_workers=4,         # ช่วยโหลดข้อมูลเร็วขึ้น
        pin_memory=True
    )

    model = ZeroShotJointModel(config.MODEL_NAME).to(device)

    # 2. Setup Optimizer & Loss
    criterion = SigmoidFocalLoss(alpha=config.ALPHA, gamma=config.GAMMA, reduction='none') # ใช้ 'none' เพื่อจัดการ Padding เอง
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')

    # 3. Training Loop
    num_epochs = config.NUM_EPOCHS

    model.train()
    print(f"Start Training on {len(train_dataset)} samples...")

    for epoch in range(num_epochs):
        total_loss = 0
        
        # วนลูปจาก DataLoader ของจริง
        loop = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for step, batch in enumerate(loop):
            
            # ย้ายข้อมูลพื้นฐานเข้า GPU
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            # เตรียม Labels และ Targets (จัดการ Padding)
            (ent_lbl_ids, ent_lbl_mask), ent_targets, \
            (rel_lbl_ids, rel_lbl_mask), rel_targets = prepare_batch_inputs(batch, tokenizer, device)
            
            optimizer.zero_grad()
            
            with autocast():
                # Forward Pass
                # หมายเหตุ: Model ต้องรองรับ Label input แบบ 3D [Batch, Num_Labels, Seq]
                # ถ้า Model ไม่รองรับ ต้องแก้ Model ให้ Flatten ก่อนเข้า Encoder
                ent_logits, rel_logits = model(
                    text_ids, text_mask,
                    ent_lbl_ids, ent_lbl_mask,
                    rel_lbl_ids, rel_lbl_mask,
                    batch['spans'],
                    batch['pairs']
                )
                
                # --- Calculate Loss (Custom Loop) ---
                # เนื่องจาก ent_logits และ targets เป็น List of Tensors (ขนาดไม่เท่ากันตาม Spans)
                # เราต้องวนลูปคำนวณ Loss ทีละ Sample ใน Batch (วิธีที่ปลอดภัยสุดสำหรับ Dynamic Data)
                
                loss_ent = 0
                loss_rel = 0
                valid_samples = 0
                
                # วนลูปทีละ Sample ใน Batch
                for b in range(len(batch['spans'])):
                    # Entity Loss
                    # Logit: [Num_Spans, Max_Labels] -> ตัดเอาแค่ Num_Labels จริง
                    if len(batch['spans'][b]) > 0: # ถ้ามี Span
                        num_real_labels = ent_targets[b].shape[1] # จำนวน Label จริง (ก่อน Pad)
                        
                        # ตัด Logit ส่วนเกิน (Padding) ออก
                        curr_ent_logits = ent_logits[b, :len(batch['spans'][b]), :num_real_labels]
                        curr_ent_targets = ent_targets[b][:, :num_real_labels]
                        
                        l_ent = criterion(curr_ent_logits, curr_ent_targets)
                        loss_ent += l_ent.mean()
                        valid_samples += 1
                    
                    # Relation Loss (ถ้ามี)
                    if rel_logits is not None and len(batch['pairs'][b]) > 0:
                        # (Logic เดียวกัน)
                        l_rel = criterion(rel_logits[b, :len(batch['pairs'][b]), :], rel_targets[b])
                        loss_rel += l_rel.mean()

                # Average Loss
                if valid_samples > 0:
                    loss = (loss_ent + loss_rel) / valid_samples
                else:
                    loss = torch.tensor(0.0, requires_grad=True).to(device)

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_dataloader)
        print(f"--- Epoch {epoch+1} Finished. Avg Loss: {avg_loss:.4f} ---")
        
        # (Optional) Save Checkpoint here...

    print("Training Complete!")

    # --- Save Model ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/pytorch_model.bin")
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    # ... Save Config ...
    print(f"Model saved to {config.OUTPUT_DIR}")