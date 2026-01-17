import os
import sys
import json
# 1. หา path ของโฟลเดอร์แม่ (NERRE)
# (__file__ คือไฟล์ปัจจุบัน -> dirname ครั้งที่ 1 ได้ 'train/' -> dirname ครั้งที่ 2 ได้ 'NERRE/')
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. เพิ่ม path นั้นเข้าไปในระบบ
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- Imports ของเรา ---
from model.loss_fn.focal_loss import SigmoidFocalLoss 
from model.model import ZeroShotJointModel 
import train_config as config

# นำเข้า Dataset - ใช้ GraphRAGDataset สำหรับ Graph RAG
from data.GraphRAGDataset import GraphRAGDataset, graph_rag_collate_fn

# ตรวจสอบว่ามีไฟล์ training data หรือไม่
if not os.path.exists(config.TRAIN_FILE):
    print(f"❌ Train file not found: {config.TRAIN_FILE}")
    print("   Please create a training dataset first.")
    sys.exit(1)
else:
    print(f"✅ Found training file: {config.TRAIN_FILE}")



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

    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # 1. Setup Data & Model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # สร้าง Dataset สำหรับ Graph RAG
    train_dataset = GraphRAGDataset(
        json_file=config.TRAIN_FILE,
        tokenizer=tokenizer,
        max_len=256,
        neg_sample_ratio=0.0, # ไม่ต้องใช้ negative label sampling เพราะมี labels ครบแล้ว
        neg_span_ratio=2.0    # ✅ เพิ่มเป็น 2.0 - 2x negative spans ต่อ positive spans
    )

    # สร้าง DataLoader
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        collate_fn=graph_rag_collate_fn,
        num_workers=0,  # ลดเพื่อ debug
        pin_memory=True
    )

    model = ZeroShotJointModel(config.MODEL_NAME).to(device)

    # 2. Setup Optimizer & Loss
    # ✅ เปลี่ยนเป็น CrossEntropyLoss สำหรับ single-label classification
    # CrossEntropy ใช้ softmax + log likelihood ทำให้ model เรียนรู้ที่จะ discriminate ระหว่าง classes
    # ✅ เพิ่ม label_smoothing=0.1 เพื่อป้องกัน overconfidence (scores ไม่ไปถึง 1.0 หมด)
    ent_criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)  # For entity type classification
    rel_criterion = SigmoidFocalLoss(alpha=config.ALPHA, gamma=config.GAMMA, reduction='none')  # Relations can be multi-label
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
                valid_ent_samples = 0
                valid_rel_samples = 0
                
                # วนลูปทีละ Sample ใน Batch
                for b in range(len(batch['spans'])):
                    # Entity Loss (CrossEntropy - single label per span)
                    # Logit: [Num_Spans, Max_Labels] -> ตัดเอาแค่ Num_Labels จริง
                    if len(batch['spans'][b]) > 0: # ถ้ามี Span
                        num_real_labels = ent_targets[b].shape[1] # จำนวน Label จริง (ก่อน Pad)
                        
                        # ตัด Logit ส่วนเกิน (Padding) ออก
                        curr_ent_logits = ent_logits[b, :len(batch['spans'][b]), :num_real_labels]
                        curr_ent_targets = ent_targets[b][:, :num_real_labels]
                        
                        # ✅ Convert one-hot to class indices for CrossEntropyLoss
                        # Shape: [Num_Spans, Num_Labels] -> [Num_Spans] (class indices)
                        target_indices = curr_ent_targets.argmax(dim=1).long()  # Get the class index for each span
                        
                        l_ent = ent_criterion(curr_ent_logits, target_indices)
                        loss_ent += l_ent
                        valid_ent_samples += 1
                    
                    # Relation Loss (Focal Loss - can be multi-label)
                    if rel_logits is not None and len(batch['pairs'][b]) > 0:
                        l_rel = rel_criterion(rel_logits[b, :len(batch['pairs'][b]), :], rel_targets[b])
                        loss_rel += l_rel.mean()
                        valid_rel_samples += 1

                # Average Loss
                total_samples = max(1, valid_ent_samples + valid_rel_samples)
                if valid_ent_samples > 0:
                    loss_ent = loss_ent / valid_ent_samples
                else:
                    loss_ent = torch.tensor(0.0, device=device)
                if valid_rel_samples > 0:
                    loss_rel = loss_rel / valid_rel_samples
                else:
                    loss_rel = torch.tensor(0.0, device=device)
                    
                loss = loss_ent + loss_rel
                
                # Handle case where both are zero (no valid samples at all)
                if valid_ent_samples == 0 and valid_rel_samples == 0:
                    loss = torch.tensor(0.0, requires_grad=True, device=device)

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
    
    # Save Config - ✅ ใช้ all_ent_labels_with_O ที่รวม "O" label
    with open(f"{config.OUTPUT_DIR}/config.json", "w", encoding='utf-8') as f:
        json.dump({
            "model_name": config.MODEL_NAME,
            "ent_labels": train_dataset.all_ent_labels_with_O,  # ✅ รวม "O"
            "rel_labels": sorted(list(train_dataset.all_rel_labels))
        }, f, ensure_ascii=False, indent=4)
        
    print(f"Model saved to {config.OUTPUT_DIR}")
    print(f"✅ Entity labels (with O): {train_dataset.all_ent_labels_with_O}")