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
from loss_fn.focal_loss import SigmoidFocalLoss
from loss_fn.AsymmetricFocalLoss import AsymmetricFocalLoss
from model.model import ZeroShotJointModel 
import train_config as config
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
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


# Inside your GraphRAGDataset or as a wrapper
def get_curriculum_indices(dataset, stage):
    indices = []
    for i in range(len(dataset)):
        sample = dataset.data[i] # Assuming raw data is accessible
        text_len = len(sample['text'].split())
        num_ents = len(sample['entities'])
        
        # Define Stage logic
        if stage == 1: # Easy
            if text_len < 15 and num_ents <= 2:
                indices.append(i)
        elif stage == 2: # Medium
            if text_len < 30 and num_ents <= 5:
                indices.append(i)
        else: # Stage 3: All data (Hard)
            indices.append(i)
    return indices


def get_sample_weights(model, dataloader, device, tokenizer):
    model.eval()
    all_losses = []
    
    # ent_criterion = SigmoidFocalLoss(alpha=config.ALPHA, gamma=config.GAMMA, reduction='mean')
    ent_criterion = AsymmetricFocalLoss(alpha=config.ALPHA, gamma_pos=config.POS_GAMMA, gamma_neg=config.NEG_GAMMA, reduction='mean')
    
    print("🔍 Mining Hard Negatives (Calculating Sample Weights)...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            (ent_lbl_ids, ent_lbl_mask), ent_targets, \
            (rel_lbl_ids, rel_lbl_mask), rel_targets = prepare_batch_inputs(batch, tokenizer, device)

            # รัน Forward เพื่อดู Logits
            ent_logits, _ = model(
                text_ids, text_mask, ent_lbl_ids, ent_lbl_mask,
                rel_lbl_ids, rel_lbl_mask, batch['spans'], batch['pairs']
            )
            
            # คำนวณ Loss ราย Batch (หรือราย Sample)
            batch_loss = 0
            num_spans_total = 0
            for b in range(len(batch['spans'])):
                if len(batch['spans'][b]) > 0:
                    num_real = ent_targets[b].shape[1]
                    curr_logits = ent_logits[b, :len(batch['spans'][b]), :num_real]
                    curr_targets = ent_targets[b][:, :num_real]
                    
                    l_ent = ent_criterion(curr_logits, curr_targets)
                    # เก็บค่า Loss ของ sample นี้ไว้
                    all_losses.append(l_ent.item() + 1e-6) # + epsilon กันค่าเป็น 0
                else:
                    all_losses.append(0.0) # ประโยคไม่มี entity เลย (มักจะเป็น easy negative)

    # แปลง Loss เป็น Weights: ประโยคที่ Loss สูงจะมีโอกาสถูกสุ่มมาเทรนบ่อยขึ้น
    weights = torch.tensor(all_losses)
    # ทำให้ค่าอยู่ในช่วงที่เหมาะสม (Normalization)
    weights = weights / weights.sum()
    
    return weights

def evaluate(model, dataloader, device, tokenizer, num_ent_labels):
    model.eval()
    
    # 1. ใช้ Criterion เดียวกับตอนเทรน
    # ent_criterion = SigmoidFocalLoss(alpha=config.ALPHA, gamma=config.GAMMA, reduction='mean')
    # rel_criterion = SigmoidFocalLoss(alpha=config.ALPHA, gamma=config.GAMMA, reduction='none')


    ent_criterion = AsymmetricFocalLoss(alpha=config.ALPHA, gamma_pos=config.POS_GAMMA, gamma_neg=config.NEG_GAMMA, reduction='mean')
    rel_criterion = AsymmetricFocalLoss(alpha=config.ALPHA, gamma_pos=config.POS_GAMMA, gamma_neg=config.NEG_GAMMA, reduction='none')
    total_loss = 0
    correct_ent = 0
    total_ent = 0
    num_batches = 0 
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            (ent_lbl_ids, ent_lbl_mask), ent_targets, \
            (rel_lbl_ids, rel_lbl_mask), rel_targets = prepare_batch_inputs(batch, tokenizer, device)

            ent_logits, rel_logits = model(
                text_ids, text_mask, ent_lbl_ids, ent_lbl_mask,
                rel_lbl_ids, rel_lbl_mask, batch['spans'], batch['pairs']
            )
            
            batch_loss_ent = 0
            batch_loss_rel = 0
            valid_ent_samples = 0
            valid_rel_samples = 0
            
            for b in range(len(batch['spans'])):
                # --- Entity Val Loss & Acc ---
                if len(batch['spans'][b]) > 0:
                    num_real = ent_targets[b].shape[1]
                    curr_logits = ent_logits[b, :len(batch['spans'][b]), :num_real]
                    curr_targets = ent_targets[b][:, :num_real]
                    
                    # Focal Loss (One-hot)
                    batch_loss_ent += ent_criterion(curr_logits, curr_targets).item()
                    valid_ent_samples += 1
                    
                    # Accuracy (ยังใช้ Argmax เพื่อเช็คว่าทายคลาสถูกไหม)
                    preds = curr_logits.argmax(dim=1)
                    target_indices = curr_targets.argmax(dim=1)
                    correct_ent += (preds == target_indices).sum().item()
                    total_ent += len(target_indices)

                # --- Relation Val Loss ---
                if rel_logits is not None and len(batch['pairs'][b]) > 0:
                    l_rel_raw = rel_criterion(rel_logits[b, :len(batch['pairs'][b]), :], rel_targets[b])
                    batch_loss_rel += l_rel_raw.mean().item()
                    valid_rel_samples += 1
            
            # รวม Loss แบบถ่วงน้ำหนักเหมือนตอนเทรน
            if valid_ent_samples > 0 or valid_rel_samples > 0:
                l_e = (batch_loss_ent / valid_ent_samples) if valid_ent_samples > 0 else 0
                l_r = (batch_loss_rel / valid_rel_samples) * 1.5 if valid_rel_samples > 0 else 0
                total_loss += (l_e + l_r)
                num_batches += 1

    final_avg_loss = total_loss / num_batches if num_batches > 0 else 0
    accuracy = correct_ent / total_ent if total_ent > 0 else 0
    
    model.train()
    return final_avg_loss, accuracy


if __name__ == "__main__":
    # ==========================================
    # Main Training Script
    # ==========================================

    device = "cuda:2" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {device}")

    # 1. Setup Data & Model
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # สร้าง Dataset สำหรับ Graph RAG
    full_dataset = GraphRAGDataset(
        json_file=config.TRAIN_FILE,
        tokenizer=tokenizer,
        max_len=256,
        neg_sample_ratio=0.0, # ไม่ต้องใช้ negative label sampling เพราะมี labels ครบแล้ว
        neg_span_ratio=2.0    # ✅ เพิ่มเป็น 2.0 - 2x negative spans ต่อ positive spans
    )

    val_dataset_raw = GraphRAGDataset(
        json_file=config.VAL_FILE,  # เช่น 'val_data.json'
        tokenizer=tokenizer,
        max_len=256,
        neg_sample_ratio=0.0,
        neg_span_ratio=2.0
    )

    train_set = full_dataset
    val_set = val_dataset_raw

    # # # 🔥 [NEW] แบ่งข้อมูล: Train 90% / Val 10%
    # train_size = int(0.9 * len(full_dataset))
    # val_size = len(full_dataset) - train_size
    # train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # print(f"📊 Data Split: Train {len(train_set)} / Val {len(val_set)}")

    # # สร้าง Loader แยกกัน
    # train_dataloader = DataLoader(
    #     train_set, 
    #     batch_size=config.BATCH_SIZE, 
    #     shuffle=True, 
    #     collate_fn=graph_rag_collate_fn
    # )

    # Validation ไม่ต้อง Shuffle และควร Batch Size ใหญ่หน่อยได้ (เพราะไม่ต้องเก็บ Gradient)
    val_dataloader = DataLoader(
        val_set, 
        batch_size=config.BATCH_SIZE * 2, 
        shuffle=False, 
        collate_fn=graph_rag_collate_fn
    )

    model = ZeroShotJointModel(config.MODEL_NAME).to(device)

    # # 2. Setup Optimizer & Loss
    # # ✅ เปลี่ยนเป็น CrossEntropyLoss สำหรับ single-label classification
    # # CrossEntropy ใช้ softmax + log likelihood ทำให้ model เรียนรู้ที่จะ discriminate ระหว่าง classes
    # num_ent_labels = len(train_set.dataset.all_ent_labels_with_O)
    # class_weights = torch.ones(num_ent_labels).to(device)

    # class_weights[0] = 0.05   # 🔥 ลดความสำคัญของ O ลงเหลือ 10%
    # class_weights[1:] = 5.0  # 🔥 เพิ่มความสำคัญของ Entity ทุกตัว

    # # 2. ใส่เข้าไปใน Loss Function
    # ent_criterion = nn.CrossEntropyLoss(
    #     weight=class_weights, 
    #     reduction='mean',
    #     label_smoothing=0.0 # ปิด smoothing เพื่อให้คมชัด
    # )
    # ent_criterion = SigmoidFocalLoss(alpha=config.ALPHA, gamma=config.GAMMA, reduction='mean')  # Entities เป็น single-label
    # rel_criterion = SigmoidFocalLoss(alpha=config.ALPHA, gamma=config.GAMMA, reduction='none')  # Relations can be multi-label

    ent_criterion = AsymmetricFocalLoss(alpha=config.ALPHA, gamma_pos=config.POS_GAMMA, gamma_neg=config.NEG_GAMMA, reduction='mean')  # Entities เป็น single-label
    rel_criterion = AsymmetricFocalLoss(alpha=config.ALPHA, gamma_pos=config.POS_GAMMA, gamma_neg=config.NEG_GAMMA, reduction='none')  # Relations can be multi-label
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')


    # --- Curriculum Configuration ---
    STAGES = [1,2,3] # 1: Easy, 2: Medium, 3: Hard (All data)
    EPOCHS_PER_STAGE = [1, 3, 3] # Total 7 epochs
    current_global_epoch = 0


    # # คำนวณจำนวน Step ทั้งหมด
    # total_steps = len(train_dataloader) * config.NUM_EPOCHS

    # --- [NEW] คำนวณ Total Steps ล่วงหน้าสำหรับทุก Stages ---
    total_training_steps = 0
    for s_idx, s_val in enumerate(STAGES):
        # จำลองการกรองข้อมูลเพื่อหาจำนวน Batch ใน Stage นั้นๆ
        s_indices = get_curriculum_indices(full_dataset, s_val)
        n_train = int(0.9 * len(s_indices))
        # จำนวน batches ต่อ epoch = (n_train / batch_size) ปัดขึ้น
        steps_in_stage = ((n_train + config.BATCH_SIZE - 1) // config.BATCH_SIZE) * EPOCHS_PER_STAGE[s_idx]
        total_training_steps += steps_in_stage

    print(f"Total planned training steps: {total_training_steps}")

    # สร้าง Scheduler (Warmup 10% แรก แล้วค่อยๆ ลด LR ลงจนเหลือ 0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_training_steps * 0.1), 
        num_training_steps=total_training_steps
    )
    # 3. Training Loop
    num_epochs = config.NUM_EPOCHS

    model.train()
    print(f"Start Training on {len(train_set)} samples...")


    best_val_acc = 0.0
    best_val_loss = float('inf')
    checkpoint_dir = config.OUTPUT_DIR
    os.makedirs(checkpoint_dir, exist_ok=True)

    for stage_idx, stage in enumerate(STAGES):
        print(f"\n🚀 Entering Curriculum Stage {stage}: " + 
            ["Easy (Short)", "Medium (Normal)", "Hard (All/Rare)"][stage_idx])
        
        # Filter dataset for this stage
        stage_indices = get_curriculum_indices(full_dataset, stage)
        stage_subset = Subset(full_dataset, stage_indices)
        
        # Re-split for Train/Val
        train_size = int(0.9 * len(stage_subset))
        val_size = len(stage_subset) - train_size
        train_set, val_set = random_split(stage_subset, [train_size, val_size])

        # 🔥 เงื่อนไขพิเศษสำหรับ Stage 3 (Hard Stage)
        sampler = None
        if stage == 3:
            # 1. รัน Mining เพื่อหาประโยคยากใน Stage 3
            # ใช้ DataLoader ชั่วคราวเพื่อหา Weights
            temp_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=graph_rag_collate_fn)
            sample_weights = get_sample_weights(model, temp_loader, device, tokenizer)
            
            # 2. สร้าง Sampler: จะสุ่มประโยคยาก (Loss สูง) มาให้โมเดลเห็นบ่อยกว่าประโยคง่าย
            sampler = WeightedRandomSampler(
                weights=sample_weights, 
                num_samples=len(sample_weights), 
                replacement=True # อนุญาตให้สุ่มประโยคเดิมซ้ำได้ (เน้นย้ำประโยคที่ผิด)
            )
        
        train_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE,sampler=sampler, 
                                     collate_fn=graph_rag_collate_fn)
        
        print(f"📊 Stage {stage} Data: {len(train_set)} samples")


        for stage_epoch in range(EPOCHS_PER_STAGE[stage_idx]):
            total_loss = 0
            
            # วนลูปจาก DataLoader ของจริง
            loop = tqdm(train_dataloader, desc=f"Epoch {stage_epoch+1}/{EPOCHS_PER_STAGE[stage_idx]}")
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
                        # --- 1. Entity Loss (Focal Loss) ---
                        if len(batch['spans'][b]) > 0:
                            num_real_labels = ent_targets[b].shape[1]
                            curr_ent_logits = ent_logits[b, :len(batch['spans'][b]), :num_real_labels]
                            
                            # 🎯 เป้าหมาย: ent_targets[b] ตอนนี้เป็น One-hot อยู่แล้วจาก Dataset 
                            # (ไม่ต้องทำ argmax เหมือน CrossEntropy)
                            curr_ent_targets = ent_targets[b][:, :num_real_labels]
                            
                            # คำนวณ Focal Loss (จะได้ค่า 'mean' ตามที่ตั้งไว้)
                            l_ent = ent_criterion(curr_ent_logits, curr_ent_targets)
                            loss_ent += l_ent
                            valid_ent_samples += 1

                        # --- 2. Relation Loss (Focal Loss + Masking) ---
                        if rel_logits is not None and len(batch['pairs'][b]) > 0:
                            # logits: [Num_Pairs, Num_Rel_Labels]
                            curr_rel_logits = rel_logits[b, :len(batch['pairs'][b]), :]
                            curr_rel_targets = rel_targets[b]
                            
                            # คำนวณแบบ 'none' จะได้ Tensor ขนาด [Num_Pairs, Num_Rel_Labels]
                            l_rel_raw = rel_criterion(curr_rel_logits, curr_rel_targets)
                            
                            # 🎯 การทำ Masking: 
                            # ในที่นี้คือการกรองคู่ที่เป็น Padding หรือไม่สมบูรณ์ (ถ้ามี)
                            # หาก dataset ส่งคู่ที่ valid มาแล้ว เราสามารถใช้ .mean() ได้เลย
                            # แต่ถ้าต้องการเน้น Relation ที่ไม่ใช่คลาสว่าง (ถ้าคลาส 0 คือ No-Relation)
                            # ตัวอย่าง: l_rel = l_rel_raw.mean()
                            
                            loss_rel += l_rel_raw.mean()
                            valid_rel_samples += 1

                    # Average Loss
                    if valid_ent_samples > 0:
                        loss_ent = loss_ent / valid_ent_samples
                    else:
                        loss_ent = torch.tensor(0.0, device=device)

                    if valid_rel_samples > 0:
                        # เพิ่มน้ำหนักให้ Relation (เช่น 1.5 หรือ 2.0) เพื่อให้โมเดลตั้งใจเรียนความสัมพันธ์มากขึ้น
                        loss_rel = (loss_rel / valid_rel_samples) * 1.5 
                    else:
                        loss_rel = torch.tensor(0.0, device=device)

                    # รวมเป็น Total Loss
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
                scheduler.step()
                current_lr = optimizer.param_groups[0]['lr']
                loop.set_description(f"Epoch {stage_epoch+1}/{EPOCHS_PER_STAGE[stage_idx]} [LR: {current_lr:.2e}]")
                
                total_loss += loss.item()
                
                loop.set_postfix(loss=loss.item())


            
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"--- Epoch {stage_epoch+1} Finished. Avg Loss: {avg_train_loss:.4f} ---")
            # 🔥 [NEW] เริ่มกระบวนการ Validation
            print(f"\n--- Validating Epoch {stage_epoch+1} ---")
            num_ent_labels = len(full_dataset.all_ent_labels_with_O)
            val_loss, val_acc = evaluate(model, val_dataloader, device, tokenizer,num_ent_labels)
            
            print(f"✅ Epoch {stage_epoch+1} Summary:")
            print(f"   - LR: {current_lr:.6f}")
            print(f"   - Train Loss: {avg_train_loss:.4f}")
            print(f"   - Val Loss:   {val_loss:.4f} (ยิ่งต่ำยิ่งดี)")
            print(f"   - Val Acc:    {val_acc*100:.2f}% (ยิ่งสูงยิ่งดี)")
            print("-" * 40)

            # --- 🔥 [NEW] Best Model Checkpoint (Zero-shot Optimization) ---
            # เราจะเซฟเมื่อ Val Acc เพิ่มขึ้น หรือ Val Loss ต่ำลงอย่างมีนัยสำคัญ
            is_best = False
            
            # เงื่อนไขที่ 1: Accuracy ดีที่สุดเท่าที่เคยมีมา
            if val_acc > best_val_acc:
                print(f"🌟 New Best Acc! ({val_acc*100:.2f}% > {best_val_acc*100:.2f}%)")
                best_val_acc = val_acc
                is_best = True
            
            # เงื่อนไขที่ 2: Loss ต่ำที่สุด (ช่วยยืนยันเรื่อง Generalization)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # ถ้า Loss ต่ำลงมาก แม้ Acc จะไม่พุ่ง เราก็อาจจะเซฟไว้เป็นทางเลือก
                if not is_best and val_acc > (best_val_acc * 0.95): # ยอมรับ Acc ตกได้นิดหน่อยถ้า Loss สวย
                     is_best = True

            if is_best:
                save_path = f"{checkpoint_dir}/best_model.bin"
                torch.save(model.state_dict(), save_path)
                
                # เซฟ Config ควบคู่ไปด้วยเพื่อให้พร้อมใช้งานทันที
                with open(f"{checkpoint_dir}/config.json", "w", encoding='utf-8') as f:
                    json.dump({
                        "model_name": config.MODEL_NAME,
                        "ent_labels": full_dataset.all_ent_labels_with_O,
                        "rel_labels": sorted(list(full_dataset.all_rel_labels)),
                        "best_epoch": stage_epoch + 1,
                        "stage": stage,
                        "val_acc": val_acc
                    }, f, ensure_ascii=False, indent=4)
                print(f"💾 Saved Best Model to: {save_path}")
            
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
            "ent_labels": train_set.dataset.dataset.all_ent_labels_with_O,  # ✅ รวม "O"
            "rel_labels": sorted(list(train_set.dataset.dataset.all_rel_labels))
        }, f, ensure_ascii=False, indent=4)
        
    print(f"Model saved to {config.OUTPUT_DIR}")
    print(f"✅ Entity labels (with O): {train_set.dataset.dataset.all_ent_labels_with_O}")