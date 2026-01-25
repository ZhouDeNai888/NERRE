import os
import sys
import json
# 1. หา path ของโฟลเดอร์แม่ (NERRE)
# (__file__ คือไฟล์ปัจจุบัน -> dirname ครั้งที่ 1 ได้ 'train/' -> dirname ครั้งที่ 2 ได้ 'NERRE/')
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 2. เพิ่ม path นั้นเข้าไปในระบบ
sys.path.append(parent_dir)
import random
import numpy as np
from sklearn.metrics import f1_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast 
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# --- Imports ของเรา ---
from model.model import ZeroShotJointModel 
import train_config as config
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import Subset
from torch.utils.data import random_split
from torch.utils.data import WeightedRandomSampler
import torch.optim.swa_utils as swa_utils
# นำเข้า Dataset - ใช้ GraphRAGDataset สำหรับ Graph RAG
from data.GraphRAGDataset import GraphRAGDataset, graph_rag_collate_fn

# ตรวจสอบว่ามีไฟล์ training data หรือไม่
if not all(os.path.exists(f) for f in config.TRAIN_FILE):
    print(f"❌ Train file not found in: {config.TRAIN_FILE}")
    print("   Please create a training dataset first.")
    sys.exit(1)
else:
    print(f"✅ Found training files: {config.TRAIN_FILE}")



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
        sample = dataset.data[i]
        text_len = len(sample['text'].split())
        num_ents = len(sample['entities'])
        num_rels = len(sample.get('relations', []))

        if stage == 1: # Easy but Meaningful
            # เลือกประโยคที่ยาวพอดีๆ และมีความสัมพันธ์ชัดเจน (ไม่ใช่แค่คำลอยๆ)
            if 10 < text_len < 35 and 2 <= num_ents <= 3 and num_rels >= 1:
                indices.append(i)
        elif stage == 2: # Medium - เพิ่มความซับซ้อน
            if text_len < 60 and num_ents <= 6:
                indices.append(i)
        else: # Stage 3: All data (Hard / Rare cases)
            indices.append(i)
            
    # ถ้า Stage 1 ได้ข้อมูลน้อยเกินไป (เช่น < 5000) ให้สุ่มประโยคสั้นอื่นๆ มาเสริม
    if stage == 1 and len(indices) < 5000:
        return list(range(len(dataset)))[:10000] 
        
    return indices

def set_trainable_layers(model, stage):
    if stage == 1:
        # Stage 1: Mastering Entity
        # Train: Encoder, Entity Projector, Width Embedding, Span Attention
        # Freeze: Relation Projector, Distance Embedding, Directional Gates
        print("🔓 [Stage 1] Training Encoder + Entity Layout (Relation logic Frozen)")
        for param in model.parameters():
            param.requires_grad = True
            
        # Freeze Relation specific components
        for param in model.relation_proj.parameters():
            param.requires_grad = False
        if hasattr(model, 'dist_emb'):
            for param in model.dist_emb.parameters():
                param.requires_grad = False
        if hasattr(model, 's_gate'):
            for param in model.s_gate.parameters():
                param.requires_grad = False
        if hasattr(model, 'o_gate'):
            for param in model.o_gate.parameters():
                param.requires_grad = False
            
    elif stage == 2:
        # Stage 2: Relation Injection 
        # Freeze: Encoder, Entity Projector, Width Embedding
        # Train: Relation Projector, Distance, Gates, Temps
        print("🔒 [Stage 2] Freezing Encoder/Entity (Training Relation Logic ONLY)")
        for param in model.parameters():
            param.requires_grad = False
            
        # Unfreeze Relation components
        for param in model.relation_proj.parameters():
            param.requires_grad = True
        if hasattr(model, 'dist_emb'):
            for param in model.dist_emb.parameters():
                param.requires_grad = True
        if hasattr(model, 's_gate'):
            for param in model.s_gate.parameters():
                param.requires_grad = True
        if hasattr(model, 'o_gate'):
            for param in model.o_gate.parameters():
                param.requires_grad = True
        
        # Allow Temperatures to tune
        if hasattr(model, 'log_temp_ent'): model.log_temp_ent.requires_grad = True
        if hasattr(model, 'log_temp_rel'): model.log_temp_rel.requires_grad = True

    else:
        # Stage 3: Joint Fine-tuning
        print("🔓 [Stage 3] Joint Fine-tuning (All Layers Trainable)")
        for param in model.parameters():
            param.requires_grad = True

def get_sample_weights(model, dataloader, device, tokenizer):
    model.eval()
    all_losses = []
    
    # ใช้ CrossEntropyLoss แบบลดทอน (None reduction) เพื่อดู Loss รายตัว
    ent_criterion = nn.CrossEntropyLoss(reduction='none')

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
                    
                    # Convert to Indices
                    curr_target_indices = curr_targets.argmax(dim=1)
                    
                    l_ent = ent_criterion(curr_logits, curr_target_indices)
                    
                    # Mean Loss of this sample
                    loss_val = l_ent.mean().item()
                    all_losses.append(loss_val)
                else:
                    # ประโยคไม่มี entity ให้ค่าน้ำหนักต่ำๆ ไว้ (แต่ไม่เป็น 0)
                    all_losses.append(0.01)

    # แปลง Loss เป็น Weights: ประโยคที่ Loss สูงจะมีโอกาสถูกสุ่มมาเทรนบ่อยขึ้น
    weights = torch.tensor(all_losses)
    # ทำให้ค่าอยู่ในช่วงที่เหมาะสม (Normalization)
    weights = weights / weights.sum()
    
    return weights

def evaluate(model, dataloader, device, tokenizer, num_ent_labels):
    model.eval()
    
    # 1. Setup Criterions
    # Entity: CrossEntropyLoss (matching training)
    ent_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # Relation: CrossEntropyLoss (updated)
    rel_criterion = nn.CrossEntropyLoss(reduction='mean')
    
    total_loss = 0
    correct_ent, total_ent = 0, 0
    
    # Store all relation preds and targets for F1 calculation
    all_rel_preds = []
    all_rel_targets = []
    
    num_batches = 0 

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            
            (ent_lbl_ids, ent_lbl_mask), ent_targets, \
            (rel_lbl_ids, rel_lbl_mask), rel_targets = prepare_batch_inputs(batch, tokenizer, device)

            # Forward pass
            ent_logits, rel_logits = model(
                text_ids, text_mask, ent_lbl_ids, ent_lbl_mask,
                rel_lbl_ids, rel_lbl_mask, batch['spans'], batch['pairs']
            )
            
            batch_loss = 0
            valid_samples_in_batch = 0
            
            for b in range(len(batch['spans'])):
                sample_loss = 0
                has_valid_task = False

                # --- 1. Entity Metrics & Loss ---
                if len(batch['spans'][b]) > 0:
                    num_real = ent_targets[b].shape[1]
                    curr_logits = ent_logits[b, :len(batch['spans'][b]), :num_real]
                    curr_targets = ent_targets[b][:, :num_real]
                    
                    # Convert to Indices for CrossEntropy
                    curr_target_indices = curr_targets.argmax(dim=1)

                    # Calculate Entity Loss
                    l_ent = ent_criterion(curr_logits, curr_target_indices)
                    sample_loss += l_ent
                    
                    # Accuracy
                    preds = curr_logits.argmax(dim=1)
                    correct_ent += (preds == curr_target_indices).sum().item()
                    total_ent += len(curr_target_indices)
                    has_valid_task = True

                # --- 2. Relation Metrics & Loss ---
                if rel_logits is not None and len(batch['pairs'][b]) > 0:
                    curr_rel_logits = rel_logits[b, :len(batch['pairs'][b]), :]
                    curr_rel_targets = rel_targets[b]
                    
                    # Convert to Indices
                    curr_rel_targets_idx = curr_rel_targets.argmax(dim=1)
                    
                    l_rel = rel_criterion(curr_rel_logits, curr_rel_targets_idx)
                    
                    sample_loss += l_rel

                    # Store for global F1
                    rel_preds = curr_rel_logits.argmax(dim=-1)
                    all_rel_preds.extend(rel_preds.cpu().numpy())
                    all_rel_targets.extend(curr_rel_targets_idx.cpu().numpy())
                    
                    has_valid_task = True
                
                if has_valid_task:
                    batch_loss += sample_loss
                    valid_samples_in_batch += 1

            if valid_samples_in_batch > 0:
                total_loss += (batch_loss / valid_samples_in_batch).item()
                num_batches += 1

    # Final Aggregation
    final_avg_loss = total_loss / num_batches if num_batches > 0 else 0
    ent_acc = correct_ent / total_ent if total_ent > 0 else 0
    
    # Calculate Relation F1 (Macro/Micro) excluding NO_RELATION (index 0)
    # Assuming index 0 is NO_RELATION
    if len(all_rel_targets) > 0:
        # labels=list(range(1, max(all_rel_targets)+1)) if using dynamics, 
        # but let's just use unique labels present in data minus 0
        unique_labels = list(set(all_rel_targets) | set(all_rel_preds))
        if 0 in unique_labels: unique_labels.remove(0)
        
        rel_f1 = f1_score(all_rel_targets, all_rel_preds, labels=unique_labels, average='micro', zero_division=0)
        rel_acc = (np.array(all_rel_preds) == np.array(all_rel_targets)).mean() # Raw accuracy
    else:
        rel_f1 = 0.0
        rel_acc = 0.0
    
    print(f"\n📊 Validation Results:")
    print(f"   - Entity Acc:   {ent_acc*100:.2f}%")
    print(f"   - Relation Acc: {rel_acc*100:.2f}% (Includes NO_RELATION)")
    print(f"   - Relation F1:  {rel_f1*100:.2f}% (Micro, Excludes NO_RELATION)")
    print(f"   - Total Loss:   {final_avg_loss:.4f}")

    model.train()
    
    # 🔥 [User Request] Combined Score for Best Model
    # ใช้วิธีเฉลี่ย (Entity Accuracy + Relation F1) เพื่อให้โมเดลเก่งทั้งคู่
    combined_score = (ent_acc + rel_f1) / 2
    
    return final_avg_loss, combined_score


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
        neg_span_ratio=6.0    # ✅ เพิ่มเป็น 2.0 - 2x negative spans ต่อ positive spans
    )

    val_dataset_raw = GraphRAGDataset(
        json_file=config.VAL_FILE,
        tokenizer=tokenizer,
        max_len=256,
        neg_sample_ratio=0.0,
        neg_span_ratio=2.0
    )

    train_set = full_dataset
    val_set = val_dataset_raw

    # Validation ไม่ต้อง Shuffle และควร Batch Size ใหญ่หน่อยได้ (เพราะไม่ต้องเก็บ Gradient)
    val_dataloader = DataLoader(
        val_set, 
        batch_size=config.BATCH_SIZE * 2, 
        shuffle=False, 
        collate_fn=graph_rag_collate_fn
    )

    model = ZeroShotJointModel(config.MODEL_NAME).to(device)

    if hasattr(train_set, 'dataset'):
        all_labels = train_set.dataset.all_ent_labels_with_O
        all_rel_labels = train_set.dataset.all_rel_labels_with_NO_REL
    else:
        all_labels = train_set.all_ent_labels_with_O
        all_rel_labels = train_set.all_rel_labels_with_NO_REL
        
    num_ent_labels = len(all_labels)
    class_weights = torch.ones(num_ent_labels).to(device)
    class_weights[0] = 0.1  # ✅ Further increased O weight to 0.5 to improve Precision
    class_weights[1:] = 2.0 # Keep Real Entities higher
    
    ent_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='mean')

    # 🔥 [SOTA Rel Fix] Switch Relation to CrossEntropy too
    num_rel_labels = len(all_rel_labels)
    rel_class_weights = torch.ones(num_rel_labels).to(device)
    rel_class_weights[0] = 0.8   # ✅ เพิ่มน้ำหนัก NO_RELATION เป็น 0.8 เพื่อกด False Positives (แก้ปัญหาทิศทางมั่ว)
    
    rel_criterion = nn.CrossEntropyLoss(weight=rel_class_weights, reduction='mean', label_smoothing=0.05) 
    
    optimizer = optim.AdamW(model.parameters(), lr=config.LR, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler('cuda')


    # --- Curriculum Configuration ---
    STAGES = [1,2,3] # 1: Easy, 2: Medium, 3: Hard (All data)
    EPOCHS_PER_STAGE = [3, 5, 7] # Total 15 epochs
    current_global_epoch = 0


    # --- [SWA Config] ---
    swa_model = swa_utils.AveragedModel(model)
    # SWA Learning Rate มักจะใช้ค่าคงที่ต่ำๆ (เช่น 10% ของ LR ปกติ)
    swa_scheduler = swa_utils.SWALR(optimizer, swa_lr=config.LR * 0.1)

    # กำหนดจุดเริ่ม SWA: ใน Stage 3 (Epoch 11-12 จากทั้งหมด 12)
    # หรือคิดเป็น 2 Epoch สุดท้ายของ Stage 3
    swa_start_epoch = EPOCHS_PER_STAGE[2] - 2

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
        
        # if os.path.exists(f"{config.OUTPUT_DIR}/best_model.bin"):
        #     model.load_state_dict(torch.load(f"{config.OUTPUT_DIR}/best_model.bin", map_location=device))
        #     print("✅ Loaded previous best weights. Ready for next stage!")
        
        set_trainable_layers(model, stage)


        # Filter dataset for this stage
        stage_indices = get_curriculum_indices(full_dataset, stage)
        stage_subset = Subset(full_dataset, stage_indices)
        
        # Re-split for Train/Val
        train_size = int(1 * len(stage_subset))
        val_size = len(stage_subset) - train_size
        train_set, val_set = random_split(stage_subset, [train_size, val_size])



        # --- 🚀 [V19 FIX] RESET OPTIMIZER & SCHEDULER FOR NEW STAGE ---
        # การ Reset ตรงนี้จะช่วยให้ Stage 2 และ 3 เริ่มต้นด้วย LR ที่สูงอีกครั้ง (Warmup)
        
        # คำนวณ Steps เฉพาะของ Stage นี้
        num_epochs_this_stage = EPOCHS_PER_STAGE[stage_idx]
        steps_per_epoch = (train_size + config.BATCH_SIZE - 1) // config.BATCH_SIZE
        total_stage_steps = steps_per_epoch * num_epochs_this_stage
        current_lr = config.LR if stage < 3 else config.LR * 0.2
        # Re-initialize Optimizer (ล้างค่า momentum/velocity เก่าที่อาจจะค้างจาก Stage ก่อน)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=current_lr, weight_decay=config.WEIGHT_DECAY)
        
        # Re-create Scheduler ให้มี Warmup ใหม่สำหรับ Stage นี้โดยเฉพาะ
        # ใช้ 10-15% ของ steps ใน stage นี้เป็น warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_stage_steps * 0.1), 
            num_training_steps=total_stage_steps
        )

        # 🔥 เงื่อนไขพิเศษสำหรับ Stage 3 (Hard Stage)
        sampler = None
        if stage == 3:
            # 1. หา Weights ทั้งหมด
            temp_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=graph_rag_collate_fn)
            sample_weights = get_sample_weights(model, temp_loader, device, tokenizer)
            
            # 2. 🔥 [Step 2] Active Hard Sampling Logic
            # เราจะสร้าง 2 รายการอินเด็กซ์
            num_samples = len(train_set)
            
            # - กลุ่มที่ 1: Top 25% ของประโยคที่ Loss สูงที่สุด (จุดอ่อนสะสม)
            num_hard = int(num_samples * 0.25)
            top_hard_indices = torch.topk(sample_weights, k=num_hard).indices.tolist()
            
            # - กลุ่มที่ 2: สุ่มจากประโยคทั้งหมดตามปกติ
            all_indices = list(range(num_samples))
            
            # สร้าง Weighted Sampler ที่เน้น 'จุดอ่อน' เป็นพิเศษ
            # โดยการบวกน้ำหนักเพิ่มให้กลุ่ม Top Hard
            active_weights = sample_weights.clone()
            active_weights[top_hard_indices] *= 2.0 # เน้นกลุ่มที่ทำผิดซ้ำซากเพิ่ม 2 เท่า
            
            sampler = WeightedRandomSampler(
                weights=active_weights, 
                num_samples=num_samples, 
                replacement=True 
            )
            
            train_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE, sampler=sampler, collate_fn=graph_rag_collate_fn)
        else:
            train_dataloader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=graph_rag_collate_fn)
        print(f"📊 Stage {stage} Data: {len(train_set)} samples")


        for stage_epoch in range(EPOCHS_PER_STAGE[stage_idx]):


            # ถ้าอยู่ใน Stage 3 และเป็น Epoch ที่ 1, 3, 5... ให้คำนวณน้ำหนักใหม่
            if stage == 3 and (stage_epoch % 2 == 0):
                print(f"\n🔄 Stage 3: Re-calculating Sample Weights for Epoch {stage_epoch+1}...")
                
                # ใช้ DataLoader ชั่วคราวสแกนหาประโยคที่โมเดล 'ในขณะนี้' ยังทายผิดอยู่
                temp_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=graph_rag_collate_fn)
                sample_weights = get_sample_weights(model, temp_loader, device, tokenizer)
                
                # สร้าง Sampler ตัวใหม่ที่เน้นจุดอ่อนล่าสุด
                sampler = WeightedRandomSampler(
                    weights=sample_weights, 
                    num_samples=len(sample_weights), 
                    replacement=True
                )
                
                # อัปเดต DataLoader หลักที่ใช้เทรน
                train_dataloader = DataLoader(
                    train_set, 
                    batch_size=config.BATCH_SIZE, 
                    sampler=sampler, 
                    collate_fn=graph_rag_collate_fn
                )
                print("✅ DataLoader updated with fresh hard samples!")

            # กรณี Stage อื่นๆ หรือ Epoch ที่ไม่ต้อง Re-sample ให้ใช้ DataLoader เดิม
            elif stage_epoch == 0: # สร้างครั้งแรกสำหรับ Stage 1 และ 2
                 train_dataloader = DataLoader(
                    train_set, 
                    batch_size=config.BATCH_SIZE, 
                    shuffle=True, 
                    collate_fn=graph_rag_collate_fn
                )




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
                
                # optimizer.zero_grad()
                
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
                        # --- 1. Entity Loss (CrossEntropy) ---
                        if len(batch['spans'][b]) > 0:
                            num_real_labels = ent_targets[b].shape[1]
                            curr_ent_logits = ent_logits[b, :len(batch['spans'][b]), :num_real_labels]
                            
                            # แปลง One-hot กลับเป็น Class Indices สำหรับ CrossEntropyLoss
                            curr_ent_targets_onehot = ent_targets[b][:, :num_real_labels]
                            curr_ent_targets_idx = curr_ent_targets_onehot.argmax(dim=1)
                            
                            l_ent = ent_criterion(curr_ent_logits, curr_ent_targets_idx)
                            loss_ent += l_ent
                            valid_ent_samples += 1

                        # --- 2. Relation Loss (CrossEntropy) ---
                        if rel_logits is not None and len(batch['pairs'][b]) > 0:
                            # logits: [Num_Pairs, Num_Rel_Labels]
                            curr_rel_logits = rel_logits[b, :len(batch['pairs'][b]), :]
                            curr_rel_targets = rel_targets[b]
                            
                            # Convert to Indices
                            curr_rel_targets_idx = curr_rel_targets.argmax(dim=1)
                            
                            l_rel = rel_criterion(curr_rel_logits, curr_rel_targets_idx)
                            
                            loss_rel += l_rel
                            valid_rel_samples += 1

                    # Average Loss
                    if valid_ent_samples > 0:
                        loss_ent = loss_ent / valid_ent_samples
                    else:
                        loss_ent = torch.tensor(0.0, requires_grad=True, device=device)

                    if valid_rel_samples > 0:
                        loss_rel = loss_rel / valid_rel_samples
                    else:
                        loss_rel = torch.tensor(0.0, requires_grad=True, device=device)

                    # # รวมเป็น Total Loss
                    # loss = (loss_ent) + (loss_rel)

                    # 3. 🔥 หัวใจสำคัญ: การผสม Loss ตาม Stage
                    if stage == 1:
                        # เน้น Entity อย่างเดียวเพื่อให้ถึงเป้า 80%
                        loss = loss_ent * 1.0 
                    elif stage == 2:
                        # สอน RE โดยเฉพาะ (เพราะ NER ถูก Freeze ไว้แล้ว)
                        # ใช้ตัวคูณสูงเพื่อให้ Gradient แรงพอจะปรับหัวใหม่
                        loss = loss_rel * 50.0 
                    else:
                        # จูนพร้อมกัน (Balance)
                        loss = (loss_ent * 1.0) + (loss_rel * 2.0)
                    
                    # Handle case where both are zero (no valid samples at all)
                    if valid_ent_samples == 0 and valid_rel_samples == 0:
                        loss = torch.tensor(0.0, requires_grad=True, device=device)

                    # กรณีกันพลาด: ถ้าไม่มีตัวอย่างเลยใน Batch นั้น
                    if valid_ent_samples == 0 and valid_rel_samples == 0:
                        loss = torch.tensor(0.0, requires_grad=True, device=device)

                # Backward
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()

                # --- 🔥 [V3 SWA Upgrade] วางตรงนี้ 🔥 ---
                # เช็คว่าอยู่ใน Stage 3 และถึง Epoch ที่กำหนดให้เริ่ม SWA หรือยัง
                if stage == 3 and stage_epoch >= swa_start_epoch:
                    # 1. อัปเดตค่าเฉลี่ยน้ำหนักโมเดล (เก็บสะสมน้ำหนักเข้า swa_model)
                    swa_model.update_parameters(model)
                    # 2. ใช้ swa_scheduler (ค่า LR คงที่ต่ำๆ) แทนตัวหลัก
                    swa_scheduler.step()
                else:
                    # กรณีอื่นๆ (Stage 1, 2 หรือต้น Stage 3) ให้ใช้ Linear Scheduler ปกติ
                    scheduler.step()
                
                # ---------------------------------------

                optimizer.zero_grad() # ล้าง gradient หลังอัปเดตน้ำหนักแล้ว

                current_lr = optimizer.param_groups[0]['lr']
                loop.set_description(f"Epoch {stage_epoch+1}/{EPOCHS_PER_STAGE[stage_idx]} [LR: {current_lr:.2e}]")
                
                total_loss += loss.item()
                
                # loop.set_postfix(loss=loss.item())
                loop.set_postfix(
                    loss=f"{loss.item():.3f}",
                    ent_loss=f"{loss_ent.item():.3f}",
                    rel_loss=f"{loss_rel.item():.3f}" 
                )


            
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
            print(f"   - Val Score:  {val_acc*100:.2f}% (Average of Ent Acc & Rel F1)")
            print("-" * 40)

            # --- 🔥 [NEW] Best Model Checkpoint (Zero-shot Optimization) ---
            # เราจะเซฟเมื่อ combined score เพิ่มขึ้น
            is_best = False
            
            # เงื่อนไขที่ 1: Score ดีที่สุดเท่าที่เคยมีมา
            if val_acc > best_val_acc:
                print(f"🌟 New Best Score! ({val_acc*100:.2f}% > {best_val_acc*100:.2f}%)")
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


    # --- 🔥 [V3 SWA Finalize] วางตรงนี้ (นอกลูป Stage ใหญ่) 🔥 ---
    # หลังจากวนเทรนจนครบทุก Stage (จบ Stage 3)
    
    print("\n🚀 All Stages Complete. Finalizing SWA Model...")
    
    # 1. อัปเดต BatchNorm/LayerNorm statistics
    # สำคัญมาก: เพราะน้ำหนัก SWA คือค่าเฉลี่ย ต้องรันข้อมูลผ่านอีกรอบเพื่อให้โมเดลปรับค่าสถิติภายใน
    swa_utils.update_bn(train_dataloader, swa_model, device=device)

    # 2. ใช้ swa_model ในการ Evaluate ครั้งสุดท้ายเพื่อดูผลลัพธ์ 90%+
    # หมายเหตุ: swa_model มักจะถูกห่อ (wrap) ไว้ ถ้าใช้ DataParallel ต้องดึง .module ออกมา
    model_to_eval = swa_model.module if hasattr(swa_model, 'module') else swa_model
    
    # ปิดการทำงานของ BatchNorm update ภายใน evaluate
    val_loss, val_acc = evaluate(model_to_eval, val_dataloader, device, tokenizer, num_ent_labels)
    
    print(f"📊 Final SWA Validation Accuracy: {val_acc*100:.2f}%")

    # 3. เซฟ SWA Model เป็นไฟล์หลักสำหรับการใช้งานจริง
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.save(swa_model.state_dict(), f"{config.OUTPUT_DIR}/swa_model.bin")
    
    # -------------------------------------------------------



    print("Training Complete!")

    # --- Save Model ---
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    torch.save(model.state_dict(), f"{config.OUTPUT_DIR}/pytorch_model.bin")
    tokenizer.save_pretrained(config.OUTPUT_DIR)
    
    # Find base dataset
    base_ds = train_set
    while hasattr(base_ds, 'dataset'):
        base_ds = base_ds.dataset

    # Save Config - ✅ ใช้ all_ent_labels_with_O ที่รวม "O" label
    # Also save descriptions for inference
    with open(f"{config.OUTPUT_DIR}/config.json", "w", encoding='utf-8') as f:
        json.dump({
            "model_name": config.MODEL_NAME,
            "ent_labels": base_ds.all_ent_labels_with_O,  # ✅ รวม "O"
            "rel_labels": base_ds.all_rel_labels_with_NO_REL,
            "ent_label_descriptions": getattr(base_ds, 'ent_label_texts', []),
            "rel_label_descriptions": getattr(base_ds, 'rel_label_texts', []),
            "max_len": 256
        }, f, ensure_ascii=False, indent=4)
        
    print(f"Model saved to {config.OUTPUT_DIR}")
    print(f"✅ Entity labels (with O): {base_ds.all_ent_labels_with_O}")