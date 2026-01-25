import os
import sys
import json
import torch
import random
import numpy as np
from transformers import AutoTokenizer

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# 1. Setup Paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from model.model import ZeroShotJointModel

# --- CONFIG ---
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model_path = "saved_model_v24" 

print(f"⏳ Loading model from {model_path}...")

# 2. Load Config & Labels
with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

ent_labels = config["ent_labels"] # ต้องมี "O" อยู่ตัวแรก
rel_labels = config["rel_labels"]

# Check for descriptions (Zero-Shot Support)
ent_label_texts = config.get("ent_label_descriptions", ent_labels)
rel_label_texts = config.get("rel_label_descriptions", rel_labels)

if "ent_label_descriptions" in config:
    print(f"✅ Found descriptions for labels. Using semantic descriptions for inference.")

print(f"✅ Loaded {len(ent_labels)} Entity types and {len(rel_labels)} Relation types.")

# 3. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path,fix_mistral_regex=True)
model = ZeroShotJointModel(config["model_name"]).to(device)
model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
model.eval()

def predict(text):
    print(f"\n" + "="*50)
    print(f"📝 Text: {text}")
    print("="*50)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    
    # Helper: Encode Labels
    def encode_list(labels):
        t = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(device)
        return t["input_ids"].unsqueeze(0), t["attention_mask"].unsqueeze(0)

    # Use descriptions for encoding if available
    ent_input, ent_mask = encode_list(ent_label_texts)
    rel_input, rel_mask = encode_list(rel_label_texts)

    # 1. Generate Spans (Sliding Window)
    spans = []
    max_span_width = 8
    for w in range(1, max_span_width + 1):
        for i in range(seq_len - w + 1):
            spans.append((i, i + w - 1))
    
    if not spans: return

    with torch.no_grad():
        # -------------------------------------------------
        # STEP 1: Predict Entities
        # -------------------------------------------------
        e_logits, _ = model(
            inputs["input_ids"], inputs["attention_mask"],
            ent_input, ent_mask,
            rel_input, rel_mask,
            entity_spans=[spans], relation_pairs=None
        )

        tau = model.temperature.item() 
        print(f"🌡️ Using Learned Temperature: {tau:.4f}")
        
        # [FIXED PROBABILITIES]
        # The logits from model() are ALREADY scaled by 1/temperature.
        # So we should NOT divide by tau again here.
        e_probs = torch.softmax(e_logits, dim=-1) # Remove / tau
        
        # 2. Separate "O" score from "Entity" scores for filtering
        o_probs = e_probs[0, :, 0]
        entity_only_probs = e_probs[0, :, 1:] 
        
        best_ent_scores, best_ent_indices = torch.max(entity_only_probs, dim=-1)
        best_ent_indices += 1 # Shift index back
        
        found_entities = []
        
        # --- DEBUG SCAN ---
        print(f"--- 🔍 DEBUG SCAN (Entities) ---")
        keywords = ["Elon", "Musk", "SpaceX", "ลิซ่า", "BLACKPINK", "Harry", "Potter"]
        
        for idx in range(len(spans)):
            score = best_ent_scores[idx].item()
            label_idx = best_ent_indices[idx].item()
            start, end = spans[idx]
            span_text = tokenizer.decode(inputs["input_ids"][0][start:end+1]).strip()
            label_text = ent_labels[label_idx]
            
            # Debug Logic
            if any(k in span_text for k in keywords) and len(span_text.split()) <= 2:
                 o_score = o_probs[idx].item()
                 # print(f"Token: '{span_text}' | Pred: {label_text} ({score:.4f}) | O-Score: {o_score:.4f}")

            # [FIXED FILTER LOGIC]
            # Even if O class probability is higher, if the best Entity class probability > Threshold, take it.
            # This improves Recall significantly.
            if score < 0.25: continue    # Threshold (Matched infer2.py)
            
            # Additional Check: Skip purely special tokens
            if span_text in ["<s>", "</s>", "<pad>", ".", ","]: continue

            # Simple NMS (ตัดตัวซ้ำ)
            is_overlap = False
            for exist in found_entities:
                e_s, e_e = exist['span']
                if not (end < e_s or start > e_e): # Overlap?
                    if score > exist['score']: # ถ้าตัวใหม่มั่นใจกว่า
                        exist['active'] = False # ปิดตัวเก่า
                    else:
                        is_overlap = True
                        break # Stop checking once overlapped
            
            if not is_overlap:
                found_entities.append({
                    "idx": idx, "span": (start, end), 
                    "text": span_text, "label": label_text, "score": score, "active": True
                })
        
        # Filter active entities only
        found_entities = [e for e in found_entities if e['active']]
        print("--- END DEBUG ---\n")

        print("📍 Entities found:")
        if not found_entities:
            print("  (None)")


        for ent in found_entities:
            print(f"  - [{ent['label']}] {ent['text']} ({ent['score']:.2f})")

        # -------------------------------------------------
        # STEP 2: Predict Relations
        # -------------------------------------------------
        if len(found_entities) > 1:
            pairs = []
            pair_map = [] # เก็บ mapping ว่า pair index นี้คือ entity ตัวไหนกับตัวไหน
            
            for i in range(len(found_entities)):
                for j in range(len(found_entities)):
                    if i == j: continue
                    # ใช้ index ดั้งเดิมใน spans list
                    pairs.append((found_entities[i]['idx'], found_entities[j]['idx']))
                    pair_map.append((i, j))
            
            # Run Model อีกรอบเพื่อหา Relation
            _, r_logits = model(
                inputs["input_ids"], inputs["attention_mask"],
                ent_input, ent_mask,
                rel_input, rel_mask,
                entity_spans=[spans], relation_pairs=[pairs]
            )
            # IMPORTANT: Switch from Sigmoid to Softmax because we trained with CrossEntropy
            # Check if model has temperature for relation
            if hasattr(model, 'log_temperature'):
                # Use model's internal temperature scaling if available
                pass
            
            # Apply Softmax (Logic: Single best relation per pair, including NO_RELATION)
            # Logits are already scaled by temperature in model.forward()
            r_probs = torch.softmax(r_logits, dim=-1)  
            
            print("\n🔗 Relations found:")
            has_rel = False
            for k, (i, j) in enumerate(pair_map):
                # หา Relation ที่คะแนนสูงสุด
                val, r_idx = torch.max(r_probs[0][k], dim=-1)
                
                # Filter Logic:
                # 1. ต้องไม่ใช่ Class 0 (NO_RELATION)
                # 2. Score ต้อง > Threshold (0.05 is permissive for filtered entities)
                if r_idx.item() > 0 and val.item() > 0.05:
                    has_rel = True
                    h = found_entities[i]
                    t = found_entities[j]
                    r_name = rel_labels[r_idx]
                    print(f"  - {h['text']} --[{r_name}]--> {t['text']} ({val:.2f})")
            
            if not has_rel:
                print("  (No relations found)")

# --- RUN TEST ---
predict("Elon Musk founded SpaceX in 2002.")
predict("ลิซ่า เป็นสมาชิกของวง BLACKPINK")
predict("Harry Potter used his magic wand to fight Voldemort at Hogwarts.")
predict("ลิซ่า BLACKPINK เดินทางไปร่วมงานแฟชั่นโชว์ที่ปารีส ประเทศฝรั่งเศส")
predict("Tony Stark is also known as Iron Man.")
predict("สมเด็จพระนางเจ้าสิริกิติ์ พระบรมราชินีนาถ ทรงเป็นพระราชินีของประเทศไทย")
predict("周德耐在慈濟大學讀書")
predict("I am Chatdanai from Thailand.")