import os
import sys
import json
import torch
from transformers import AutoTokenizer

# 1. Setup Paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from model.model import ZeroShotJointModel

# --- CONFIG ---
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model_path = "saved_model_v16" 

print(f"⏳ Loading model from {model_path}...")

# 2. Load Config & Labels
with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
    config = json.load(f)

ent_labels = config["ent_labels"] # ต้องมี "O" อยู่ตัวแรก
rel_labels = config["rel_labels"]

print(f"✅ Loaded {len(ent_labels)} Entity types and {len(rel_labels)} Relation types.")

# 3. Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ZeroShotJointModel(config["model_name"]).to(device)
model.load_state_dict(torch.load(os.path.join(model_path, "best_model.bin"), map_location=device))
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

    ent_input, ent_mask = encode_list(ent_labels)
    rel_input, rel_mask = encode_list(rel_labels)

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
        
        e_probs = torch.sigmoid(e_logits / tau)
        scores, preds = torch.max(e_probs, dim=-1)
        
        found_entities = []
        
        # --- DEBUG SCAN ---
        print(f"--- 🔍 DEBUG SCAN (Entities) ---")
        keywords = ["Elon", "Musk", "SpaceX", "ลิซ่า", "BLACKPINK", "Harry", "Potter"]
        
        for idx, (score, label_idx) in enumerate(zip(scores[0], preds[0])):
            start, end = spans[idx]
            span_text = tokenizer.decode(inputs["input_ids"][0][start:end+1]).strip()
            label_text = ent_labels[label_idx]
            
            # Debug Logic
            if any(k in span_text for k in keywords) and len(span_text.split()) <= 2:
                 o_score = e_probs[0, idx, 0].item() 
                 print(f"Token: '{span_text}' | Pred: {label_text} ({score:.4f}) | O-Score: {o_score:.4f}")

            # Filter Logic
            if label_idx == 0: continue # Skip O
            if score < 0.1: continue    # Threshold (ปรับตามใจชอบ)
            
            # Simple NMS (ตัดตัวซ้ำ)
            is_overlap = False
            for exist in found_entities:
                e_s, e_e = exist['span']
                if not (end < e_s or start > e_e): # Overlap?
                    if score > exist['score']: # ถ้าตัวใหม่มั่นใจกว่า
                        exist['active'] = False # ปิดตัวเก่า
                    else:
                        is_overlap = True
            
            if not is_overlap:
                found_entities.append({
                    "idx": idx, "span": (start, end), 
                    "text": span_text, "label": label_text, "score": score.item(), "active": True
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
            
            r_probs = torch.sigmoid(r_logits / tau)
            
            print("\n🔗 Relations found:")
            has_rel = False
            for k, (i, j) in enumerate(pair_map):
                # หา Relation ที่คะแนนสูงสุดในคู่นั้น
                val, r_idx = torch.max(r_probs[0][k], dim=-1)
                
                if val > 0.05: # Threshold สำหรับ Relation
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