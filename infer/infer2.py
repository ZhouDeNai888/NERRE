import os
import sys
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# 1. Setup Paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from model.model import ZeroShotJointModel

# --- CONFIG ---
device = "cuda:2" if torch.cuda.is_available() else "cpu"
model_path = "saved_model_v23" 
THRESHOLD_ENT = 0.25  # Lowered to catch entities like "Harry Potter"
THRESHOLD_REL = 0.05  # Lowered to see "weak" but valid relations
TEMP_OVERRIDE = 1.0   
DEBUG_MODE = True     # Show rejected candidates

print(f"⏳ Loading model from {model_path}...")

# 2. Load Config & Labels
with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
    config_data = json.load(f)

ent_labels = config_data["ent_labels"] 
rel_labels = config_data["rel_labels"]

print(f"✅ Loaded {len(ent_labels)} Entity types and {len(rel_labels)} Relation types.")

# 3. Load Model & Tokenizer
# [Updated] Removed fix_mistral_regex which might cause tokenizer glitches for XLM-R
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = ZeroShotJointModel(config_data["model_name"]).to(device)
model.load_state_dict(torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location=device))
model.eval()

# Show Calibration Status
print(f"🌡️  Learned Calibration Temperatures:")
if hasattr(model, 'ent_temperature'):
    print(f"   - Entity Temp:   {model.ent_temperature.item():.4f}")
if hasattr(model, 'rel_temperature'):
    print(f"   - Relation Temp: {model.rel_temperature.item():.4f}")

def predict(text):
    print(f"\n" + "="*60)
    print(f"📝 Text: {text}")
    print("="*60)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    
    # Helper: Encode Labels for Zero-shot
    def encode_list(labels):
        t = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(device)
        return t["input_ids"].unsqueeze(0), t["attention_mask"].unsqueeze(0)

    ent_input, ent_mask = encode_list(ent_labels)
    rel_input, rel_mask = encode_list(rel_labels)

    # 1. Generate Spans (Sliding Window)
    spans = []
    max_span_width = 5 # Reduced to 5 to enforce tighter boundaries (Boundary Precision)
    for w in range(1, max_span_width + 1):
        for i in range(seq_len - w + 1):
            spans.append((i, i + w - 1))
    
    if not spans: return

    with torch.no_grad():
        # -------------------------------------------------
        # STEP 1: Predict Entities
        # -------------------------------------------------
        e_logits, _ = model(
            input_ids, inputs["attention_mask"],
            ent_input, ent_mask,
            rel_input, rel_mask,
            entity_spans=[spans], relation_pairs=None
        )

        # 💡 ปรับปรุงส่วนการคำนวณ Probability
        # ใช้ Softmax ตามที่เทรนมาด้วย CrossEntropyLoss
        # ไม่ต้องปรับ Temperature เพิ่มเติมแล้ว เพราะโมเดลเรียนรู้ Scale ที่เหมาะสมมาแล้ว
        e_probs = torch.softmax(e_logits, dim=-1) # [1, num_spans, num_classes]
        
        found_entities = []
        
        # ดึงคะแนนคลาส O และคะแนน Entity อื่นๆ แยกกัน
        o_probs = e_probs[0, :, 0]
        entity_only_probs = e_probs[0, :, 1:] # ตัดคลาส 0 ออก
        
        best_ent_scores, best_ent_indices = torch.max(entity_only_probs, dim=-1)
        best_ent_indices += 1 # ปรับ index กลับคืน

        for idx in range(len(spans)):
            score = best_ent_scores[idx].item()
            label_idx = best_ent_indices[idx].item()
            
            # 🎯 หัวใจหลัก: แม้ O จะสูงกว่า แต่ถ้า Entity Score เกิน Threshold ให้ดึงออกมา
            if score > THRESHOLD_ENT:
                start, end = spans[idx]
                span_text = tokenizer.decode(input_ids[0][start:end+1]).strip()
                # ป้องกันการเก็บตัวอักษรขยะหรือรหัสควบคุม
                if span_text in ["<s>", "</s>", "<pad>"]: continue
                
                label_text = ent_labels[label_idx]
                
                # Simple NMS (Non-Maximum Suppression)
                is_overlap = False
                for exist in found_entities:
                    e_s, e_e = exist['span']
                    if not (end < e_s or start > e_e): # มีการทับซ้อนกัน
                        if score > exist['score']:
                            exist['active'] = False 
                        else:
                            is_overlap = True
                            break
                
                if not is_overlap:
                    found_entities.append({
                        "idx": idx, "span": (start, end), 
                        "text": span_text, "label": label_text, 
                        "score": score, "active": True
                    })
        
        found_entities = [e for e in found_entities if e['active']]

        print("📍 Entities found:")
        if not found_entities:
            print("  (None)")
        for ent in found_entities:
            print(f"  - [{ent['label']}] {ent['text']} (Conf: {ent['score']:.2f})")

        # -------------------------------------------------
        # STEP 2: Predict Relations
        # -------------------------------------------------
        if len(found_entities) > 1:
            pairs = []
            pair_map = [] 
            
            for i in range(len(found_entities)):
                for j in range(len(found_entities)):
                    if i == j: continue
                    pairs.append((found_entities[i]['idx'], found_entities[j]['idx']))
                    pair_map.append((i, j))
            
            _, r_logits = model(
                input_ids, inputs["attention_mask"],
                ent_input, ent_mask,
                rel_input, rel_mask,
                entity_spans=[spans], relation_pairs=[pairs]
            )
            
            # [FIXED] Use Softmax because specific relations are mutually exclusive (Single-Label)
            # Logits are already scaled by internal temperature in forward()
            r_probs = torch.softmax(r_logits, dim=-1) # [1, num_pairs, num_classes]
            
            print("\n🔗 Relations found:")
            has_rel = False
            # Loop for checking results
            for k, (i, j) in enumerate(pair_map):
                prob_vec = r_probs[0][k]
                val, r_idx = torch.max(prob_vec, dim=-1)
                
                subj = found_entities[i]['text']
                obj = found_entities[j]['text']
                pred_label = rel_labels[r_idx.item()]
                confidence = val.item()

                # Filter logic
                is_valid = (r_idx.item() > 0) and (confidence > THRESHOLD_REL)
                
                if is_valid:
                    has_rel = True
                    print(f"  - {subj} --[{pred_label}]--> {obj} (Conf: {confidence:.2f})")
                elif DEBUG_MODE:
                    # Show why it was rejected
                    if r_idx.item() == 0:
                        # If top guess is O, find the 2nd best (the suppressed relation)
                        # excluding index 0
                        top2_val, top2_idx = torch.topk(prob_vec[1:], k=1)
                        # mapping index back (since we sliced [1:])
                        real_top2_idx = top2_idx.item() + 1
                        real_top2_name = rel_labels[real_top2_idx]
                        
                        print(f"  [DEBUG] {subj} -> {obj} : Best was NO_RELation ({confidence:.2f}). "
                              f"Next best: {real_top2_name} ({top2_val.item():.2f})")
                    else:
                        print(f"  [DEBUG] {subj} -> {obj} : Rejected '{pred_label}' (Conf {confidence:.2f} < {THRESHOLD_REL})")

            
            if not has_rel:
                print("  (No relations found)")

# --- RUN TEST ---
test_sentences = [
    "Elon Musk founded SpaceX in 2002.",
    "ลิซ่า เป็นสมาชิกของวง BLACKPINK",
    "Harry Potter used his magic wand to fight Voldemort at Hogwarts.",
    "周德耐在慈濟大學讀書"
]

for sent in test_sentences:
    predict(sent)