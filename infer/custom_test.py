import torch
import json
from transformers import AutoTokenizer
import os
import sys
from typing import List

# 1. Setup Paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from model.model import ZeroShotJointModel 

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "saved_model_v16"

# Load Model & Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
with open(f"{model_path}/config.json", "r", encoding="utf-8") as f: 
    config = json.load(f)

model = ZeroShotJointModel(config["model_name"]).to(device)
model.load_state_dict(torch.load(f"{model_path}/best_model.bin", map_location=device))
model.eval()

def zero_shot_test(text: str, custom_ents: List[str], custom_rels: List[str], debug: bool = True):
    print(f"\n" + "="*70)
    print(f"📝 Testing: {text}")
    print(f"🏷️ Custom Entities: {custom_ents}")
    print(f"🔗 Custom Relations: {custom_rels}")
    print("="*70)
    
    inputs = tokenizer(text, return_tensors="pt").to(device)
    seq_len = inputs["input_ids"].shape[1]
    
    # 1. Encode Custom Labels
    def encode(labels):
        t = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(device)
        return t["input_ids"].unsqueeze(0), t["attention_mask"].unsqueeze(0)

    # ใส่ "O" ไว้ตัวแรกเสมอ และใช้ Label ที่คุณส่งมา
    ent_labels = ["O"] + custom_ents
    ent_in, ent_mask = encode(ent_labels)
    rel_in, rel_mask = encode(custom_rels)

    # 2. Generate Spans (Sliding Window)
    spans = []
    max_span_width = 8
    for w in range(1, max_span_width + 1):
        for i in range(seq_len - w + 1):
            spans.append((i, i + w - 1))

    found_entities = []

    with torch.no_grad():
        # --- STEP 1: Predict Entities ---
        e_logits, _ = model(
            inputs["input_ids"], inputs["attention_mask"],
            ent_in, ent_mask, rel_in, rel_mask,
            entity_spans=[spans], relation_pairs=None
        )
        
        # ดึงค่า Temperature (ถ้าเทรนมาดี ค่านี้จะช่วยบีบ Probability)
        tau = model.temperature.item()
        e_probs = torch.sigmoid(e_logits / tau)
        # --- [DEBUG SECTION] ---
        if debug:
            print(f"\n🔍 [DEBUG SCAN] Top Span Candidates (Sorted by Entity Confidence):")
            print(f"{'Text Content':<25} | {'Predicted':<12} | {'Prob':<8} | {'O-Score':<8} | {'Best Alt'}")
            print("-" * 80)
            
            debug_list = []
            for idx, span_probs in enumerate(e_probs[0]):
                max_prob, label_idx = torch.max(span_probs, dim=-1)
                o_score = span_probs[0].item()
                
                # หาคลาส Entity ที่ไม่ใช่ "O" ที่มีคะแนนสูงสุด
                ent_only_probs = span_probs[1:]
                best_ent_score, best_ent_idx = torch.max(ent_only_probs, dim=-1)
                best_ent_name = custom_ents[best_ent_idx.item()]
                
                start, end = spans[idx]
                span_text = tokenizer.decode(inputs["input_ids"][0][start:end+1]).strip()
                
                debug_list.append({
                    "text": span_text,
                    "label": ent_labels[label_idx],
                    "prob": max_prob.item(),
                    "o_score": o_score,
                    "best_ent": f"{best_ent_name} ({best_ent_score.item():.4f})"
                })
            
            # เรียงลำดับโชว์เฉพาะตัวที่ Best Entity Score สูงสุด 10 อันดับแรก
            debug_list.sort(key=lambda x: float(x['best_ent'].split('(')[1][:-1]), reverse=True)
            for d in debug_list[:10]:
                print(f"{d['text'][:25]:<25} | {d['label']:<12} | {d['prob']:.4f} | {d['o_score']:.4f} | {d['best_ent']}")
        # max_ent_score, max_ent_idx = torch.max(e_probs[1:], dim=-1)
        
        # หาตัวที่คะแนนสูงสุดในแต่ละ Span
        scores, preds = torch.max(e_probs, dim=-1)

        print(f"🌡️ Temperature: {tau:.4f} | Scanning spans...")

        for idx, (score, label_idx) in enumerate(zip(scores[0], preds[0])):
            # ในงาน Zero-shot คะแนนอาจไม่สูงเท่าที่เทรนมา 
            # หากยังไม่เจอ ให้ลองลด 0.05 เหลือ 0.01
            if label_idx == 0 or score < 0.05: 
                continue 
            
            start, end = spans[idx]
            entity_text = tokenizer.decode(inputs["input_ids"][0][start:end+1], skip_special_tokens=True).strip()
            label_text = ent_labels[label_idx.item()]

            # --- System NMS: เลือก Span ที่ดีที่สุด ---
            is_overlap = False
            current_entity = {
                "idx": idx, "span": (start, end), "text": entity_text,
                "label": label_text, "score": score.item(), "active": True
            }
            
            for exist in found_entities:
                if not exist['active']: continue
                e_s, e_e = exist['span']
                # Check Overlap
                if not (end < e_s or start > e_e):
                    is_overlap = True
                    if score.item() > exist['score']:
                        exist['active'] = False
                        is_overlap = False # ตัวใหม่ชนะ ให้ข้ามไปเพิ่มในลิสต์
                    else:
                        break # ตัวเก่าชนะ ไม่ต้องทำอะไรเพิ่ม
            
            if not is_overlap:
                found_entities.append(current_entity)

        # กรองเฉพาะตัวที่ Active
        final_entities = [e for e in found_entities if e['active']]
        
        print(f"\n📍 Found Entities:")
        if not final_entities: 
            print("   (None) - Try lowering the threshold or checking if 'O' class is too dominant.")
        for ent in final_entities:
            print(f"   - [{ent['label']}] '{ent['text']}' (Score: {ent['score']:.4f})")

        # --- STEP 2: Predict Relations (Zero-shot) ---
        if len(final_entities) > 1:
            pairs, pair_map = [], []
            for i in range(len(final_entities)):
                for j in range(len(final_entities)):
                    if i == j: continue
                    pairs.append((final_entities[i]['idx'], final_entities[j]['idx']))
                    pair_map.append((i, j))
            
            # Predict Relations โดยใช้ pairs จาก Entities ที่เจอ
            _, r_logits = model(
                inputs["input_ids"], inputs["attention_mask"],
                ent_in, ent_mask, rel_in, rel_mask,
                entity_spans=[spans], relation_pairs=[pairs]
            )
            
            r_probs = torch.sigmoid(r_logits / tau)
            print(f"\n🔗 Found Relations:")
            has_rel = False
            
            for k, (i, j) in enumerate(pair_map):
                val, r_idx = torch.max(r_probs[0][k], dim=-1)
                if val > 0.02: # Relation threshold ในงาน Zero-shot มักจะต่ำ
                    has_rel = True
                    h = final_entities[i]
                    t = final_entities[j]
                    print(f"   - {h['text']} --[{custom_rels[r_idx]}]--> {t['text']} (Score: {val:.4f})")
            
            if not has_rel: 
                print("   (No relations found)")

# --- EXECUTE TEST ---
zero_shot_test(
    "Hermione is a brilliant witch from Gryffindor.",
    custom_ents=["witch", "wizard", "house"],
    custom_rels=["belongs_to", "member_of"]
)

zero_shot_test(
    "ลิซ่า เป็นสมาชิกของวง BLACKPINK",
    custom_ents=["person", "girl_group", "organisation"], 
    custom_rels=["member_of", "part_of"]
)