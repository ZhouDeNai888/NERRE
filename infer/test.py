import torch
import torch.nn.functional as F
import json
import os
import re
import sys
# 1. Setup Paths
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from transformers import AutoTokenizer
from model.model import ZeroShotJointModel  # ไฟล์ model.py ต้องอยู่ในโฟลเดอร์เดียวกัน

class NERREPredictor:
    def __init__(self, model_dir):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. โหลดข้อมูล Labels จาก Config
        with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
            self.config = json.load(f)
        
        self.ent_labels = self.config["ent_labels"]
        self.rel_labels = self.config["rel_labels"]
        
        # 2. โหลด Tokenizer และโครงสร้างโมเดล
        self.tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        self.model = ZeroShotJointModel(model_name="xlm-roberta-base")
        
        # 3. โหลด Weights (pytorch_model.bin)
        model_path = os.path.join(model_dir, "best_model.bin")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        # 4. แคช Labels ไว้ในโมเดลเพื่อความเร็ว
        self._cache_labels()
        print(f"✅ โหลดโมเดลสำเร็จ: พร้อมใช้งาน {len(self.ent_labels)} Entities และ {len(self.rel_labels)} Relations")

    def _cache_labels(self):
        """เตรียมความหมายของ Label ไว้ล่วงหน้า"""
        def tokenize(label_list):
            tokens = self.tokenizer(label_list, return_tensors="pt", padding=True, truncation=True).to(self.device)
            return tokens["input_ids"].unsqueeze(0), tokens["attention_mask"].unsqueeze(0)

        ent_ids, ent_mask = tokenize(self.ent_labels)
        rel_ids, rel_mask = tokenize(self.rel_labels)
        self.model.set_global_labels(ent_ids, ent_mask, rel_ids, rel_mask)

    def _generate_spans(self, tokens, max_width=8): # เพิ่มความกว้างเป็น 8
        spans = []
        for i in range(len(tokens)):
            # กรองเอาเฉพาะ Token ที่ไม่ใช่ Special Tokens (<s>, </s>, <pad>)
            if tokens[i] in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            for width in range(1, max_width + 1):
                if i + width <= len(tokens):
                    spans.append((i, i + width - 1))
        return spans

    def predict(self, text, conf_threshold=0.1): # ลด threshold ตามที่คุณใช้
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        token_ids = inputs["input_ids"][0]
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        
        spans = self._generate_spans(tokens)
        with torch.no_grad():
            ent_logits, _ = self.model(inputs["input_ids"], inputs["attention_mask"], entity_spans=[spans])
        
        # --- [จุดแก้ที่ 1: Scaling] ---
        # ถ้าคะแนนเกาะกลุ่มที่ 0.12 ให้ลองคูณ Temperature หรือขยาย Scale ก่อน Softmax
        # หรือใช้ logits โดยตรงเพื่อดู Margin
        logits = ent_logits[0] # [num_spans, num_labels]
        
        # 🔥 จุดเปลี่ยนสำคัญ: ใช้ Temperature Scaling เพื่อถ่างคะแนน
        # ลองปรับจาก 0.07 ไปจนถึง 0.01 ถ้าคะแนนยังเกาะกลุ่ม
        tau = 0.03 
        probs = torch.softmax(logits / tau, dim=-1)
        
        max_probs, pred_ids = torch.max(probs, dim=-1)
        
        extracted_entities = []
        for i, (span, p_id, conf) in enumerate(zip(spans, pred_ids, max_probs)):
            label = self.ent_labels[p_id]
            
            # Debug: ดูอันดับที่ 2 สำหรับ Elon Musk
            span_text = self.tokenizer.decode(token_ids[span[0] : span[1] + 1]).strip()
            if "Elon" in span_text and label == "company":
                top2_val, top2_idx = torch.topk(probs[i], 2)
                # print(f"DEBUG: {span_text} -> Top1: {self.ent_labels[top2_idx[0]]} ({top2_val[0]:.4f}), Top2: {self.ent_labels[top2_idx[1]]} ({top2_val[1]:.4f})")

            if label != "O" and conf > conf_threshold:
                extracted_entities.append({
                    "text": span_text,
                    "type": label,
                    "conf": conf.item(),
                    "span": span
                })

        # --- ใช้ NMS เพื่อกำจัด Span ซ้อน (เช่น He is a famous engineer) ---
        final_entities = self._apply_nms(extracted_entities)
        return final_entities

    def _apply_nms(self, entities):
        """เลือกเฉพาะ Span ที่สั้นที่สุด (สำหรับ NER) หรือที่มี Conf สูงสุด"""
        entities.sort(key=lambda x: x['conf'], reverse=True)
        final = []
        occupied = set()
        for ent in entities:
            s, e = ent['span']
            indices = set(range(s, e + 1))
            if not (indices & occupied):
                final.append(ent)
                occupied.update(indices)
        return sorted(final, key=lambda x: x['span'][0])

# --- การใช้งาน ---
if __name__ == "__main__":
    MODEL_DIR = "saved_model_v16"  # ระบุโฟลเดอร์ของคุณ
    predictor = NERREPredictor(MODEL_DIR)
    
    raw_text = "Elon Musk founded SpaceX in 2002. He is a famous engineer from USA."
    
    ents = predictor.predict(raw_text)
    
    print(f"\n📝 ข้อความ: {raw_text}")
    print("\n🔍 ตรวจพบ Entities:")
    for e in ents:
        print(f"   - {e['text']} ({e['type']}) [Conf: {e['conf']}]")
        
    # print("\n🔗 ตรวจพบ Relations:")
    # for r in rels:
    #     print(f"   - {r['subject']} --[{r['relation']}]--> {r['object']} (Conf: {r['conf']})")