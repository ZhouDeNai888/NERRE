import torch
import json
import os
import sys

# Add parent directory to path to allow importing from model/
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from transformers import AutoTokenizer
from model.model import ZeroShotJointModel

class NERREInference:
    def __init__(self, model_dir, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_dir} to {self.device}...")
        
        # 1. Load Config
        with open(f"{model_dir}/config.json", "r", encoding='utf-8') as f:
            self.config = json.load(f)
            
        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 3. Load Model
        self.model = ZeroShotJointModel(self.config['model_name'])
        self.model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin", map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Pre-compute Label Embeddings (Set Global)
        # นี่คือส่วนที่ทำให้เราไม่ต้องส่ง Labels บ่อยๆ
        self._set_labels(self.config['ent_labels'], self.config['rel_labels'])
        
    def _set_labels(self, ent_labels, rel_labels):
        print("Caching labels...")
        self.ent_labels = ent_labels
        self.rel_labels = rel_labels
        
        def tokenize_list(labels):
            tokenized = self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(self.device)
            return tokenized

        ent_inputs = tokenize_list(ent_labels)
        rel_inputs = tokenize_list(rel_labels)
        
        # Add batch dimension [1, Num_Labels, Seq_Len] to match model expectation
        self.model.set_global_labels(
            ent_inputs["input_ids"].unsqueeze(0), ent_inputs["attention_mask"].unsqueeze(0),
            rel_inputs["input_ids"].unsqueeze(0), rel_inputs["attention_mask"].unsqueeze(0)
        )

    def _nms(self, entities):
        """Non-Maximum Suppression to remove overlapping entities"""
        if not entities: return []
        
        # Sort by score descending (high confidence first)
        sorted_ents = sorted(entities, key=lambda x: x['score'], reverse=True)
        keep = []
        
        while sorted_ents:
            # Pick the best entity
            current = sorted_ents.pop(0)
            keep.append(current)
            
            # Remove entities that overlap with 'current'
            c_start, c_end = current['span_char']
            
            non_overlapping = []
            for ent in sorted_ents:
                e_start, e_end = ent['span_char']
                
                # Check overlap: (StartA < EndB) and (EndA > StartB)
                # Note: span_char is usually inclusive or slightly distinct depending on implementation
                # but standard intersection logic is: max(s1, s2) < min(e1, e2)
                # In python slice terms: text[s:e], so s is inclusive, e is exclusive.
                # Here we used text[char_start:char_end] so it is [s, e).
                
                if max(c_start, e_start) < min(c_end, e_end):
                    continue # Discard overlapping entity
                
                non_overlapping.append(ent)
            
            sorted_ents = non_overlapping
            
        return sorted(keep, key=lambda x: x['span_char'][0])

    def predict(self, text, threshold=0.5, max_span_width=8):
        # 1. Tokenize Text
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        input_ids = inputs["input_ids"]
        offset_mapping = inputs["offset_mapping"][0] # เพื่อแปลง token idx กลับเป็น char idx
        
        seq_len = input_ids.shape[1]
        
        # 2. Auto-Generate Candidate Spans (Sliding Window)
        # สร้าง Span ทุกรูปแบบที่เป็นไปได้ (ยาวไม่เกิน max_span_width)
        candidate_spans = []
        # เริ่มที่ 1 (ข้าม [CLS]) จบก่อนตัวสุดท้าย (ข้าม [SEP]) 
        # *หมายเหตุ: ต้องเช็คดีๆ ว่า model tokenizer ใช้ special token แบบไหน (RoBERTa ใช้ <s>, </s>)
        # RoBERTa: <s>...text...</s> -> indices 1 to len-2
        
        for width in range(1, max_span_width + 1):
            for i in range(1, seq_len - width): 
                # i คือ start index, i+width-1 คือ end index
                candidate_spans.append((i, i + width)) # format ของ model คือ use (start, end) แบบ inclusive? 
                # เช็ค model.py -> _get_span_embeddings ใช้ s_indices และ e_indices
                # ดังนั้น (start, end) ควรเป็น index ของ token ตัวแรก และ token ตัวสุดท้าย
                # ซึ่ง logic ผมข้างบน i คือ start, i+width-1 เป็น end (inclusive)
                # ดังนั้น span คู่คือ (i, i + width - 1)
                
        # แก้ไข format ให้ตรงกับ model: inclusive (start, end)
        model_spans = [(s, s+w-1) for w in range(1, max_span_width+1) for s in range(1, seq_len - w)]
        
        if not model_spans:
            return {"entities": [], "relations": []}

        # 3. Model Inference (Entities)
        with torch.no_grad():
            # ส่งแค่ Spans, ไม่ต้องส่ง Labels (ใช้ cache)
            e_logits, _ = self.model(
                input_ids, inputs["attention_mask"],
                entity_spans=[model_spans] # Batch size 1
            )
            
        # 4. Filter Entities (Thresholding)
        # e_logits: [1, Num_Spans, Num_Ent_Labels]
        probs = torch.sigmoid(e_logits[0]) # Convert to probability
        
        found_entities = []
        
        for idx, (start, end) in enumerate(model_spans):
            scores = probs[idx]
            max_score, label_idx = torch.max(scores, dim=0)
            
            if max_score > threshold:
                # แปลง token index กลับเป็น string จริง
                char_start = offset_mapping[start][0].item()
                char_end = offset_mapping[end][1].item()
                span_text = text[char_start:char_end]
                
                ent_obj = {
                    "id": len(found_entities), # Temporary ID
                    "text": span_text,
                    "label": self.ent_labels[label_idx],
                    "score": max_score.item(),
                    "span_token": (start, end),
                    "span_char": (char_start, char_end)
                }
                found_entities.append(ent_obj)
        
        # Apply NMS (Non-Maximum Suppression)
        found_entities = self._nms(found_entities)

        # 5. Model Inference (Relations)
        # สร้าง Pairs จาก Entities ที่หาเจอเท่านั้น
        if len(found_entities) < 2:
            return {"entities": found_entities, "relations": []}
            
        pairs = []
        pair_mapping = [] # เก็บ mapping ว่า pair นี้มาจาก entity id ไหน (subj_id, obj_id)
        
        for i in range(len(found_entities)):
            for j in range(len(found_entities)):
                if i == j: continue
                # สร้างคู่ (Subj_Token_Idx, Obj_Token_Idx) ตาม format ของ Model
                subj = found_entities[i]
                obj = found_entities[j]
                
                # Model ต้องการ index ใน list ของ spans 
                # แต่เดี๋ยวก่อน! Model ต้องการ index ของ span ใน embedding list
                # ซึ่งเราต้องส่ง spans ของ found_entities เข้าไปใหม่ เพื่อความแม่นยำและประหยัด
                pass

        # เพื่อความง่ายและถูกต้อง: 
        # เราควรเรียก model อีกรอบ โดยส่งเฉพาะ Spans ของ Entities ที่ "คัดเลือกแล้ว"
        # จะได้ไม่ต้องจัดการ index mapping ให้ปวดหัว
        
        final_spans = [e["span_token"] for e in found_entities]
        
        # สร้าง Pair indices (0,1), (0,2), (1,0)... relative to final_spans list
        final_pairs = []
        for i in range(len(final_spans)):
            for j in range(len(final_spans)):
                if i != j:
                    final_pairs.append((i, j))
        
        with torch.no_grad():
            _, r_logits = self.model(
                input_ids, inputs["attention_mask"],
                entity_spans=[final_spans], 
                relation_pairs=[final_pairs]
            )
            
        # 6. Filter Relations
        found_relations = []
        if r_logits is not None:
            r_probs = torch.sigmoid(r_logits[0])
            
            for idx, (subj_idx, obj_idx) in enumerate(final_pairs):
                scores = r_probs[idx]
                max_score, label_idx = torch.max(scores, dim=0)
                
                if max_score > threshold:
                    rel_obj = {
                        "subject": found_entities[subj_idx]["text"],
                        "object": found_entities[obj_idx]["text"],
                        "relation": self.rel_labels[label_idx],
                        "score": max_score.item()
                    }
                    found_relations.append(rel_obj)
                    
        return {"entities": found_entities, "relations": found_relations}

# --- Example Usage ---
if __name__ == "__main__":
    # จำลองการใช้งาน
    inference = NERREInference("saved_model_v1")
    
    text = "Elon Musk founded SpaceX in 2002."
    result = inference.predict(text, threshold=0.5)
    
    print("\n--- Formatted Predictions ---")
    print("Entities:")
    for ent in result["entities"]:
        print(f"  - {ent['text']} : {ent['label']}") # (Score: {ent['score']:.4f})
        
    print("\nRelations:")
    for rel in result["relations"]:
        print(f"  - {rel['subject']} --[{rel['relation']}]--> {rel['object']}") # (Score: {rel['score']:.4f})
