import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
class ZeroShotJointModel(nn.Module):
    def __init__(self, model_name, hidden_size=768):
        super(ZeroShotJointModel, self).__init__()
        
        # 1. Shared Encoder (ใช้ตัวเดียวกันทั้ง Text และ Labels เพื่อประหยัดเมม A100)
        # แนะนำ: 'xlm-roberta-large' หรือ 'microsoft/deberta-v3-large' สำหรับ A100
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 2. Projection Layers
        # ปรับขนาด Vector ของ Entity Span ให้เหมาะกับการเทียบกับ Label
        self.entity_proj = nn.Linear(hidden_size, hidden_size)
        
        # ปรับขนาด Vector ของ Relation (Subj + Obj) ให้เหลือเท่ากับ Label Vector
        # Input size * 2 เพราะเราเอา Subject กับ Object มาต่อกัน
        self.relation_proj = nn.Linear(hidden_size * 2, hidden_size)
        
        self.hidden_size = hidden_size
        
        # เก็บ Embeddings ของ Labels ไว้ใช้ตอน Inference (ไม่ต้องคำนวณใหม่ทุกครั้ง)
        self.cached_ent_embeds = None
        self.cached_rel_embeds = None

    def set_global_labels(self, ent_label_ids, ent_label_mask, rel_label_ids, rel_label_mask):
        """เรียกใช้ครั้งเดียวตอน Inference เพื่อจำ Labels ไว้"""
        with torch.no_grad():
            self.cached_ent_embeds = self.encode_labels(ent_label_ids, ent_label_mask)
            self.cached_rel_embeds = self.encode_labels(rel_label_ids, rel_label_mask)
        print("✅ Labels cached for inference!")

    def _get_span_embeddings(self, sequence_output, starts, ends):
        """ดึง Vector ของช่วงคำ (Span) จากประโยค"""
        device = sequence_output.device
        
        span_embeddings = []
        for i in range(len(starts)): # Loop per batch sample
            # ถ้า Sample นี้ไม่มี Entity เลย ให้ใส่ Tensor ว่างๆ
            if len(starts[i]) == 0:
                # สร้าง empty tensor [0, Hidden]
                span_embeddings.append(torch.zeros(0, self.hidden_size).to(device))
                continue

            # ดึง Vector หัวคำ (Start) และ ท้ายคำ (End)
            # ใช้ Fancy Indexing ดึงทีเดียวทั้งประโยค (เร็วกว่า loop)
            # starts[i] เป็น List เช่น [1, 5] -> tensor([1, 5])
            s_indices = torch.tensor(starts[i], device=device)
            e_indices = torch.tensor(ends[i], device=device)
            
            s_vec = sequence_output[i, s_indices, :] # [Num_Spans, Hidden]
            e_vec = sequence_output[i, e_indices, :]
            
            # Mean Pooling: (Start + End) / 2
            span_embeddings.append((s_vec + e_vec) / 2)
            
        # [FIX] ใช้ pad_sequence แทน stack
        # batch_first=True จะได้ shape: [Batch, Max_Num_Spans, Hidden]
        return pad_sequence(span_embeddings, batch_first=True, padding_value=0.0)

    def encode_labels(self, label_input_ids, label_attention_mask):
        # Input shape: [Batch, Num_Labels, Seq_Len]
        batch_size, num_labels, seq_len = label_input_ids.shape
        
        # 1. Flatten: รวม Batch กับ Num_Labels เข้าด้วยกัน
        # เป็น [Batch * Num_Labels, Seq_Len]
        flat_input_ids = label_input_ids.view(-1, seq_len)
        flat_mask = label_attention_mask.view(-1, seq_len)
        
        # 2. Pass through Encoder
        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=flat_mask)
        embeddings = outputs.last_hidden_state[:, 0, :] # [Batch * Num_Labels, Hidden]
        
        # 3. Reshape กลับมาเป็น 3 มิติ
        # [Batch, Num_Labels, Hidden]
        return embeddings.view(batch_size, num_labels, -1)

    def forward(self, 
                text_input_ids, text_mask,           
                ent_label_ids=None, ent_label_mask=None,       
                rel_label_ids=None, rel_label_mask=None,       
                entity_spans=None,                        
                relation_pairs=None                       
               ):
        
        # --- A. Encode Text ---
        text_outputs = self.encoder(input_ids=text_input_ids, attention_mask=text_mask)
        text_sequence = text_outputs.last_hidden_state
        
        # --- B. Encode Labels ---
        # ถ้ามีส่งมา (Training) ให้ใช้ที่ส่งมา
        # ถ้าไม่มี (Inference) ให้ใช้ที่ Cache ไว้
        if ent_label_ids is not None:
             ent_label_embeds = self.encode_labels(ent_label_ids, ent_label_mask)
             rel_label_embeds = self.encode_labels(rel_label_ids, rel_label_mask)
        else:
             if self.cached_ent_embeds is None:
                 raise ValueError("❌ กรุณาเรียก model.set_global_labels() ก่อนใช้งานแบบไม่ส่ง Labels")
             ent_label_embeds = self.cached_ent_embeds
             rel_label_embeds = self.cached_rel_embeds

        # --- C. Zero-shot Entity Recognition ---
        if entity_spans is None: return None, None # ถ้าไม่มี Spans ก็จบเลย
        # ดึง Spans (Logic เดิมที่ถูกต้องแล้ว)
        span_starts = [[s[0] for s in batch_spans] for batch_spans in entity_spans]
        span_ends = [[s[1] for s in batch_spans] for batch_spans in entity_spans]
        span_embeds = self._get_span_embeddings(text_sequence, span_starts, span_ends)
        
        # Project
        span_embeds = self.entity_proj(span_embeds) # [Batch, Num_Spans, Hidden]
        
        # Calculate Entity Logits
        # Note: Handle Broadcasting for Cached Embeds (Batch Size = 1)
        # Matmul: [Batch, Spans, Hidden] x [..., Hidden, Labels]
        entity_logits = torch.matmul(span_embeds, ent_label_embeds.transpose(-2, -1))
        
        # --- D. Zero-shot Relation Extraction ---
        relation_logits = None

        if relation_pairs is None: return entity_logits, None
        
        # 1. หาจำนวนคู่สูงสุดใน Batch นี้ เพื่อสร้าง Tensor มารอรับ
        max_pairs = 0
        for pairs in relation_pairs:
            max_pairs = max(max_pairs, len(pairs))
            
        if max_pairs > 0:
            batch_size = len(relation_pairs)
            num_rel_labels = rel_label_embeds.shape[1]
            device = span_embeds.device
            
            # สร้าง Tensor ว่างๆ รอไว้ (Filled with -inf or large negative is safer for masking, but 0 is ok if handled in loss)
            # ใช้ 0.0 ไปก่อน เพราะใน train.py เราเลือก index มาคำนวณเฉพาะตัวที่มีค่า
            relation_logits = torch.zeros(batch_size, max_pairs, num_rel_labels).to(device)
            
            # 2. วนลูปทีละ Sample ใน Batch (b)
            for b, pairs in enumerate(relation_pairs):
                if len(pairs) == 0:
                    continue # ข้ามถ้าไม่มีคู่ความสัมพันธ์
                
                # pairs คือ list ของ tuple เช่น [(0, 1), (2, 4)]
                # เราต้องดึง Vector ของคู่ (Subject + Object) ออกมา
                
                # ดึง Span Embedding ของ Sample นี้ออกมา
                curr_span_embeds = span_embeds[b] # [Num_Spans, Hidden]
                
                pair_vecs = []
                for subj_idx, obj_idx in pairs:
                    s_vec = curr_span_embeds[subj_idx]
                    o_vec = curr_span_embeds[obj_idx]
                    # ต่อกันเป็น [Hidden * 2]
                    pair_vecs.append(torch.cat([s_vec, o_vec], dim=-1))
                
                # Stack รวมเป็น [Num_Pairs_in_Sample, Hidden*2]
                pair_batch_tensor = torch.stack(pair_vecs)
                
                # Project ลงมาเหลือ [Num_Pairs_in_Sample, Hidden]
                pair_proj = self.relation_proj(pair_batch_tensor)
                
                # Dot Product กับ Label ของ Sample นั้น
                # rel_label_embeds[b] shape is [Num_Rel_Labels, Hidden]
                
                if rel_label_embeds.shape[0] == 1: # กรณีใช้ Cached Labels (มีแค่ 1 ชุด)
                     curr_rel_embeds = rel_label_embeds[0]
                else: # กรณี Training (มี Labels แยกตาม Batch)
                     curr_rel_embeds = rel_label_embeds[b]

                # ผลลัพธ์ logits shape: [Num_Pairs_in_Sample, Num_Rel_Labels]
                logits = torch.matmul(pair_proj, curr_rel_embeds.t())
                
                # เอาไปใส่ใน Tensor ใหญ่ตามตำแหน่ง
                relation_logits[b, :len(pairs), :] = logits

        return entity_logits, relation_logits



if __name__ == "__main__":
    from transformers import AutoTokenizer
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_NAME = "xlm-roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = ZeroShotJointModel(MODEL_NAME).to(device)

    # 1. Prepare Text
    text = "Elon Musk founded SpaceX"
    text_inputs = tokenizer(text, return_tensors="pt").to(device)

    # [DEBUG] Print Tokens เพื่อดูตำแหน่งที่แท้จริง
    tokens = tokenizer.convert_ids_to_tokens(text_inputs["input_ids"][0])
    print(f"\n--- Token Mapping ---")
    for i, t in enumerate(tokens):
        print(f"Idx {i}: {t}")
    print("---------------------")

    # 2. Prepare Labels
    entity_labels = ["Person", "Organization", "Location"] 
    relation_labels = ["founded_by", "located_in", "enemy_of"]

    def tokenize_list(labels):
        tokenized = tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(device)
        return {
            "input_ids": tokenized["input_ids"].unsqueeze(0),       # [1, Num_Labels, Seq]
            "attention_mask": tokenized["attention_mask"].unsqueeze(0)
        }

    ent_inputs = tokenize_list(entity_labels)
    rel_inputs = tokenize_list(relation_labels)

    # 3. Spans & Pairs
    # จาก Token Mapping ที่ได้:
    # "Elon Musk" -> Idx 1(El), 2(on), 3(Musk) -> start=1, end=3
    # "SpaceX"    -> Idx 6(Space), 7(X)        -> start=6, end=7
    spans = [(1, 3), (6, 7)]  
    pairs = [(1, 0)]          # Index 1 (SpaceX) -> Index 0 (Elon Musk)

    # 4. Forward Pass
    with torch.no_grad():
        e_logits, r_logits = model(
            text_inputs["input_ids"], text_inputs["attention_mask"],
            ent_inputs["input_ids"], ent_inputs["attention_mask"],
            rel_inputs["input_ids"], rel_inputs["attention_mask"],
            [spans], # <--- ใส่ [] ครอบเพื่อบอกว่าเป็น Batch ที่ 1
            [pairs]  # <--- [FIX] ใส่ [] ครอบตรงนี้ด้วย!
        )

    print(f"Device: {device}")
    print("\nEntity Scores (Dot Product):")
    print(e_logits) 

    print("\nRelation Scores (Dot Product):")
    print(r_logits)

    print("\n--- Formatted Predictions ---")
    
    # 1. Decode Entities
    input_ids = text_inputs["input_ids"][0]
    e_preds = torch.argmax(e_logits, dim=-1)[0]
    
    id2text = {}
    print("Entities:")
    for i, (start, end) in enumerate(spans):
        label_idx = e_preds[i].item()
        label_text = entity_labels[label_idx]
        
        # Decode text from token IDs (start to end inclusive)
        span_token_ids = input_ids[start : end+1]
        span_text = tokenizer.decode(span_token_ids).strip()
        id2text[i] = span_text
        
        print(f"  - {span_text} : {label_text}")

    # 2. Decode Relations
    if r_logits is not None:
        print("\nRelations:")
        r_preds = torch.argmax(r_logits, dim=-1)[0]
        for i, (subj_idx, obj_idx) in enumerate(pairs):
            label_idx = r_preds[i].item()
            relation_text = relation_labels[label_idx]
            
            subj_text = id2text[subj_idx]
            obj_text = id2text[obj_idx]
            
            print(f"  - {subj_text} --[{relation_text}]--> {obj_text}")