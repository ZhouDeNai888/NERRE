import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence


class MLPProjector(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim), # เพิ่ม LayerNorm
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim) # เพิ่ม LayerNorm
        )
    
    def forward(self, x):
        return self.layers(x)


class AttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states, mask):
        # hidden_states: [Batch, Seq, Hidden]
        attn_logits = self.attention(hidden_states).squeeze(-1) # [Batch, Seq]
        # attn_logits = attn_logits.masked_fill(mask == 0, -1e9)
        attn_logits = attn_logits.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=-1).unsqueeze(-1) # [Batch, Seq, 1]
        
        pooled_output = torch.sum(hidden_states * attn_weights, dim=1)
        return pooled_output

class ZeroShotJointModel(nn.Module):
    def __init__(self, model_name="xlm-roberta-base", hidden_size=768, dropout=0.1, use_mean_pooling=True):
        super(ZeroShotJointModel, self).__init__()
        
        # 1. Shared Encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 2. Projection Heads (MLP instead of single Linear for better zero-shot)
        self.entity_proj = MLPProjector(hidden_size * 3, hidden_size, dropout)
        self.relation_proj = MLPProjector(hidden_size * 4, hidden_size, dropout)
        
        # 3. Label Projection (separate from span projection)
        self.label_proj = MLPProjector(hidden_size, hidden_size, dropout)
        
        self.hidden_size = hidden_size
        self.use_mean_pooling = use_mean_pooling  # Better for descriptions
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for scaling (optional)
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))
        
        # Cache
        self.cached_ent_embeds = None
        self.cached_rel_embeds = None

        self.attn_pool = AttentionPooling(hidden_size)

    @property
    def temperature(self):
        """Clamp temperature to reasonable range"""
        return torch.clamp(self.log_temperature.exp(), min=0.01, max=1.0)

    def set_global_labels(self, ent_label_ids, ent_label_mask, rel_label_ids, rel_label_mask):
        """Cache labels for inference"""
        with torch.no_grad():
            self.cached_ent_embeds = self.encode_labels(ent_label_ids, ent_label_mask)
            self.cached_rel_embeds = self.encode_labels(rel_label_ids, rel_label_mask)
        print("✅ Labels cached for inference!")

    def _mean_pooling(self, hidden_states, attention_mask):
        """Mean pooling - better than CLS for semantic understanding"""
        # hidden_states: [Batch, Seq, Hidden]
        # attention_mask: [Batch, Seq]
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    # def _get_span_embeddings(self, sequence_output, starts, ends):
    #     """ดึง Vector ของช่วงคำ (Span) จากประโยค"""
    #     device = sequence_output.device
        
    #     span_embeddings = []
    #     for i in range(len(starts)):
    #         if len(starts[i]) == 0:
    #             span_embeddings.append(torch.zeros(0, self.hidden_size).to(device))
    #             continue

    #         s_indices = torch.tensor(starts[i], device=device)
    #         e_indices = torch.tensor(ends[i], device=device)
            
    #         s_vec = sequence_output[i, s_indices, :]
    #         e_vec = sequence_output[i, e_indices, :]
            
    #         # Mean Pooling: (Start + End) / 2
    #         span_embeddings.append(torch.cat([s_vec, e_vec], dim=-1))
            
    #     return pad_sequence(span_embeddings, batch_first=True, padding_value=0.0)



    def _get_span_embeddings(self, sequence_output, starts, ends):
        device = sequence_output.device
        span_embeddings = []
        
        for i in range(len(starts)):
            batch_vecs = []
            for s, e in zip(starts[i], ends[i]):
                s_vec = sequence_output[i, s, :]
                e_vec = sequence_output[i, e, :]
                
                # Max Pooling ภายใน Span (Internal Context)
                internal_vec = sequence_output[i, s:e+1, :].max(dim=0)[0]
                
                # รวมพลัง: Start + End + Max 
                # ขนาดจะกลายเป็น hidden_size * 3
                combined_vec = torch.cat([s_vec, e_vec, internal_vec], dim=-1)
                batch_vecs.append(combined_vec)
                
            if len(batch_vecs) == 0:
                span_embeddings.append(torch.zeros(0, self.hidden_size * 3).to(device))
            else:
                span_embeddings.append(torch.stack(batch_vecs))
                
        return pad_sequence(span_embeddings, batch_first=True, padding_value=0.0)

    def encode_labels(self, label_input_ids, label_attention_mask):
        """Encode labels using mean pooling for better semantic understanding"""
        batch_size, num_labels, seq_len = label_input_ids.shape
        
        flat_input_ids = label_input_ids.view(-1, seq_len)
        flat_mask = label_attention_mask.view(-1, seq_len)
        
        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=flat_mask)
        
        if self.use_mean_pooling:
            # Mean pooling - better for understanding descriptions
            # embeddings = self._mean_pooling(outputs.last_hidden_state, flat_mask)
            embeddings = self.attn_pool(outputs.last_hidden_state, flat_mask)
        else:
            # CLS token
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        # Project labels through label projection head
        embeddings = self.label_proj(embeddings)
        
        # L2 normalize for cosine similarity
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        
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
        if ent_label_ids is not None:
             ent_label_embeds = self.encode_labels(ent_label_ids, ent_label_mask)
             rel_label_embeds = self.encode_labels(rel_label_ids, rel_label_mask)
        else:
             if self.cached_ent_embeds is None:
                 raise ValueError("❌ กรุณาเรียก model.set_global_labels() ก่อนใช้งานแบบไม่ส่ง Labels")
             ent_label_embeds = self.cached_ent_embeds
             rel_label_embeds = self.cached_rel_embeds

        # --- C. Zero-shot Entity Recognition ---
        if entity_spans is None: return None, None
        
        span_starts = [[s[0] for s in batch_spans] for batch_spans in entity_spans]
        span_ends = [[s[1] for s in batch_spans] for batch_spans in entity_spans]
        span_embeds = self._get_span_embeddings(text_sequence, span_starts, span_ends)
        
        # Project and normalize span embeddings
        span_embeds = self.entity_proj(span_embeds)
        span_embeds = F.normalize(span_embeds, p=2, dim=-1)  # L2 normalize
        
        # Cosine similarity (since both are normalized)
        # Optionally scale by learnable temperature
        entity_logits = torch.matmul(span_embeds, ent_label_embeds.transpose(-2, -1))
        entity_logits = entity_logits / self.temperature  # Optional: use learnable temp
        
        # --- D. Zero-shot Relation Extraction ---
        relation_logits = None

        if relation_pairs is None: return entity_logits, None
        
        max_pairs = 0
        for pairs in relation_pairs:
            max_pairs = max(max_pairs, len(pairs))
            
        if max_pairs > 0:
            batch_size = len(relation_pairs)
            num_rel_labels = rel_label_embeds.shape[1]
            device = span_embeds.device
            
            relation_logits = torch.zeros(batch_size, max_pairs, num_rel_labels).to(device)
            
            # [NEW] ดึงความมั่นใจของประเภท Entity มาใช้เป็นฟีเจอร์
            # ใช้ entity_logits ที่คำนวณมาแล้ว เพื่อหาว่าแต่ละ Span น่าจะเป็น Class ไหน
            # entity_logits: [Batch, Num_Spans, Num_Ent_Labels]
            best_entity_labels = torch.argmax(entity_logits, dim=-1) # [Batch, Num_Spans]

            for b, pairs in enumerate(relation_pairs):
                if len(pairs) == 0: continue
                
                curr_span_embeds = span_embeds[b] # [Num_Spans, Hidden]
                
                # ดึง Embedding ของประเภท Entity ที่โมเดลทายได้ใน Batch นั้นๆ
                # ent_label_embeds[b]: [Num_Ent_Labels, Hidden]
                # ent_type_feats: [Num_Spans, Hidden]
                curr_ent_label_embeds = ent_label_embeds[b]
                ent_type_feats = curr_ent_label_embeds[best_entity_labels[b]]

                pair_vecs = []
                for subj_idx, obj_idx in pairs:
                    s_vec = curr_span_embeds[subj_idx]
                    o_vec = curr_span_embeds[obj_idx]
                    
                    # [NEW] ดึงประเภทของ Subject และ Object มาผสม
                    s_type = ent_type_feats[subj_idx]
                    o_type = ent_type_feats[obj_idx]
                    
                    # รวมร่างฟีเจอร์: [S_vec, O_vec, S_type, O_type] 
                    # รวม 4 ตัว -> input_dim ต้องเป็น hidden_size * 4
                    combined_rel_feat = torch.cat([s_vec, o_vec, s_type, o_type], dim=-1)
                    pair_vecs.append(combined_rel_feat)
                
                pair_batch_tensor = torch.stack(pair_vecs) # [Num_Pairs, Hidden*4]
                
                # Project และ Normalize ตามสูตร Zero-shot
                pair_proj = self.relation_proj(pair_batch_tensor) # [Num_Pairs, Hidden]
                pair_proj = F.normalize(pair_proj, p=2, dim=-1)
                
                # ดึง Relation Label Embeddings สำหรับ Batch นี้
                if rel_label_embeds.shape[0] == 1:
                     curr_rel_embeds = rel_label_embeds[0]
                else:
                     curr_rel_embeds = rel_label_embeds[b]

                # คำนวณ Similarity (Dot Product / Temperature)
                logits = torch.matmul(pair_proj, curr_rel_embeds.t())
                logits = logits / self.temperature
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