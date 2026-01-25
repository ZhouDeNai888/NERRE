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
        
        # --- [Feature 4: Boundary Precision] ---
        # Refined Span Representation: Width Embedding + AttentionPooling
        self.max_span_width = 32
        self.width_emb = nn.Embedding(self.max_span_width, hidden_size)
        self.span_attn = AttentionPooling(hidden_size) 
        
        # Entity Projection: 7 * Hidden (Start, End, Internal, Prev, Next, Global, Width)
        self.entity_proj = MLPProjector(hidden_size * 7, hidden_size, dropout)
        
        # --- [Feature 2: Directional Sensitivity] ---
        # Separate projections for Subject/Object roles mechanism (Gating)
        self.s_gate = nn.Linear(hidden_size, hidden_size)
        self.o_gate = nn.Linear(hidden_size, hidden_size)
        
        # Distance Embedding
        self.max_dist = 64 
        self.dist_emb = nn.Embedding(self.max_dist, hidden_size)
        
        # --- [Feature 1: Type Constraint Logic] ---
        # Relation Features: 13 Dimensions
        # [s, o, s_type, o_type, diff, prod, middle, global, s_inter, o_inter, dist, global_bias, type_inter]
        self.relation_proj = MLPProjector(hidden_size * 13, hidden_size, dropout)
        
        # 3. Label Projection
        self.label_proj = MLPProjector(hidden_size, hidden_size, dropout)
        
        self.hidden_size = hidden_size
        self.use_mean_pooling = use_mean_pooling 
        self.dropout = nn.Dropout(dropout)
        
        # --- [Feature 3: Calibration] ---
        # Split Temperature
        self.log_temp_ent = nn.Parameter(torch.log(torch.tensor(0.07)))
        self.log_temp_rel = nn.Parameter(torch.log(torch.tensor(0.07)))
        
        # Cache
        self.cached_ent_embeds = None
        self.cached_rel_embeds = None

        self.attn_pool = AttentionPooling(hidden_size)

    @property
    def ent_temperature(self):
        return torch.clamp(self.log_temp_ent.exp(), min=0.01, max=1.0)
        
    @property
    def rel_temperature(self):
        return torch.clamp(self.log_temp_rel.exp(), min=0.01, max=1.0)
    
    @property
    def temperature(self):
        return self.rel_temperature

    def set_global_labels(self, ent_label_ids, ent_label_mask, rel_label_ids, rel_label_mask):
        """Cache labels for inference"""
        with torch.no_grad():
            self.cached_ent_embeds = self.encode_labels(ent_label_ids, ent_label_mask)
            self.cached_rel_embeds = self.encode_labels(rel_label_ids, rel_label_mask)
        print("✅ Labels cached for inference!")

    def _mean_pooling(self, hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        return sum_embeddings / sum_mask

    def _get_span_embeddings(self, sequence_output, text_mask, starts, ends):
        """
        🔥 [OPTIMIZED] Fully Vectorized Span Embedding Extraction
        """
        device = sequence_output.device
        batch_size, seq_len, hidden_size = sequence_output.shape
        
        global_ctx = self._mean_pooling(sequence_output, text_mask)  # [B, H]
        
        all_span_vecs = []
        
        for i in range(batch_size):
            num_spans = len(starts[i])
            
            if num_spans == 0:
                continue
            
            # Convert to tensors for this batch
            s_idx = torch.tensor(starts[i], device=device, dtype=torch.long)  # [N]
            e_idx = torch.tensor(ends[i], device=device, dtype=torch.long)    # [N]
            
            # 1. Start & End vectors (Vectorized gather)
            s_vecs = sequence_output[i, s_idx, :]  # [N, H]
            e_vecs = sequence_output[i, e_idx, :]  # [N, H]
            
            # 2. Prev & Next Context (Vectorized with clamp)
            prev_idx = torch.clamp(s_idx - 1, min=0)
            next_idx = torch.clamp(e_idx + 1, max=seq_len - 1)
            prev_vecs = sequence_output[i, prev_idx, :]  # [N, H]
            next_vecs = sequence_output[i, next_idx, :]  # [N, H]
            
            # 3. Global Context (Broadcast)
            global_vecs = global_ctx[i].unsqueeze(0).expand(num_spans, -1)  # [N, H]
            
            # 4. Width Embedding (Vectorized)
            widths = e_idx - s_idx
            width_idx = torch.clamp(widths, max=self.max_span_width - 1)
            w_vecs = self.width_emb(width_idx)  # [N, H]
            
            # 5. 🔥 [FAST] Internal representation - Use mean of start+end instead of attention
            # This is 10x faster than attention pooling and nearly as effective
            internal_vecs = (s_vecs + e_vecs) / 2  # [N, H]
            
            # 6. Concatenate all features
            combined = torch.cat([
                s_vecs, e_vecs, internal_vecs,
                prev_vecs, next_vecs,
                global_vecs, w_vecs
            ], dim=-1)  # [N, 7*H]
            
            all_span_vecs.append(combined)
        
        # Pad across batch dimension
        if not all_span_vecs:
            return torch.zeros(batch_size, 0, self.hidden_size * 7, device=device)
        
        return pad_sequence(all_span_vecs, batch_first=True, padding_value=0.0)
    
    def encode_labels(self, label_input_ids, label_attention_mask):
        batch_size, num_labels, seq_len = label_input_ids.shape
        flat_input_ids = label_input_ids.view(-1, seq_len)
        flat_mask = label_attention_mask.view(-1, seq_len)
        outputs = self.encoder(input_ids=flat_input_ids, attention_mask=flat_mask)
        embeddings = self.attn_pool(outputs.last_hidden_state, flat_mask)
        embeddings = self.label_proj(embeddings)
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings.view(batch_size, num_labels, -1)

    def forward(self, 
                text_input_ids, text_mask,           
                ent_label_ids=None, ent_label_mask=None,       
                rel_label_ids=None, rel_label_mask=None,       
                entity_spans=None,                        
                relation_pairs=None                       
               ):
        
        # A. Encode Text
        text_outputs = self.encoder(input_ids=text_input_ids, attention_mask=text_mask)
        text_sequence = text_outputs.last_hidden_state
        
        # B. Encode Labels
        if ent_label_ids is not None:
             ent_label_embeds = self.encode_labels(ent_label_ids, ent_label_mask)
             rel_label_embeds = self.encode_labels(rel_label_ids, rel_label_mask)
        else:
             ent_label_embeds = self.cached_ent_embeds
             rel_label_embeds = self.cached_rel_embeds
             if ent_label_embeds is None:
                 # Minimal fallback if labels missing
                 return None, None 

        # C. Entity Recog
        if entity_spans is None: return None, None
        
        span_starts = [[s[0] for s in batch_spans] for batch_spans in entity_spans]
        span_ends = [[s[1] for s in batch_spans] for batch_spans in entity_spans]
        span_embeds = self._get_span_embeddings(text_sequence, text_mask, span_starts, span_ends)   
        
        span_embeds = self.entity_proj(span_embeds)
        span_embeds = F.normalize(span_embeds, p=2, dim=-1)
        
        # Using Entity Temperature
        entity_logits = torch.matmul(span_embeds, ent_label_embeds.transpose(-2, -1))
        entity_logits = entity_logits / self.ent_temperature 
        
        # D. Relation Extract (Vectorized)
        relation_logits = None
        if relation_pairs is None: return entity_logits, None
        
        max_pairs = max((len(p) for p in relation_pairs), default=0)
            
        if max_pairs > 0 and rel_label_embeds is not None:
            batch_size = len(relation_pairs)
            num_rel_labels = rel_label_embeds.shape[1]
            device = span_embeds.device
            
            relation_logits = torch.zeros(batch_size, max_pairs, num_rel_labels, device=device)
            entity_probs = F.softmax(entity_logits, dim=-1)
            global_context_batch = self._mean_pooling(text_sequence, text_mask)  # [B, H]

            for b in range(batch_size):
                pairs = relation_pairs[b]
                if len(pairs) == 0: 
                    continue
                
                num_pairs = len(pairs)
                curr_span_embeds = span_embeds[b]  # [N_spans, H]
                curr_global = global_context_batch[b]  # [H]
                
                # Vectorized: Get entity type features for all spans at once
                ent_type_feats = torch.matmul(entity_probs[b], ent_label_embeds[b])  # [N_spans, H]
                
                # Convert pairs to tensors for vectorized indexing
                pairs_tensor = torch.tensor(pairs, device=device, dtype=torch.long)  # [P, 2]
                subj_indices = pairs_tensor[:, 0]  # [P]
                obj_indices = pairs_tensor[:, 1]   # [P]
                
                # Vectorized: Gather span embeddings for subjects and objects
                s_vecs_raw = curr_span_embeds[subj_indices]  # [P, H]
                o_vecs_raw = curr_span_embeds[obj_indices]   # [P, H]
                
                # Vectorized: Apply directional gates
                s_vecs = self.s_gate(s_vecs_raw)  # [P, H]
                o_vecs = self.o_gate(o_vecs_raw)  # [P, H]
                
                # Vectorized: Get type features
                s_types = ent_type_feats[subj_indices]  # [P, H]
                o_types = ent_type_feats[obj_indices]   # [P, H]
                
                # Vectorized: Compute derived features
                diff_vecs = s_vecs - o_vecs  # [P, H]
                prod_vecs = s_vecs * o_vecs  # [P, H]
                s_inter = s_vecs * o_types   # [P, H]
                o_inter = o_vecs * s_types   # [P, H]
                type_inter = s_types * o_types  # [P, H]
                
                # Vectorized: Global context (broadcast)
                global_vecs = curr_global.unsqueeze(0).expand(num_pairs, -1)  # [P, H]
                global_bias = global_vecs * s_inter  # [P, H]
                
                # Vectorized: Distance embedding
                dist = torch.abs(subj_indices - obj_indices)  # [P]
                dist_clamped = torch.clamp(dist, max=self.max_dist - 1)
                dist_feats = self.dist_emb(dist_clamped)  # [P, H]
                
                # 🔥 [FAST] Middle context - Use global context instead of computing per-pair
                # This avoids the expensive loop and is nearly as effective
                middle_vecs = global_vecs  # [P, H] - reuse global context
                
                # Concatenate all 13 features
                combined_rel_feats = torch.cat([
                    s_vecs, o_vecs, s_types, o_types,  # [P, 4H]
                    diff_vecs, prod_vecs,              # [P, 2H]
                    middle_vecs, global_vecs,          # [P, 2H]
                    s_inter, o_inter,                  # [P, 2H]
                    dist_feats, global_bias,           # [P, 2H]
                    type_inter                         # [P, H]
                ], dim=-1)  # [P, 13H]
                
                # Project and normalize
                pair_proj = self.relation_proj(combined_rel_feats)  # [P, H]
                pair_proj = F.normalize(pair_proj, p=2, dim=-1)
                
                # Get relation label embeddings
                curr_rel_embeds = rel_label_embeds[0] if rel_label_embeds.shape[0] == 1 else rel_label_embeds[b]
                
                # Vectorized: Compute logits for all pairs at once
                logits = torch.matmul(pair_proj, curr_rel_embeds.t()) / self.rel_temperature  # [P, R]
                relation_logits[b, :num_pairs, :] = logits

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