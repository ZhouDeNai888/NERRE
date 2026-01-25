"""
GraphRAG Dataset - Dataset class for NER + Relation Extraction
Designed for building Knowledge Graphs
"""

import json
import random
import torch
from torch.utils.data import Dataset


class GraphRAGDataset(Dataset):
    """
    Dataset for joint NER + Relation Extraction
    Suitable for Graph RAG / Knowledge Graph construction
    
    ✅ Key Feature: มี "O" (Outside/None) label สำหรับ spans ที่ไม่ใช่ entity
       ทำให้โมเดลเรียนรู้ว่า span ไหนควรเป็น entity และ span ไหนไม่ควร
    """
    
    # Special label for non-entity spans
    O_LABEL = "O"
    NO_REL_LABEL = "NO_RELATION"
    
    def __init__(self, json_file, tokenizer, max_len=256, neg_sample_ratio=0.3, neg_span_ratio=1.0):
        """
        Args:
            json_file: path to training data JSON (str or List[str])
            tokenizer: HuggingFace tokenizer
            max_len: max sequence length
            neg_sample_ratio: ratio of negative labels to sample
            neg_span_ratio: ratio of negative spans (non-entity) to positive spans
        """
        self.data = []
        if isinstance(json_file, str):
            json_files = [json_file]
        else:
            json_files = json_file
            
        for jf in json_files:
            print(f"Loading data from {jf}...")
            with open(jf, 'r', encoding='utf-8') as f:
                data_part = json.load(f)
                self.data.extend(data_part)
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.neg_sample_ratio = neg_sample_ratio
        self.neg_span_ratio = neg_span_ratio
        
        # Scan for all unique labels
        self.all_ent_labels = set()
        self.all_rel_labels = set()
        
        # Store descriptions
        self.ent_descriptions = {}
        self.rel_descriptions = {}

        print(f"Scanning {json_file} for labels...")
        for item in self.data:
            if 'entities' in item:
                for ent in item['entities']:
                    label = ent['label']
                    self.all_ent_labels.add(label)
                    # Capture description if available
                    if 'description' in ent and label not in self.ent_descriptions:
                        self.ent_descriptions[label] = ent['description']

            if 'relations' in item:
                for rel in item['relations']:
                    label = rel['label']
                    self.all_rel_labels.add(label)
                    # Capture description if available
                    if 'description' in rel and label not in self.rel_descriptions:
                        self.rel_descriptions[label] = rel['description']
        
        self.all_ent_labels = sorted(list(self.all_ent_labels))
        self.all_rel_labels = sorted(list(self.all_rel_labels))
        
        # ✅ เพิ่ม "O" label ไว้ตัวแรก (index 0)
        self.all_ent_labels_with_O = [self.O_LABEL] + self.all_ent_labels
        
        # 🔥 [NEW] เพิ่ม "NO_RELATION" label ไว้ตัวแรก (index 0)
        self.all_rel_labels_with_NO_REL = [self.NO_REL_LABEL] + self.all_rel_labels

        self.ent_label2id = {label: i for i, label in enumerate(self.all_ent_labels_with_O)}
        self.rel_label2id = {label: i for i, label in enumerate(self.all_rel_labels_with_NO_REL)}

        # Create Descriptive Label List for Model Input
        self.ent_label_texts = []
        for label in self.all_ent_labels_with_O:
            if label == self.O_LABEL:
                desc = "Outside: Not an entity"
            else:
                desc = self.ent_descriptions.get(label, f"{label}: Representation of {label}")
                # Ensure format "Label: Description" if not already
                if not desc.startswith(label):
                    desc = f"{label}: {desc}"
            self.ent_label_texts.append(desc)

        self.rel_label_texts = []
        for label in self.all_rel_labels_with_NO_REL:
            if label == self.NO_REL_LABEL:
                desc = "No Relation: No relationship exists between these entities"
            else:
                desc = self.rel_descriptions.get(label, f"{label}: Relation type {label}")
                if not desc.startswith(label):
                    desc = f"{label}: {desc}"
            self.rel_label_texts.append(desc)
        
        print(f"✅ Found {len(self.all_ent_labels)} entity types")
        print(f"✅ Found {len(self.all_rel_labels)} relation types")
        print(f"✅ Added special labels: '{self.O_LABEL}' and '{self.NO_REL_LABEL}'")

    def get_ent_label_list(self):
        return self.all_ent_labels_with_O

    def get_rel_label_list(self):
        return self.all_rel_labels_with_NO_REL

    def __len__(self):
        return len(self.data)

    def _char_to_token_span(self, encoding, start_char, end_char):
        """Convert character span to token span"""
        start_token = None
        end_token = None
        
        # Find start token
        for i in range(3):  # Try a few offsets
            start_token = encoding.char_to_token(start_char + i)
            if start_token is not None:
                break
                
        # Find end token (end_char is exclusive, so -1)
        for i in range(3):
            end_token = encoding.char_to_token(end_char - 1 - i)
            if end_token is not None:
                break
        
        return start_token, end_token

    def _get_word_boundaries(self, encoding, text):
        """หา word boundaries จาก offset_mapping"""
        offset_mapping = encoding["offset_mapping"].squeeze(0).tolist()
        words = []
        current_word_tokens = []
        current_char_start = None
        prev_char_end = None
        
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            if char_start == char_end == 0:
                continue
            if prev_char_end is not None and char_start > prev_char_end:
                if current_word_tokens:
                    words.append({
                        'token_start': current_word_tokens[0],
                        'token_end': current_word_tokens[-1],
                        'char_start': current_char_start,
                        'char_end': prev_char_end
                    })
                current_word_tokens = [token_idx]
                current_char_start = char_start
            else:
                if current_char_start is None:
                    current_char_start = char_start
                current_word_tokens.append(token_idx)
            prev_char_end = char_end
        
        if current_word_tokens:
            words.append({
                'token_start': current_word_tokens[0],
                'token_end': current_word_tokens[-1],
                'char_start': current_char_start,
                'char_end': prev_char_end
            })
        return words

    def _generate_negative_spans(self, words, valid_entities, num_to_sample, max_span_width=3):
        """
        สร้าง negative spans โดยเน้น "Hard Negatives" (ส่วนย่อยของ Entity จริง)
        เพื่อให้โมเดลเรียนรู้ที่จะไม่ตอบคำที่ซ้อนทับกัน
        """
        all_candidates = []
        n_words = len(words)
        
        # -------------------------------------------------------------
        # 🔥 กลยุทธ์ที่ 1: Hard Negatives (Sub-spans) - พระเอกของเรา
        # ตัด Entity จริงให้แหว่งๆ แล้วบอกว่าเป็น Negative
        # -------------------------------------------------------------
        valid_entity_ranges = set() # เก็บช่วงที่เป็น Entity จริงไว้เช็ค
        
        for ent in valid_entities:
            # ent คือ dict ที่มี 'span': (start, end)
            start, end = ent['span'] # token index
            valid_entity_ranges.add((start, end))
            
            # ถ้า Entity ยาวกว่า 1 token (เช่น "Elon Musk")
            # ให้สร้าง sub-span (เช่น "Elon", "Musk") แล้วยัดเป็น Negative
            span_len = end - start + 1
            if span_len > 1:
                # Loop สร้าง sub-spans ภายในตัวมันเอง
                for i in range(span_len):
                    for j in range(i, span_len):
                        sub_start = start + i
                        sub_end = start + j
                        
                        # ถ้า sub-span นี้ ไม่ใช่ตัวเต็ม (คือสั้นกว่าตัวเต็ม)
                        if not (sub_start == start and sub_end == end):
                            all_candidates.append((sub_start, sub_end))

        # -------------------------------------------------------------
        # กลยุทธ์ที่ 2: Random Negatives (เหมือนเดิม)
        # สุ่มจากที่ว่างๆ เพื่อให้รู้จักคำทั่วไป
        # -------------------------------------------------------------
        for width in range(1, min(max_span_width + 1, n_words + 1)):
            for start_idx in range(n_words - width + 1):
                end_idx = start_idx + width - 1
                
                # ถ้าช่วงนี้ ไม่ใช่ Entity จริง (เช็คจาก set ที่เก็บไว้)
                if (start_idx, end_idx) not in valid_entity_ranges:
                     # (Optional) กรองไม่ให้ซ้ำกับ Hard Negatives ที่ใส่ไปแล้วก็ได้ 
                     # แต่ปล่อยให้ซ้ำก็ได้ ยิ่งเน้นย้ำ
                    all_candidates.append((start_idx, end_idx))
        
        # สุ่มมาใช้ตามจำนวนที่ต้องการ
        if all_candidates and num_to_sample > 0:
            # แนะนำให้สุ่มโดยให้โอกาส Hard Negatives เยอะหน่อย 
            # (แต่ใน list all_candidates ตอนนี้ปนกันอยู่ สุ่มเลยก็ได้)
            negative_spans = random.sample(
                all_candidates, 
                k=min(len(all_candidates), num_to_sample)
            )
            return negative_spans
            
        return []
    

    def _generate_border_negatives(self, valid_entities, num_to_sample, seq_len):
        """
        จงใจสร้าง Span ที่ 'เกือบถูก' เพื่อคุมขอบเขต (Boundary Control)
        """
        border_candidates = []
        for ent in valid_entities:
            s, e = ent['span']
            
            # ตัวหลอก 1: ขยายขวาไปอีก 1 token (เช่น "Elon Musk" -> "Elon Musk founded")
            if e + 1 < seq_len:
                border_candidates.append((s, e + 1))
            
            # ตัวหลอก 2: ขยายซ้ายไปอีก 1 token (เช่น "SpaceX" -> "founded SpaceX")
            if s - 1 >= 0:
                border_candidates.append((max(0, s - 1), e))
                
            # ตัวหลอก 3: ตัดท้ายออก 1 token (เช่น "Elon Musk" -> "Elon")
            if e > s:
                border_candidates.append((s, e - 1))

        if not border_candidates:
            return []
            
        return random.sample(border_candidates, k=min(len(border_candidates), num_to_sample))

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        entities = item.get('entities', [])
        relations = item.get('relations', [])
        
        # 1. Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_offsets_mapping=True
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        
        # 2. Convert entity spans from char to token indices
        valid_entities = []  # (token_start, token_end, label, original_idx)
        valid_entity_char_ranges = []  # For negative span sampling
        
        for ent_idx, ent in enumerate(entities):
            start_token, end_token = self._char_to_token_span(
                encoding, ent['start'], ent['end']
            )
            
            if start_token is not None and end_token is not None:
                valid_entities.append({
                    'span': (start_token, end_token),
                    'label': ent['label'],
                    'original_idx': ent_idx
                })
                valid_entity_char_ranges.append((ent['start'], ent['end']))
        
        # ✅ 3. Generate negative spans (non-entity spans labeled as "O")
        words = self._get_word_boundaries(encoding, text)
        num_neg_spans = max(1, int(len(valid_entities) * self.neg_span_ratio))
        negative_spans = self._generate_negative_spans(
            words, valid_entities, num_neg_spans
        )


        # 🔥 [NEW] เพิ่ม Border Negatives ตรงนี้
        border_negatives = self._generate_border_negatives(valid_entities, len(valid_entities), self.max_len)
        
        # 4. Combine positive and negative spans
        # Positive spans (real entities)
        all_spans = []
        all_span_labels = []  # The actual label for each span
        
        for ent in valid_entities:
            all_spans.append(ent['span'])
            all_span_labels.append(ent['label'])  # e.g., "person", "organisation"
        
        # Negative spans (non-entities)
        for neg_span in negative_spans:
            all_spans.append(neg_span)
            all_span_labels.append(self.O_LABEL)  # "O" for Outside


        # 🔥 [NEW] ใส่ Border Negatives เข้าไปในชุดข้อมูล
        valid_span_set = set([ent['span'] for ent in valid_entities])
        for b_neg in border_negatives:
            if b_neg not in valid_span_set: # ตรวจสอบเพื่อไม่ให้ทับของจริง
                all_spans.append(b_neg)
                all_span_labels.append(self.O_LABEL)
        
        # ✅ 5. Use ALL labels including "O" - always use the full label set
        train_ent_labels = self.ent_label_texts  # Use descriptions
        
        # 6. Create entity target matrix
        num_spans = len(all_spans)
        num_ent_labels = len(train_ent_labels)
        
        ent_targets = torch.zeros((num_spans, num_ent_labels))

        # 🔥 [FIX] เติมค่า 1.0 ให้ตรงกับ Label จริง
        for i, label_text in enumerate(all_span_labels):
            # ใช้ .get() เพื่อความปลอดภัยและรวดเร็ว
            if label_text in self.ent_label2id:
                label_idx = self.ent_label2id[label_text]
                ent_targets[i, label_idx] = 1.0
            else:
                # กรณีกันพลาด: ถ้าไม่เจอให้โยนลง "O"
                if self.O_LABEL in self.ent_label2id:
                    o_idx = self.ent_label2id[self.O_LABEL]
                    ent_targets[i, o_idx] = 1.0
        
        # ===========================================================
        # 🔥 [UPDATED FIX] ใช้ Hybrid Mapping (ID เป็นหลัก, Text สำรอง)
        # ===========================================================
        
        # 1. สร้าง Maps เตรียมไว้ 2 แบบ
        id_to_valid_indices = {}    # ✅ แบบแม่นยำ (ใช้ original_idx จาก JSON)
        text_to_valid_indices = {}  # ⚠️ แบบสำรอง (ใช้ text)

        for new_idx, ent in enumerate(valid_entities):
            # --- A. Map by ID (Original Index) ---
            orig_idx = ent['original_idx']
            if orig_idx not in id_to_valid_indices:
                id_to_valid_indices[orig_idx] = []
            id_to_valid_indices[orig_idx].append(new_idx)
            
            # --- B. Map by Text (Fallback) ---
            # ดึง Text ออกมาจาก JSON เดิม
            orig_ent_data = entities[orig_idx] 
            # ถ้าใน JSON มี key 'text' ก็ใช้ ถ้าไม่มีก็ตัด string เอา
            entity_text = orig_ent_data.get('text', text[orig_ent_data['start']:orig_ent_data['end']])
            
            if entity_text not in text_to_valid_indices:
                text_to_valid_indices[entity_text] = []
            text_to_valid_indices[entity_text].append(new_idx)
        
        # 2. จับคู่ความสัมพันธ์ (Relation Mapping)
        positive_rel_map = {}
        
        for rel in relations:
            head_indices = []
            tail_indices = []

            # 🔥 Priority 1: เช็คว่า JSON มี 'head_idx' / 'tail_idx' หรือไม่ (แม่นยำที่สุด - รองรับ CrossRE ver เก่า)
            if 'head_idx' in rel and 'tail_idx' in rel:
                head_indices = id_to_valid_indices.get(rel['head_idx'], [])
                tail_indices = id_to_valid_indices.get(rel['tail_idx'], [])

            # 🔥 Priority 2: เช็คว่า 'head' / 'tail' เป็น Integer Index หรือไม่ (รองรับ hf_dataloader format)
            elif 'head' in rel and 'tail' in rel and isinstance(rel['head'], int) and isinstance(rel['tail'], int):
                head_indices = id_to_valid_indices.get(rel['head'], [])
                tail_indices = id_to_valid_indices.get(rel['tail'], [])

            # ⚠️ Priority 3: ถ้าไม่มี ID ให้ใช้ชื่อ (Text) เหมือนเดิม (รองรับ Data v2 เดิม)
            elif 'head' in rel and 'tail' in rel:
                head_indices = text_to_valid_indices.get(rel['head'], [])
                tail_indices = text_to_valid_indices.get(rel['tail'], [])
            
            # ถ้าหา Entity ไม่เจอเลย (เช่น ถูกตัดทิ้งตอน Tokenize) ให้ข้าม
            if not head_indices or not tail_indices:
                continue

            # เตรียม Target (One-hot vector)
            # ใช้ self.all_rel_labels_with_NO_REL (ที่มี NO_RELATION ที่ index 0)
            rel_target = torch.zeros(len(self.all_rel_labels_with_NO_REL))
            
            if rel['label'] in self.rel_label2id:
                rel_idx = self.rel_label2id[rel['label']]
                rel_target[rel_idx] = 1.0
            
            # จับคู่ทุกความเป็นไปได้ (Pairing)
            for h_idx in head_indices:
                for t_idx in tail_indices:
                    if h_idx == t_idx: continue # ข้าม Self-loop
                    
                    pair_key = (h_idx, t_idx)
                    
                    # ถ้าคู่นี้มีอยู่แล้ว ให้รวม Logic (OR) เผื่อมีความสัมพันธ์หลายแบบ
                    if pair_key in positive_rel_map:
                        positive_rel_map[pair_key] = torch.max(positive_rel_map[pair_key], rel_target)
                    else:
                        positive_rel_map[pair_key] = rel_target

        # ... (ต่อด้วยส่วน Negative Sampling โค้ดเดิมได้เลย) ...

        # -----------------------------------------------------------
        # 🔥 แก้ไขด่วน: ใช้ "Negative Sampling" แทน "All Negatives"
        # -----------------------------------------------------------
        all_pairs = []
        all_targets = []
        
        # 1. ใส่ Positive Pairs (ของจริง) ให้ครบก่อน
        # เรียงลำดับเพื่อให้ Reproducible (สำคัญมาก)
        pos_keys = sorted(list(positive_rel_map.keys()))
        for pair in pos_keys:
            all_pairs.append(pair)
            all_targets.append(positive_rel_map[pair])
            
        num_positives = len(pos_keys)
        
        # 2. เก็บ Negative Candidates (คู่ที่ไม่มีความสัมพันธ์)
        neg_candidates = []
        num_entities = len(valid_entities)
        
        if num_entities > 1:
            for i in range(num_entities):
                for j in range(num_entities):
                    if i == j: continue 
                    pair = (i, j)
                    if pair not in positive_rel_map:
                        neg_candidates.append(pair)
        
        # 3. 🔥 Hard Negative Strategy: Reversals + Random
        # กฎ: เอา Negative แค่ 3 เท่าของ Positive (Ratio 1:3)
        if num_positives > 0:
            num_neg_to_sample = num_positives * 9
        else:
            num_neg_to_sample = 15

        # 3.1 Force Reversed Pairs (สอนให้รู้ว่า A->B ไม่เท่ากับ B->A)
        forced_negatives = set()
        for (h, t) in pos_keys:
             rev_pair = (t, h)
             if rev_pair not in positive_rel_map:
                 forced_negatives.add(rev_pair)

        # 3.2 Fill the rest with Random Negatives
        neg_candidates = [p for p in neg_candidates if p not in forced_negatives] # ลบตัวซ้ำ
        
        final_negs = list(forced_negatives)
        
        # ถ้าโควตายังเหลือ ให้สุ่มเพิ่ม
        remaining_slots = num_neg_to_sample - len(final_negs)
        if remaining_slots > 0 and neg_candidates:
             final_negs += random.sample(neg_candidates, min(len(neg_candidates), remaining_slots))
        
        # ถ้า Hard Negatives เยอะเกินโควตา ก็ให้ใช้ทั้งหมดไปเลย (ยิ่งเยอะยิ่งดีสำหรับ Direction)
        
        if final_negs:
             # 🔥 [NET] Negative Target คือ class "NO_RELATION" (index 0)
            zero_target = torch.zeros(len(self.all_rel_labels_with_NO_REL))
            zero_target[0] = 1.0 # Set NO_RELATION to 1.0
            
            for pair in final_negs:
                all_pairs.append(pair)
                all_targets.append(zero_target)

        # 4. Stack Targets (เหมือนเดิม)
        if all_targets:
            rel_targets = torch.stack(all_targets)
        else:
            rel_targets = torch.zeros((0, len(self.all_rel_labels_with_NO_REL)))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": all_spans,
            "ent_labels": train_ent_labels,
            "ent_targets": ent_targets,
            "rel_pairs": all_pairs,      # ✅ ส่งไปทั้งคู่จริงและคู่หลอก
            "rel_labels": self.rel_label_texts, # Use descriptions
            "rel_targets": rel_targets,  # ✅ Target มีทั้ง 1 และ 0
            "num_positive_spans": len(valid_entities)
        }


def graph_rag_collate_fn(batch):
    """
    Collate function for variable-length data
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Keep variable-length items as lists
    spans = [item['spans'] for item in batch]
    ent_labels = [item['ent_labels'] for item in batch]
    rel_labels = [item['rel_labels'] for item in batch]
    ent_targets = [item['ent_targets'] for item in batch]
    rel_targets = [item['rel_targets'] for item in batch]
    rel_pairs = [item['rel_pairs'] for item in batch]
    num_positive = [item['num_positive_spans'] for item in batch]

    return {
        "text_ids": input_ids,
        "text_mask": attention_mask,
        "spans": spans,
        "ent_labels_text": ent_labels,
        "rel_labels_text": rel_labels,
        "ent_targets": ent_targets,
        "rel_targets": rel_targets,
        "pairs": rel_pairs,
        "num_positive_spans": num_positive  # ✅ เพิ่มสำหรับ filter relation pairs
    }


# Backward compatibility
ZeroShotDataset = GraphRAGDataset
collate_fn = graph_rag_collate_fn
