import json
import random
import torch
from torch.utils.data import Dataset

class ZeroShotDataset(Dataset):
    def __init__(self, json_file, tokenizer, max_len=512, neg_sample_ratio=0.5):
        """
        Args:
            json_file: path ไปยังไฟล์ train_data.json
            tokenizer: ตัวตัดคำของโมเดล (AutoTokenizer)
            max_len: ความยาวสูงสุดของประโยค
            neg_sample_ratio: สัดส่วนตัวหลอก (0.5 = มีตัวจริง 10 จะสุ่มตัวหลอกมา 5)
        """
        # 1. โหลดข้อมูล
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.neg_sample_ratio = neg_sample_ratio
        
        # 2. [Auto-Detect] สแกนหา Label ทั้งหมดที่มีในไฟล์ JSON
        self.all_ent_labels = set()
        self.all_rel_labels = set()
        
        print(f"Scanning {json_file} for labels...")
        for item in self.data:
            # เก็บ Entity Labels
            if 'entities' in item:
                for ent in item['entities']:
                    self.all_ent_labels.add(ent['label'])
            
            # เก็บ Relation Labels
            if 'relations' in item:
                for rel in item['relations']:
                    self.all_rel_labels.add(rel['label'])
                    
        # แปลงเป็น List เพื่อให้สุ่มได้
        self.all_ent_labels = list(self.all_ent_labels)
        self.all_rel_labels = list(self.all_rel_labels)
        
        print(f"✅ Found {len(self.all_ent_labels)} unique entities: {self.all_ent_labels[:5]}...")
        print(f"✅ Found {len(self.all_rel_labels)} unique relations: {self.all_rel_labels[:5]}...")

    def __len__(self):
        return len(self.data)

    def _align_labels_with_tokens(self, tokenized_inputs, entities):
        """แปลง Char Index (0-10) -> Token Index (1-3)"""
        span_indices = []
        valid_ent_idxs = [] 
        
        for idx, ent in enumerate(entities):
            start_char = ent['start']
            end_char = ent['end'] - 1 
            
            try:
                start_token = tokenized_inputs.char_to_token(start_char)
                end_token = tokenized_inputs.char_to_token(end_char)
                
                # ถ้าหาไม่เจอ (เป็น None) ให้ลองขยับหาตัวข้างๆ นิดหน่อย (Heuristic)
                if start_token is None:
                    start_token = tokenized_inputs.char_to_token(start_char + 1)
                if end_token is None:
                    end_token = tokenized_inputs.char_to_token(end_char - 1)

                if start_token is not None and end_token is not None:
                    span_indices.append((start_token, end_token))
                    valid_ent_idxs.append(idx)
            except:
                continue
                
        return span_indices, valid_ent_idxs

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 1. Tokenize Text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len
        )
        
        # 2. Align Spans
        span_list, valid_ent_idxs = self._align_labels_with_tokens(inputs, item.get('entities', []))
        
        # 3. Dynamic Negative Sampling (หัวใจสำคัญ)
        # 3.1 หาว่าประโยคนี้มี Label อะไรบ้าง (True Labels)
        true_ent_labels = set()
        for i in valid_ent_idxs:
            true_ent_labels.add(item['entities'][i]['label'])
            
        # 3.2 หา Label ที่ "ไม่มี" ในประโยคนี้ (Negative Candidates)
        # โดยดึงจาก Master List ที่เราสแกนไว้ตอนแรก (self.all_ent_labels)
        candidate_negatives = list(set(self.all_ent_labels) - true_ent_labels)
        
        # 3.3 สุ่ม Negative มาผสม
        # คำนวณจำนวนที่จะสุ่ม (อย่างน้อย 1 ตัว ถ้ามี)
        num_neg = max(1, int(len(true_ent_labels) * self.neg_sample_ratio))
        if len(candidate_negatives) > 0:
            chosen_negatives = random.sample(candidate_negatives, k=min(len(candidate_negatives), num_neg))
        else:
            chosen_negatives = []
            
        # 3.4 รวมเป็น Label List สำหรับเทรนรอบนี้
        train_ent_labels = list(true_ent_labels) + chosen_negatives
        random.shuffle(train_ent_labels) # สลับตำแหน่ง ไม่ให้ Model จำว่าตัวแรกคือตัวถูกเสมอ
        
        # 4. สร้าง Target Matrix (One-hot-like)
        # Shape: [Num_Spans, Num_Train_Labels]
        num_spans = len(span_list)
        num_labels = len(train_ent_labels)
        ent_targets = torch.zeros((num_spans, num_labels))
        
        for i, original_idx in enumerate(valid_ent_idxs):
            true_label_str = item['entities'][original_idx]['label']
            # ถ้า Label ของ Span นี้ อยู่ใน list ที่เราจะเทรนรอบนี้ (ซึ่งต้องอยู่แหละ)
            if true_label_str in train_ent_labels:
                label_idx = train_ent_labels.index(true_label_str)
                ent_targets[i, label_idx] = 1.0

        # --- ส่วน Relation (ใช้ Logic เดียวกัน) ---
        # (เพื่อให้โค้ดไม่ยาวเกินไป ผมขอละส่วน Relation ไว้ แต่ใช้ Logic เดียวกับ Entity ด้านบนได้เลย)
        # หรือถ้าจะทำ Relation แบบง่าย ให้ส่ง all_rel_labels ไปทั้งหมดเลยก็ได้ถ้ารายการไม่เยอะ
        
        # Return ผลลัพธ์
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "spans": span_list,                 # List of (start, end)
            "ent_labels": train_ent_labels,     # List of strings (Dynamic!)
            "ent_targets": ent_targets,         # Tensor Match กับ ent_labels
            "rel_pairs": [],                    # (ใส่ Logic Relation เพิ่มตามต้องการ)
            "rel_labels": self.all_rel_labels,  # ส่งไปทั้งหมด หรือสุ่มเอาก็ได้
            "rel_targets": torch.zeros((0, len(self.all_rel_labels))) 
        }
    

def collate_fn(batch):
        """
        หน้าที่: รวม Data หลายๆ อัน (List of Dicts) ให้เป็น Batch เดียว
        แต่ต้องระวังเรื่องข้อมูลที่มีขนาดไม่เท่ากัน (Spans, Targets)
        """
        
        # 1. ส่วนที่เป็น Tensor ขนาดคงที่ (Input Text) -> Stack รวมได้เลย
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        # 2. ส่วนที่เป็น List หรือ Tensor ขนาดไม่เท่ากัน -> ให้เก็บเป็น List ซ้อน List แทน
        # ห้าม stack!
        spans = [item['spans'] for item in batch]          # [[(1,2)], [(4,5), (6,7)]]
        
        # Labels เป็น Text (String) ซึ่งแต่ละ Batch ไม่เหมือนกัน
        ent_labels = [item['ent_labels'] for item in batch] 
        rel_labels = [item['rel_labels'] for item in batch]
        
        # Targets ขนาดไม่เท่ากัน (ตามจำนวน Spans x Labels) -> เก็บเป็น List of Tensors
        ent_targets = [item['ent_targets'] for item in batch]
        rel_targets = [item['rel_targets'] for item in batch]
        
        # Relation Pairs ก็ไม่เท่ากัน
        rel_pairs = [item['rel_pairs'] for item in batch]

        return {
            "text_ids": input_ids,
            "text_mask": attention_mask,
            "spans": spans,                   # List of List of Tuples
            "ent_labels_text": ent_labels,    # List of List of Strings
            "rel_labels_text": rel_labels,
            "ent_targets": ent_targets,       # List of Tensors
            "rel_targets": rel_targets,
            "pairs": rel_pairs
        }