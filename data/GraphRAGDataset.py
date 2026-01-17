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
    
    def __init__(self, json_file, tokenizer, max_len=256, neg_sample_ratio=0.3, neg_span_ratio=1.0):
        """
        Args:
            json_file: path to training data JSON
            tokenizer: HuggingFace tokenizer
            max_len: max sequence length
            neg_sample_ratio: ratio of negative labels to sample
            neg_span_ratio: ratio of negative spans (non-entity) to positive spans
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.neg_sample_ratio = neg_sample_ratio
        self.neg_span_ratio = neg_span_ratio
        
        # Scan for all unique labels
        self.all_ent_labels = set()
        self.all_rel_labels = set()
        
        print(f"Scanning {json_file} for labels...")
        for item in self.data:
            if 'entities' in item:
                for ent in item['entities']:
                    self.all_ent_labels.add(ent['label'])
            if 'relations' in item:
                for rel in item['relations']:
                    self.all_rel_labels.add(rel['label'])
        
        self.all_ent_labels = sorted(list(self.all_ent_labels))
        self.all_rel_labels = sorted(list(self.all_rel_labels))
        
        # ✅ เพิ่ม "O" label ไว้ตัวแรก (index 0)
        self.all_ent_labels_with_O = [self.O_LABEL] + self.all_ent_labels
        
        print(f"✅ Found {len(self.all_ent_labels)} entity types: {self.all_ent_labels}")
        print(f"✅ Added 'O' label for non-entity spans")
        print(f"✅ Found {len(self.all_rel_labels)} relation types: {self.all_rel_labels}")

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

    def _generate_negative_spans(self, words, valid_entity_char_ranges, num_to_sample, max_span_width=3):
        """สร้าง negative spans (ที่ไม่ใช่ entity) สำหรับ training"""
        negative_spans = []
        n_words = len(words)
        
        all_candidates = []
        for width in range(1, min(max_span_width + 1, n_words + 1)):
            for start_idx in range(n_words - width + 1):
                end_idx = start_idx + width - 1
                span_words = words[start_idx:end_idx + 1]
                char_start = span_words[0]['char_start']
                char_end = span_words[-1]['char_end']
                token_start = span_words[0]['token_start']
                token_end = span_words[-1]['token_end']
                
                # Check if overlaps with any entity
                is_entity = False
                for ent_start, ent_end in valid_entity_char_ranges:
                    # Overlap check
                    if not (char_end <= ent_start or char_start >= ent_end):
                        is_entity = True
                        break
                
                if not is_entity:
                    all_candidates.append((token_start, token_end))
        
        # Sample some negatives
        if all_candidates and num_to_sample > 0:
            negative_spans = random.sample(
                all_candidates, 
                k=min(len(all_candidates), num_to_sample)
            )
        
        return negative_spans

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
            words, valid_entity_char_ranges, num_neg_spans
        )
        
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
        
        # ✅ 5. Use ALL labels including "O" - always use the full label set
        train_ent_labels = self.all_ent_labels_with_O  # ["O", "algorithm", "date", ...]
        
        # 6. Create entity target matrix
        num_spans = len(all_spans)
        num_ent_labels = len(train_ent_labels)
        
        ent_targets = torch.zeros((num_spans, num_ent_labels))
        
        for i, label in enumerate(all_span_labels):
            if label in train_ent_labels:
                label_idx = train_ent_labels.index(label)
                ent_targets[i, label_idx] = 1.0
        
        # 5. Prepare relation pairs and targets
        # Map original entity idx -> new idx in valid_entities
        original_to_valid = {}
        for new_idx, ent in enumerate(valid_entities):
            original_to_valid[ent['original_idx']] = new_idx
        
        rel_pairs = []
        rel_targets_list = []
        
        for rel in relations:
            head_orig = rel['head']
            tail_orig = rel['tail']
            
            # Check if both entities are valid
            if head_orig in original_to_valid and tail_orig in original_to_valid:
                head_new = original_to_valid[head_orig]
                tail_new = original_to_valid[tail_orig]
                
                rel_pairs.append((head_new, tail_new))
                
                # Create one-hot for this relation
                rel_target = torch.zeros(len(self.all_rel_labels))
                if rel['label'] in self.all_rel_labels:
                    rel_idx = self.all_rel_labels.index(rel['label'])
                    rel_target[rel_idx] = 1.0
                rel_targets_list.append(rel_target)
        
        # Stack relation targets
        if rel_targets_list:
            rel_targets = torch.stack(rel_targets_list)
        else:
            rel_targets = torch.zeros((0, len(self.all_rel_labels)))
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "spans": all_spans,  # ✅ ใช้ all_spans ที่รวม positive + negative
            "ent_labels": train_ent_labels,
            "ent_targets": ent_targets,
            "rel_pairs": rel_pairs,
            "rel_labels": self.all_rel_labels,
            "rel_targets": rel_targets,
            "num_positive_spans": len(valid_entities)  # บอกว่า span แรก N ตัวเป็น entity จริง
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
