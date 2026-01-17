"""
NERRE Inference v2 - Word-Level Entity & Relation Extraction for Graph RAG
============================================================================
ปรับปรุงจาก v1 ให้ทำงานแบบ word-level แทน subword tokenization
เพื่อให้ได้ผลลัพธ์ที่เหมาะกับการสร้าง Knowledge Graph
"""

import torch
import json
import os
import sys
import re

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from transformers import AutoTokenizer
from model.model import ZeroShotJointModel


class NERREInference:
    def __init__(self, model_dir, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_dir} to {self.device}...")
        
        # ✅ O_LABEL for filtering non-entities (set first!)
        self.O_LABEL = "O"
        
        # 1. Load Config
        with open(f"{model_dir}/config.json", "r", encoding='utf-8') as f:
            self.config = json.load(f)
            
        # 2. Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # 3. Load Model
        self.model = ZeroShotJointModel(self.config['model_name'])
        self.model.load_state_dict(
            torch.load(f"{model_dir}/pytorch_model.bin", map_location=self.device, weights_only=True)
        )
        self.model.to(self.device)
        self.model.eval()
        
        # 4. Pre-compute Label Embeddings
        self._set_labels(self.config['ent_labels'], self.config['rel_labels'])
        
    def _set_labels(self, ent_labels, rel_labels):
        """Cache label embeddings for faster inference"""
        print("Caching labels...")
        self.ent_labels = ent_labels
        self.rel_labels = rel_labels
        
        # ✅ Check if "O" is in labels (for trained model with O support)
        self.has_O_label = self.O_LABEL in ent_labels
        if self.has_O_label:
            self.O_idx = ent_labels.index(self.O_LABEL)
            print(f"✅ Found 'O' label at index {self.O_idx}")
        else:
            print("⚠️  No 'O' label found - model may predict all spans as entities")
        
        def tokenize_list(labels):
            return self.tokenizer(labels, return_tensors="pt", padding=True, truncation=True).to(self.device)

        ent_inputs = tokenize_list(ent_labels)
        rel_inputs = tokenize_list(rel_labels)
        
        self.model.set_global_labels(
            ent_inputs["input_ids"].unsqueeze(0), ent_inputs["attention_mask"].unsqueeze(0),
            rel_inputs["input_ids"].unsqueeze(0), rel_inputs["attention_mask"].unsqueeze(0)
        )

    def _get_word_boundaries(self, text, offset_mapping):
        """
        หา word boundaries จาก offset_mapping
        Return: list of (word_text, char_start, char_end, token_start, token_end)
        
        Word คือช่วงที่ติดต่อกัน - ถ้ามี gap ใน character positions = คำใหม่
        """
        words = []
        current_word_tokens = []
        current_char_start = None
        prev_char_end = None
        
        for token_idx, (char_start, char_end) in enumerate(offset_mapping):
            # Skip special tokens (offset = (0, 0))
            if char_start == char_end == 0:
                continue
                
            # Check if this is a new word (gap in characters = space/punctuation)
            if prev_char_end is not None and char_start > prev_char_end:
                # Save previous word
                if current_word_tokens:
                    word_text = text[current_char_start:prev_char_end]
                    token_start = current_word_tokens[0]
                    token_end = current_word_tokens[-1]
                    words.append({
                        'text': word_text,
                        'char_start': current_char_start,
                        'char_end': prev_char_end,
                        'token_start': token_start,
                        'token_end': token_end
                    })
                # Start new word
                current_word_tokens = [token_idx]
                current_char_start = char_start
            else:
                # Continue current word
                if current_char_start is None:
                    current_char_start = char_start
                current_word_tokens.append(token_idx)
            
            prev_char_end = char_end
        
        # Don't forget the last word
        if current_word_tokens:
            word_text = text[current_char_start:prev_char_end]
            words.append({
                'text': word_text,
                'char_start': current_char_start,
                'char_end': prev_char_end,
                'token_start': current_word_tokens[0],
                'token_end': current_word_tokens[-1]
            })
        
        return words

    def _generate_word_spans(self, words, max_span_width=5):
        """
        สร้าง candidate spans จาก word combinations
        เช่น ["Elon", "Musk", "founded"] -> [("Elon",), ("Musk",), ("founded",), ("Elon Musk",), ("Musk founded",), ...]
        
        Return: list of dicts with combined word info
        """
        spans = []
        n_words = len(words)
        
        for width in range(1, min(max_span_width + 1, n_words + 1)):
            for start_idx in range(n_words - width + 1):
                end_idx = start_idx + width - 1
                
                # Combine words
                span_words = words[start_idx:end_idx + 1]
                span_text = ' '.join([w['text'] for w in span_words])
                
                # Clean up text (remove extra spaces)
                span_text = ' '.join(span_text.split())
                
                spans.append({
                    'text': span_text,
                    'char_start': span_words[0]['char_start'],
                    'char_end': span_words[-1]['char_end'],
                    'token_start': span_words[0]['token_start'],
                    'token_end': span_words[-1]['token_end'],
                    'word_indices': (start_idx, end_idx)
                })
        
        return spans

    def _nms(self, entities, prefer_shorter=True, max_preferred_len=3, score_tolerance=0.05):
        """
        Non-Maximum Suppression - keep best non-overlapping entities
        
        Strategy: 
        1. Group by overlap
        2. If scores within tolerance, prefer longer spans (e.g., "Elon Musk" > "Elon")
        3. Penalize very long spans (> max_preferred_len words)
        """
        if not entities:
            return []
        
        # Compute effective score with length penalty for very long spans
        for ent in entities:
            num_words = len(ent['text'].split())
            if num_words > max_preferred_len:
                # Heavy penalty for spans longer than max_preferred_len words
                ent['effective_score'] = ent['score'] * (max_preferred_len / num_words)
            else:
                ent['effective_score'] = ent['score']
        
        # Sort by: (1) effective score descending, (2) length descending (prefer longer when scores close)
        # ✅ Key insight: multiply effective_score by 100 then add length/100 
        # This way length only matters when scores are within 1%
        sorted_ents = sorted(
            entities, 
            key=lambda x: (round(x['effective_score'], 2), len(x['text'])),  # Round to 2 decimal, then longer text wins
            reverse=True
        )
        
        keep = []
        
        while sorted_ents:
            current = sorted_ents.pop(0)
            keep.append(current)
            
            c_start, c_end = current['char_start'], current['char_end']
            
            # Remove overlapping or containing entities
            non_overlapping = []
            for ent in sorted_ents:
                e_start, e_end = ent['char_start'], ent['char_end']
                
                # Check for any overlap: two spans overlap if one doesn't end before the other starts
                has_overlap = not (e_end <= c_start or e_start >= c_end)
                
                if not has_overlap:
                    non_overlapping.append(ent)
            
            sorted_ents = non_overlapping
        
        return sorted(keep, key=lambda x: x['char_start'])

    def predict(self, text, ent_threshold=0.5, rel_threshold=0.5, max_span_width=5, debug=False):
        """
        Main prediction function
        
        Args:
            text: Input text
            ent_threshold: Threshold for entity detection
            rel_threshold: Threshold for relation detection
            max_span_width: Maximum number of words in a span
            debug: If True, print raw scores
            
        Returns:
            dict with 'entities' and 'relations' suitable for Graph RAG
        """
        # 1. Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", return_offsets_mapping=True).to(self.device)
        input_ids = inputs["input_ids"]
        offset_mapping = inputs["offset_mapping"][0].tolist()
        
        # 2. Get word boundaries
        words = self._get_word_boundaries(text, offset_mapping)
        
        if not words:
            return {"entities": [], "relations": [], "triplets": []}
        
        # 3. Generate word-level candidate spans
        candidate_spans = self._generate_word_spans(words, max_span_width)
        
        if not candidate_spans:
            return {"entities": [], "relations": [], "triplets": []}
        
        # 4. Prepare model spans (token indices)
        model_spans = [(s['token_start'], s['token_end']) for s in candidate_spans]
        
        # 5. Entity Recognition
        with torch.no_grad():
            e_logits, _ = self.model(
                input_ids, inputs["attention_mask"],
                entity_spans=[model_spans]
            )
        
        # 6. Filter entities by threshold
        # ✅ ใช้ softmax แทน sigmoid เพราะเทรนด้วย CrossEntropyLoss
        probs = torch.softmax(e_logits[0], dim=-1)
        
        found_entities = []
        if debug:
            print("\n🔍 DEBUG: Top candidate spans:")
            
        for idx, span in enumerate(candidate_spans):
            scores = probs[idx]
            
            if self.has_O_label:
                # ✅ วิธีใหม่: แยก O score ออก แล้วหา max จาก entity scores เท่านั้น
                O_score = scores[self.O_idx].item()
                
                # ✅ ถ้า O score สูงมาก (> 0.5) = ไม่ใช่ entity แน่นอน, skip เลย
                if O_score > 0.5:
                    if debug:
                        max_ent_score, max_ent_idx = torch.max(scores, dim=0)
                        # Find best non-O label for debug
                        entity_scores = scores.clone()
                        entity_scores[self.O_idx] = -1e9
                        best_ent_score, best_ent_idx = torch.max(entity_scores, dim=0)
                        if best_ent_score.item() > 0.01:
                            print(f"   - '{span['text']}': O ({O_score:.4f}) [SKIP]")
                    continue
                
                # Mask out O score and find max among entity labels only
                entity_scores = scores.clone()
                entity_scores[self.O_idx] = -1e9  # Mask O
                
                max_ent_score, max_ent_idx = torch.max(entity_scores, dim=0)
                label_name = self.ent_labels[max_ent_idx.item()]
                
                if debug and max_ent_score.item() > 0.01:
                    print(f"   - '{span['text']}': {label_name} ({max_ent_score.item():.4f}) vs O ({O_score:.4f})")
                
                # Only add if entity score is above threshold
                if max_ent_score.item() > ent_threshold:
                    found_entities.append({
                        "id": len(found_entities),
                        "text": span['text'],
                        "label": label_name,
                        "score": max_ent_score.item(),
                        "char_start": span['char_start'],
                        "char_end": span['char_end'],
                        "token_span": (span['token_start'], span['token_end'])
                    })
            else:
                # Fallback: No O label (old behavior)
                max_score, label_idx = torch.max(scores, dim=0)
                label_name = self.ent_labels[label_idx.item()]
                
                if debug and max_score.item() > 0.01:
                    print(f"   - '{span['text']}': {label_name} ({max_score.item():.4f})")
                    
                if max_score.item() > ent_threshold:
                    found_entities.append({
                        "id": len(found_entities),
                        "text": span['text'],
                        "label": label_name,
                        "score": max_score.item(),
                        "char_start": span['char_start'],
                        "char_end": span['char_end'],
                        "token_span": (span['token_start'], span['token_end'])
                    })
        
        # 7. Apply NMS to remove overlapping entities
        found_entities = self._nms(found_entities)
        
        # 7.5 Normalize text (remove trailing punctuation) and deduplicate
        for ent in found_entities:
            # Strip trailing punctuation like . , ! ?
            ent['text'] = ent['text'].rstrip('.,!?;:')
        
        # Deduplicate by (text, label) - keep highest score
        seen_entities = {}
        for ent in found_entities:
            key = (ent['text'].lower(), ent['label'])
            if key not in seen_entities or ent['score'] > seen_entities[key]['score']:
                seen_entities[key] = ent
        found_entities = list(seen_entities.values())
        
        # 7.6 Remove substrings: if "Elon" and "Elon Musk" both exist, keep only "Elon Musk"
        entity_texts = [(ent['text'].lower(), ent) for ent in found_entities]
        entities_to_remove = set()
        for text1, ent1 in entity_texts:
            for text2, ent2 in entity_texts:
                if text1 != text2 and text1 in text2 and ent1['label'] == ent2['label']:
                    # text1 is substring of text2, mark text1 for removal
                    entities_to_remove.add(text1)
        
        found_entities = [ent for ent in found_entities if ent['text'].lower() not in entities_to_remove]
        
        # Re-assign IDs after NMS
        for i, ent in enumerate(found_entities):
            ent['id'] = i
        
        # 8. Relation Extraction (if >= 2 entities)
        found_relations = []
        triplets = []  # For Graph RAG: (subject, relation, object)
        
        if len(found_entities) >= 2:
            final_spans = [e["token_span"] for e in found_entities]
            
            # Create all possible pairs
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
            
            if r_logits is not None:
                r_probs = torch.sigmoid(r_logits[0])
                
                # ✅ Define valid relation constraints based on entity types
                # Format: relation -> (allowed_subject_types, allowed_object_types)
                valid_relations = {
                    'founder_of': (['person'], ['organisation']),
                    'ceo_of': (['person'], ['organisation']),
                    'developed': (['person', 'organisation'], ['product', 'programlang']),
                    'creator_of': (['person', 'organisation'], ['product', 'programlang']),
                    'founded_in': (['organisation', 'product'], ['date', 'location']),
                    'released_in': (['product', 'programlang'], ['date', 'location']),
                    'located_in': (['organisation', 'person'], ['location']),
                }
                
                for idx, (subj_idx, obj_idx) in enumerate(final_pairs):
                    scores = r_probs[idx]
                    max_score, label_idx = torch.max(scores, dim=0)
                    
                    if max_score.item() > rel_threshold:
                        subj = found_entities[subj_idx]
                        obj = found_entities[obj_idx]
                        rel_label = self.rel_labels[label_idx.item()]
                        
                        # ✅ Validate relation against entity types
                        if rel_label in valid_relations:
                            allowed_subj, allowed_obj = valid_relations[rel_label]
                            if subj['label'] not in allowed_subj or obj['label'] not in allowed_obj:
                                continue  # Skip invalid relation
                        
                        found_relations.append({
                            "subject_id": subj_idx,
                            "object_id": obj_idx,
                            "subject": subj["text"],
                            "subject_label": subj["label"],
                            "object": obj["text"],
                            "object_label": obj["label"],
                            "relation": rel_label,
                            "score": max_score.item()
                        })
                        
                        # Graph RAG triplet format
                        triplets.append({
                            "head": subj["text"],
                            "head_type": subj["label"],
                            "relation": rel_label,
                            "tail": obj["text"],
                            "tail_type": obj["label"],
                            "confidence": max_score.item()
                        })
        
        # 9. Deduplicate triplets by (head, relation, tail) - keep highest confidence
        seen_triplets = {}
        for t in triplets:
            key = (t['head'].lower(), t['relation'], t['tail'].lower())
            if key not in seen_triplets or t['confidence'] > seen_triplets[key]['confidence']:
                seen_triplets[key] = t
        triplets = list(seen_triplets.values())
        
        return {
            "entities": found_entities,
            "relations": found_relations,
            "triplets": triplets  # Ready for Graph RAG
        }
    
    def extract_knowledge_graph(self, text, ent_threshold=0.5, rel_threshold=0.5):
        """
        Extract knowledge graph in a format ready for Neo4j or NetworkX
        
        Returns:
            nodes: list of unique entities
            edges: list of relations between entities
        """
        result = self.predict(text, ent_threshold, rel_threshold)
        
        # Deduplicate nodes by text
        nodes_dict = {}
        for ent in result["entities"]:
            key = ent["text"].lower()
            if key not in nodes_dict or ent["score"] > nodes_dict[key]["score"]:
                nodes_dict[key] = {
                    "id": ent["text"],
                    "label": ent["label"],
                    "score": ent["score"]
                }
        
        nodes = list(nodes_dict.values())
        
        # Edges
        edges = []
        for rel in result["relations"]:
            edges.append({
                "source": rel["subject"],
                "target": rel["object"],
                "relation": rel["relation"],
                "score": rel["score"]
            })
        
        return {"nodes": nodes, "edges": edges}


# --- Example Usage ---
if __name__ == "__main__":
    inference = NERREInference("saved_model_v8")
    
    text = "Elon Musk founded SpaceX in 2002. Tesla was also founded by Elon Musk."
    
    print(f"\nInput: {text}")
    print("=" * 60)
    
    # ใช้ threshold ต่ำกว่าเพื่อทดสอบ model ที่เพิ่ง train
    # ✅ Enable debug mode
    # ✅ max_span_width=3 เพื่อจับชื่อหลายคำอย่าง "Elon Musk" แต่ไม่ยาวเกินไป
    result = inference.predict(text, ent_threshold=0.7, rel_threshold=0.8, max_span_width=3, debug=True)
    
    print("\n📌 Entities Found:")
    if result["entities"]:
        for ent in result["entities"]:
            print(f"  [{ent['label']}] {ent['text']} (score: {ent['score']:.3f})")
    else:
        print("  (No entities - model needs training!)")
    
    print("\n🔗 Relations Found:")
    if result["relations"]:
        for rel in result["relations"]:
            print(f"  {rel['subject']} --[{rel['relation']}]--> {rel['object']} (score: {rel['score']:.3f})")
    else:
        print("  (No relations detected above threshold)")
    
    print("\n📊 Knowledge Graph Triplets (for Graph RAG):")
    if result["triplets"]:
        for t in result["triplets"]:
            print(f"  ({t['head']}, {t['relation']}, {t['tail']})")
    else:
        print("  (No triplets - model needs training!)")
    
    print("\n" + "=" * 60)
    
    # ทดสอบประโยคที่ 2
    text2 = "Mark Zuckerberg is the CEO of Meta. He also created Facebook in 2004."
    print(f"\nInput: {text2}")
    print("=" * 60)
    result2 = inference.predict(text2, ent_threshold=0.7, rel_threshold=0.8, max_span_width=3, debug=False)
    
    print("\n📌 Entities Found:")
    for ent in result2["entities"]:
        print(f"  [{ent['label']}] {ent['text']} (score: {ent['score']:.3f})")
    
    print("\n📊 Knowledge Graph Triplets:")
    for t in result2["triplets"]:
        print(f"  ({t['head']}, {t['relation']}, {t['tail']})")
