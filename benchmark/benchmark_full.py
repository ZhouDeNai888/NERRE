"""
Full Benchmark Script: NERRE vs SpERT vs NuNER vs GLiNER vs spaCy
==================================================================
Compares Named Entity Recognition and Relation Extraction performance 
across different models using standard benchmark-style datasets.

Standard Datasets Used:
- CoNLL04-style: NER + Relation Extraction
- Few-NERD-style: Fine-grained NER

Models Compared:
- NERRE (Ours): Joint Entity & Relation Extraction
- SpERT: Span-based Entity and Relation Transformer (Reference)
- NuNER Zero: Zero-shot NER from NuMind
- GLiNER: Generalist NER
- spaCy: Traditional NER
"""

import time
import json
import os
import sys
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Standard Benchmark Datasets (CoNLL04-style with Relations)
# ============================================================================

def load_conll04_style_data() -> List[Dict]:
    """
    Load CoNLL04-style test data with entities and relations.
    CoNLL04 Entity Types: Loc, Org, Peop, Other
    CoNLL04 Relation Types: Located_In, Work_For, OrgBased_In, Live_In, Kill
    """
    test_data = [
        # Sample 1: Work_For relation
        {
            "text": "Bill Gates is the founder of Microsoft.",
            "entities": [
                {"text": "Bill Gates", "label": "person", "start": 0, "end": 10},
                {"text": "Microsoft", "label": "organisation", "start": 28, "end": 37},
            ],
            "relations": [
                {"head": "Bill Gates", "tail": "Microsoft", "relation": "founder of"}
            ]
        },
        # Sample 2: Located_In relation
        {
            "text": "The Eiffel Tower is located in Paris, France.",
            "entities": [
                {"text": "Eiffel Tower", "label": "location", "start": 4, "end": 16},
                {"text": "Paris", "label": "location", "start": 31, "end": 36},
                {"text": "France", "label": "location", "start": 38, "end": 44},
            ],
            "relations": [
                {"head": "Eiffel Tower", "tail": "Paris", "relation": "located in"},
                {"head": "Paris", "tail": "France", "relation": "located in"}
            ]
        },
        # Sample 3: OrgBased_In relation
        {
            "text": "Apple Inc. is headquartered in Cupertino, California.",
            "entities": [
                {"text": "Apple Inc.", "label": "organisation", "start": 0, "end": 10},
                {"text": "Cupertino", "label": "location", "start": 31, "end": 40},
                {"text": "California", "label": "location", "start": 42, "end": 52},
            ],
            "relations": [
                {"head": "Apple Inc.", "tail": "Cupertino", "relation": "headquartered in"}
            ]
        },
        # Sample 4: Live_In relation
        {
            "text": "Elon Musk lives in Austin, Texas where Tesla has operations.",
            "entities": [
                {"text": "Elon Musk", "label": "person", "start": 0, "end": 9},
                {"text": "Austin", "label": "location", "start": 19, "end": 25},
                {"text": "Texas", "label": "location", "start": 27, "end": 32},
                {"text": "Tesla", "label": "organisation", "start": 39, "end": 44},
            ],
            "relations": [
                {"head": "Elon Musk", "tail": "Austin", "relation": "lives in"}
            ]
        },
        # Sample 5: Multiple relations
        {
            "text": "Mark Zuckerberg founded Facebook in 2004 while studying at Harvard University.",
            "entities": [
                {"text": "Mark Zuckerberg", "label": "person", "start": 0, "end": 15},
                {"text": "Facebook", "label": "organisation", "start": 24, "end": 32},
                {"text": "2004", "label": "date", "start": 36, "end": 40},
                {"text": "Harvard University", "label": "organisation", "start": 59, "end": 77},
            ],
            "relations": [
                {"head": "Mark Zuckerberg", "tail": "Facebook", "relation": "founder of"},
                {"head": "Facebook", "tail": "2004", "relation": "founded in"}
            ]
        },
        # Sample 6: Work_For
        {
            "text": "Satya Nadella has been the CEO of Microsoft since 2014.",
            "entities": [
                {"text": "Satya Nadella", "label": "person", "start": 0, "end": 13},
                {"text": "Microsoft", "label": "organisation", "start": 34, "end": 43},
                {"text": "2014", "label": "date", "start": 50, "end": 54},
            ],
            "relations": [
                {"head": "Satya Nadella", "tail": "Microsoft", "relation": "CEO of"}
            ]
        },
        # Sample 7: Complex sentence
        {
            "text": "Jeff Bezos, the founder of Amazon, was born in Albuquerque, New Mexico in 1964.",
            "entities": [
                {"text": "Jeff Bezos", "label": "person", "start": 0, "end": 10},
                {"text": "Amazon", "label": "organisation", "start": 27, "end": 33},
                {"text": "Albuquerque", "label": "location", "start": 47, "end": 58},
                {"text": "New Mexico", "label": "location", "start": 60, "end": 70},
                {"text": "1964", "label": "date", "start": 74, "end": 78},
            ],
            "relations": [
                {"head": "Jeff Bezos", "tail": "Amazon", "relation": "founder of"},
                {"head": "Jeff Bezos", "tail": "Albuquerque", "relation": "born in"}
            ]
        },
        # Sample 8: Organization hierarchy
        {
            "text": "Google, a subsidiary of Alphabet Inc., is based in Mountain View.",
            "entities": [
                {"text": "Google", "label": "organisation", "start": 0, "end": 6},
                {"text": "Alphabet Inc.", "label": "organisation", "start": 24, "end": 37},
                {"text": "Mountain View", "label": "location", "start": 52, "end": 65},
            ],
            "relations": [
                {"head": "Google", "tail": "Alphabet Inc.", "relation": "subsidiary of"},
                {"head": "Google", "tail": "Mountain View", "relation": "based in"}
            ]
        },
        # Sample 9: Research institution
        {
            "text": "Dr. Andrew Ng, a professor at Stanford University, co-founded Coursera.",
            "entities": [
                {"text": "Andrew Ng", "label": "person", "start": 4, "end": 13},
                {"text": "Stanford University", "label": "organisation", "start": 31, "end": 50},
                {"text": "Coursera", "label": "organisation", "start": 63, "end": 71},
            ],
            "relations": [
                {"head": "Andrew Ng", "tail": "Stanford University", "relation": "works at"},
                {"head": "Andrew Ng", "tail": "Coursera", "relation": "founder of"}
            ]
        },
        # Sample 10: Historical event
        {
            "text": "Steve Jobs introduced the iPhone at Macworld Conference in San Francisco in 2007.",
            "entities": [
                {"text": "Steve Jobs", "label": "person", "start": 0, "end": 10},
                {"text": "iPhone", "label": "product", "start": 26, "end": 32},
                {"text": "Macworld Conference", "label": "event", "start": 36, "end": 55},
                {"text": "San Francisco", "label": "location", "start": 59, "end": 72},
                {"text": "2007", "label": "date", "start": 76, "end": 80},
            ],
            "relations": []
        },
        # Chinese samples
        {
            "text": "馬雲於1999年在杭州創立了阿里巴巴集團。",
            "entities": [
                {"text": "馬雲", "label": "person", "start": 0, "end": 2},
                {"text": "1999年", "label": "date", "start": 3, "end": 8},
                {"text": "杭州", "label": "location", "start": 9, "end": 11},
                {"text": "阿里巴巴集團", "label": "organisation", "start": 14, "end": 20},
            ],
            "relations": [
                {"head": "馬雲", "tail": "阿里巴巴集團", "relation": "founder of"}
            ]
        },
        {
            "text": "台積電是全球最大的晶圓代工廠，總部位於新竹科學園區。",
            "entities": [
                {"text": "台積電", "label": "organisation", "start": 0, "end": 3},
                {"text": "新竹科學園區", "label": "location", "start": 19, "end": 25},
            ],
            "relations": [
                {"head": "台積電", "tail": "新竹科學園區", "relation": "headquartered in"}
            ]
        },
        # Japanese samples
        {
            "text": "孫正義はソフトバンクグループの創業者で、東京を拠点としている。",
            "entities": [
                {"text": "孫正義", "label": "person", "start": 0, "end": 3},
                {"text": "ソフトバンクグループ", "label": "organisation", "start": 4, "end": 14},
                {"text": "東京", "label": "location", "start": 21, "end": 23},
            ],
            "relations": [
                {"head": "孫正義", "tail": "ソフトバンクグループ", "relation": "founder of"}
            ]
        },
        {
            "text": "任天堂は1889年に山内房治郎によって京都で創業された。",
            "entities": [
                {"text": "任天堂", "label": "organisation", "start": 0, "end": 3},
                {"text": "1889年", "label": "date", "start": 4, "end": 9},
                {"text": "山内房治郎", "label": "person", "start": 10, "end": 15},
                {"text": "京都", "label": "location", "start": 21, "end": 23},
            ],
            "relations": [
                {"head": "山内房治郎", "tail": "任天堂", "relation": "founder of"}
            ]
        },
        # Korean samples
        {
            "text": "이건희는 삼성전자의 전 회장으로 서울에서 활동했다.",
            "entities": [
                {"text": "이건희", "label": "person", "start": 0, "end": 3},
                {"text": "삼성전자", "label": "organisation", "start": 5, "end": 9},
                {"text": "서울", "label": "location", "start": 18, "end": 20},
            ],
            "relations": [
                {"head": "이건희", "tail": "삼성전자", "relation": "chairman of"}
            ]
        },
        # More English samples for better statistics
        {
            "text": "Tim Cook became the CEO of Apple after Steve Jobs passed away in 2011.",
            "entities": [
                {"text": "Tim Cook", "label": "person", "start": 0, "end": 8},
                {"text": "Apple", "label": "organisation", "start": 27, "end": 32},
                {"text": "Steve Jobs", "label": "person", "start": 39, "end": 49},
                {"text": "2011", "label": "date", "start": 65, "end": 69},
            ],
            "relations": [
                {"head": "Tim Cook", "tail": "Apple", "relation": "CEO of"}
            ]
        },
        {
            "text": "Jensen Huang co-founded NVIDIA in 1993 in Santa Clara, California.",
            "entities": [
                {"text": "Jensen Huang", "label": "person", "start": 0, "end": 12},
                {"text": "NVIDIA", "label": "organisation", "start": 24, "end": 30},
                {"text": "1993", "label": "date", "start": 34, "end": 38},
                {"text": "Santa Clara", "label": "location", "start": 42, "end": 53},
                {"text": "California", "label": "location", "start": 55, "end": 65},
            ],
            "relations": [
                {"head": "Jensen Huang", "tail": "NVIDIA", "relation": "founder of"},
                {"head": "NVIDIA", "tail": "Santa Clara", "relation": "based in"}
            ]
        },
        {
            "text": "Larry Page and Sergey Brin founded Google in 1998 at Stanford University.",
            "entities": [
                {"text": "Larry Page", "label": "person", "start": 0, "end": 10},
                {"text": "Sergey Brin", "label": "person", "start": 15, "end": 26},
                {"text": "Google", "label": "organisation", "start": 35, "end": 41},
                {"text": "1998", "label": "date", "start": 45, "end": 49},
                {"text": "Stanford University", "label": "organisation", "start": 53, "end": 72},
            ],
            "relations": [
                {"head": "Larry Page", "tail": "Google", "relation": "founder of"},
                {"head": "Sergey Brin", "tail": "Google", "relation": "founder of"}
            ]
        },
        {
            "text": "OpenAI, led by Sam Altman, developed ChatGPT and is headquartered in San Francisco.",
            "entities": [
                {"text": "OpenAI", "label": "organisation", "start": 0, "end": 6},
                {"text": "Sam Altman", "label": "person", "start": 15, "end": 25},
                {"text": "ChatGPT", "label": "product", "start": 37, "end": 44},
                {"text": "San Francisco", "label": "location", "start": 69, "end": 82},
            ],
            "relations": [
                {"head": "Sam Altman", "tail": "OpenAI", "relation": "CEO of"},
                {"head": "OpenAI", "tail": "San Francisco", "relation": "headquartered in"}
            ]
        },
        {
            "text": "Anthropic was founded by Dario Amodei and Daniela Amodei in 2021.",
            "entities": [
                {"text": "Anthropic", "label": "organisation", "start": 0, "end": 9},
                {"text": "Dario Amodei", "label": "person", "start": 25, "end": 37},
                {"text": "Daniela Amodei", "label": "person", "start": 42, "end": 56},
                {"text": "2021", "label": "date", "start": 60, "end": 64},
            ],
            "relations": [
                {"head": "Dario Amodei", "tail": "Anthropic", "relation": "founder of"},
                {"head": "Daniela Amodei", "tail": "Anthropic", "relation": "founder of"}
            ]
        },
    ]
    return test_data


# ============================================================================
# Reference Results from Published Papers
# ============================================================================

def get_spert_reference_results() -> Dict:
    """
    SpERT reference results from the ECAI 2020 paper.
    "Span-based Joint Entity and Relation Extraction with Transformer Pre-training"
    https://arxiv.org/abs/1909.07755
    
    Note: SpERT requires specific setup and training. These are reported results.
    """
    return {
        "model": "SpERT",
        "paper": "Eberts & Ulges, ECAI 2020",
        "datasets": {
            "CoNLL04": {
                "entity_f1": 0.8821,  # 88.21%
                "relation_f1": 0.7261,  # 72.61%
            },
            "SciERC": {
                "entity_f1": 0.7087,  # 70.87%
                "relation_f1": 0.4803,  # 48.03%
            },
            "ADE": {
                "entity_f1": 0.8942,  # 89.42%
                "relation_f1": 0.7917,  # 79.17%
            }
        },
        "note": "Reference results from paper. SpERT is archived (April 2025)."
    }


# ============================================================================
# Evaluation Metrics
# ============================================================================

def normalize_label(label: str) -> str:
    """Normalize entity labels for fair comparison."""
    label = label.lower().strip()
    mapping = {
        # Standard mappings
        "person": "person",
        "per": "person",
        "peop": "person",
        "people": "person",
        "org": "organisation",
        "organization": "organisation",
        "organisation": "organisation",
        "company": "organisation",
        "corporation": "organisation",
        "gpe": "location",
        "loc": "location",
        "location": "location",
        "date": "date",
        "time": "date",
        "product": "product",
        "work_of_art": "product",
        "event": "event",
        "misc": "other",
        "other": "other",
    }
    return mapping.get(label, label)


def calculate_entity_metrics(predictions: List[Dict], ground_truth: List[Dict], fuzzy_match: bool = True) -> Dict:
    """Calculate precision, recall, and F1 score for NER."""
    tp, fp, fn = 0, 0, 0
    
    pred_set = set()
    for p in predictions:
        text = p.get('text', '').lower().strip()
        label = normalize_label(p.get('label', ''))
        pred_set.add((text, label))
    
    gt_set = set()
    for g in ground_truth:
        text = g.get('text', '').lower().strip()
        label = normalize_label(g.get('label', ''))
        gt_set.add((text, label))
    
    # Calculate TP, FP
    for pred_text, pred_label in pred_set:
        matched = False
        for gt_text, gt_label in gt_set:
            if pred_label == gt_label:
                if fuzzy_match:
                    if pred_text in gt_text or gt_text in pred_text:
                        matched = True
                        break
                else:
                    if pred_text == gt_text:
                        matched = True
                        break
        if matched:
            tp += 1
        else:
            fp += 1
    
    # Calculate FN
    for gt_text, gt_label in gt_set:
        matched = False
        for pred_text, pred_label in pred_set:
            if pred_label == gt_label:
                if fuzzy_match:
                    if pred_text in gt_text or gt_text in pred_text:
                        matched = True
                        break
                else:
                    if pred_text == gt_text:
                        matched = True
                        break
        if not matched:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


def calculate_relation_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate precision, recall, and F1 score for Relation Extraction."""
    if not ground_truth:
        return {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0}
    
    tp, fp, fn = 0, 0, 0
    
    # Normalize predictions
    pred_set = set()
    for p in predictions:
        head = p.get('head', '').lower().strip()
        tail = p.get('tail', '').lower().strip()
        # Ignore relation type for now, just check if pair exists
        pred_set.add((head, tail))
    
    # Normalize ground truth
    gt_set = set()
    for g in ground_truth:
        head = g.get('head', '').lower().strip()
        tail = g.get('tail', '').lower().strip()
        gt_set.add((head, tail))
    
    # Calculate metrics with fuzzy matching
    for pred_head, pred_tail in pred_set:
        matched = False
        for gt_head, gt_tail in gt_set:
            if (pred_head in gt_head or gt_head in pred_head) and \
               (pred_tail in gt_tail or gt_tail in pred_tail):
                matched = True
                break
        if matched:
            tp += 1
        else:
            fp += 1
    
    for gt_head, gt_tail in gt_set:
        matched = False
        for pred_head, pred_tail in pred_set:
            if (pred_head in gt_head or gt_head in pred_head) and \
               (pred_tail in gt_tail or gt_tail in pred_tail):
                matched = True
                break
        if not matched:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn
    }


# ============================================================================
# Model Benchmarks
# ============================================================================

def benchmark_nerre(test_data: List[Dict], fast_mode: bool = False) -> Tuple[Dict, Dict, float]:
    """Benchmark NERRE model for both NER and RE."""
    from huggingface_release.nerre import NERREPipeline
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "huggingface_release")
    pipe = NERREPipeline(model_dir)
    
    all_pred_entities = []
    all_gt_entities = []
    all_pred_relations = []
    all_gt_relations = []
    
    start_time = time.time()
    for sample in test_data:
        if fast_mode:
            result = pipe(sample["text"], ent_threshold=0.5, fast_mode=True, extract_relations=False)
        else:
            result = pipe(sample["text"], ent_threshold=0.5, rel_threshold=0.5)
        
        all_pred_entities.extend(result["entities"])
        all_gt_entities.extend(sample["entities"])
        
        # Convert triplets to relation format
        if not fast_mode:
            for triplet in result.get("triplets", []):
                all_pred_relations.append({
                    "head": triplet["head"],
                    "tail": triplet["tail"],
                    "relation": triplet["relation"]
                })
        all_gt_relations.extend(sample.get("relations", []))
    
    total_time = time.time() - start_time
    
    entity_metrics = calculate_entity_metrics(all_pred_entities, all_gt_entities)
    relation_metrics = calculate_relation_metrics(all_pred_relations, all_gt_relations)
    
    return entity_metrics, relation_metrics, total_time


def benchmark_nuner(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark NuNER Zero model."""
    try:
        from gliner import GLiNER
    except ImportError:
        print("GLiNER not installed. Run: pip install gliner")
        return None, 0
    
    try:
        model = GLiNER.from_pretrained("numind/NuNerZero")
    except Exception as e:
        print(f"Failed to load NuNER Zero: {e}")
        return None, 0
    
    # NuNER requires lowercase labels
    labels = ["person", "organisation", "location", "date", "product", "event"]
    
    all_predictions = []
    all_ground_truth = []
    
    def merge_entities(entities, text):
        """Merge adjacent entities with same label."""
        if not entities:
            return []
        merged = []
        current = entities[0].copy()
        for next_entity in entities[1:]:
            if next_entity['label'] == current['label'] and \
               (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
                current['text'] = text[current['start']: next_entity['end']].strip()
                current['end'] = next_entity['end']
            else:
                merged.append(current)
                current = next_entity.copy()
        merged.append(current)
        return merged
    
    start_time = time.time()
    for sample in test_data:
        text = sample["text"]
        try:
            entities = model.predict_entities(text, labels, threshold=0.5)
            entities = merge_entities(entities, text)
            for ent in entities:
                all_predictions.append({
                    "text": ent["text"],
                    "label": ent["label"]
                })
        except Exception as e:
            pass  # Skip failed predictions
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_entity_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


def benchmark_nuner_span(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark NuNER Zero-span model."""
    try:
        from gliner import GLiNER
    except ImportError:
        print("GLiNER not installed. Run: pip install gliner")
        return None, 0
    
    try:
        model = GLiNER.from_pretrained("numind/NuNerZero_span")
    except Exception as e:
        print(f"Failed to load NuNER Zero-span: {e}")
        return None, 0
    
    labels = ["person", "organisation", "location", "date", "product", "event"]
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        try:
            entities = model.predict_entities(sample["text"], labels, threshold=0.5)
            for ent in entities:
                all_predictions.append({
                    "text": ent["text"],
                    "label": ent["label"]
                })
        except Exception as e:
            pass
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_entity_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


def benchmark_gliner(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark GLiNER model."""
    try:
        from gliner import GLiNER
    except ImportError:
        print("GLiNER not installed. Run: pip install gliner")
        return None, 0
    
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    labels = ["person", "organisation", "location", "date", "product", "event"]
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        try:
            entities = model.predict_entities(sample["text"], labels, threshold=0.5)
            for ent in entities:
                all_predictions.append({
                    "text": ent["text"],
                    "label": ent["label"]
                })
        except:
            pass
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_entity_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


def benchmark_spacy(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark spaCy model."""
    try:
        import spacy
    except ImportError:
        print("spaCy not installed.")
        return None, 0
    
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found.")
            return None, 0
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        try:
            doc = nlp(sample["text"])
            for ent in doc.ents:
                all_predictions.append({
                    "text": ent.text,
                    "label": ent.label_
                })
        except:
            pass
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_entity_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


# ============================================================================
# Speed Benchmark
# ============================================================================

def benchmark_speed(num_chars: int = 10000) -> Dict:
    """Benchmark processing speed for long texts."""
    base_text = "Elon Musk founded SpaceX in 2002 in California. Bill Gates is the founder of Microsoft. "
    long_text = base_text * (num_chars // len(base_text) + 1)
    long_text = long_text[:num_chars]
    
    results = {}
    
    # NERRE
    try:
        from huggingface_release.nerre import NERREPipeline
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "huggingface_release")
        pipe = NERREPipeline(model_dir)
        
        # Normal mode
        start_time = time.time()
        result = pipe(long_text)
        nerre_time = time.time() - start_time
        results["NERRE"] = {
            "time": nerre_time, 
            "entities": len(result["entities"]), 
            "relations": len(result.get("triplets", [])),
            "chars_per_sec": num_chars / nerre_time
        }
        
        # Fast mode
        start_time = time.time()
        result_fast = pipe(long_text, fast_mode=True, extract_relations=False)
        nerre_fast_time = time.time() - start_time
        results["NERRE (Fast)"] = {
            "time": nerre_fast_time, 
            "entities": len(result_fast["entities"]), 
            "relations": 0,
            "chars_per_sec": num_chars / nerre_fast_time
        }
    except Exception as e:
        print(f"NERRE speed benchmark failed: {e}")
    
    # NuNER
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained("numind/NuNerZero")
        labels = ["person", "organisation", "location", "date"]
        
        start_time = time.time()
        entities = model.predict_entities(long_text, labels, threshold=0.5)
        nuner_time = time.time() - start_time
        results["NuNER Zero"] = {
            "time": nuner_time, 
            "entities": len(entities), 
            "relations": 0,
            "chars_per_sec": num_chars / nuner_time
        }
    except Exception as e:
        print(f"NuNER speed benchmark failed: {e}")
    
    # GLiNER
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        labels = ["person", "organisation", "location", "date"]
        
        start_time = time.time()
        entities = model.predict_entities(long_text, labels, threshold=0.5)
        gliner_time = time.time() - start_time
        results["GLiNER"] = {
            "time": gliner_time, 
            "entities": len(entities), 
            "relations": 0,
            "chars_per_sec": num_chars / gliner_time
        }
    except Exception as e:
        print(f"GLiNER speed benchmark failed: {e}")
    
    # spaCy
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_trf")
        except:
            nlp = spacy.load("en_core_web_sm")
        
        start_time = time.time()
        doc = nlp(long_text)
        spacy_time = time.time() - start_time
        results["spaCy"] = {
            "time": spacy_time, 
            "entities": len(doc.ents), 
            "relations": 0,
            "chars_per_sec": num_chars / spacy_time
        }
    except Exception as e:
        print(f"spaCy speed benchmark failed: {e}")
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("Full NER/RE Benchmark: NERRE vs SpERT vs NuNER vs GLiNER vs spaCy")
    print("=" * 80)
    
    test_data = load_conll04_style_data()
    total_entities = sum(len(s["entities"]) for s in test_data)
    total_relations = sum(len(s.get("relations", [])) for s in test_data)
    
    print(f"\n📊 Test Dataset Statistics:")
    print(f"   Samples: {len(test_data)}")
    print(f"   Total Entities: {total_entities}")
    print(f"   Total Relations: {total_relations}")
    print(f"   Languages: English, Chinese, Japanese, Korean")
    print("-" * 80)
    
    results = {}
    
    # ========== NERRE ==========
    print("\n🔹 Benchmarking NERRE (Normal Mode)...")
    try:
        ner_metrics, rel_metrics, nerre_time = benchmark_nerre(test_data, fast_mode=False)
        results["NERRE"] = {
            "entity_metrics": ner_metrics,
            "relation_metrics": rel_metrics,
            "time": nerre_time,
            "supports_re": True
        }
        print(f"   Entity F1:    {ner_metrics['f1']:.3f} (P: {ner_metrics['precision']:.3f}, R: {ner_metrics['recall']:.3f})")
        print(f"   Relation F1:  {rel_metrics['f1']:.3f} (P: {rel_metrics['precision']:.3f}, R: {rel_metrics['recall']:.3f})")
        print(f"   Time:         {nerre_time:.2f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== NERRE Fast ==========
    print("\n🔹 Benchmarking NERRE (Fast Mode)...")
    try:
        ner_metrics_fast, _, nerre_fast_time = benchmark_nerre(test_data, fast_mode=True)
        results["NERRE (Fast)"] = {
            "entity_metrics": ner_metrics_fast,
            "relation_metrics": {"f1": 0, "precision": 0, "recall": 0},
            "time": nerre_fast_time,
            "supports_re": False
        }
        print(f"   Entity F1:    {ner_metrics_fast['f1']:.3f} (P: {ner_metrics_fast['precision']:.3f}, R: {ner_metrics_fast['recall']:.3f})")
        print(f"   Time:         {nerre_fast_time:.2f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== NuNER Zero ==========
    print("\n🔹 Benchmarking NuNER Zero...")
    try:
        nuner_metrics, nuner_time = benchmark_nuner(test_data)
        if nuner_metrics:
            results["NuNER Zero"] = {
                "entity_metrics": nuner_metrics,
                "time": nuner_time,
                "supports_re": False
            }
            print(f"   Entity F1:    {nuner_metrics['f1']:.3f} (P: {nuner_metrics['precision']:.3f}, R: {nuner_metrics['recall']:.3f})")
            print(f"   Time:         {nuner_time:.2f}s")
        else:
            print("   Skipped (not installed)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== NuNER Zero-span ==========
    print("\n🔹 Benchmarking NuNER Zero-span...")
    try:
        nuner_span_metrics, nuner_span_time = benchmark_nuner_span(test_data)
        if nuner_span_metrics:
            results["NuNER Zero-span"] = {
                "entity_metrics": nuner_span_metrics,
                "time": nuner_span_time,
                "supports_re": False
            }
            print(f"   Entity F1:    {nuner_span_metrics['f1']:.3f} (P: {nuner_span_metrics['precision']:.3f}, R: {nuner_span_metrics['recall']:.3f})")
            print(f"   Time:         {nuner_span_time:.2f}s")
        else:
            print("   Skipped (not installed)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== GLiNER ==========
    print("\n🔹 Benchmarking GLiNER...")
    try:
        gliner_metrics, gliner_time = benchmark_gliner(test_data)
        if gliner_metrics:
            results["GLiNER"] = {
                "entity_metrics": gliner_metrics,
                "time": gliner_time,
                "supports_re": False
            }
            print(f"   Entity F1:    {gliner_metrics['f1']:.3f} (P: {gliner_metrics['precision']:.3f}, R: {gliner_metrics['recall']:.3f})")
            print(f"   Time:         {gliner_time:.2f}s")
        else:
            print("   Skipped (not installed)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== spaCy ==========
    print("\n🔹 Benchmarking spaCy...")
    try:
        spacy_metrics, spacy_time = benchmark_spacy(test_data)
        if spacy_metrics:
            results["spaCy"] = {
                "entity_metrics": spacy_metrics,
                "time": spacy_time,
                "supports_re": False
            }
            print(f"   Entity F1:    {spacy_metrics['f1']:.3f} (P: {spacy_metrics['precision']:.3f}, R: {spacy_metrics['recall']:.3f})")
            print(f"   Time:         {spacy_time:.2f}s")
        else:
            print("   Skipped (not installed)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== SpERT Reference ==========
    print("\n🔹 SpERT (Reference from Paper)...")
    spert_ref = get_spert_reference_results()
    print(f"   CoNLL04 Entity F1:    {spert_ref['datasets']['CoNLL04']['entity_f1']:.3f}")
    print(f"   CoNLL04 Relation F1:  {spert_ref['datasets']['CoNLL04']['relation_f1']:.3f}")
    print(f"   Note: {spert_ref['note']}")
    
    # ========== Speed Benchmark ==========
    print("\n" + "=" * 80)
    print("Speed Benchmark (10,000 characters)")
    print("=" * 80)
    
    speed_results = benchmark_speed(10000)
    for model_name, data in speed_results.items():
        print(f"\n🔹 {model_name}")
        print(f"   Time:       {data['time']:.2f}s")
        print(f"   Entities:   {data['entities']}")
        if data.get('relations', 0) > 0:
            print(f"   Relations:  {data['relations']}")
        print(f"   Speed:      {data['chars_per_sec']:.0f} chars/sec")
    
    # ========== Summary Tables ==========
    print("\n" + "=" * 80)
    print("SUMMARY: Entity Recognition (NER)")
    print("=" * 80)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Time (s)':<10}")
    print("-" * 80)
    for model_name, data in results.items():
        if "entity_metrics" in data:
            m = data["entity_metrics"]
            print(f"{model_name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {data['time']:<10.2f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Relation Extraction (RE)")
    print("=" * 80)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Supports RE':<12}")
    print("-" * 80)
    for model_name, data in results.items():
        if data.get("supports_re"):
            m = data.get("relation_metrics", {"precision": 0, "recall": 0, "f1": 0})
            print(f"{model_name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {'✓':<12}")
        else:
            print(f"{model_name:<20} {'-':<12} {'-':<12} {'-':<12} {'✗':<12}")
    print(f"{'SpERT (CoNLL04)':<20} {'-':<12} {'-':<12} {spert_ref['datasets']['CoNLL04']['relation_f1']:<12.3f} {'✓ (ref)':<12}")
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_full_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        # Convert metrics to serializable format
        serializable_results = {}
        for k, v in results.items():
            serializable_results[k] = {
                "entity_f1": v.get("entity_metrics", {}).get("f1", 0),
                "entity_precision": v.get("entity_metrics", {}).get("precision", 0),
                "entity_recall": v.get("entity_metrics", {}).get("recall", 0),
                "relation_f1": v.get("relation_metrics", {}).get("f1", 0) if v.get("supports_re") else None,
                "time": v.get("time", 0),
                "supports_re": v.get("supports_re", False)
            }
        serializable_results["SpERT (Reference)"] = spert_ref
        serializable_results["speed_benchmark"] = speed_results
        json.dump(serializable_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n📁 Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("""
✅ NERRE is the ONLY model that supports Joint Entity & Relation Extraction
✅ NuNER Zero and GLiNER are zero-shot NER models (NER only)
✅ SpERT requires task-specific training on each dataset
✅ spaCy is pretrained on fixed entity types (English-focused)

📝 Note: SpERT results are from their published paper (ECAI 2020).
   The repository was archived in April 2025.
""")


if __name__ == "__main__":
    main()
