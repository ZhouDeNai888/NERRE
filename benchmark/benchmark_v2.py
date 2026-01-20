"""
NERRE Benchmark V2: Aligned with Model's Relation Labels
==========================================================
Uses the new 100-sample dataset with proper relation label matching.

This benchmark is designed to fairly evaluate NERRE's Relation Extraction
capability by using the exact relation labels the model was trained on.
"""

import time
import json
import os
import sys
from typing import List, Dict, Tuple, Set
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our aligned dataset
from benchmark.benchmark_dataset import get_benchmark_dataset, get_dataset_statistics


# ============================================================================
# SpERT Reference Results
# ============================================================================

def get_spert_reference_results() -> Dict:
    """
    SpERT reference results from the ECAI 2020 paper.
    https://arxiv.org/abs/1909.07755
    """
    return {
        "model": "SpERT",
        "paper": "Eberts & Ulges, ECAI 2020",
        "datasets": {
            "CoNLL04": {"entity_f1": 0.8821, "relation_f1": 0.7261},
            "SciERC": {"entity_f1": 0.7087, "relation_f1": 0.4803},
            "ADE": {"entity_f1": 0.8942, "relation_f1": 0.7917}
        },
        "note": "SpERT requires task-specific training. Repository archived April 2025."
    }


# ============================================================================
# Evaluation Metrics
# ============================================================================

def normalize_label(label: str) -> str:
    """Normalize entity labels for fair comparison."""
    label = label.lower().strip()
    mapping = {
        "person": "person", "per": "person", "peop": "person",
        "org": "organisation", "organization": "organisation", "organisation": "organisation",
        "gpe": "location", "loc": "location", "location": "location",
        "date": "date", "time": "date",
        "product": "product", "work_of_art": "product",
        "programlang": "programlang",
    }
    return mapping.get(label, label)


def normalize_relation(relation: str) -> str:
    """Normalize relation labels for matching."""
    rel = relation.lower().strip().replace(" ", "_").replace("-", "_")
    # Map common variations to model's labels
    mapping = {
        "founder_of": "founder_of",
        "ceo_of": "ceo_of",
        "developed": "developed",
        "creator_of": "creator_of",
        "founded_in": "founded_in",
        "released_in": "released_in",
        "located_in": "located_in",
        "headquartered_in": "located_in",  # Map to located_in
        "based_in": "located_in",
    }
    return mapping.get(rel, rel)


def calculate_entity_metrics(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Calculate precision, recall, and F1 for NER with fuzzy matching."""
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
    
    # Match predictions to ground truth
    matched_gt = set()
    for pred_text, pred_label in pred_set:
        found_match = False
        for gt_text, gt_label in gt_set:
            if pred_label == gt_label and (pred_text in gt_text or gt_text in pred_text):
                if (gt_text, gt_label) not in matched_gt:
                    tp += 1
                    matched_gt.add((gt_text, gt_label))
                    found_match = True
                    break
        if not found_match:
            fp += 1
    
    # Count unmatched ground truth as FN
    for gt_text, gt_label in gt_set:
        if (gt_text, gt_label) not in matched_gt:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def calculate_relation_metrics_strict(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Calculate RE metrics with strict matching: head, tail, AND relation type must match.
    This is how SpERT and other RE papers evaluate.
    """
    if not ground_truth:
        return {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0}
    
    # Normalize and create sets
    pred_triplets = set()
    for p in predictions:
        head = p.get('head', '').lower().strip()
        tail = p.get('tail', '').lower().strip()
        rel = normalize_relation(p.get('relation', ''))
        pred_triplets.add((head, tail, rel))
    
    gt_triplets = set()
    for g in ground_truth:
        head = g.get('head', '').lower().strip()
        tail = g.get('tail', '').lower().strip()
        rel = normalize_relation(g.get('relation', ''))
        gt_triplets.add((head, tail, rel))
    
    # Match with fuzzy entity matching but strict relation matching
    tp, fp, fn = 0, 0, 0
    matched_gt = set()
    
    for pred_h, pred_t, pred_r in pred_triplets:
        found = False
        for gt_h, gt_t, gt_r in gt_triplets:
            # Relation must match exactly
            if pred_r != gt_r:
                continue
            # Entities can have fuzzy match
            head_match = pred_h in gt_h or gt_h in pred_h
            tail_match = pred_t in gt_t or gt_t in pred_t
            if head_match and tail_match:
                if (gt_h, gt_t, gt_r) not in matched_gt:
                    tp += 1
                    matched_gt.add((gt_h, gt_t, gt_r))
                    found = True
                    break
        if not found:
            fp += 1
    
    fn = len(gt_triplets) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def calculate_relation_metrics_relaxed(predictions: List[Dict], ground_truth: List[Dict]) -> Dict:
    """
    Calculate RE metrics with relaxed matching: only head and tail must match.
    Ignores relation type - just checks if the entity pair was detected.
    """
    if not ground_truth:
        return {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0}
    
    pred_pairs = set()
    for p in predictions:
        head = p.get('head', '').lower().strip()
        tail = p.get('tail', '').lower().strip()
        pred_pairs.add((head, tail))
    
    gt_pairs = set()
    for g in ground_truth:
        head = g.get('head', '').lower().strip()
        tail = g.get('tail', '').lower().strip()
        gt_pairs.add((head, tail))
    
    tp, fp = 0, 0
    matched_gt = set()
    
    for pred_h, pred_t in pred_pairs:
        found = False
        for gt_h, gt_t in gt_pairs:
            head_match = pred_h in gt_h or gt_h in pred_h
            tail_match = pred_t in gt_t or gt_t in pred_t
            if head_match and tail_match:
                if (gt_h, gt_t) not in matched_gt:
                    tp += 1
                    matched_gt.add((gt_h, gt_t))
                    found = True
                    break
        if not found:
            fp += 1
    
    fn = len(gt_pairs) - len(matched_gt)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


# ============================================================================
# Model Benchmarks
# ============================================================================

def benchmark_nerre(test_data: List[Dict], fast_mode: bool = False, threshold: float = 0.5) -> Tuple[Dict, Dict, Dict, float]:
    """Benchmark NERRE model."""
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
            result = pipe(sample["text"], ent_threshold=threshold, fast_mode=True, extract_relations=False)
        else:
            result = pipe(sample["text"], ent_threshold=threshold, rel_threshold=threshold)
        
        all_pred_entities.extend(result["entities"])
        all_gt_entities.extend(sample["entities"])
        
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
    rel_metrics_strict = calculate_relation_metrics_strict(all_pred_relations, all_gt_relations)
    rel_metrics_relaxed = calculate_relation_metrics_relaxed(all_pred_relations, all_gt_relations)
    
    return entity_metrics, rel_metrics_strict, rel_metrics_relaxed, total_time


def benchmark_gliner(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark GLiNER model."""
    try:
        from gliner import GLiNER
    except ImportError:
        return None, 0
    
    try:
        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    except:
        return None, 0
    
    labels = ["person", "organisation", "location", "date", "product", "programlang"]
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        try:
            entities = model.predict_entities(sample["text"], labels, threshold=0.5)
            for ent in entities:
                all_predictions.append({"text": ent["text"], "label": ent["label"]})
        except:
            pass
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_entity_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


def benchmark_nuner(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark NuNER Zero model."""
    try:
        from gliner import GLiNER
    except ImportError:
        return None, 0
    
    try:
        model = GLiNER.from_pretrained("numind/NuNerZero")
    except:
        return None, 0
    
    labels = ["person", "organisation", "location", "date", "product", "programlang"]
    
    def merge_entities(entities, text):
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
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        text = sample["text"]
        try:
            entities = model.predict_entities(text, labels, threshold=0.5)
            entities = merge_entities(entities, text)
            for ent in entities:
                all_predictions.append({"text": ent["text"], "label": ent["label"]})
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
        try:
            nlp = spacy.load("en_core_web_trf")
        except:
            nlp = spacy.load("en_core_web_sm")
    except:
        return None, 0
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        try:
            doc = nlp(sample["text"])
            for ent in doc.ents:
                all_predictions.append({"text": ent.text, "label": ent.label_})
        except:
            pass
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_entity_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


# ============================================================================
# Detailed Analysis
# ============================================================================

def analyze_relation_errors(test_data: List[Dict], predictions_by_sample: List[List[Dict]]) -> Dict:
    """Analyze what kinds of relation errors occur."""
    analysis = {
        "correct": defaultdict(int),
        "missed": defaultdict(int),
        "wrong_type": defaultdict(int),
        "hallucinated": defaultdict(int)
    }
    
    for i, sample in enumerate(test_data):
        gt_rels = sample.get("relations", [])
        pred_rels = predictions_by_sample[i] if i < len(predictions_by_sample) else []
        
        gt_set = {(r["head"].lower(), r["tail"].lower(), r["relation"]): r for r in gt_rels}
        pred_set = {(r["head"].lower(), r["tail"].lower(), r["relation"]): r for r in pred_rels}
        
        # Check predictions
        for key, pred in pred_set.items():
            if key in gt_set:
                analysis["correct"][pred["relation"]] += 1
            else:
                # Check if pair exists with wrong relation
                pair = (key[0], key[1])
                pair_found = False
                for gt_key in gt_set:
                    if (gt_key[0], gt_key[1]) == pair:
                        analysis["wrong_type"][f"{gt_key[2]}->{pred['relation']}"] += 1
                        pair_found = True
                        break
                if not pair_found:
                    analysis["hallucinated"][pred["relation"]] += 1
        
        # Check missed
        for key in gt_set:
            if key not in pred_set:
                analysis["missed"][key[2]] += 1
    
    return dict(analysis)


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 80)
    print("NERRE Benchmark V2: Aligned Dataset (100 samples, 155 relations)")
    print("=" * 80)
    
    # Load dataset
    test_data = get_benchmark_dataset()
    stats = get_dataset_statistics(test_data)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Samples:   {stats['total_samples']}")
    print(f"   Entities:  {stats['total_entities']}")
    print(f"   Relations: {stats['total_relations']}")
    print(f"   Languages: English, Chinese, Japanese")
    print(f"\n   Relation Distribution:")
    for rel, count in sorted(stats['relation_counts'].items()):
        print(f"     {rel}: {count}")
    print("-" * 80)
    
    results = {}
    
    # ========== NERRE ==========
    print("\n🔹 Benchmarking NERRE...")
    try:
        ent_metrics, rel_strict, rel_relaxed, nerre_time = benchmark_nerre(test_data, fast_mode=False, threshold=0.5)
        results["NERRE"] = {
            "entity": ent_metrics,
            "relation_strict": rel_strict,
            "relation_relaxed": rel_relaxed,
            "time": nerre_time,
            "supports_re": True
        }
        print(f"   Entity F1:           {ent_metrics['f1']:.3f} (P: {ent_metrics['precision']:.3f}, R: {ent_metrics['recall']:.3f})")
        print(f"   Relation F1 (Strict): {rel_strict['f1']:.3f} (P: {rel_strict['precision']:.3f}, R: {rel_strict['recall']:.3f})")
        print(f"   Relation F1 (Pairs):  {rel_relaxed['f1']:.3f} (P: {rel_relaxed['precision']:.3f}, R: {rel_relaxed['recall']:.3f})")
        print(f"   Time: {nerre_time:.2f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== NERRE with lower threshold ==========
    print("\n🔹 Benchmarking NERRE (threshold=0.3 for better recall)...")
    try:
        ent_metrics2, rel_strict2, rel_relaxed2, nerre_time2 = benchmark_nerre(test_data, fast_mode=False, threshold=0.3)
        results["NERRE (t=0.3)"] = {
            "entity": ent_metrics2,
            "relation_strict": rel_strict2,
            "relation_relaxed": rel_relaxed2,
            "time": nerre_time2,
            "supports_re": True
        }
        print(f"   Entity F1:           {ent_metrics2['f1']:.3f} (P: {ent_metrics2['precision']:.3f}, R: {ent_metrics2['recall']:.3f})")
        print(f"   Relation F1 (Strict): {rel_strict2['f1']:.3f} (P: {rel_strict2['precision']:.3f}, R: {rel_strict2['recall']:.3f})")
        print(f"   Relation F1 (Pairs):  {rel_relaxed2['f1']:.3f} (P: {rel_relaxed2['precision']:.3f}, R: {rel_relaxed2['recall']:.3f})")
        print(f"   Time: {nerre_time2:.2f}s")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== GLiNER ==========
    print("\n🔹 Benchmarking GLiNER...")
    try:
        gliner_metrics, gliner_time = benchmark_gliner(test_data)
        if gliner_metrics:
            results["GLiNER"] = {"entity": gliner_metrics, "time": gliner_time, "supports_re": False}
            print(f"   Entity F1: {gliner_metrics['f1']:.3f} (P: {gliner_metrics['precision']:.3f}, R: {gliner_metrics['recall']:.3f})")
            print(f"   Time: {gliner_time:.2f}s")
            print(f"   ⚠️  No Relation Extraction support")
        else:
            print("   Skipped (not installed)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== NuNER ==========
    print("\n🔹 Benchmarking NuNER Zero...")
    try:
        nuner_metrics, nuner_time = benchmark_nuner(test_data)
        if nuner_metrics:
            results["NuNER Zero"] = {"entity": nuner_metrics, "time": nuner_time, "supports_re": False}
            print(f"   Entity F1: {nuner_metrics['f1']:.3f} (P: {nuner_metrics['precision']:.3f}, R: {nuner_metrics['recall']:.3f})")
            print(f"   Time: {nuner_time:.2f}s")
            print(f"   ⚠️  No Relation Extraction support")
        else:
            print("   Skipped (not installed)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== spaCy ==========
    print("\n🔹 Benchmarking spaCy...")
    try:
        spacy_metrics, spacy_time = benchmark_spacy(test_data)
        if spacy_metrics:
            results["spaCy"] = {"entity": spacy_metrics, "time": spacy_time, "supports_re": False}
            print(f"   Entity F1: {spacy_metrics['f1']:.3f} (P: {spacy_metrics['precision']:.3f}, R: {spacy_metrics['recall']:.3f})")
            print(f"   Time: {spacy_time:.2f}s")
            print(f"   ⚠️  No Relation Extraction support")
        else:
            print("   Skipped (not installed)")
    except Exception as e:
        print(f"   Error: {e}")
    
    # ========== SpERT Reference ==========
    spert_ref = get_spert_reference_results()
    
    # ========== Summary Tables ==========
    print("\n" + "=" * 80)
    print("SUMMARY: Named Entity Recognition (NER)")
    print("=" * 80)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Time (s)':<10}")
    print("-" * 80)
    for name, data in results.items():
        if "entity" in data:
            m = data["entity"]
            print(f"{name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {data['time']:<10.2f}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Relation Extraction (RE) - Strict Matching")
    print("=" * 80)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Note':<20}")
    print("-" * 80)
    for name, data in results.items():
        if data.get("supports_re"):
            m = data.get("relation_strict", {"precision": 0, "recall": 0, "f1": 0})
            print(f"{name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {'Zero-shot':<20}")
        else:
            print(f"{name:<20} {'-':<12} {'-':<12} {'-':<12} {'NER only':<20}")
    print(f"{'SpERT (CoNLL04)':<20} {'-':<12} {'-':<12} {spert_ref['datasets']['CoNLL04']['relation_f1']:<12.3f} {'Supervised':<20}")
    
    print("\n" + "=" * 80)
    print("SUMMARY: Relation Extraction (RE) - Pair Detection Only")
    print("=" * 80)
    print(f"{'Model':<20} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
    print("-" * 80)
    for name, data in results.items():
        if data.get("supports_re"):
            m = data.get("relation_relaxed", {"precision": 0, "recall": 0, "f1": 0})
            print(f"{name:<20} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f}")
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_v2_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        serializable = {}
        for k, v in results.items():
            serializable[k] = {
                "entity_f1": v.get("entity", {}).get("f1", 0),
                "relation_strict_f1": v.get("relation_strict", {}).get("f1", 0) if v.get("supports_re") else None,
                "relation_relaxed_f1": v.get("relation_relaxed", {}).get("f1", 0) if v.get("supports_re") else None,
                "time": v.get("time", 0)
            }
        serializable["SpERT (Reference)"] = spert_ref
        json.dump(serializable, f, indent=2, ensure_ascii=False)
    print(f"\n📁 Results saved to: {output_path}")
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
📌 Evaluation Metrics:
   - Strict: Head, Tail, AND Relation Type must all match (standard RE metric)
   - Relaxed: Only Head-Tail pair must match (relation detection)

📌 Key Differences:
   - NERRE: Zero-shot, supports both NER and RE
   - SpERT: Requires supervised training per dataset (CoNLL04, SciERC, etc.)
   - GLiNER/NuNER: Zero-shot NER only, no relation extraction

📌 SpERT Comparison:
   SpERT's F1=0.726 is on CoNLL04 dataset with supervised training.
   NERRE is zero-shot - no training required for new entity/relation types.
   
💡 To improve NERRE RE further:
   1. Lower the threshold (e.g., 0.3) to improve recall
   2. Fine-tune on domain-specific data
   3. Add more relation types to training data
""")


if __name__ == "__main__":
    main()
