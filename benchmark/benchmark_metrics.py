import json
import os
import sys
from tqdm import tqdm
from collections import defaultdict
import numpy as np

try:
    from gliner import GLiNER
except ImportError:
    GLiNER = None
    print("⚠️ GLiNER not installed. pip install gliner")

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from huggingface_release.nerre import NERREPipeline

# ============================================================================
# LABEL MAPPING: Map External Dataset Labels → Model's Known Labels
# ช่วยให้ Model ที่เทรนด้วย fine-grained labels ทำงานกับ coarse-grained datasets ได้
# ============================================================================

# Entity Label Mapping (External → Model)
ENTITY_LABEL_MAP = {}

# Relation Label Mapping (External → Model)  
RELATION_LABEL_MAP = {}

def normalize_label(label: str, label_map: dict) -> str:
    """Normalize label using the mapping, case-insensitive."""
    label_lower = label.lower().strip()
    return label_map.get(label_lower, label_lower)

def calculate_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def evaluate_model(model_path, test_files, use_label_mapping=True):
    print(f"🚀 Loading model from {model_path}...")
    if use_label_mapping:
        print(f"📋 Label Mapping: ENABLED (will normalize external labels)")

    try:
        model_path = os.path.abspath(model_path)
        print(f"🔍 Resolved model path: {model_path}")
        # Check if saved_model_v23 exists, otherwise maybe it's in a different folder or fallback
        if os.path.exists(model_path):
            model_path = model_path
        else:
            print("❌ Model path invalid.")
            return

        pipe = NERREPipeline(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load All Data
    data = []
    if isinstance(test_files, str): test_files = [test_files]
    for tf in test_files:
        if os.path.exists(tf):
            print(f"📚 Loading test data: {tf}")
            with open(tf, 'r', encoding='utf-8') as f:
                data.extend(json.load(f))
        else:
            print(f"⚠️ File not found: {tf}")

    # =========================================================================
    # 🔥 [FEATURE] Dynamic Label Discovery
    # Automatically add labels found in the dataset to the map (as Identity)
    # This prevents 'unknown' labels from disappearing if they are valid.
    # =========================================================================
    found_ent_labels = set()
    found_rel_labels = set()
    
    print("🔍 Scanning dataset for labels...")
    for item in data:
        for e in item.get('entities', []):
            label = e.get('label')
            if label: found_ent_labels.add(label.lower())
        for r in item.get('relations', []):
            label = r.get('label')
            if label: found_rel_labels.add(label.lower())

    print(f"✅ Found {len(found_ent_labels)} Entity Labels: {sorted(list(found_ent_labels))}")
    print(f"✅ Found {len(found_rel_labels)} Relation Labels: {sorted(list(found_rel_labels))}")

    # Update maps with Identity Mappings for any labels not explicitly mapped
    for l in found_ent_labels:
        if l not in ENTITY_LABEL_MAP:
            ENTITY_LABEL_MAP[l] = l  # e.g., "aircraft" -> "aircraft"
    
    for l in found_rel_labels:
        if l not in RELATION_LABEL_MAP:
            RELATION_LABEL_MAP[l] = l # e.g., "made_of" -> "made_of"

    # 🔥 [User Request] Tell the pipeline to ONLY look for these labels
    # This acts as a strict schema constraint for Zero-Shot
    print(f"🔧 Configuring Pipeline with Active Labels from Dataset...")
    pipe.set_active_labels(
        ent_labels=sorted(list(found_ent_labels)), 
        rel_labels=sorted(list(found_rel_labels))
    )

    print(f"📊 Evaluating on {len(data)} samples...")
    
    # Metrics Counters
    ent_tp, ent_fp, ent_fn = 0, 0, 0
    rel_tp, rel_fp, rel_fn = 0, 0, 0
    
    ent_type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    rel_type_metrics = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for item in tqdm(data, desc="Inferencing"):
        text = item['text']
        gold_entities = item.get('entities', [])
        gold_relations = item.get('relations', [])

        # --- 1. Get Predictions ---
        # 🔥 [PERF] Use fast_mode if no relations in test set
        use_fast = len(found_rel_labels) == 0
        output = pipe(text, ent_threshold=0.5, rel_threshold=0.5, fast_mode=use_fast)
        
        # --- 2. Process Entities ---
        # Convert to sets of tuples for easy comparison: (start, end, label)
        pred_ent_set = set()
        for e in output['entities']:
            # Normalize predicted labels for comparison
            pred_label = normalize_label(e['label'], ENTITY_LABEL_MAP) if use_label_mapping else e['label']
            
            # 🔥 [FILTER] Only allow labels that exist in the test dataset
            # (User Request: "let the model follow dataset found")
            if pred_label in found_ent_labels:
                pred_ent_set.add((e['start'], e['end'], pred_label))
            
        gold_ent_set = set()
        for e in gold_entities:
            # Normalize gold labels for comparison
            gold_label = normalize_label(e['label'], ENTITY_LABEL_MAP) if use_label_mapping else e['label']
            gold_ent_set.add((e['start'], e['end'], gold_label))
            
        # Compare
        tp_set = pred_ent_set.intersection(gold_ent_set)
        fp_set = pred_ent_set - gold_ent_set
        fn_set = gold_ent_set - pred_ent_set
        
        ent_tp += len(tp_set)
        ent_fp += len(fp_set)
        ent_fn += len(fn_set)
        
        # Per-type metrics
        for (s, e, l) in tp_set: ent_type_metrics[l]["tp"] += 1
        for (s, e, l) in fp_set: ent_type_metrics[l]["fp"] += 1
        for (s, e, l) in fn_set: ent_type_metrics[l]["fn"] += 1

        # --- 3. Process Relations ---
        # Convert to tuples: (head_start, head_end, tail_start, tail_end, label)
        pred_rel_set = set()
        for r in output['triplets']:
            # From modified nerre.py, we have spans in triplets now
            if 'head_start' in r:
                # Normalize predicted relation labels
                pred_rel_label = normalize_label(r['relation'], RELATION_LABEL_MAP) if use_label_mapping else r['relation']
                
                # 🔥 [FILTER] Only allow labels that exist in the test dataset
                if pred_rel_label in found_rel_labels:
                    pred_rel_set.add((r['head_start'], r['head_end'], r['tail_start'], r['tail_end'], pred_rel_label))
            else:
                 # Fallback if modification didn't work (should not happen if file updated correctly)
                 pass
            
        gold_rel_set = set()
        for r in gold_relations:
            h, t = None, None
            
            # Priority 1: Indices (head/tail as int)
            if 'head' in r and 'tail' in r and isinstance(r['head'], int) and isinstance(r['tail'], int):
                 if r['head'] < len(gold_entities) and r['tail'] < len(gold_entities):
                      h = gold_entities[r['head']]
                      t = gold_entities[r['tail']]
            
            # Priority 2: Indices with _idx keys
            elif 'head_idx' in r and 'tail_idx' in r:
                 if r['head_idx'] < len(gold_entities) and r['tail_idx'] < len(gold_entities):
                      h = gold_entities[r['head_idx']]
                      t = gold_entities[r['tail_idx']]
            
            # Priority 3: String Matching (head/tail as str)
            elif 'head' in r and 'tail' in r and isinstance(r['head'], str) and isinstance(r['tail'], str):
                # Find entity by text. WARNING: Ambiguity if multiple entities have same text.
                # We take the first match as fallback.
                h_candidates = [e for e in gold_entities if e['text'] == r['head']]
                t_candidates = [e for e in gold_entities if e['text'] == r['tail']]
                
                if h_candidates and t_candidates:
                    h = h_candidates[0]
                    t = t_candidates[0]

            if h and t:
                # Normalize gold relation labels
                gold_rel_label = normalize_label(r['label'], RELATION_LABEL_MAP) if use_label_mapping else r['label']
                gold_rel_set.add((h['start'], h['end'], t['start'], t['end'], gold_rel_label))

        # Compare
        r_tp_set = pred_rel_set.intersection(gold_rel_set)
        r_fp_set = pred_rel_set - gold_rel_set
        r_fn_set = gold_rel_set - pred_rel_set
        
        rel_tp += len(r_tp_set)
        rel_fp += len(r_fp_set)
        rel_fn += len(r_fn_set)

        for (*_, l) in r_tp_set: rel_type_metrics[l]["tp"] += 1
        for (*_, l) in r_fp_set: rel_type_metrics[l]["fp"] += 1
        for (*_, l) in r_fn_set: rel_type_metrics[l]["fn"] += 1

    # Print Results
    e_prec, e_rec, e_f1 = calculate_f1(ent_tp, ent_fp, ent_fn)
    r_prec, r_rec, r_f1 = calculate_f1(rel_tp, rel_fp, rel_fn)

    print("\n" + "="*60)
    print("🏆 Overall Performance Metrics")
    print("="*60)
    print(f"Entities (NER):")
    print(f"  - Precision: {e_prec*100:.2f}%")
    print(f"  - Recall:    {e_rec*100:.2f}%")
    print(f"  - F1 Score:  {e_f1*100:.2f}%")
    print(f"Relations (RE):")
    print(f"  - Precision: {r_prec*100:.2f}%")
    print(f"  - Recall:    {r_rec*100:.2f}%")
    print(f"  - F1 Score:  {r_f1*100:.2f}%")

    print("\n" + "-"*60)
    print("📋 Detailed Entity Metrics (Top 20 by Support)")
    print("-"*60)
    print(f"{'Label':<30} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Supp'}")
    
    ent_types_sorted = sorted(ent_type_metrics.keys(), key=lambda k: ent_type_metrics[k]['tp'] + ent_type_metrics[k]['fn'], reverse=True)
    
    for l in ent_types_sorted[:20]:
        m = ent_type_metrics[l]
        p, r, f = calculate_f1(m['tp'], m['fp'], m['fn'])
        supp = m['tp'] + m['fn']
        print(f"{l:<30} | {p*100:<8.2f} | {r*100:<8.2f} | {f*100:<8.2f} | {supp}")

    print("\n" + "-"*60)
    print("📋 Detailed Relation Metrics (Top 20 by Support)")
    print("-"*60)
    print(f"{'Label':<30} | {'Prec':<8} | {'Rec':<8} | {'F1':<8} | {'Supp'}")
    
    rel_types_sorted = sorted(rel_type_metrics.keys(), key=lambda k: rel_type_metrics[k]['tp'] + rel_type_metrics[k]['fn'], reverse=True)
    
    for l in rel_types_sorted[:20]:
        m = rel_type_metrics[l]
        p, r, f = calculate_f1(m['tp'], m['fp'], m['fn'])
        supp = m['tp'] + m['fn']
        print(f"{l:<30} | {p*100:<8.2f} | {r*100:<8.2f} | {f*100:<8.2f} | {supp}")


def evaluate_gliner(model_name, test_files, device='cuda'):
    if GLiNER is None: 
        print("❌ GLiNER library not found. Skipping GLiNER evaluation.")
        return
    import torch

    print(f"\n" + "="*60)
    print(f"🚀 Evaluating GLiNER Model: {model_name}")
    print("="*60)
    
    try:
        model = GLiNER.from_pretrained(model_name).to(device)
    except Exception as e:
        print(f"Failed to load GLiNER: {e}")
        return

    # Load Data (Duplicate logic for safety)
    data = []
    if isinstance(test_files, str): test_files = [test_files]
    for tf in test_files:
        if os.path.exists(tf):
            with open(tf, 'r', encoding='utf-8') as f:
                data.extend(json.load(f))

    # Scan labels
    found_ent_labels = set()
    for item in data:
        for e in item.get('entities', []):
            if e.get('label'): found_ent_labels.add(e.get('label').lower())
            
    target_labels = sorted(list(found_ent_labels))
    print(f"✅ GLiNER will look for {len(target_labels)} labels: {target_labels}")

    # Eval Loop
    ent_tp, ent_fp, ent_fn = 0, 0, 0
            
    for item in tqdm(data, desc=f"GLiNER Inference"):
        text = item['text']
        gold_entities = item.get('entities', [])
        
        # GLiNER Predict
        with torch.no_grad():
             preds = model.predict_entities(text, target_labels, threshold=0.5)
        
        # Process Preds
        pred_ent_set = set()
        for e in preds:
            pred_ent_set.add((e['start'], e['end'], e['label'].lower()))
            
        # Process Gold
        gold_ent_set = set()
        for e in gold_entities:
            gold_ent_set.add((e['start'], e['end'], e['label'].lower()))
            
        # Metrics
        tp_set = pred_ent_set.intersection(gold_ent_set)
        fp_set = pred_ent_set - gold_ent_set
        fn_set = gold_ent_set - pred_ent_set
        
        ent_tp += len(tp_set)
        ent_fp += len(fp_set)
        ent_fn += len(fn_set)

    # Print
    p, r, f1 = calculate_f1(ent_tp, ent_fp, ent_fn)
    print(f"\n🏆 GLiNER '{model_name}' Results (NER Only)")
    print(f"  - Precision: {p*100:.2f}%")
    print(f"  - Recall:    {r*100:.2f}%")
    print(f"  - F1 Score:  {f1*100:.2f}%")

if __name__ == "__main__":
    MODEL_PATH = "saved_model_v24"
    # Adjust file paths to match your actual validation set
    TEST_FILES = [
        "dataset/conll04_test.json"
    ]
    
    evaluate_model(MODEL_PATH, TEST_FILES)
    
    # Compare with GLiNER
    evaluate_gliner("urchade/gliner_multi-v2.1", TEST_FILES)

