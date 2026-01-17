"""
Benchmark Script: NERRE vs GLiNER vs spaCy
==========================================
Compares Named Entity Recognition performance across different models.
"""

import time
import json
import os
import sys
from typing import List, Dict, Tuple
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_test_data() -> List[Dict]:
    """Load test data with ground truth labels."""
    test_data = [
        # English
        {
            "text": "Elon Musk founded SpaceX in 2002 in California.",
            "language": "en",
            "entities": [
                {"text": "Elon Musk", "label": "person", "start": 0, "end": 9},
                {"text": "SpaceX", "label": "organisation", "start": 18, "end": 24},
                {"text": "2002", "label": "date", "start": 28, "end": 32},
                {"text": "California", "label": "location", "start": 36, "end": 46},
            ]
        },
        {
            "text": "Mark Zuckerberg is the CEO of Meta. He created Facebook in 2004.",
            "language": "en",
            "entities": [
                {"text": "Mark Zuckerberg", "label": "person", "start": 0, "end": 15},
                {"text": "Meta", "label": "organisation", "start": 30, "end": 34},
                {"text": "Facebook", "label": "organisation", "start": 48, "end": 56},
                {"text": "2004", "label": "date", "start": 60, "end": 64},
            ]
        },
        {
            "text": "Bill Gates and Paul Allen founded Microsoft in 1975 in Redmond.",
            "language": "en",
            "entities": [
                {"text": "Bill Gates", "label": "person", "start": 0, "end": 10},
                {"text": "Paul Allen", "label": "person", "start": 15, "end": 25},
                {"text": "Microsoft", "label": "organisation", "start": 34, "end": 43},
                {"text": "1975", "label": "date", "start": 47, "end": 51},
                {"text": "Redmond", "label": "location", "start": 55, "end": 62},
            ]
        },
        {
            "text": "Steve Jobs co-founded Apple in 1976 in Cupertino, California.",
            "language": "en",
            "entities": [
                {"text": "Steve Jobs", "label": "person", "start": 0, "end": 10},
                {"text": "Apple", "label": "organisation", "start": 22, "end": 27},
                {"text": "1976", "label": "date", "start": 31, "end": 35},
                {"text": "Cupertino", "label": "location", "start": 39, "end": 48},
                {"text": "California", "label": "location", "start": 50, "end": 60},
            ]
        },
        {
            "text": "Jeff Bezos started Amazon in 1994 in Seattle, Washington.",
            "language": "en",
            "entities": [
                {"text": "Jeff Bezos", "label": "person", "start": 0, "end": 10},
                {"text": "Amazon", "label": "organisation", "start": 19, "end": 25},
                {"text": "1994", "label": "date", "start": 29, "end": 33},
                {"text": "Seattle", "label": "location", "start": 37, "end": 44},
                {"text": "Washington", "label": "location", "start": 46, "end": 56},
            ]
        },
        {
            "text": "Sundar Pichai is the CEO of Google and Alphabet Inc.",
            "language": "en",
            "entities": [
                {"text": "Sundar Pichai", "label": "person", "start": 0, "end": 13},
                {"text": "Google", "label": "organisation", "start": 28, "end": 34},
                {"text": "Alphabet Inc", "label": "organisation", "start": 39, "end": 51},
            ]
        },
        {
            "text": "Jensen Huang founded NVIDIA in 1993 in Santa Clara.",
            "language": "en",
            "entities": [
                {"text": "Jensen Huang", "label": "person", "start": 0, "end": 12},
                {"text": "NVIDIA", "label": "organisation", "start": 21, "end": 27},
                {"text": "1993", "label": "date", "start": 31, "end": 35},
                {"text": "Santa Clara", "label": "location", "start": 39, "end": 50},
            ]
        },
        {
            "text": "Tim Cook became the CEO of Apple in 2011 after Steve Jobs.",
            "language": "en",
            "entities": [
                {"text": "Tim Cook", "label": "person", "start": 0, "end": 8},
                {"text": "Apple", "label": "organisation", "start": 27, "end": 32},
                {"text": "2011", "label": "date", "start": 36, "end": 40},
                {"text": "Steve Jobs", "label": "person", "start": 47, "end": 57},
            ]
        },
        {
            "text": "Satya Nadella has been the CEO of Microsoft since 2014.",
            "language": "en",
            "entities": [
                {"text": "Satya Nadella", "label": "person", "start": 0, "end": 13},
                {"text": "Microsoft", "label": "organisation", "start": 34, "end": 43},
                {"text": "2014", "label": "date", "start": 50, "end": 54},
            ]
        },
        {
            "text": "Larry Page and Sergey Brin founded Google in 1998 at Stanford University.",
            "language": "en",
            "entities": [
                {"text": "Larry Page", "label": "person", "start": 0, "end": 10},
                {"text": "Sergey Brin", "label": "person", "start": 15, "end": 26},
                {"text": "Google", "label": "organisation", "start": 35, "end": 41},
                {"text": "1998", "label": "date", "start": 45, "end": 49},
                {"text": "Stanford University", "label": "organisation", "start": 53, "end": 72},
            ]
        },
        # Traditional Chinese
        {
            "text": "台積電是全球最大的晶片製造商，張忠謀於1987年在新竹創立。",
            "language": "zh",
            "entities": [
                {"text": "台積電", "label": "organisation", "start": 0, "end": 3},
                {"text": "張忠謀", "label": "person", "start": 14, "end": 17},
                {"text": "1987年", "label": "date", "start": 18, "end": 23},
                {"text": "新竹", "label": "location", "start": 24, "end": 26},
            ]
        },
        {
            "text": "馬克·祖克柏是Meta的執行長，他在2004年創建了Facebook。",
            "language": "zh",
            "entities": [
                {"text": "馬克·祖克柏", "label": "person", "start": 0, "end": 6},
                {"text": "Meta", "label": "organisation", "start": 7, "end": 11},
                {"text": "2004年", "label": "date", "start": 18, "end": 23},
                {"text": "Facebook", "label": "organisation", "start": 26, "end": 34},
            ]
        },
        {
            "text": "蘋果公司由史蒂夫·賈伯斯於1976年在加州創立。",
            "language": "zh",
            "entities": [
                {"text": "蘋果公司", "label": "organisation", "start": 0, "end": 4},
                {"text": "史蒂夫·賈伯斯", "label": "person", "start": 5, "end": 12},
                {"text": "1976年", "label": "date", "start": 13, "end": 18},
                {"text": "加州", "label": "location", "start": 19, "end": 21},
            ]
        },
        {
            "text": "黃仁勳於1993年創立輝達，現為全球領先的GPU製造商。",
            "language": "zh",
            "entities": [
                {"text": "黃仁勳", "label": "person", "start": 0, "end": 3},
                {"text": "1993年", "label": "date", "start": 4, "end": 9},
                {"text": "輝達", "label": "organisation", "start": 11, "end": 13},
                {"text": "GPU", "label": "product", "start": 21, "end": 24},
            ]
        },
        {
            "text": "伊隆·馬斯克於2002年創立SpaceX，目標是實現火星殖民。",
            "language": "zh",
            "entities": [
                {"text": "伊隆·馬斯克", "label": "person", "start": 0, "end": 6},
                {"text": "2002年", "label": "date", "start": 7, "end": 12},
                {"text": "SpaceX", "label": "organisation", "start": 14, "end": 20},
            ]
        },
        # Japanese
        {
            "text": "ソニーは盛田昭夫によって1946年に東京で設立されました。",
            "language": "ja",
            "entities": [
                {"text": "ソニー", "label": "organisation", "start": 0, "end": 3},
                {"text": "盛田昭夫", "label": "person", "start": 4, "end": 8},
                {"text": "1946年", "label": "date", "start": 13, "end": 18},
                {"text": "東京", "label": "location", "start": 19, "end": 21},
            ]
        },
        {
            "text": "任天堂は山内房治郎によって1889年に京都で創業されました。",
            "language": "ja",
            "entities": [
                {"text": "任天堂", "label": "organisation", "start": 0, "end": 3},
                {"text": "山内房治郎", "label": "person", "start": 4, "end": 9},
                {"text": "1889年", "label": "date", "start": 14, "end": 19},
                {"text": "京都", "label": "location", "start": 20, "end": 22},
            ]
        },
        {
            "text": "トヨタ自動車は豊田喜一郎によって1937年に愛知県で設立されました。",
            "language": "ja",
            "entities": [
                {"text": "トヨタ自動車", "label": "organisation", "start": 0, "end": 6},
                {"text": "豊田喜一郎", "label": "person", "start": 7, "end": 12},
                {"text": "1937年", "label": "date", "start": 17, "end": 22},
                {"text": "愛知県", "label": "location", "start": 23, "end": 26},
            ]
        },
    ]
    return test_data


def normalize_label(label: str) -> str:
    """Normalize entity labels for fair comparison."""
    label = label.lower().strip()
    mapping = {
        # spaCy labels
        "person": "person",
        "per": "person",
        "org": "organisation",
        "organization": "organisation",
        "organisation": "organisation",
        "gpe": "location",
        "loc": "location",
        "location": "location",
        "date": "date",
        "time": "date",
        "product": "product",
        "work_of_art": "product",
        # GLiNER labels
        "company": "organisation",
        "corporation": "organisation",
    }
    return mapping.get(label, label)


def calculate_metrics(predictions: List[Dict], ground_truth: List[Dict], fuzzy_match: bool = True) -> Dict:
    """Calculate precision, recall, and F1 score."""
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
    
    # Calculate TP, FP, FN
    for pred_text, pred_label in pred_set:
        matched = False
        for gt_text, gt_label in gt_set:
            if pred_label == gt_label:
                if fuzzy_match:
                    # Fuzzy: check if one contains the other
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


def benchmark_nerre(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark NERRE model."""
    from huggingface_release.nerre import NERREPipeline
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "huggingface_release")
    pipe = NERREPipeline(model_dir)
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        result = pipe(sample["text"], ent_threshold=0.5)
        all_predictions.extend(result["entities"])
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


def benchmark_gliner(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark GLiNER model."""
    try:
        from gliner import GLiNER
    except ImportError:
        print("GLiNER not installed. Run: pip install gliner")
        return None, 0
    
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    labels = ["person", "organisation", "location", "date", "product"]
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        entities = model.predict_entities(sample["text"], labels, threshold=0.5)
        for ent in entities:
            all_predictions.append({
                "text": ent["text"],
                "label": ent["label"]
            })
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


def benchmark_spacy(test_data: List[Dict]) -> Tuple[Dict, float]:
    """Benchmark spaCy model."""
    try:
        import spacy
    except ImportError:
        print("spaCy not installed. Run: pip install spacy && python -m spacy download en_core_web_trf")
        return None, 0
    
    # Try to load transformer model, fallback to small
    try:
        nlp = spacy.load("en_core_web_trf")
    except OSError:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Run: python -m spacy download en_core_web_sm")
            return None, 0
    
    all_predictions = []
    all_ground_truth = []
    
    start_time = time.time()
    for sample in test_data:
        doc = nlp(sample["text"])
        for ent in doc.ents:
            all_predictions.append({
                "text": ent.text,
                "label": ent.label_
            })
        all_ground_truth.extend(sample["entities"])
    total_time = time.time() - start_time
    
    metrics = calculate_metrics(all_predictions, all_ground_truth)
    return metrics, total_time


def benchmark_speed_only(num_chars: int = 10000) -> Dict:
    """Benchmark processing speed for long texts."""
    from huggingface_release.nerre import NERREPipeline
    
    # Generate long text
    base_text = "Elon Musk founded SpaceX in 2002. Tesla was founded by Elon Musk in California. "
    long_text = base_text * (num_chars // len(base_text) + 1)
    long_text = long_text[:num_chars]
    
    model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "huggingface_release")
    pipe = NERREPipeline(model_dir)
    
    # NERRE Normal Mode (NER + Relations)
    start_time = time.time()
    result = pipe(long_text)
    nerre_time = time.time() - start_time
    nerre_entities = len(result["entities"])
    nerre_triplets = len(result["triplets"])
    
    results = {
        "nerre": {"time": nerre_time, "entities": nerre_entities, "triplets": nerre_triplets, "chars_per_sec": num_chars / nerre_time}
    }
    
    # NERRE Fast Mode (NER only, no relations)
    start_time = time.time()
    result_fast = pipe(long_text, fast_mode=True, extract_relations=False)
    nerre_fast_time = time.time() - start_time
    nerre_fast_entities = len(result_fast["entities"])
    results["nerre_fast"] = {"time": nerre_fast_time, "entities": nerre_fast_entities, "triplets": 0, "chars_per_sec": num_chars / nerre_fast_time}
    
    # GLiNER
    try:
        from gliner import GLiNER
        model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        labels = ["person", "organisation", "location", "date", "product"]
        
        start_time = time.time()
        entities = model.predict_entities(long_text, labels, threshold=0.5)
        gliner_time = time.time() - start_time
        results["gliner"] = {"time": gliner_time, "entities": len(entities), "chars_per_sec": num_chars / gliner_time}
    except:
        pass
    
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
        results["spacy"] = {"time": spacy_time, "entities": len(doc.ents), "chars_per_sec": num_chars / spacy_time}
    except:
        pass
    
    return results


def main():
    print("=" * 70)
    print("NER Benchmark: NERRE vs GLiNER vs spaCy")
    print("=" * 70)
    
    test_data = load_test_data()
    print(f"\nTest Data: {len(test_data)} samples")
    print(f"Languages: English, Chinese, Japanese")
    print("-" * 70)
    
    results = {}
    
    # NERRE
    print("\n🔹 Benchmarking NERRE...")
    nerre_metrics, nerre_time = benchmark_nerre(test_data)
    results["NERRE"] = {"metrics": nerre_metrics, "time": nerre_time}
    print(f"   Precision: {nerre_metrics['precision']:.3f}")
    print(f"   Recall:    {nerre_metrics['recall']:.3f}")
    print(f"   F1 Score:  {nerre_metrics['f1']:.3f}")
    print(f"   Time:      {nerre_time:.2f}s")
    
    # GLiNER
    print("\n🔹 Benchmarking GLiNER...")
    gliner_metrics, gliner_time = benchmark_gliner(test_data)
    if gliner_metrics:
        results["GLiNER"] = {"metrics": gliner_metrics, "time": gliner_time}
        print(f"   Precision: {gliner_metrics['precision']:.3f}")
        print(f"   Recall:    {gliner_metrics['recall']:.3f}")
        print(f"   F1 Score:  {gliner_metrics['f1']:.3f}")
        print(f"   Time:      {gliner_time:.2f}s")
    else:
        print("   Skipped (not installed)")
    
    # spaCy
    print("\n🔹 Benchmarking spaCy...")
    spacy_metrics, spacy_time = benchmark_spacy(test_data)
    if spacy_metrics:
        results["spaCy"] = {"metrics": spacy_metrics, "time": spacy_time}
        print(f"   Precision: {spacy_metrics['precision']:.3f}")
        print(f"   Recall:    {spacy_metrics['recall']:.3f}")
        print(f"   F1 Score:  {spacy_metrics['f1']:.3f}")
        print(f"   Time:      {spacy_time:.2f}s")
    else:
        print("   Skipped (not installed)")
    
    # Speed benchmark
    print("\n" + "=" * 70)
    print("Speed Benchmark (10,000 characters)")
    print("=" * 70)
    speed_results = benchmark_speed_only(10000)
    for model_name, data in speed_results.items():
        print(f"\n🔹 {model_name.upper()}")
        print(f"   Time:       {data['time']:.2f}s")
        print(f"   Entities:   {data['entities']}")
        print(f"   Speed:      {data['chars_per_sec']:.0f} chars/sec")
    
    # Summary Table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12} {'Time (s)':<12}")
    print("-" * 70)
    for model_name, data in results.items():
        m = data["metrics"]
        print(f"{model_name:<12} {m['precision']:<12.3f} {m['recall']:<12.3f} {m['f1']:<12.3f} {data['time']:<12.2f}")
    
    # Save results
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
