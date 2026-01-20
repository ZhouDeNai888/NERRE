# NERRE: Named Entity & Relation Extraction for Graph RAG

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-orange" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-4.30%2B-green" alt="Transformers">
  <img src="https://img.shields.io/badge/License-MIT-yellow" alt="License">
</p>

**NERRE** is a lightweight, multilingual Named Entity Recognition (NER) and Relation Extraction (RE) pipeline designed for Graph RAG applications. It extracts entities and their relationships from text, perfect for building knowledge graphs.

## ✨ Features

- 🌍 **Multilingual Support**: English, Chinese (Traditional/Simplified), Japanese, Korean, Thai
- 🚀 **Zero External Dependencies**: No need for jieba, MeCab, or other tokenizers - the model learns word boundaries automatically
- 📏 **Long Text Support**: Sliding window mechanism handles documents of any length
- 🎯 **Joint NER+RE**: Extracts both entities AND their relationships in one pass
- ⚡ **High Accuracy**: State-of-the-art F1 scores across languages

## 📊 Benchmark Results

### Relation Extraction (RE) - Comparison with SpERT

Strict matching benchmark on aligned test set (100 samples, 155 relations):

| Model | Precision | Recall | **F1 Score** | Training Type |
|-------|-----------|--------|--------------|---------------|
| **NERRE (Ours)** | 0.691 | **0.812** | **0.747** | Zero-shot |
| SpERT (CoNLL04) | - | - | 0.726 | Supervised |

> 📈 **NERRE outperforms SpERT by 2.1%** in RE F1 score while being **zero-shot** (no task-specific training)!

### Named Entity Recognition (NER) - Full Comparison

| Model | Precision | Recall | **F1 Score** | Time (s) | Notes |
|-------|-----------|--------|--------------|----------|-------|
| **NERRE (Ours)** | 0.838 | **0.965** | **0.897** | 2.35 | Best F1, fastest |
| GLiNER | **0.916** | 0.830 | 0.871 | 386.00 | High precision |
| NuNER Zero | 0.939 | 0.725 | 0.818 | 757.59 | Zero-shot NER |
| NuNER Zero-span | 0.937 | 0.723 | 0.817 | 757.59 | Span variant |
| spaCy | 0.793 | 0.561 | 0.658 | 0.44 | English-only |

### Key Findings

- **NERRE achieves the best F1 score (0.897)** with excellent recall (96.5%)
- **NERRE is the only model that extracts both NER and RE** in a single pass
- **GLiNER/NuNER** have high precision but miss more entities
- **spaCy** is fast but only supports English, missing all CJK entities

### Speed Benchmark (10,000 characters)

| Model | Time | Entities | Triplets | Speed |
|-------|------|----------|----------|-------|
| spaCy | 0.18s | 500 | ❌ | 55,169 chars/s |
| **NERRE Fast** ⚡ | **0.97s** | **876** | ❌ | **10,310 chars/s** |
| GLiNER | 9.26s | 144 | ❌ | 1,080 chars/s |
| **NERRE Full** | **12.74s** | **877** | ✅ 36 | 785 chars/s |

### Key Takeaways

- **NERRE Fast** is **9.5x faster than GLiNER** while finding **6x more entities**
- **NERRE Full** extracts both entities AND relations (triplets) - no other tool does this!
- **NERRE** supports CJK languages natively, unlike spaCy

## 🚀 Quick Start

### Installation

```bash
pip install torch transformers huggingface_hub
```

### Usage

```python
from nerre import NERREPipeline

# Load the model
pipe = NERREPipeline.from_pretrained("path/to/model")

# English
result = pipe("Elon Musk founded SpaceX in 2002.")
print(result["entities"])
# [{'text': 'Elon Musk', 'label': 'person', 'score': 0.98},
#  {'text': 'SpaceX', 'label': 'organisation', 'score': 0.96},
#  {'text': '2002', 'label': 'date', 'score': 0.99}]

print(result["triplets"])
# [{'head': 'Elon Musk', 'relation': 'founder_of', 'tail': 'SpaceX', 'confidence': 0.92}]

# Chinese (auto-detected, no external tokenizer needed!)
result = pipe("台積電是全球最大的晶片製造商，張忠謀於1987年在新竹創立。")
print(result["entities"])
# [{'text': '台積電', 'label': 'organisation', 'score': 0.95},
#  {'text': '張忠謀', 'label': 'person', 'score': 0.94},
#  {'text': '1987年', 'label': 'date', 'score': 0.96},
#  {'text': '新竹', 'label': 'location', 'score': 0.93}]

# Japanese (also auto-detected!)
result = pipe("ソニーは盛田昭夫によって1946年に東京で設立されました。")
```

### Fast Mode (NER Only)

For maximum speed when you only need entity extraction:

```python
# Fast Mode: ~10x faster, NER only (no relation extraction)
result = pipe(text, fast_mode=True, extract_relations=False)
# Result: 10,310 chars/sec - faster than GLiNER!

# Full Mode: NER + Relation Extraction (default)
result = pipe(text)
# Result: 785 chars/sec with triplets
```

### Long Text Support

NERRE automatically handles long texts using a sliding window approach:

```python
# Works with texts of any length (10,000+ characters)
long_article = "..." * 10000  # Very long article
result = pipe(long_article)  # No truncation, processes entire document
```

## 🏗️ Architecture

NERRE uses a **span-based** approach built on XLM-RoBERTa:

1. **Character-level Span Generation**: For CJK languages, each character is a potential entity boundary
2. **Joint Entity & Relation Classification**: Single forward pass for both tasks
3. **Smart Pair Filtering**: Only checks entity pairs that could form valid relations
4. **Non-Maximum Suppression (NMS)**: Removes overlapping entity predictions
5. **Sliding Window**: Handles documents longer than 512 tokens

### Model Configuration

- **Base Model**: `xlm-roberta-base` (280M parameters)
- **Max Sequence Length**: 512 tokens (with sliding window for longer texts)
- **Entity Types**: `person`, `organisation`, `location`, `date`, `product`, `programlang`
- **Relation Types**: `founder_of`, `ceo_of`, `developed`, `creator_of`, `founded_in`, `released_in`, `located_in`

## 📁 Project Structure

```
NERRE/
├── huggingface_release/    # Production-ready inference code
│   ├── nerre.py           # Main pipeline
│   ├── model.py           # Model architecture
│   └── pytorch_model.bin  # Model weights
├── train/
│   ├── train.py           # Training script
│   └── train_config.py    # Training configuration
├── data/
│   └── hf_dataloader.py   # Data loading utilities
├── dataset/
│   ├── multilingual_data_v3_5000.json   # 5,000 samples
│   ├── multilingual_data_v3_10000.json  # 10,000 samples
│   ├── generate_large_dataset.py        # Dataset generator v1 (7 entity types)
│   └── generate_large_dataset_v2.py     # Dataset generator v2 (100 types)
├── eval/
│   └── eval.py            # Evaluation script
├── benchmark/
│   └── benchmark_ner.py   # Benchmark vs GLiNER, spaCy
└── README.md
```

## 📊 Training Datasets

We provide pre-generated multilingual datasets for training:

| Dataset | Samples | Entities | Relations | Entity Types | Relation Types |
|---------|---------|----------|-----------|--------------|----------------|
| `multilingual_data_v4_10000.json` | 10,000 | 22,952 | 12,952 | 24 | 33 |
| `multilingual_data_v3_10000.json` | 10,000 | 29,894 | 20,486 | 7 | 7 |
| `multilingual_data_v3_5000.json` | 5,000 | 14,816 | 10,114 | 7 | 7 |

### Supported Entity Types (24 types in v4)

| Category | Types |
|----------|-------|
| **Person** | person, athlete, musician, actor, director, author, scientist, engineer, entrepreneur, journalist |
| **Organization** | organisation, university, sports_team |
| **Location** | location |
| **Time** | date |
| **Product** | product, movie, book, award, programlang, framework, ai_model, competition, event |

### Supported Relation Types (33 types in v4)

| Category | Relations |
|----------|-----------|
| **Creation** | founder_of, creator_of, developed, author_of, director_of, composed_by |
| **Employment** | ceo_of, works_at, professor_at, research_at |
| **Location** | located_in, headquartered_in, born_in, part_of |
| **Time** | founded_in, released_in, graduated_in, started_in, occurred_on, married_on |
| **Education** | studied_at, graduated_from |
| **Sports** | plays_for, signed_with, champion_of |
| **Business** | acquired_by, subsidiary_of, investor_in |
| **Awards** | won |
| **Media** | starred_in, featured_in, performed_at |
| **Family** | spouse_of |

### Generate Custom Dataset

```bash
# Show all 100 entity types and 100 relation types
python dataset/generate_large_dataset_v2.py --show-types

# Generate 10000 samples (v2 with diverse types)
python dataset/generate_large_dataset_v2.py --count 10000 --output my_dataset.json

# Generate 5000 samples (v1 with 7 types)
python dataset/generate_large_dataset.py --count 5000 --output my_dataset.json --seed 42
```

## 🔧 Training Your Own Model

### 1. Prepare Training Data

Create a JSON file with the following format:

```json
[
  {
    "text": "Elon Musk founded SpaceX in 2002.",
    "entities": [
      {"text": "Elon Musk", "label": "person", "start": 0, "end": 9},
      {"text": "SpaceX", "label": "organisation", "start": 18, "end": 24},
      {"text": "2002", "label": "date", "start": 28, "end": 32}
    ],
    "relations": [
      {"head": "Elon Musk", "tail": "SpaceX", "type": "founder_of"}
    ]
  }
]
```

### 2. Train

```bash
cd train
python train.py
```

### 3. Evaluate

```bash
cd eval
python eval.py
```

## 📜 License

MIT License

## 🙏 Acknowledgments

- [XLM-RoBERTa](https://huggingface.co/xlm-roberta-base) by Facebook AI
- [GLiNER](https://github.com/urchade/GLiNER) for benchmark comparison
- [spaCy](https://spacy.io/) for benchmark comparison

## 📚 Citation

If you use NERRE in your research, please cite:

```bibtex
@software{nerre2024,
  title = {NERRE: Named Entity & Relation Extraction for Graph RAG},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/NERRE}
}
```