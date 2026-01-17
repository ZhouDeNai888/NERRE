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

Comparison against popular NER tools on multilingual test set (English + Chinese + Japanese):

| Model | Precision | Recall | **F1 Score** | Notes |
|-------|-----------|--------|--------------|-------|
| **NERRE (Ours)** | 0.959 | **0.986** | **0.973** | Best overall |
| GLiNER | **0.978** | 0.738 | 0.841 | Lower recall |
| spaCy | 0.971 | 0.557 | 0.708 | English-only |

### Key Findings

- **NERRE achieves the highest F1 score (0.973)** with balanced precision and recall
- **GLiNER** has high precision but misses many entities (lower recall)
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
│   └── *.json             # Training data
├── eval/
│   └── eval.py            # Evaluation script
├── benchmark/
│   └── benchmark_ner.py   # Benchmark vs GLiNER, spaCy
└── README.md
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