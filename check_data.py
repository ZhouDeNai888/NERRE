import json
from transformers import AutoTokenizer
from data.GraphRAGDataset import GraphRAGDataset # Import class Dataset ของคุณ



# Config
MODEL_NAME = "xlm-roberta-base" # หรือ Large ตามที่คุณใช้
DATA_FILE = "dataset/multilingual_data_v5_10000.json"

print("⏳ Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("📂 Loading Dataset...")
# โหลดแบบไม่สุ่ม Negative (เพื่อเช็คของจริงเน้นๆ)
ds = GraphRAGDataset(DATA_FILE, tokenizer, neg_span_ratio=0.0)

print(f"\n--- 🔍 Validating {len(ds)} samples ---")
total_entities_json = 0
total_entities_valid = 0

# สุ่มเช็คสัก 1000 ตัวอย่าง หรือเช็คหมดเลยก็ได้
for i in range(min(len(ds), 1000)):
    item = ds.data[i]
    raw_entities_count = len(item.get('entities', []))
    total_entities_json += raw_entities_count
    
    # ดูว่า Dataset Class แปลงเป็น Token ได้กี่ตัว
    processed_item = ds[i]
    valid_count = processed_item['num_positive_spans']
    total_entities_valid += valid_count
    
    # ถ้าหายไปเยอะ ให้โวยวาย
    if valid_count < raw_entities_count:
        print(f"⚠️ Sample {i} Lost Entities: Has {raw_entities_count}, Kept {valid_count}")
        print(f"   Text: {item['text'][:100]}...")

print("\n" + "="*40)
print(f"📊 SUMMARY REPORT")
print(f"Total Entities in JSON: {total_entities_json}")
print(f"Total Entities Validated: {total_entities_valid}")
loss_rate = (total_entities_json - total_entities_valid) / total_entities_json * 100
print(f"📉 Data Loss Rate: {loss_rate:.2f}%")
print("="*40)

if loss_rate > 10:
    print("❌ Critical Issue: Tokenizer cannot align with your entity indices.")
    print("   Solution: Add spaces around entities in your templates.")
else:
    print("✅ Data looks good. The issue might be Learning Rate.")