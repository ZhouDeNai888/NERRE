import json
import os
from datasets import load_dataset
from tqdm import tqdm
import traceback # เพิ่มบรรทัดนี้ด้านบนสุด


def reconstruct_text_from_tokens(tokens):
    """
    ฟังก์ชันช่วย: แปลง List of Tokens กลับเป็น String และคำนวณตำแหน่ง Char Offset
    เช่น ["I", "love", "AI"] -> "I love AI", [(0,1), (2,6), (7,9)]
    """
    text = ""
    token_map = [] # เก็บ mapping ว่า token ที่ i เริ่มและจบที่ตัวอักษรไหน
    
    current_char = 0
    for token in tokens:
        # สมมติว่าเชื่อมด้วย Space (อาจจะไม่เป๊ะ 100% สำหรับทุกภาษาแต่ใช้เทรนได้)
        start = current_char
        end = start + len(token)
        text += token + " " # เติม space ท้ายคำ
        token_map.append((start, end))
        current_char = end + 1 # +1 สำหรับ space
        
    return text.strip(), token_map

def process_cross_re(dataset_name="DFKI-SLT/cross_re", split="train"):
    """
    รองรับโครงสร้างใหม่ของ CrossRE:
    - Text: item['sentence']
    - NER: item['ner'] [{'id-start', 'id-end', 'entity-type'}]
    - Rel: item['relations'] [{'id_1-start', ...}]
    """
    
    # กำหนด Domain ที่ต้องการโหลด (ตามที่คุณแปะมามี ai, literature, music ฯลฯ)
    # เราจะวนลูปโหลดทีละโดเมนเลยครับ จะได้ข้อมูลเยอะๆ
    domains = ["ai", "literature", "music", "politics", "science"]
    all_formatted_data = []

    for domain in domains:
        print(f"Loading CrossRE domain: {domain}...")
        try:
            # โหลด Dataset (สังเกต: ไม่ต้องใช้ trust_remote_code แล้วสำหรับเวอร์ชันนี้)
            ds = load_dataset(dataset_name, domain, split=split)
        except Exception as e:
            print(f"⚠️ Warning: Could not load domain {domain}. Skipping. Error: {e}")
            continue

        for item in tqdm(ds, desc=f"Processing {domain}"):
            # 1. แก้ชื่อคอลัมน์จาก 'tokens' เป็น 'sentence'
            tokens = item['sentence'] 
            text, token_map = reconstruct_text_from_tokens(tokens)
            
            # 2. Extract Entities (จาก List of Dicts)
            entities = []
            
            # item['ner'] หน้าตาแบบ: [{'id-start': 0, 'id-end': 0, 'entity-type': '...'}]
            # หมายเหตุ: id-end ใน dataset นี้เป็นแบบ Inclusive (รวมตัวสุดท้ายด้วย)
            raw_ner_list = item['ner']
            
            for ent_data in raw_ner_list:
                start_token_idx = ent_data['id-start']
                end_token_idx = ent_data['id-end']
                label = ent_data['entity-type']
                
                # แปลง Token Index -> Char Offset
                start_char = token_map[start_token_idx][0]
                end_char = token_map[end_token_idx][1]
                
                entities.append({
                    "start": start_char,
                    "end": end_char,
                    "label": label,
                    # เก็บ token span ไว้เทียบกับ relation (เดี๋ยวลบทิ้งตอนจบ)
                    "_token_span": (start_token_idx, end_token_idx) 
                })
            
            # 3. Extract Relations (Mapping Spans)
            processed_relations = []
            raw_rel_list = item['relations']
            
            for rel_data in raw_rel_list:
                # Relation ระบุเป็น Span ของคู่กรณี
                # arg1 span
                arg1_span = (rel_data['id_1-start'], rel_data['id_1-end'])
                # arg2 span
                arg2_span = (rel_data['id_2-start'], rel_data['id_2-end'])
                rel_label = rel_data['relation-type']
                
                # ค้นหาว่า Span นี้ตรงกับ Entity ตัวไหนในลิสต์ของเรา
                head_idx = -1
                tail_idx = -1
                
                for i, ent in enumerate(entities):
                    if ent['_token_span'] == arg1_span:
                        head_idx = i
                    if ent['_token_span'] == arg2_span:
                        tail_idx = i
                
                # ถ้าเจอคู่ครบ ก็บันทึก
                if head_idx != -1 and tail_idx != -1:
                    processed_relations.append({
                        "head": head_idx,
                        "tail": tail_idx,
                        "label": rel_label
                    })
            
            # Clean up temporary keys
            for ent in entities:
                if '_token_span' in ent:
                    del ent['_token_span']

            all_formatted_data.append({
                "text": text,
                "entities": entities,
                "relations": processed_relations
            })
            
    return all_formatted_data

# แก้ไขฟังก์ชัน process_few_rel
def process_few_rel(dataset_name="thunlp/few_rel", split="train_wiki"):
    print(f"Loading {dataset_name}...")
    try:
        # [FIX 1] เพิ่ม trust_remote_code=True
        ds = load_dataset(dataset_name, split=split, trust_remote_code=True)
    except Exception as e:
        # [FIX 2] ปริ้นท์ Error ตัวจริงออกมาดู
        print(f"❌ Error loading {dataset_name}: {e}")
        traceback.print_exc()
        return []
    
    formatted_data = []
    
    for item in tqdm(ds, desc="Processing FewRel"):
        tokens = item['tokens']
        text, token_map = reconstruct_text_from_tokens(tokens)
        
        # FewRel โครงสร้าง: {'head': {'text': '...', 'indices': [[1, 2]]}, ...}
        head_info = item['head']
        tail_info = item['tail']
        rel_labels = item['names'] # อาจมีหลายชื่อ
        rel_label = rel_labels[0] if isinstance(rel_labels, list) else rel_labels
        
        entities = []
        
        # Helper เพื่อสร้าง Entity จาก Indices ของ FewRel
        def add_entity(info, label_placeholder="Entity"):
            # FewRel indices เป็น list of lists (เผื่อคำแยกกัน) เอาตัวแรกพอ
            token_idxs = info['indices'][0] 
            start_token = token_idxs[0]
            end_token = token_idxs[-1] # inclusive?
            
            start_char = token_map[start_token][0]
            end_char = token_map[end_token][1]
            
            entities.append({
                "start": start_char,
                "end": end_char,
                "label": "Entity" # FewRel อาจไม่มี Type ให้ใช้ Generic ไปก่อน
            })
            return len(entities) - 1 # คืนค่า index

        head_idx = add_entity(head_info, "Head")
        tail_idx = add_entity(tail_info, "Tail")
        
        formatted_data.append({
            "text": text,
            "entities": entities,
            "relations": [{
                "head": head_idx,
                "tail": tail_idx,
                "label": rel_label
            }]
        })
        
    return formatted_data

def process_tacred(dataset_name="tacred", split="train"):
    """
    TACRED มักจะต้องโหลดผ่าน LDC หรือเวอร์ชันชุมชน
    ถ้าใช้ 'DFKI-SLT/tacred' ไม่ได้ อาจต้องใช้ 'knowledgator/tacred' หรือไฟล์ json local
    ฟังก์ชันนี้เขียนเผื่อไว้สำหรับ Structure ทั่วไป
    """
    # ... (Logic คล้าย FewRel โดยดู field subj_start, obj_start) ...
    return []

def generate_merged_dataset(output_file="train_data.json"):
    all_data = []
    
    # 1. Load CrossRE (ตัวอย่าง load 2 โดเมน)
    all_data.extend(process_cross_re("DFKI-SLT/cross_re", split="train"))
    
    # 2. Load FewRel
    # all_data.extend(process_few_rel("thunlp/few_rel", split="train_wiki"))
    
    # 3. Load TACRED (ถ้ามี)
    # all_data.extend(process_tacred())

    print(f"Total samples collected: {len(all_data)}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved merged dataset to {output_file}")

if __name__ == "__main__":
    # รันไฟล์นี้ตรงๆ เพื่อสร้างไฟล์ JSON
    generate_merged_dataset(output_file="dataset/data_v2.json")