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
    domains = ["ai", "literature", "music", "news", "politics", "science"]
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
                
                # Check bounds to avoid errors
                if start_token_idx >= len(token_map) or end_token_idx >= len(token_map):
                    continue

                # แปลง Token Index -> Char Offset
                start_char = token_map[start_token_idx][0]
                end_char = token_map[end_token_idx][1]
                
                # Extract entity text
                entity_text = text[start_char:end_char]
                
                entities.append({
                    "start": start_char,
                    "end": end_char,
                    "label": label,
                    "text": entity_text,  # ✅ Added text field
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

def process_tacred(dataset_name="DFKI-SLT/tacred", split="train"):
    print(f"Loading {dataset_name} ({split})...")
    try:
        # Note: TACRED is usually licensed. If this fails, user might need to login or use local files.
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return []

    processed_data = []
    
    for item in tqdm(ds, desc="Processing TACRED"):
        # 1. Get Tokens
        tokens = item.get('tokens') or item.get('sentence')
        if not tokens: continue
        
        text, token_map = reconstruct_text_from_tokens(tokens)
        
        # 2. Extract Entities & Relations
        # Check standard TACRED format (subj/obj indices) vs DFKI format (ner/relations lists)
        
        entities = []
        relations = []
        
        # --- Case A: DFKI Unified Format (like CrossRE) ---
        if 'ner' in item and 'relations' in item:
            # Re-use logic similar to CrossRE/Conll04
            raw_ner = item['ner']
            
            for ent in raw_ner:
                s_idx = ent.get('id-start', ent.get('start'))
                e_idx = ent.get('id-end', ent.get('end'))
                label = ent.get('entity-type', ent.get('type'))
                
                if s_idx is None or e_idx is None: continue
                if s_idx >= len(token_map) or e_idx >= len(token_map): continue
                
                start_char = token_map[s_idx][0]
                end_char = token_map[e_idx][1]
                entity_text = text[start_char:end_char]
                
                entities.append({
                    "start": start_char,
                    "end": end_char,
                    "label": label,
                    "text": entity_text,
                    "_token_span": (s_idx, e_idx)
                })

            raw_rels = item['relations']
            for rel in raw_rels:
                # Find indices matching spans
                # DFKI relations usually point to token spans
                if 'id_1-start' in rel:
                    span1 = (rel['id_1-start'], rel['id_1-end'])
                    span2 = (rel['id_2-start'], rel['id_2-end'])
                elif 'head' in rel and isinstance(rel['head'], list): # Check if head is [start, end]
                    # This case depends on format
                    pass 
                
                head_idx = -1
                tail_idx = -1
                
                for i, e in enumerate(entities):
                    if e.get('_token_span') == span1: head_idx = i
                    if e.get('_token_span') == span2: tail_idx = i
                    
                if head_idx != -1 and tail_idx != -1:
                    relations.append({
                        "head": head_idx,
                        "tail": tail_idx,
                        "label": rel['relation-type']
                    })

        # --- Case B: Standard TACRED Format (subj_start, obj_start...) ---
        elif 'subj_start' in item and 'obj_start' in item:
            # TACRED has 1 relation per sample, between SUBJ and OBJ
            
            # Subj
            s_start, s_end = item['subj_start'], item['subj_end']
            s_type = item['subj_type']
            
            # Obj
            o_start, o_end = item['obj_start'], item['obj_end']
            o_type = item['obj_type']
            
            rel_label = item['relation']
            
            # Helper to add entity if not exists (or just add both)
            # In TACRED, we focus on these two. Other entities might be in 'ner' list if available?
            # Standard TACRED has 'ner' field which is list of tags matching tokens length
            
            # Let's extract ALL entities from 'stanford_ner' / 'ner' tags if present, otherwise just subj/obj
            
            # Using Subj/Obj is safest for relation extraction
            # 1. Subj
            subj_char_start = token_map[s_start][0]
            subj_char_end = token_map[s_end][1]
            subj_text = text[subj_char_start:subj_char_end]
            
            entities.append({
                "start": subj_char_start,
                "end": subj_char_end,
                "label": s_type,
                "text": subj_text
            })
            head_idx = 0
            
            # 2. Obj
            obj_char_start = token_map[o_start][0]
            obj_char_end = token_map[o_end][1]
            obj_text = text[obj_char_start:obj_char_end]
            
            entities.append({
                "start": obj_char_start,
                "end": obj_char_end,
                "label": o_type,
                "text": obj_text
            })
            tail_idx = 1
            
            # Add relation (if not 'no_relation', but we keep it and let dataset class handle negative ratio)
            # Or filtering 'no_relation' depends on user preference. GraphRAGDataset uses "NO_RELATION"
            
            relations.append({
                "head": head_idx,
                "tail": tail_idx,
                "label": rel_label
            })
            
        # Cleanup
        for e in entities:
            if '_token_span' in e: del e['_token_span']
            
        processed_data.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })

    return processed_data

def process_conll04(dataset_name="DFKI-SLT/conll04", split="train"):
    print(f"Loading {dataset_name} ({split})...")
    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception as e:
        print(f"❌ Error loading {dataset_name}: {e}")
        return []

    processed_data = []
    for item in tqdm(ds, desc="Processing Conll04"):
        # Adapt keys if needed. Assuming DFKI structure.
        tokens = item.get('tokens') or item.get('sentence')
        if not tokens: continue
        
        text, token_map = reconstruct_text_from_tokens(tokens)
        
        # NER
        entities = []
        ner_list = item.get('entities') or item.get('ner') or []
        for ent in ner_list:
            # Check keys
            if 'id-start' in ent: # DFKI style (Inclusive)
                s_idx = ent['id-start']
                e_idx = ent['id-end']
                label = ent['entity-type']
                
                if s_idx >= len(token_map) or e_idx >= len(token_map): continue
                start_char = token_map[s_idx][0]
                end_char = token_map[e_idx][1]
                
            elif 'start' in ent: # Generic/Standard (Exclusive End)
                s_idx = ent['start']
                e_idx = ent['end'] 
                label = ent['type']
                
                # CoNLL04 is Exclusive End for 'end' field
                if s_idx >= len(token_map): continue
                if e_idx > len(token_map): continue # e_idx can be equal to len
                
                start_char = token_map[s_idx][0]
                if e_idx > 0:
                     end_char = token_map[e_idx - 1][1]
                else:
                     continue
            else:
                continue
                
            entities.append({
                "start": start_char,
                "end": end_char,
                "label": label,
                "text": text[start_char:end_char], # Keep aligned text
                "_token_span": (s_idx, e_idx) # Save original for rel matching
            })
            
        # Relations
        relations = []
        rel_list = item.get('relations') or []
        for rel in rel_list:
             label = rel.get('relation-type') or rel.get('type')
             
             head_idx = -1
             tail_idx = -1
             
             if 'id_1-start' in rel: # DFKI
                 h_span = (rel['id_1-start'], rel['id_1-end'])
                 t_span = (rel['id_2-start'], rel['id_2-end'])
                 
                 for i, e in enumerate(entities):
                     if e.get('_token_span') == h_span: head_idx = i
                     if e.get('_token_span') == t_span: tail_idx = i
             elif 'head' in rel:
                 # Check if int index
                 if isinstance(rel['head'], int):
                     head_idx = rel['head']
                     tail_idx = rel['tail']
            
             if head_idx != -1 and tail_idx != -1:
                 relations.append({
                     "head": head_idx,
                     "tail": tail_idx,
                     "label": label
                 })
        
        # Clean entities
        for e in entities:
            if '_token_span' in e: del e['_token_span']
            
        processed_data.append({
            "text": text,
            "entities": entities,
            "relations": relations
        })
        
    return processed_data

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
    # 1. Download CrossRE (DFKI-SLT/cross_re)
    print("🚀 Processing CrossRE Dataset...")
    crossre_data_train = process_cross_re("DFKI-SLT/cross_re", split="train")
    crossre_data_val = process_cross_re("DFKI-SLT/cross_re", split="validation")
    crossre_data_test = process_cross_re("DFKI-SLT/cross_re", split="test")



    if crossre_data_train:
        os.makedirs("dataset", exist_ok=True)
        with open("dataset/cross_re_train.json", "w", encoding="utf-8") as f:
            json.dump(crossre_data_train, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(crossre_data_train)} samples to dataset/cross_re_train.json")

    if crossre_data_val:
        with open("dataset/cross_re_validation.json", "w", encoding="utf-8") as f:
            json.dump(crossre_data_val, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(crossre_data_val)} samples to dataset/cross_re_validation.json")

    if crossre_data_test:
        with open("dataset/cross_re_test.json", "w", encoding="utf-8") as f:
            json.dump(crossre_data_test, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(crossre_data_test)} samples to dataset/cross_re_test.json")


    print("\n🚀 Processing conll04 Dataset...")

    # 2. Download CoNLL04 (optional)
    conll_data_train = process_conll04("DFKI-SLT/conll04", split="train")
    # 3. Download TACRED (DFKI-SLT/tacred)
    print("🚀 Processing TACRED Dataset...")
    tacred_data = process_tacred("DFKI-SLT/tacred", split="train")
    if tacred_data:
        # Save
        with open("dataset/tacred_train.json", "w", encoding="utf-8") as f:
            json.dump(tacred_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(tacred_data)} samples to dataset/tacred_train.json")   
    conll_data_test = process_conll04("DFKI-SLT/conll04", split="test")
    conll_data_val = process_conll04("DFKI-SLT/conll04", split="validation")
    
    # Save separately or merge? The user just said "download".
    # Saving to 'dataset/conll04_train.json' and 'dataset/conll04_test.json'
    
    if conll_data_train:
        with open("dataset/conll04_train.json", "w", encoding="utf-8") as f:
            json.dump(conll_data_train, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(conll_data_train)} samples to dataset/conll04_train.json")
        
    if conll_data_test:
        with open("dataset/conll04_test.json", "w", encoding="utf-8") as f:
            json.dump(conll_data_test, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(conll_data_test)} samples to dataset/conll04_test.json")

    if conll_data_val:
        with open("dataset/conll04_validation.json", "w", encoding="utf-8") as f:
            json.dump(conll_data_val, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(conll_data_val)} samples to dataset/conll04_validation.json")

    # generate_merged_dataset(output_file="dataset/data_v2.json")
