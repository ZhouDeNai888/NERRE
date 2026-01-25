import json
import os
import argparse

# ==========================================
# 🔧 NORMALIZATION CONFIGURATION
# ==========================================

# 1. Entity Mappings (Canonical Names)
# Format: "Original Label": "New Unified Label"
ENTITY_MAPPING = {
    # CoNLL04
    "Peop": "person",
    "Loc": "location",
    "Org": "organization",
    "Other": "other",
    
    # TACRED / Common NER
    "PER": "person",
    "ORG": "organization",
    "LOC": "location",
    "GPE": "location", # Geopolitical Entity often merged with Location
    "FAC": "facility",
    "VEH": "vehicle",
    
    # CrossRE / Wikipedia
    "person": "person", # Keep existing
    "organisation": "organization", # Unified spelling
    "organization": "organization",
    "location": "location",
    "place": "location"
}

# 2. Relation Mappings
RELATION_MAPPING = {
    # CoNLL04
    "Work_For": "employee_of",
    "Kill": "killed_by",        # Check directionality! (Usually Subject Kills Object) -> but "killed_by" is clearer if passive?? 
                                # Actually CoNLL04 "Kill" is [Assassin, Victim]. So "killed" or "victim_is" might be better. 
                                # Let's use "kill" or "killed_by" depending on preference.
                                # Let's stick to lower_snake_case for now.
    "OrgBased_In": "org_based_in",
    "Live_In": "lived_in",
    "Located_In": "located_in",

    # TACRED (examples - cleaning up colons)
    "per:title": "title",
    "per:employee_of": "employee_of",
    "org:founded_by": "founded_by",
    "org:top_members/employees": "top_members",
}

def normalize_label(label, mapping):
    """
    Look up label in mapping. 
    1. Try exact match.
    2. Try lower case match.
    3. Return original (lowercased/cleaned) if not found.
    """
    if not label: 
        return "unknown"
        
    # 1. Exact map
    if label in mapping:
        return mapping[label]
        
    # 2. Case insensitive map
    # (Create a temp dict for this lookup if efficient, but loop is fine for small schemas)
    for k, v in mapping.items():
        if k.lower() == label.lower():
            return v
            
    # 3. Fallback: Lower case & snake_case
    return label.lower().replace(":", "_").replace("-", "_").replace(" ", "_")

def normalize_file(input_path, output_path):
    print(f"🔄 Processing: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"❌ File not found: {input_path}")
        return

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    normalized_data = []
    
    stats = {
        "entities_changed": 0,
        "relations_changed": 0
    }
    
    for item in data:
        new_item = item.copy()
        
        # 1. Normalize Entities
        new_ents = []
        if 'entities' in item:
            for ent in item['entities']:
                new_ent = ent.copy()
                old_lbl = ent['label']
                new_lbl = normalize_label(old_lbl, ENTITY_MAPPING)
                
                if old_lbl != new_lbl:
                    stats["entities_changed"] += 1
                
                new_ent['label'] = new_lbl
                new_ents.append(new_ent)
        new_item['entities'] = new_ents
        
        # 2. Normalize Relations
        new_rels = []
        if 'relations' in item:
            for rel in item['relations']:
                new_rel = rel.copy()
                old_lbl = rel.get('label', rel.get('type')) # Handle various naming keys
                
                if old_lbl:
                    new_lbl = normalize_label(old_lbl, RELATION_MAPPING)
                    
                    if old_lbl != new_lbl:
                        stats["relations_changed"] += 1
                        
                    new_rel['label'] = new_lbl
                    # Ensure consistency keys
                    if 'type' in new_rel: del new_rel['type'] 
                    
                new_rels.append(new_rel)
        new_item['relations'] = new_rels
        
        normalized_data.append(new_item)
        
    # Save
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_data, f, ensure_ascii=False, indent=2)
        
    print(f"✅ Saved to: {output_path}")
    print(f"   Entities Normalized: {stats['entities_changed']}")
    print(f"   Relations Normalized: {stats['relations_changed']}")

if __name__ == "__main__":
    # Example Usage
    files_to_process = [
        ("dataset/conll04_train.json", "dataset/conll04_train_normalized.json"),
        ("dataset/conll04_test.json", "dataset/conll04_test_normalized.json"),
        ("dataset/cross_re_train.json", "dataset/cross_re_train_normalized.json"),
        ("dataset/cross_re_test.json", "dataset/cross_re_test_normalized.json"),
        ("dataset/cross_re_validation.json", "dataset/cross_re_validation_normalized.json"),
        ("dataset/conll04_validation.json", "dataset/conll04_validation_normalized.json"),
    ]
    
    for inp, out in files_to_process:
        normalize_file(inp, out)
