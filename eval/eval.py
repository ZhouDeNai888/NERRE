import os
import sys
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import classification_report
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from model.model import ZeroShotJointModel
from data.ZeroShotDataset import ZeroShotDataset
import train.train_config as config
from transformers import AutoTokenizer

class ZeroShotEvalDataset(ZeroShotDataset):
    """
    Subclass of ZeroShotDataset for Evaluation.
    Instead of sampling negatives, it returns targets for ALL labels.
    """
    def __init__(self, json_file, tokenizer, max_len=512):
        # Initialize parent (which scans for all labels)
        super().__init__(json_file, tokenizer, max_len, neg_sample_ratio=0.0) 
        
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        
        # 1. Tokenize Text
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=self.max_len
        )
        
        # 2. Align Spans
        span_list, valid_ent_idxs = self._align_labels_with_tokens(inputs, item.get('entities', []))
        
        # 3. Create Targets for ALL labels
        # Shape: [Num_Spans, Num_All_Labels]
        num_spans = len(span_list)
        num_all_labels = len(self.all_ent_labels)
        
        ent_targets = torch.zeros((num_spans, num_all_labels))
        
        # For each valid span, mark the correct label
        for i, original_idx in enumerate(valid_ent_idxs):
            true_label_str = item['entities'][original_idx]['label']
            if true_label_str in self.all_ent_labels:
                label_idx = self.all_ent_labels.index(true_label_str)
                ent_targets[i, label_idx] = 1.0
                
        # (Optional) Handle Relations similarly if needed
        # For now, we focus on Entity Evaluation
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "spans": span_list,                 
            "ent_targets": ent_targets,
            "ent_labels_list": self.all_ent_labels # Just for reference if needed
        }

def eval_collate_fn(batch):
    """
    Custom collate for Evaluation
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    spans = [item['spans'] for item in batch]
    ent_targets = [item['ent_targets'] for item in batch] # List of Tensors [Num_Spans, Num_Classes]
    
    return {
        "text_ids": input_ids,
        "text_mask": attention_mask,
        "spans": spans,
        "ent_targets": ent_targets,
    }

def evaluate(model_path, data_file):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Evaluating on: {device}")
    
    # 1. Load Resources
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    if not os.path.exists(data_file):
        print(f"Dataset not found at {data_file}")
        return

    dataset = ZeroShotEvalDataset(data_file, tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, collate_fn=eval_collate_fn)
    
    model = ZeroShotJointModel(config.MODEL_NAME).to(device)
    
    # Load Weights
    weight_path = os.path.join(model_path, "pytorch_model.bin")
    if os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"Loaded weights from {weight_path}")
    else:
        print(f"Warning: No weights found at {weight_path}, using random init.")

    model.eval()
    
    # 2. Set Global Labels (Cache them)
    all_labels = dataset.all_ent_labels
    print(f"Evaluating on {len(all_labels)} labels: {all_labels}")
    
    # Tokenize labels
    # We treat all labels as a single batch for encoding
    label_inputs = tokenizer(all_labels, return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Prepare global label inputs
    # Shape needed for encode_labels: [Batch, Num_Labels, Seq]
    # Here we can just treat it as Batch=1, Num_Labels=Total
    ent_label_ids = label_inputs['input_ids'].unsqueeze(0)
    ent_label_mask = label_inputs['attention_mask'].unsqueeze(0)
    
    # Dummy Relation Labels (since we focus on NER for now)
    rel_label_ids = torch.zeros_like(ent_label_ids) 
    rel_label_mask = torch.zeros_like(ent_label_mask)

    model.set_global_labels(ent_label_ids, ent_label_mask, rel_label_ids, rel_label_mask)
    
    # 3. Inference Loop
    all_preds = []
    all_trues = []
    
    # Mapping index to label
    idx2label = {i: label for i, label in enumerate(all_labels)}
    
    # Include 'O' (Outside/No Label) if your logic supports it. 
    # Current ZeroShot model treats everything as a potential match via Dot Product.
    # Usually standard NER eval requires 'O' tag for non-entities, 
    # but here we are evaluating *given spans*.
    # So we are classifying: "Given this span, what is its label?"
    # If the span is not an entity, the dataset wouldn't have it (or label it O).
    # Since we use `valid_ent_idxs` from dataset, we only evaluate on specific spans that *are* entities in GT.
    # If the model predicts a specific label with highest score, we check if it matches.
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            text_ids = batch['text_ids'].to(device)
            text_mask = batch['text_mask'].to(device)
            spans = batch['spans']
            targets_list = batch['ent_targets'] # List of [Num_Spans, Num_Classes]
            
            # Forward (No labels passed, uses cached)
            ent_logits, _ = model(
                text_ids, text_mask,
                ent_label_ids=None, ent_label_mask=None,
                rel_label_ids=None, rel_label_mask=None,
                entity_spans=spans
            )
            # ent_logits: [Batch, Max_Spans, Num_Classes]
            
            # Process results per sample
            for b in range(len(spans)):
                num_spans = len(spans[b])
                if num_spans == 0: continue
                
                # Get logits for this sample's spans
                # Logits shape: [Max_Spans, Num_Classes] -> slice to [Num_Spans, Num_Classes]
                logits = ent_logits[b, :num_spans, :]
                
                # Get Targets
                targets = targets_list[b].to(device) # [Num_Spans, Num_Classes] (One-hot)
                
                # Convert One-hot targets to Class Indices
                # Assuming single-label per span for now
                true_indices = torch.argmax(targets, dim=-1).cpu().numpy()
                
                # Predictions
                pred_indices = torch.argmax(logits, dim=-1).cpu().numpy()
                
                all_preds.extend(pred_indices)
                all_trues.extend(true_indices)

    # 4. Metrics
    print("\nClassification Report:")
    print(classification_report(all_trues, all_preds, target_names=all_labels))
    
if __name__ == "__main__":
    # Default to data.json if exists, else whatever is in config
    DATA_FILE = config.TRAIN_FILE
    MODEL_DIR = config.OUTPUT_DIR # "saved_model_v1"
    
    evaluate(MODEL_DIR, DATA_FILE)
