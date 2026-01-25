import sys
import os
import collections

# Add dataset folder to path
sys.path.append(os.path.join(os.getcwd(), 'dataset'))

import generate_large_dataset_v2

def check():
    print("Generating 500 samples to check label consolidation...")
    # Generate small batch
    samples = generate_large_dataset_v2.generate_dataset(500)
    
    rel_counts = collections.Counter()
    for s in samples:
        for r in s.get("relations", []):
            rel_counts[r["label"]] += 1
    
    print("\nRelation Label Distribution:")
    for label, count in rel_counts.most_common():
        print(f"{label}: {count}")

if __name__ == "__main__":
    check()
