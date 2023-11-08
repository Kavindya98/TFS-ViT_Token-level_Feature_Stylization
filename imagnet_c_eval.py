import json
import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    
    parser.add_argument('--input_dir', type=str, default="test_ViT_RB")
    parser.add_argument('--cases', type=int, default=0)
    args = parser.parse_args()

    data = json.load(open(os.path.join(args.input_dir,'results.jsonl')))

    domain = {"blur":[], "digital":[], "noise":[], "weather":[], "real":[]}

    for k in data:
        if "env" in k:
            domain[k.split("_")[1]].append(data[k])
            
    print(domain)

    result = dict()
    for d in domain:
        result[d] = sum(domain[d])/len(domain[d])

    epochs_path = os.path.join(args.input_dir, 'Overall_results.jsonl')
    with open(epochs_path, 'a') as f:
        f.write(json.dumps(result, sort_keys=True) + "\n")
