import sys
import json
import numpy as np
import argparse
import pickle
from collections import defaultdict
"""
Tool to visualize predictions
"""
parser = argparse.ArgumentParser()
parser.add_argument("predictions_path", help="Path to predictions pickle file")
parser.add_argument("-dataset_path", help="Path to smaug dataset that contains input instances and label names")
parser.add_argument("-label_names", help="Path to override label names")
parser.add_argument("-tag", help="Tag to match indexes to the prediction container.", default="test", required=False)
parser.add_argument("-role", help="Role (i.e. train/test/val) to match the dataset container.", default="test", required=False)
parser.add_argument("--only_mistakes", help="Whether to only display misclassified instances", default=True)


args = parser.parse_args()
with open(args.predictions_path, "rb") as f:
    data = pickle.load(f)

preds, usages, tags, label_info = data
label_names = label_info["labelnames"]
match_idx = [t for t, tg in enumerate(tags) if tg == args.tag]
if len(match_idx) != 1:
    print(f"Found {len(match_idx)} matching tags equal to {args.tag}, need exactly 1")

match_idx = match_idx[0]
indexes = usages[match_idx]
predictions = preds[indexes, :]

with open(args.dataset_path) as f:
    data = json.load(f)
    data = data['data'][args.role]

perlabel = defaultdict(list)
num_misclassified = 0
for i, dat in enumerate(data):
    true_label = dat["labels"]
    assert len(true_label) == 1, f"Multilabel sample #{i+1}/len(data)"
    true_label = true_label[0]
    true_numeric = label_names.index(true_label)
    if not true_label in perlabel:
        perlabel[true_label] = []

    obj = {}
    obj["text"] = dat["text"]
    obj["label"] = true_label
    obj["predictions"] = predictions[i]
    prediction = np.argmax(obj["predictions"])

    if prediction == true_numeric and args.only_mistakes:
        continue
    num_misclassified += 1
    perlabel[true_label].append(obj)

    print("Text:", obj["text"])
    print("True label:", obj["label"])
    labels_preds = sorted(zip(label_names, obj["predictions"]), key=lambda x: x[1], reverse=True)
    for label, prob in labels_preds:
        print(f"\t{label:<20}: {prob:.3f}")
print(f"A total of {num_misclassified} items were misclassified")
