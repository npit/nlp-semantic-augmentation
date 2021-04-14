import sys
import os
import json

dataset_path = sys.argv[1]

with open(dataset_path) as f:
    data = json.load(f)["data"]

while True:
    term = input("Enter term to search dataset text, qq to quit: ")
    if term == "":
        os.system("clear")
        continue
    if term == "qq":
        break
    for ttest in data:
        print(ttest, ":")
        for x in data[ttest]:
            if term.lower() in x["text"].lower():
                print(x)
