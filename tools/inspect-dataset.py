#!/usr/bin/env python3
import argparse
import json

import pandas

"""Read and output stat. information on a jsonic dataset"""


def inspect(dataset_path):
    with open(dataset_path) as f:
        dset = json.load(f)

    traintest = ('train', 'test')
    samples = {x: 0 for x in traintest}
    mean_num_words = {x: 0 for x in traintest}

    # for summarization datasets
    label_samples = {x: {} for x in traintest}
    mean_sentences_per_doc = {x: 0 for x in traintest}
    mean_summary_sentences_per_doc = {x: 0 for x in traintest}

    for ttest in traintest:
        if ttest not in dset['data']:
            continue
        data = dset['data'][ttest]
        if len(data) == 0:
            print("no samples!")
            continue
        samples[ttest] = len(data)

        num_words = []
        num_sents = {}
        num_summary_sents = {}
        for datum in data:
            words = datum['text'].split()
            num_words.append(len(words))
            labels = datum['labels']
            for lab in labels:
                if lab not in label_samples[ttest]:
                    label_samples[ttest][lab] = 0
                label_samples[ttest][lab] += 1


            if "document_index" in datum:
                # num sentences
                doc_idx = datum["document_index"]
                if doc_idx not in num_sents:
                    num_sents[doc_idx] = 0
                num_sents[doc_idx] += 1

                # num summary sentences
                if 1 in datum['labels']:
                    if doc_idx not in num_summary_sents:
                        num_summary_sents[doc_idx] = 0
                    num_summary_sents[doc_idx] += 1

        mean_num_words[ttest] = sum(num_words) / len(data)
        if num_sents:
            mean_sentences_per_doc[ttest] = sum(
                num_sents.values()) / len(num_sents)
        if num_summary_sents:
            mean_summary_sentences_per_doc[ttest] = sum(
                num_summary_sents.values()) / len(num_summary_sents)

    stats = {
        "samples": {
            "train": samples['train'],
            "test": samples['test']
        },
        "mean_num_words": {
            "train": mean_num_words['train'],
            "test": mean_num_words['test']
        },
        "mean_num_sentences": {
            "train": mean_sentences_per_doc["train"],
            "test": mean_sentences_per_doc["test"]
        },
        "mean_num_summary_sentences": {
            "train": mean_summary_sentences_per_doc["train"],
            "test": mean_summary_sentences_per_doc["test"]
        },
        "samples_per_label": {
            "train": label_samples["train"],
            "test": label_samples["test"]
        }

    }
    df = pandas.DataFrame.from_dict(stats, orient='index')

    print(dataset_path, "\n")
    print(df.round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Read and output stat. information on a jsonic dataset')
    parser.add_argument('dataset_path')
    args = parser.parse_args()
    inspect(args.dataset_path)
