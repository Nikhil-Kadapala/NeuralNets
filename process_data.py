import json
import pandas as pd
import numpy as np
import torch
import argparse
import os
from typing import List, Dict, Any

def parse_data(file_path):
    data = []                                               # Initialize an empty list to store the dictionaries

    with open(file_path, 'r') as file:                      # Open the .jsonl file and read it line by line
        for line in file:
            annotation = json.loads(line)                   # Parse each line as JSON and append it to the list
            id = annotation["annotation_id"]
            annotation["classification"] = 1 if annotation['classification'] == "POS" else 0

            with open(f"./movies/docs/{id}", 'r') as file:  # open the file named by annotation_id to extract the review text
                content = file.read()
                annotation['content'] = content.replace('\n', ' ')
                data.append(annotation)
    return data

def print_example(data, index, print_content=True, print_classification=True, print_rationales=True ):
    print(f'Retrieving Training Example [{index}].................\n')
    item = data[index]
    classification = item['classification']
    evidences = item['evidences']
    content = item['content']
    if print_content: print(f'Review content:\n{content}\n')
    if print_classification: print('----------------------------',
                                   '\n| Sentiment class:',
                                   classification,
                                   ("- NEG" if not classification else "- POS"),
                                   '|', '\n----------------------------')
    if print_rationales:
        print('\nHuman rationales / Supporting Evidence:')
        for evidence in evidences:
            print('     - ', evidence[0]['text'])

def get_content(data, index):
    item = data[index]
    content = item['content']
    return content

def get_classes(data, index):
    item = data[index]
    classification = item['classification']
    return classification

def get_annotations(data, index):
    item = data[index]
    content = item['evidences']
    annotations: Dict[List[str], List[int]] = {'evidence': [], 'index': []}
    for evidence in content:
        annotations['evidence'].append(evidence[0]['text']) 
        annotations['index'].append(evidence[0]['start_sentence'])
    return annotations


def get_rationales(train_data, val_data, test_data):
    train_rationales = []
    for i in range(len(train_data)):
        train_data[i]['evidences'] =get_annotations(train_data, i)
        train_rationales.append(train_data[i]['evidences'])

    val_rationales = []
    for i in range(len(val_data)):
        val_data[i]['evidences'] = get_annotations(val_data, i)
        val_rationales.append(val_data[i]['evidences'])

    test_rationales = []
    for i in range(len(test_data)):
        test_data[i]['evidences'] = get_annotations(test_data, i)
        test_rationales.append(test_data[i]['evidences'])

    return train_rationales, val_rationales, test_rationales

def get_reviews(train_data, val_data, test_data):
    train_reviews = [get_content(train_data, i) for i in range(len(train_data))]
    val_reviews = [get_content(val_data, i) for i in range(len(val_data))]
    test_reviews = [get_content(test_data, i) for i in range(len(test_data))]
    
    return train_reviews, val_reviews, test_reviews
    
def get_labels(train_data, val_data, test_data):
    train_classes = [get_classes(train_data, i) for i in range(len(train_data))]
    val_classes = [get_classes(val_data, i) for i in range(len(val_data))]
    test_classes = [get_classes(test_data, i) for i in range(len(test_data))]
    
    return train_classes, val_classes, test_classes

def write_to_jsonl(file_path, data, type):
    with open(file_path, 'w', encoding='utf-8') as f:
        if type == 'review':
            for item in data:
                jsonobj = {f'{type}': item}
                jsonline = json.dumps(jsonobj, ensure_ascii=False)
                f.write(jsonline + '\n')
        elif type == 'rationale':
            for item in data:
                jsonline = json.dumps(item, ensure_ascii=False)
                f.write(jsonline + '\n')

def write_classes_to_jsonl(file_path, data, type):
    with open(file_path, 'w', encoding='utf-8') as f:
        jsonobj = {f'{type}': data}
        jsonline = json.dumps(jsonobj, ensure_ascii=False)
        f.write(jsonline + '\n')
        
def main():
    parser = argparse.ArgumentParser(description="Processing the initial dataset to extract the reviews, labels, and rationales.")

    # add the option or flag to specify the system prompt for the fine-tuning data and the path to the training data file.
    parser.add_argument("-trn", "--train", type=str, help="path to the training data in the following format: ./data/train.jsonl")
    parser.add_argument("-val", "--val", type=str, help="path to the validation data in the following format: ./data/val.jsonl")
    parser.add_argument("-tst", "--test", type=str, help="path to the test data in the following format: ./data/test.jsonl")

    # use parse_args() method to parse the command line arguments 
    args = parser.parse_args()
    
    train_file_path: str
    val_file_path: str
    test_file_path: str

    if args.train is None:
        train_file_path = './movies/train.jsonl'
    else:
        train_file_path = args.train
    if args.val is None:
        val_file_path = './movies/val.jsonl'
    else:
        val_file_path = args.val
    if args.test is None:
        test_file_path = './movies/test.jsonl'
    else:
        test_file_path = args.test

    train_data = parse_data(train_file_path)
    val_data = parse_data(val_file_path)
    test_data = parse_data(test_file_path)

    train_reviews, val_reviews, test_reviews = get_reviews(train_data, val_data, test_data)
    train_classes, val_classes, test_classes = get_labels(train_data, val_data, test_data)
    train_rationales, val_rationales, test_rationales = get_rationales(train_data, val_data, test_data)
    os.makedirs('./data', exist_ok=True)  # create the directory if it does not exist
    # save the reviews, labels, and rationales to the files
    write_to_jsonl('./data/train_reviews.jsonl', train_reviews, 'review')
    write_to_jsonl('./data/val_reviews.jsonl', val_reviews, 'review')
    write_to_jsonl('./data/test_reviews.jsonl', test_reviews, 'review')
    write_classes_to_jsonl('./data/train_labels.jsonl', train_classes, 'label')
    write_classes_to_jsonl('./data/val_labels.jsonl', val_classes, 'label')
    write_classes_to_jsonl('./data/test_labels.jsonl', test_classes, 'label')
    write_to_jsonl('./data/train_rationales.jsonl', train_rationales, 'rationale')
    write_to_jsonl('./data/val_rationales.jsonl', val_rationales, 'rationale')
    write_to_jsonl('./data/test_rationales.jsonl', test_rationales, 'rationale')
    print(f'Data processing completed successfully!\nThe processed data has been saved to the ./data directory.')

if __name__ == "__main__":
    main()