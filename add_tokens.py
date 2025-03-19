import json
import pandas as pd
import numpy as np
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

def add_tokens(reviews: List, annotations: List) -> List:
    """
    Add tokens to the reviews based on the annotations.
    :param reviews: List of reviews
    :param annotations: List of annotations
    :return: List of reviews with added tokens
    """
    for i in tqdm(range(len(reviews))):
        review = reviews[i]
        evidences = annotations[i].split(',') if type(annotations[i]) == str else ["[None]"]
        evi_len = len(evidences)
        indexes = {i: 0 for i in range(evi_len)}
        print(indexes)
        for i, evidence in enumerate(reversed(evidences)):
            if indexes[evi_len - 1 - i] == 0:
                review = review.replace(evidence, f"<evidence>{evidence}</evidence>")
                indexes[evi_len - 1 - i] = 1
            reviews[i] = review
            
    return reviews

def main():
    parser = argparse.ArgumentParser(description="Processing the initial dataset to extract the reviews, labels, and rationales.")

    # add the option or flag to specify the system prompt for the fine-tuning data and the path to the training data file.
    parser.add_argument("-trn", "--train", nargs="*", type=str, help="path to the training data in the following format: ./data/train.jsonl")
    parser.add_argument("-val", "--val", nargs="*", type=str, help="path to the validation data in the following format: ./data/val.jsonl")

    # use parse_args() method to parse the command line arguments 
    args = parser.parse_args()
    
    train_reviews_filepath: str
    val_reviews_filepath: str
    test_reviews_filepath: str

    train_annotations_filepath: str
    val_annotations_filepath: str
    test_annotations_filepath: str

    if args.train is None:
        train_reviews_filepath = './data/train_reviews.csv'
        train_annotations_filepath = './data/train_rationales.csv'
    else:
        train_reviews_filepath = args.train[0]
        train_annotations_filepath = args.train[1]
    if args.val is None:
        val_reviews_filepath = './data/val_reviews.csv'
        val_annotations_filepath = './data/val_rationales.csv'
    else:
        val_reviews_filepath = args.val[0]
        val_annotations_filepath = args.val[1]

    train_reviews = pd.read_csv(train_reviews_filepath)
    val_reviews = pd.read_csv(val_reviews_filepath)

    train_annotations = pd.read_csv(train_annotations_filepath)
    val_annotations = pd.read_csv(val_annotations_filepath)
    
    print(f"Adding special tokens to the training reviews...")
    special_train_reviews = add_tokens(train_reviews['reviews'].tolist(), train_annotations['annotations'].tolist())
    print(f"Adding special tokens to the validation reviews...")
    #special_val_reviews = add_tokens(val_reviews['reviews'].tolist(), val_annotations['annotations'].tolist())
    
    with open('./data/mrkd_train_reviews.jsonl', 'w', encoding='utf-8') as f:
        for review in special_train_reviews:
            jsonobj = {'review': review}
            jsonline = json.dumps(jsonobj, ensure_ascii=False)
            f.write(jsonline + '\n')
    """
    with open('./data/mrkd_val_reviews.jsonl', 'w', encoding='utf-8') as f:
        for review in special_val_reviews: # type: ignore
            jsonobj = {'review': review}
            jsonline = json.dumps(jsonobj, ensure_ascii=False)
            f.write(jsonline + '\n')
    """   
    print(f'Addition of special tokens completed successfully!\nThe processed data has been saved to the ./data directory.')

if __name__ == "__main__":
    main()