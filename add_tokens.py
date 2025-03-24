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
    reviews_list = []
    evidences = annotations['evidence']
    indexes = annotations['index']
    for i in tqdm(range(len(reviews))):
        review = reviews[i]['review']
        rationales = evidences[i]
        idx = indexes[i]
        review_split = review.split('.')
        for i, rationale in enumerate(rationales):
            review_split[idx] = review_split[idx].replace(rationale, f"<evidence>{rationale}</evidence>")
        reviews_list.append(" ".join(review_split))
    return reviews_list

def read_reviews_and_rationales(filepath: str, type: str) -> Any:
    """
    Read the reviews and rationales from the file.
    :param filepath: Path to the file
    :param type: Type of the file (reviews or rationales)
    :return: List of reviews or rationales
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        if type == 'reviews':
            return [json.loads(line) for line in f]
        elif type == 'rationales':
            rationales = {'evidence': [], 'index': []}
            for line in f:
                data = json.loads(line)
                rationales['evidence'].append(data['evidence'])
                rationales['index'].append(data['index'])
            return rationales
        else:
            raise ValueError(f"Invalid type: {type}")

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
        train_reviews_filepath = './data/train_reviews.jsonl'
        train_annotations_filepath = './data/train_rationales.jsonl'
    else:
        train_reviews_filepath = args.train[0]
        train_annotations_filepath = args.train[1]
    if args.val is None:
        val_reviews_filepath = './data/val_reviews.jsonl'
        val_annotations_filepath = './data/val_rationales.jsonl'
    else:
        val_reviews_filepath = args.val[0]
        val_annotations_filepath = args.val[1]

    train_reviews = read_reviews_and_rationales(train_reviews_filepath, 'reviews')
    train_annotations = read_reviews_and_rationales(train_annotations_filepath, 'rationales')
    #val_reviews = read_reviews_and_rationales(val_reviews_filepath, 'reviews')
    #val_annotations = read_reviews_and_rationales(val_annotations_filepath, 'rationales')
    
    print(f"Adding special tokens to the training reviews...")
    special_train_reviews = add_tokens(train_reviews[0:1], train_annotations)
    print(f"Adding special tokens to the validation reviews...")
    #special_val_reviews = add_tokens(val_reviews['reviews'], val_annotations)

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
