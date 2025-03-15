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
        annotation = annotations[i]
        for evidence in annotation['evidences']:
            evidence_text = evidence[0]['text']
            review = review.replace(evidence_text, f"<evidence>{evidence_text}</evidence>")
    return reviews

def main():
    parser = argparse.ArgumentParser(description="Processing the initial dataset to extract the reviews, labels, and rationales.")

    # add the option or flag to specify the system prompt for the fine-tuning data and the path to the training data file.
    parser.add_argument("-trn", "--train", nargs="*", type=str, help="path to the training data in the following format: ./data/train.jsonl")
    parser.add_argument("-val", "--val", nargs="*", type=str, help="path to the validation data in the following format: ./data/val.jsonl")
    parser.add_argument("-tst", "--test", nargs="*", type=str, help="path to the test data in the following format: ./data/test.jsonl")

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
    if args.test is None:
        test_reviews_filepath = './data/test_reviews.csv'
        test_annotations_filepath = './data/test_rationales.csv'
    else:
        test_reviews_filepath = args.test[0]
        test_annotations_filepath = args.test[1]

    train_reviews = pd.read_csv(train_reviews_filepath)
    val_reviews = pd.read_csv(val_reviews_filepath)
    test_reviews = pd.read_csv(test_reviews_filepath)

    train_annotations = pd.read_csv(train_annotations_filepath)
    val_annotations = pd.read_csv(val_annotations_filepath)
    test_annotations = pd.read_csv(test_annotations_filepath)

    # Convert the reviews and annotations to lists
    train_reviews.columns = ['reviews']
    val_reviews.columns = ['reviews']
    test_reviews.columns = ['reviews']
    train_annotations.columns = ['annotations']
    val_annotations.columns = ['annotations']
    test_annotations.columns = ['annotations']
    train_anns = pd.read_csv('./movies/train.jsonl')
    special_train_reviews = add_tokens(train_reviews['reviews'].tolist(), train_annotations['annotations'].tolist())
    special_val_reviews = add_tokens(val_reviews['reviews'].tolist(), val_annotations['annotations'].tolist())
    special_test_reviews = add_tokens(test_reviews['reviews'].tolist(), test_annotations['annotations'].tolist())

    print(f'Addition of special tokens completed successfully!\nThe processed data has been saved to the ./data directory.')

if __name__ == "__main__":
    main()