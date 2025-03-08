import pandas as pd
import numpy as np
import argparse
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import pickle

def create_glove_dict(sequence, wv, set, embed_dim=50):
    """
    Creates a dictionary mapping words in the vocabulary to their GloVe embeddings.
    Words that don't exist are mapped to zero vectors.
    """
    glove_dict = {}
    empty_vec = np.zeros(embed_dim, dtype=np.float64)

    for word in tqdm(sequence, desc=f"Building {set} GloVe dictionary"):
        glove_dict[word] = wv[word] if word in wv else empty_vec

    return glove_dict

def get_w2GloVe(data, glove_dict, set, embed_dim=50, rationale=False):
    """
    Retrieves the GloVe embeddings using the custom-built GloVe dictionary.
    Args:
        data: List of text reviews.
        glove_dict (dict): custom-built GloVe dictionary.
        embed_dim (int): Dimensions of GloVe embeddings.
    Returns:
        torch.Tensor: Padded tensor of GloVe embeddings to maintain uniform length.
    """
    glove_embeds = []
    max_len = 2048

    for review in data:   #tqdm(data, desc=f"Retrieving {set} GloVe Word Embeddings"):
      words = word_tokenize(review.lower())
      words = words[:max_len]
      embeddings = [glove_dict.get(word, np.zeros(embed_dim)) for word in words]
      if not embeddings:
        embeddings = [np.zeros(embed_dim)]
      glove_embeds.append(torch.tensor(embeddings, dtype=torch.float))

    return pad_sequence(glove_embeds, batch_first=True)

def get_reviews_dict(train_reviews, val_reviews, test_reviews, wv):
    print(f"----------------------------------------------------------------------------------------\nProcessing Reviews\n----------------------------------------------------------------------------------------\n")

    # Extract vocabulary(distinct words) from training, validation, and test data
    train_vocab = set(word for review in train_reviews for word in word_tokenize(review.lower()))
    val_vocab = set(word for review in val_reviews for word in word_tokenize(review.lower()))
    test_vocab = set(word for review in test_reviews for word in word_tokenize(review.lower()))

    # Build the GloVe dictionary for the reviews
    reviews_dict = create_glove_dict(train_vocab, wv, "training")
    reviews_dict.update(create_glove_dict(val_vocab, wv, "validation"))
    reviews_dict.update(create_glove_dict(test_vocab, wv, "test"))

    return reviews_dict

def get_rationales_dict(train_rationales, val_rationales, test_rationales, wv):
    print(f"\n----------------------------------------------------------------------------------------\nProcessing Rationales\n----------------------------------------------------------------------------------------\n")

    #Extract vocabulary(distinct words) from training, validation, and test data for the rationales
    train_rationale_vocab = set(word for rationale in train_rationales for word in word_tokenize(rationale.lower()))
    test_rationale_vocab = set(word for rationale in test_rationales for word in word_tokenize(rationale.lower()))
    val_rationale_vocab = set(word for rationale in val_rationales for word in word_tokenize(rationale.lower()))

    # Build the GloVe dictionary for the rationales
    rationales_dict = create_glove_dict(train_rationale_vocab, wv, "training")
    rationales_dict.update(create_glove_dict(val_rationale_vocab, wv, "validation"))
    rationales_dict.update(create_glove_dict(test_rationale_vocab, wv, "test"))

    return rationales_dict

def get_rev_embeddings(train_reviews, val_reviews, test_reviews, reviews_dict):
    print(f"----------------------------------------------------------------------------------------\nProcessing Reviews\n----------------------------------------------------------------------------------------\n")
    # Convert reviews to glove embeddings
    train_review_gloves = get_w2GloVe(train_reviews, reviews_dict, "training")
    val_review_gloves = get_w2GloVe(val_reviews, reviews_dict, "validation")
    test_review_gloves = get_w2GloVe(test_reviews, reviews_dict, "test")

    return train_review_gloves, val_review_gloves, test_review_gloves

def get_rat_embeddings(train_rationales, val_rationales, test_rationales, rationales_dict):
    print(f"\n----------------------------------------------------------------------------------------\nProcessing Rationales\n----------------------------------------------------------------------------------------\n")
    # Convert rationales to glove embeddings
    train_rationale_gloves = get_w2GloVe(train_rationales, rationales_dict, "training")
    val_rationale_gloves = get_w2GloVe(val_rationales, rationales_dict, "validation")
    test_rationale_gloves = get_w2GloVe(test_rationales, rationales_dict, "test")
    return train_rationale_gloves, val_rationale_gloves, test_rationale_gloves

def main():
    parser = argparse.ArgumentParser(description="Processing the initial dataset to extract the reviews, labels, and rationales.")

    # add the option or flag to specify the system prompt for the fine-tuning data and the path to the training data file.
    parser.add_argument("-trn", "--train", type=str, help="path to the training data in the following format: ./data/train_reviews.csv")
    parser.add_argument("-val", "--val", type=str, help="path to the validation data in the following format: ./data/val_reviews.csv")
    parser.add_argument("-tst", "--test", type=str, help="path to the test data in the following format: ./data/test_reviews.csv")

    # use parse_args() method to parse the command line arguments 
    args = parser.parse_args()
    
    trn_rev_path: str
    val_rev_path: str
    tst_rev_path: str
    trn_rat_path: str
    val_rat_path: str
    tst_rat_path: str

    if args.train is None:
        trn_rev_path = './data/train_reviews.csv'
        trn_rat_path = './data/train_rationales.csv'
    else:
        trn_rev_path = args.train
    if args.val is None:
        val_rev_path = './data/val_reviews.csv'
        val_rat_path = './data/val_rationales.csv'
    else:
        val_rev_path = args.val
    if args.test is None:
        tst_rev_path = './data/test_reviews.csv'
        tst_rat_path = './data/test_rationales.csv'
    else:
        tst_rev_path = args.test
    
    train_reviews = pd.read_csv(trn_rev_path, header=None).values.flatten().tolist()
    val_reviews = pd.read_csv(val_rev_path, header=None).values.flatten().tolist()
    test_reviews = pd.read_csv(tst_rev_path, header=None).values.flatten().tolist()
    train_rationales = pd.read_csv(trn_rat_path, header=None).values.flatten().tolist()
    val_rationales = pd.read_csv(val_rat_path, header=None).values.flatten().tolist()
    test_rationales = pd.read_csv(tst_rat_path, header=None).values.flatten().tolist()
    
    with open("./glove_embeddings.pkl", "rb") as f:
        wv = pickle.load(f)

    reviews_dict = get_reviews_dict(train_reviews, val_reviews, test_reviews, wv)
    rationales_dict = get_rationales_dict(train_rationales, val_rationales, test_rationales, wv)

    train_review_gloves, val_review_gloves, test_review_gloves = get_rev_embeddings(train_reviews, val_reviews, test_reviews, reviews_dict)
    train_rationale_gloves, val_rationale_gloves, test_rationale_gloves = get_rat_embeddings(train_rationales, val_rationales, test_rationales, rationales_dict)
    
    # Save the GloVe embeddings
    with open("./gloves/train_reviews.pkl", "wb") as f:
        pickle.dump(train_review_gloves, f)

    with open("./gloves/val_reviews.pkl", "wb") as f:
        pickle.dump(val_review_gloves, f)

    with open("./gloves/test_reviews.pkl", "wb") as f:
        pickle.dump(test_review_gloves, f)

    with open("./gloves/train_rationales.pkl", "wb") as f:
        pickle.dump(train_rationale_gloves, f)

    with open("./gloves/val_rationales.pkl", "wb") as f:
        pickle.dump(val_rationale_gloves, f)

    with open("./gloves/test_rationales.pkl", "wb") as f:
        pickle.dump(test_rationale_gloves, f)

    print(f"\n GloVe embeddings saved successfully!\nThe processed GloVe embeddings have been saved to the ./gloves directory.")

if __name__ == "__main__":
    main()
