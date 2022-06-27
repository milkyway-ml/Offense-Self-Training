import os
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import random
from typing import Optional, Tuple

import numpy as np
import torch
import sys


def get_logger(level: Optional[str] = "debug", filename: Optional[str] = None) -> logging.Logger:
    level_map = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "fatal": logging.FATAL,
    }
    loglevel = level_map.get(level)
    if loglevel is None:
        raise TypeError

    logging.basicConfig(
        level=loglevel,
        filename=filename,
        filemode="w" if filename else None,
        format="%(levelname)s | %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))
    return logger


def set_seed(seed_value: int) -> None:
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)


def load_mhs(seed: Optional[int] = None) -> Tuple[pd.DataFrame]:
    path = os.path.join("datasets", "measuring_hate_speech")

    unlabeled_path = os.path.join("datasets", "tweets_augmented.csv")
    train_path = os.path.join(path, "measuring_hate_speech.csv")

    data_df = pd.read_csv(train_path)
    data_df.loc[data_df["hate_speech_score"] >= 1, "label"] = 1
    data_df.loc[data_df["hate_speech_score"] < 1, "label"] = 0
    data_df = data_df[["text", "label"]]
    data_df["label"] = data_df["label"].astype(int)

    train_df, dev_df = train_test_split(data_df, train_size=0.7, stratify=data_df["label"], random_state=seed)
    dev_df, test_df = train_test_split(dev_df, train_size=0.5, stratify=dev_df["label"], random_state=seed)

    train_df = train_df.reset_index(drop=True)
    dev_df = dev_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    unlabeled_df = pd.read_csv(unlabeled_path)[["text", "text_augmented"]]
    unlabeled_df["text"] = unlabeled_df["text"]
    unlabeled_df["text_augmented"] = unlabeled_df["text_augmented"]

    unlabeled_df = unlabeled_df.drop_duplicates("text")

    return train_df, dev_df, test_df, unlabeled_df


def load_convabuse() -> Tuple[pd.DataFrame]:
    path = os.path.join("datasets", "ConvAbuse")

    unlabeled_path = os.path.join("datasets", "tweets_augmented.csv")
    train_path = os.path.join(path, "ConvAbuseEMNLPtrain.csv")
    dev_path = os.path.join(path, "ConvAbuseEMNLPvalid.csv")
    test_path = os.path.join(path, "ConvAbuseEMNLPtest.csv")

    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    test_df = pd.read_csv(test_path)

    unlabeled_df = pd.read_csv(unlabeled_path)[["text", "text_augmented"]]
    unlabeled_df["text"] = unlabeled_df["text"]
    unlabeled_df["text_augmented"] = unlabeled_df["text_augmented"]

    unlabeled_df = unlabeled_df.drop_duplicates("text")

    train_df["text"] = train_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"], axis=1
    )
    dev_df["text"] = dev_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"], axis=1
    )
    test_df["text"] = test_df.apply(
        lambda x: x["prev_agent"] + "\n" + x["prev_user"] + "\n" + x["agent"] + "\n" + x["user"], axis=1
    )

    train_df = train_df[["text", "is_abuse_majority"]]
    dev_df = dev_df[["text", "is_abuse_majority"]]
    test_df = test_df[["text", "is_abuse_majority"]]

    return train_df, dev_df, test_df, unlabeled_df


def load_olid() -> Tuple[pd.DataFrame]:
    path = os.path.join("datasets", "OLIDv1.0")

    train_path = os.path.join(path, "olid-training-v1.0.tsv")
    test_path = os.path.join(path, "testset-levela.tsv")
    test_labels_path = os.path.join(path, "labels-levela.csv")
    unlabeled_path = os.path.join("datasets", "tweets_augmented.csv")

    train_df = pd.read_csv(train_path, engine="python", sep="\t")[["tweet", "subtask_a"]]
    train_df["subtask_a"] = train_df["subtask_a"].apply(lambda x: 1 if x == "OFF" else 0)
    train_df = train_df.rename({"tweet": "text", "subtask_a": "toxic"}, axis=1)

    test_df = pd.read_csv(test_path, engine="python", sep="\t")
    test_labels = pd.read_csv(test_labels_path, header=None)
    test_df["toxic"] = test_labels[1].apply(lambda x: 1 if x == "OFF" else 0)
    test_df = test_df[["tweet", "toxic"]]
    test_df = test_df.rename({"tweet": "text"}, axis=1)

    unlabeled_df = pd.read_csv(unlabeled_path)[["text", "text_augmented"]]
    unlabeled_df["text"] = unlabeled_df["text"]
    unlabeled_df["text_augmented"] = unlabeled_df["text_augmented"]

    unlabeled_df = unlabeled_df.drop_duplicates("text")

    return train_df, None, test_df, unlabeled_df


def get_stratified_split(df: pd.DataFrame, num_split: int, seed: Optional[int] = None):
    """splits the dataset into 4 equal sized stratified parts and returns one of them"""
    splits = []
    left_half, right_half = train_test_split(
        df, train_size=0.5, shuffle=True, stratify=df.iloc[:, 1], random_state=seed
    )
    splits.extend(
        train_test_split(left_half, train_size=0.5, shuffle=True, stratify=left_half.iloc[:, 1], random_state=seed)
    )
    splits.extend(
        train_test_split(right_half, train_size=0.5, shuffle=True, stratify=right_half.iloc[:, 1], random_state=seed)
    )

    return splits[num_split]


def load_dataset(dataset_name):
    if dataset_name == "olidv1":
        train_df, dev_df, test_df, unlabeled_df = load_olid()
    elif dataset_name == "convabuse":
        train_df, dev_df, test_df, unlabeled_df = load_convabuse()
    elif dataset_name == "mhs":
        train_df, dev_df, test_df, unlabeled_df = load_mhs()
    else:
        raise Exception(f"{dataset_name} is not a valid dataset.")

    loaded_log = f"""
            Loaded {dataset_name}
        Train Size: {len(train_df)}
            Positives: {len(train_df[train_df.iloc[:, 1] == 1])}
            Negatives: {len(train_df[train_df.iloc[:, 1] == 0])}
        """

    if dev_df is not None:
        loaded_log += f"""
        Dev Size: {len(dev_df)}
            Positives: {len(dev_df[dev_df.iloc[:, 1] == 1])}
            Negatives: {len(dev_df[dev_df.iloc[:, 1] == 0])}
        """

    loaded_log += f"""
        Test Size: {len(test_df)}
            Positives: {len(test_df[test_df.iloc[:, 1] == 1])}
            Negatives: {len(test_df[test_df.iloc[:, 1] == 0])}
        Augmented Data: {len(unlabeled_df)}
        """
    logging.info(loaded_log)

    return train_df, dev_df, test_df, unlabeled_df
