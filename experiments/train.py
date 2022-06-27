import sys

sys.path.append("..")

import os
from selftraining import SelfTrainer
import argparse
import time
from experiments import load_dataset, set_seed, get_logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--exp_name", default="experiment", type=str)
    parser.add_argument("--loglevel", default="info", type=str)

    # bert args
    parser.add_argument("--pretrained_bert_name", default="bert-base-cased", type=str)
    parser.add_argument("--max_seq_len", default=128, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--num_train_epochs", default=2, type=int)
    parser.add_argument("--learning_rate", default=5e-5, type=float)
    parser.add_argument("--warmup_ratio", default=0.15, type=float)
    parser.add_argument("--classifier_dropout", default=0.1, type=float)
    parser.add_argument("--attention_dropout", default=0.1, type=float)

    # ST args
    parser.add_argument("--min_confidence_threshold", default=0.51, type=float)
    parser.add_argument("--num_st_iters", default=3, type=int)
    parser.add_argument("--use_augmentation", const=True, default=False, nargs="?", type=bool)
    parser.add_argument("--increase_attention_dropout_amount", default=None, type=float)
    parser.add_argument("--increase_classifier_dropout_amount", default=None, type=float)
    parser.add_argument("--increase_confidence_threshold_amount", default=None, type=float)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    set_seed(args.seed)

    log_path = os.path.join("logs", f"{args.exp_name}.log")
    logger = get_logger(level=args.loglevel, filename=log_path)

    train_df, dev_df, test_df, weak_label_df = load_dataset(args.dataset)

    st = SelfTrainer(
        pretrained_bert_name=args.pretrained_bert_name,
        device=args.device,
        classifier_dropout=args.classifier_dropout,
        attention_dropout=args.attention_dropout,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
    )

    start = time.time()
    st.fit(
        train_df=train_df,
        dev_df=dev_df,
        test_df=test_df,
        unlabeled_df=weak_label_df,
        num_iters=args.num_st_iters,
        min_confidence_threshold=args.min_confidence_threshold,
        increase_attention_dropout_amount=args.increase_attention_dropout_amount,
        increase_classifier_dropout_amount=args.increase_classifier_dropout_amount,
        increase_confidence_threshold_amount=args.increase_confidence_threshold_amount,
        use_augmentation=args.use_augmentation,
    )
    end = time.time()
    runtime = end - start

    logger.info(f"\nTotal Runtime: {runtime/60:.2f} minutes.")
