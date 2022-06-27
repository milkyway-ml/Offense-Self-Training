import os
import json
import logging
import time
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.special import softmax
from torch.utils.data import DataLoader, RandomSampler, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BatchEncoding, get_scheduler


class BertDataset(Dataset):
    def __init__(self, encodings: BatchEncoding, labels: List[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self) -> int:
        return len(self.labels)


class WeakLabelDataset(Dataset):
    def __init__(self, text, labels=None):
        self.text = text
        self.labels = labels

    def __getitem__(self, idx: int) -> Dict:
        item = {"text": self.text[idx]}
        if self.labels is not None:
            item["labels"] = self.labels[idx]

        return item

    def __len__(self) -> int:
        return len(self.text)


class SelfTrainer:
    def __init__(
        self,
        pretrained_bert_name: Optional[str] = "bert-base-cased",
        max_seq_len: Optional[int] = 128,
        attention_dropout: Optional[float] = 0.1,
        classifier_dropout: Optional[float] = 0.1,
        weight_decay: Optional[float] = 1e-2,
        num_train_epochs: Optional[int] = 2,
        batch_size: Optional[int] = 32,
        learning_rate: Optional[float] = 5e-5,
        warmup_ratio: Optional[float] = 0.15,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            self.device = torch.device(device)
        self.pretrained_bert_name = pretrained_bert_name
        self.max_seq_len = max_seq_len
        self.attention_dropout = attention_dropout
        self.classifier_dropout = classifier_dropout
        self.weight_decay = weight_decay
        self.num_train_epochs = num_train_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.tokenizer = self.__init_tokenizer()
        self.num_st_iter = 0
        self.model = self.__init_model(self.attention_dropout, self.classifier_dropout)

    def __init_model(
        self, attention_dropout: Optional[float], classifier_dropout: Optional[float]
    ) -> AutoModelForSequenceClassification:
        model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_bert_name)

        # class attributes referring to dropout are not the same for bert and distilbert
        if "distilbert" in self.pretrained_bert_name:
            if attention_dropout is not None:
                model.config.attention_dropout = attention_dropout
            if self.classifier_dropout is not None:
                model.config.seq_classif_dropout = classifier_dropout

        else:
            if attention_dropout is not None:
                model.config.attention_probs_dropout_prob = attention_dropout
            if classifier_dropout is not None:
                model.config.classifier_dropout = classifier_dropout

        return model

    def __init_tokenizer(self) -> AutoTokenizer:
        # bertweet needs normalized inputs
        if self.pretrained_bert_name == "vinai/bertweet-base":
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_name, normalize=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.pretrained_bert_name)

        return tokenizer

    def __get_dataloader_from_df(self, df: pd.DataFrame) -> DataLoader:
        texts = df.iloc[:, 0].astype("str").to_list()
        targets = df.iloc[:, 1].astype("category").to_list()

        tokenized_train = self.tokenize(texts)
        dataset = BertDataset(tokenized_train, labels=targets)

        sampler = RandomSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.batch_size)

        return dataloader

    def __get_optimizer(self, train_dataloader: DataLoader) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        total_steps = len(train_dataloader) * self.num_train_epochs
        num_warmup_steps = int(total_steps * self.warmup_ratio)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_scheduler(
            "linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
        )

        self.model.to(self.device)
        return optimizer, scheduler

    def __train(
        self,
        train_dataloader: DataLoader,
        evaluate_during_training=False,
        is_student=False,
        dump_train_history: Optional[bool] = True,
        clip_grad: Optional[bool] = True,
        dev_dataloader: Optional[DataLoader] = None,
        weak_label_dataloader: Optional[DataLoader] = None,
        unl_to_label_batch_ratio: Optional[float] = None,
    ):
        optimizer, scheduler = self.__get_optimizer(train_dataloader)
        progress_bar = tqdm(range(self.num_train_epochs * len(train_dataloader)), desc="Training")
        print_each_n_steps = int(len(train_dataloader) // 4)
        logging.debug("Start training...\n")

        historic_loss = {"loss": [], "labeled_loss": [], "unlabeled_loss": [], "steps": [], "unl_steps": []}
        for epoch_i in range(self.num_train_epochs):
            if is_student:
                logging.debug(
                    f"{'Epoch':^7} | {'Labeled Batch':^14} | {'Unlabeled Batch':^16} | "
                    f"{'Train Loss':^11} | {'Labeled Loss':^13} | "
                    f"{'Unlabeled Loss':^15} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
                )
                logging.debug("-" * 130)
            else:
                logging.debug(
                    f"{'Epoch':^7} | {'Train Batch':^12} | "
                    f"{'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}"
                )
                logging.debug("-" * 80)

            # measure the elapsed time of each epoch
            t0_epoch, t0_batch = time.time(), time.time()

            # reset tracking variables at the beginning of each epoch
            total_loss, batch_loss, batch_unl_loss, batch_lab_loss, batch_counts, = (
                0,
                0,
                0,
                0,
                0,
            )

            loss_list = []
            unl_loss_list = []
            lab_loss_list = []
            step_list = []
            unl_step_list = []

            # train loop
            self.model.train()
            loss_fn = torch.nn.CrossEntropyLoss()
            for step, batch in enumerate(train_dataloader):
                batch_counts += 1
                batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()
                output = self.model(**batch_inputs)
                # if model is student, train with the noised data aswell
                if is_student:
                    unl_logits = []
                    unl_labels = []

                    for _ in range(unl_to_label_batch_ratio):
                        unl_batch = next(iter(weak_label_dataloader))

                        unl_texts = unl_batch["text"]
                        unl_inputs = self.tokenize(unl_texts)
                        unl_inputs["labels"] = unl_batch["labels"].clone().detach()
                        unl_batch_inputs = {k: v.to(self.device) for k, v in unl_inputs.items()}
                        unl_output = self.model(**unl_batch_inputs)

                        unl_logits.append(unl_output.logits.cpu().detach().numpy())
                        unl_labels.append(unl_inputs["labels"].cpu().detach().numpy())

                        del unl_batch_inputs
                        del unl_output

                    # concatenate the unlabeled batch outputs into a single tensor
                    unl_labels = torch.cat([torch.as_tensor(t) for t in unl_labels])
                    unl_logits = torch.cat([torch.as_tensor(t) for t in unl_logits])

                    # combine unlabeled + labeled loss
                    unl_loss = loss_fn(unl_logits, unl_labels)
                    lab_loss = output.loss
                    loss = lab_loss + unl_loss

                    batch_lab_loss += lab_loss.item()
                    batch_unl_loss += unl_loss.item()

                else:
                    loss = output.loss

                batch_loss += loss.item()
                total_loss += loss.item()

                loss.backward()

                # historic data
                loss_list.append(batch_loss / batch_counts)
                step_list.append(step)
                if is_student:
                    unl_loss_list.append(batch_unl_loss / batch_counts)
                    lab_loss_list.append(batch_lab_loss / batch_counts)
                    unl_step_list.append(unl_to_label_batch_ratio * step)

                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                progress_bar.update(1)

                if (step % print_each_n_steps == 0 and step != 0) or (step == len(train_dataloader) - 1):
                    time_elapsed = time.time() - t0_batch

                    # Print training results
                    if is_student:
                        logging.debug(
                            f"{epoch_i + 1:^7} | {step:^14} | {(step*unl_to_label_batch_ratio):^16} | "
                            f"{batch_loss / batch_counts:^11.6f} | "
                            f"{batch_lab_loss / batch_counts:^15.6f} | "
                            f"{batch_unl_loss / batch_counts :^13.6f} | "
                            f"{'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                        )

                    else:
                        logging.debug(
                            f"{epoch_i + 1:^7} | {step:^12} | {batch_loss / batch_counts:^12.6f} | "
                            f"{'-':^10} | {'-':^9} | {time_elapsed:^9.2f}"
                        )

                    batch_loss, batch_lab_loss, batch_unl_loss, batch_counts = 0, 0, 0, 0
                    t0_batch = time.time()

            # Calculate the average loss over the entire training data
            avg_train_loss = total_loss / len(train_dataloader)
            if evaluate_during_training:
                val_loss, val_accuracy, _, _ = self.score(dev_dataloader, dump_test_history=False)
                time_elapsed = time.time() - t0_epoch

                if is_student:
                    logging.debug("-" * 130)
                    logging.debug(
                        f"{epoch_i + 1:^7} | {'-':^14} | {'-':^16} | {avg_train_loss:^11.6f} | "
                        f"{'-':^15} | {'-':^13}| {val_loss:^10.6f} | "
                        f"{val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
                    )
                    logging.debug("-" * 130)
                else:
                    logging.debug("-" * 80)
                    logging.debug(
                        f"{epoch_i + 1:^7} | {'-':^12} | {avg_train_loss:^12.6f} | "
                        f"{val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}"
                    )
                    logging.debug("-" * 80)
            logging.debug("\n")

            historic_loss["loss"].append(loss_list)
            historic_loss["labeled_loss"].append(lab_loss_list)
            historic_loss["unlabeled_loss"].append(unl_loss_list)
            historic_loss["unl_steps"].append(unl_step_list)
            historic_loss["steps"].append(step_list)

        if dump_train_history:
            with open(os.path.join("logs", f"train_history-model{self.num_st_iter}.json"), "a+") as f:
                json.dump(historic_loss, f)

    def predict_batch(self, dataloader: DataLoader) -> List[np.array]:
        self.model.eval()
        all_logits = []

        for batch in dataloader:
            batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                output = self.model(**batch_inputs)
                logits = output.logits
            all_logits.append(logits)

        all_logits = torch.cat(all_logits, dim=0)

        probs = torch.nn.functional.softmax(all_logits, dim=1).cpu().numpy()
        labels = np.argmax(probs, axis=1)

        return probs, labels

    def score(
        self, test_dataloader: DataLoader, dump_test_history: Optional[bool] = True
    ) -> Tuple[float, float, float]:
        self.model.eval()

        logits, preds, true_labels, val_loss = [], [], [], []
        for batch in test_dataloader:
            batch_labels = batch["labels"]
            batch_inputs = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                batch_output = self.model(**batch_inputs)
                batch_logits = batch_output.logits

            # get accuracy and loss
            batch_probs = torch.nn.functional.softmax(batch_logits, dim=1).cpu().numpy()
            batch_preds = np.argmax(batch_probs, axis=1)

            val_loss.append(batch_output.loss.item())
            preds.extend(batch_preds)
            logits.append(batch_logits)
            true_labels.extend(batch_labels)

        logits = torch.cat(logits, dim=0)
        true_labels = np.array(true_labels)
        preds = np.array(preds)

        clf_report = classification_report(true_labels, preds, zero_division=0)
        f1 = f1_score(true_labels, preds, average="macro", zero_division=0)
        acc = accuracy_score(true_labels, preds)
        val_loss = np.mean(val_loss)

        if dump_test_history:
            history = {"y_true": [], "y_pred": [], "logits_0": [], "logits_1": []}
            history["y_true"] = [int(v) for v in true_labels.tolist()]
            history["y_pred"] = [int(v) for v in preds.tolist()]
            history["logits_0"] = logits.detach().cpu().numpy()[:, 0].tolist()
            history["logits_1"] = logits.detach().cpu().numpy()[:, 1].tolist()
            history["f1_score"] = f1
            history["accuracy"] = acc
            history["loss"] = val_loss

            with open(os.path.join("logs", f"test_history-model{self.num_st_iter}.json"), "w") as f:
                json.dump(history, f)

        return val_loss, acc, f1, clf_report

    def __get_weak_labels(
        self, unlabeled_dataloader: DataLoader, min_confidence_threshold: float
    ) -> Tuple[DataLoader, int, int]:
        texts = []
        labels = []
        logits = []
        for unl_batch in tqdm(unlabeled_dataloader, desc="Inferring Silver Labels"):
            unl_texts = unl_batch["text"]
            unl_inputs = self.tokenize(unl_texts)

            # get model predictions
            batch_inputs = {k: v.to(self.device) for k, v in unl_inputs.items()}
            self.model.to(self.device)
            with torch.no_grad():
                unl_outputs = self.model(**batch_inputs)

            unl_logits = unl_outputs.logits
            batch_labels = unl_logits.argmax(dim=-1).cpu().detach().numpy()

            logits.append(unl_logits.cpu().detach().numpy())
            texts.extend(unl_texts)
            labels.extend(batch_labels)

        logits = np.concatenate(logits)
        # get all examples with high confidence
        unl_softmax = softmax(logits, axis=1)
        high_confidence_positive_idxs = np.where(unl_softmax[:, 1] >= min_confidence_threshold)[0]
        high_confidence_negative_idxs = np.where(unl_softmax[:, 0] >= min_confidence_threshold)[0]

        # select same amount of positives and negatives (limited by the class with least examples)
        size = min(len(high_confidence_positive_idxs), len(high_confidence_negative_idxs))

        # both classes must have at least one example
        if size > 0:
            high_confidence_negative_idxs = np.random.choice(high_confidence_negative_idxs, size=size, replace=False)
            high_confidence_positive_idxs = np.random.choice(high_confidence_positive_idxs, size=size, replace=False)

        else:
            raise Exception(f"No examples predicted for the one of the classes.")

        high_confidence_idxs = np.append(high_confidence_positive_idxs, high_confidence_negative_idxs)

        # get selected elements from each data field by their idxs
        selected_text = list(map(texts.__getitem__, high_confidence_idxs.tolist()))
        selected_label = np.argmax(unl_softmax[high_confidence_idxs], axis=1)
        selected_confidence = np.max(unl_softmax[high_confidence_idxs], axis=1)

        augmented_df = pd.DataFrame(
            {
                "text": selected_text,
                "label": selected_label,
                "confidence": selected_confidence,
            }
        )

        augmentedset = WeakLabelDataset(text=selected_text, labels=selected_label)
        augmented_sampler = RandomSampler(augmentedset)
        augmented_dataloader = DataLoader(augmentedset, sampler=augmented_sampler, batch_size=self.batch_size)

        amnt_new_samples_pos = len(augmented_df[augmented_df["label"] == 1])
        amnt_new_samples_neg = len(augmented_df[augmented_df["label"] == 0])

        return augmented_dataloader, amnt_new_samples_pos, amnt_new_samples_neg

    def tokenize(self, texts: List[str]) -> BatchEncoding:
        tokenized = self.tokenizer(
            texts, truncation=True, padding="max_length", max_length=self.max_seq_len, return_tensors="pt"
        )

        return tokenized

    def fit(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        unlabeled_df: pd.DataFrame,
        min_confidence_threshold: float,
        num_iters: int,
        dev_df: Optional[pd.DataFrame] = None,
        increase_attention_dropout_amount: Optional[float] = None,
        increase_classifier_dropout_amount: Optional[float] = None,
        increase_confidence_threshold_amount: Optional[float] = None,
        use_augmentation: Optional[bool] = True,
    ):
        # get dataloaders
        train_dataloader = self.__get_dataloader_from_df(train_df)
        test_dataloader = self.__get_dataloader_from_df(test_df)

        if use_augmentation:
            text = unlabeled_df.iloc[:, 0].to_list()
            text.extend(unlabeled_df.iloc[:, 1].to_list())
        else:
            text = unlabeled_df.iloc[:, 0].to_list()
        logging.debug(f"Weakly Labelled Set Size: {len(text)}")

        weaklabelset = WeakLabelDataset(text=text)
        sampler = RandomSampler(weaklabelset)
        unlabeled_dataloader = DataLoader(weaklabelset, sampler=sampler, batch_size=self.batch_size)

        if dev_df is not None:
            dev_dataloader = self.__get_dataloader_from_df(dev_df)
        else:
            dev_dataloader = test_dataloader

        current_attention_dropout = self.attention_dropout
        current_classifier_dropout = self.classifier_dropout
        current_confidence_threshold = min_confidence_threshold

        # train teacher model
        logging.info("Training Base Classifier...")
        start = time.time()
        self.__train(
            train_dataloader=train_dataloader,
            dev_dataloader=dev_dataloader,
            evaluate_during_training=True,
            is_student=False,
        )

        _, acc, f1, clf_report = self.score(test_dataloader)
        end = time.time()
        logging.info("Classification Report\n" + clf_report)
        logging.info(f"Macro F1-Score: {f1*100:.2f}% - Accuracy: {acc*100:.2f}%")
        logging.info(f"Model {self.num_st_iter} runtime: {(end-start)/60:.2f} minutes.")

        for i in range(num_iters):
            start = time.time()
            self.num_st_iter += 1

            logging.debug(f"Inferring silver labels for student {i+1}...")
            weak_label_dataloader, num_new_examples_pos, num_new_examples_neg = self.__get_weak_labels(
                unlabeled_dataloader, current_confidence_threshold
            )
            logging.info(f"Added {num_new_examples_neg} Negative and {num_new_examples_pos} Positive samples.")

            trainset_steps = int(np.ceil(len(train_dataloader.dataset) / self.batch_size))
            weaklabelset_steps = int(np.ceil(len(weak_label_dataloader.dataset) / self.batch_size))
            unl_to_label_batch_ratio = int(np.ceil(weaklabelset_steps / trainset_steps))

            if increase_attention_dropout_amount is not None:
                current_attention_dropout += increase_attention_dropout_amount
            if increase_classifier_dropout_amount is not None:
                current_classifier_dropout += increase_classifier_dropout_amount
            if increase_confidence_threshold_amount is not None:
                current_confidence_threshold += increase_confidence_threshold_amount

            # instantiate new student model
            self.model = self.__init_model(current_attention_dropout, current_classifier_dropout)

            # train student model
            logging.info(f"Training Student {i+1} Classifier...")
            self.__train(
                train_dataloader=train_dataloader,
                dev_dataloader=dev_dataloader,
                evaluate_during_training=True,
                is_student=True,
                weak_label_dataloader=weak_label_dataloader,
                unl_to_label_batch_ratio=unl_to_label_batch_ratio,
            )

            _, acc, f1, clf_report = self.score(test_dataloader)
            end = time.time()
            logging.info("Classification Report:\n" + clf_report)
            logging.info(f"Macro F1-Score: {f1*100:.2f}% - Accuracy: {acc*100:.2f}%")
            logging.info(f"Model {self.num_st_iter} runtime: {(end-start)/60:.2f} minutes.")
