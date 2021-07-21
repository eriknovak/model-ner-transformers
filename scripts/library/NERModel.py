import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModelForTokenClassification, AutoTokenizer
import pytorch_lightning as pl
import torchmetrics

# import format named entities
from library.format import format_named_entities

# python types
from typing import Dict


class NERModel(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        label2id: Dict[str, str],
        eps: float = 1e-5,
        lr: float = 1e-5,
        wd: float = 1e-2,
    ):
        super().__init__()

        # save the model hyperparameters
        self.save_hyperparameters("model_name", "lr", "wd", "eps")

        # set the placeholder for the entities
        num_classes = len(label2id.keys())

        # prepare the model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_classes
        )
        self.model.config.id2label = {value: key for key, value in label2id.items()}
        self.model.config.label2id = label2id

        # prepare the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
        )

        # add evaluation metrics
        self.acc = torchmetrics.classification.Accuracy(
            num_classes=num_classes, average="macro"
        )
        self.prec = torchmetrics.classification.Precision(
            num_classes=num_classes, average="macro"
        )
        self.rec = torchmetrics.classification.Recall(
            num_classes=num_classes, average="macro"
        )

    def forward(self, text: str):
        # encode and calculate the label logits
        encodings = self.tokenizer(text, truncation=True, return_tensors="pt")
        outputs = self.model(**encodings)

        # get the tokens and labels from the outputs
        tokens = self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])
        labels = outputs["logits"].argmax(dim=2)[0]

        # calculate the entities
        entities = format_named_entities(self.model, tokens[1:-1], labels[1:-1])

        return labels, tokens, entities

    def on_training_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, train_batch, batch_idx):
        outputs = self.model(**train_batch)
        loss = outputs["loss"]
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_eval_step(val_batch, batch_idx)
        self.log("val_loss", loss)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self._shared_log("val_metrics", validation_step_outputs)

    def test_step(self, test_batch, batch_idx):
        loss = self._shared_eval_step(test_batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def test_epoch_end(self, test_step_outputs):
        self._shared_log("test_metrics", test_step_outputs)

    def _shared_eval_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        # get the loss value
        loss = outputs["loss"]
        # get the prediction and true labels
        true, pred = batch["labels"], outputs["logits"].argmax(dim=2)

        # get the attention mask
        attention_mask = batch["attention_mask"]

        # iterate through the labels
        for idx in range(true.shape[0]):
            # get the values that are actually corresponding to the values
            last_idx = attention_mask[idx].sum()
            curr_pred = pred[idx][:last_idx]
            curr_true = true[idx][:last_idx]
            # measure the performance
            self.prec(curr_pred, curr_true)
            self.rec(curr_pred, curr_true)
            self.acc(curr_pred, curr_true)
        return loss

    def _shared_log(self, state, step_outputs):
        self.log_dict(
            {
                f"{state}_accuracy": self.acc,
                f"{state}_precision": self.prec,
                f"{state}_recall": self.rec,
            }
        )

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.wd,
            eps=self.hparams.eps,
        )
        return optimizer

