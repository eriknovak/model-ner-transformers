import torch
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
        lr: float = 1e-3,
        wd: float = 1e-2,
        eps=1e-6,
    ):
        super().__init__()

        # learning parameters
        self.lr = lr
        self.wd = wd
        self.eps = eps

        # set the placeholder for the entities
        num_labels = len(label2id.keys())

        # prepare the model
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )
        self.model.config.id2label = {value: key for key, value in label2id.items()}
        self.model.config.label2id = label2id

        # prepare the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, add_prefix_space=True
        )

    def forward(self, text: str):
        # encode and calculate the label logits
        encodings = self.tokenizer(text, truncation=True, return_tensors="pt")
        outputs = self.model(**encodings)

        # get the tokens and labels from the outputs
        tokens = self.tokenizer.convert_ids_to_tokens(encodings["input_ids"][0])
        labels = torch.argmax(outputs["logits"], dim=2)[0]

        # calculate the entities
        entities = format_named_entities(self.model, tokens[1:-1], labels[1:-1])

        return labels, tokens, entities

    def training_step(self, train_batch, batch_idx):
        outputs = self.model(**train_batch)
        loss = outputs["loss"]
        self.log("training_loss", loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self._shared_eval_step(val_batch, batch_idx)
        self.log("validation_loss", loss, on_step=True, prog_bar=True)
        return loss

    def test_step(self, test_batch, batch_idx):
        loss = self._shared_eval_step(test_batch, batch_idx)

        return loss

    def _shared_eval_step(self, batch, batch_idx):
        outputs = self.model(**batch)
        loss = outputs["loss"]

        # TODO: add validation checking

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd, eps=self.eps
        )
        return optimizer

