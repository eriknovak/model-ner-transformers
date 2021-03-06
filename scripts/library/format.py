import re

import torch

# python types
from typing import List, Tuple


def format_named_entities(
    model, tokens: List[str], labels: torch.Tensor
) -> List[Tuple[str, str]]:
    """Formats and joins the tokens and labels
        Args:
            model (nn.Module): The NER model.
            tokens (List[str]): The list of tokenized words.
            labels (torch.Tensor): The token labels extracted from the model.
                The labels are generated with the following function:
                `labels = torch.argmax(scores, dim=2)[0]` where `scores` are
                the label scores provided by `self.model`.
        Returns:
            ner (List[Tuple[str, str]]): The list of all named entity pairs.
        """

    regex_pattern = "▁|Ġ|##"

    def has_special_token(token: str) -> bool:
        return re.match(regex_pattern, token)

    def format_token(token: str) -> str:
        """Formats the token by removing the underscore
            Args:
                token (str): The token.
            Returns:
                str: The formatted string.
            """
        return re.sub(regex_pattern, "", token)

    # initialize the named entity array
    entities = []
    # initialize the variables with the first token and label
    tk = format_token(tokens[0])
    lb = model.config.id2label[labels[0]]

    # iterate through the tokens, labels pairs
    for token, label in zip(tokens[1:], labels[1:]):

        # get the current label
        l = model.config.id2label[label]

        # if the token is a new word or if the label
        # has changed, then save the previous NER example
        # and reset the token variable
        if has_special_token(token) or l != lb:
            entities.append((tk.strip(), lb))
            tk = ""

        # merge and update the token and label, respectively
        tk += format_token(token)
        lb = l

    # add the last named entity pair
    entities.append((tk.strip(), lb))

    # return the list of named entities
    return entities


def aggregate_entities(labels):
    """Merges and returns the entities that are not 'O'

    Args:
        labels (List[Tuple[str, str]]) - The list of (token, label) tuples.

    Returns:
        entities (List[str]) - The list of entities and types.

    """

    entities = []
    pt = None
    pl = None
    # iterate through the labels
    for (token, label) in labels:
        cl = label.split("-")[1] if label != "O" else label
        if cl != pl and pt != None:
            # add the entities
            entities.append((pt, pl))
            pt = None
            pl = None

        if cl == "O":
            # skip the other values
            continue

        if cl == pl:
            # add the token to the previous token
            pt += f" {token}"
        else:
            pt = token
            pl = cl

    if pt != None and pl != None:
        entities.append((pt, pl))
    # return the entities
    return entities

