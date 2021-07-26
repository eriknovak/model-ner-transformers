import os
import re
import sys
import math
import yaml
import torch

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =====================================
# Import Model Checkpoint
# =====================================

checkpoint_path = sys.argv[1]

# =====================================
# Define the Model
# =====================================

from library.NER import NER

# initialize the (trained) model
model = NER.load_from_checkpoint(checkpoint_path=checkpoint_path)

# set it in evaluation mode
model.eval()

# prepare text strings
examples = [
    "Janez Novak is a researcher working at the Jožef Stefan Institute located in Ljubljana, Slovenia.",
    "Janez Novak je raziskovalec, ki dela na Institutu Jožef Stefan, ki se nahaja v Ljubljani, Sloveniji.",
    "Janez Novak ist ein Suchender am Jožef Stefan Institut in Ljubljana, Slowenien.",
    "Janez Novak ni mtafuta kazi anayefanya kazi katika Taasisi ya Jožef Stefan iliyoko Ljubljana, Slovenia.",
    "Janez Novak je pretraživač koji radi na Institutu Jožef Stefan smještenom u Ljubljani, Slovenija.",
    "Janez Novak jest poszukiwaczem pracującym w Instytucie Jožefa Stefana w Lublanie w Słowenii.",
]

for example in examples:
    # get the entities from the text
    entities = model(example)
    print(entities)

