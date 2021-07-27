# 📜 Named Entity Recognition

This repository contains the code to build a named entity recognition model
using 🤗 huggingface, pytorch, pytorch lightning.

## ☑️ Requirements

Before starting the project make sure these requirements are available:

- [conda][conda]. For setting up your research environment and python dependencies.
- [git][git]. For versioning your code.

## 🛠️ Setup

### Install the conda environment

First create the new conda environment:

```bash
conda env create -f environment.yml
```

### Activate the environment

To activate the newly set environment run:

```bash
conda activate ner
```

### Install PyTorch

Based on your CUDA drivers install the appropriate pytorch version. Please
reference the instructions [here][pytorch].

```bash
conda install pytorch cudatoolkit=11.1 -c pytorch -c nvidia
```

### Deactivate the environment

When the environment is not in use anymore deactivate it by running:

```bash
conda deactivate
```

## 🥼 Experiment Setup

**NOTE:** Training a single model requires approximate 10 GB of GPU space.

### Parameters

To run the experiments one can manually change `params.yaml` file with different
parameters (the provided parameters are good default values). The default parameters
are presented in Table 1.

| Param                  | Default Value    | Description                                                                       |
| ---------------------- | ---------------- | --------------------------------------------------------------------------------- |
| model_name             | xlm-roberta-base | The 🤗 Transformers pretrained model. Supports `roberta` and `xlm-roberta` models |
| training.seed          | 1                | The seed used to create the experiments deterministics                            |
| training.epochs        | 5                | The number of epochs the model is trained                                         |
| training.batch_size    | 8                | The number of examples in a batch                                                 |
| training.grad_step     | 4                | The number of gradients accumulated before updating the model weights             |
| training.epsilon       | 0.00001          | The epsilon used in the optimizer                                                 |
| training.learning_rate | 0.00001          | The starting learning rate used in the optimizer                                  |
| training.weight_decay  | 0.01             | The weight decay used in the optimizer                                            |

_Table 1. The default training parameters._

The [XLM-RoBERTa][xlm-roberta] is a cross-lingual language model proposed in the following paper:

```bibtex
@inproceedings{conneau-etal-2020-unsupervised,
  title = "Unsupervised Cross-lingual Representation Learning at Scale",
  author = "Conneau, Alexis and
    Khandelwal, Kartikay and
    Goyal, Naman and
    Chaudhary, Vishrav and
    Wenzek, Guillaume and
    Guzm{\'a}n, Francisco and
    Grave, Edouard and
    Ott, Myle and
    Zettlemoyer, Luke and
    Stoyanov, Veselin",
  booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics",
  month = jul,
  year = "2020",
  publisher = "Association for Computational Linguistics",
  url = "https://aclanthology.org/2020.acl-main.747",
  doi = "10.18653/v1/2020.acl-main.747",
  pages = "8440--8451",
}
```

The default dataset used to train the Named Entity Recognition model is
[CoNLL-2003][conll2003]. The dataset concerns language-independent named entity
recognition. The paper introducing the dataset is:

```bibtex
@inproceedings{tjong-kim-sang-de-meulder-2003-introduction,
  title = "Introduction to the {C}o{NLL}-2003 Shared Task: Language-Independent Named Entity Recognition",
  author = "Tjong Kim Sang, Erik F. and De Meulder, Fien",
  booktitle = "Proceedings of the Seventh Conference on Natural Language Learning at {HLT}-{NAACL} 2003",
  year = "2003",
  url = "https://www.aclweb.org/anthology/W03-0419",
  pages = "142--147",
}
```

**NOTE:** To train a NER model using a different dataset, one must
first check its structure and change the `./scripts/train.py` file accordingly.

### Training the Model

Then simply run the following command to train the model:

```bash
python scripts/train.py
```

After about 1 hour of training the above command will build a model and will
be located in the `/models` folder. The default format of the model path is:

```bash
./models/xlm-roberta-base-conll2003.ckpt
```

## 📋 Experiment Results

After the model is trained it is automatically evaluated with the
predefined `validation` and `test` set. The results of the validation and test
scores of the trained models are found in Table 2.

| Model                      | Cross-Lingual |  Accuracy   |  Precision  |   Recall    |
| -------------------------- | :-----------: | :---------: | :---------: | :---------: |
| xlm-roberta-base-conll2003 |     True      | 93.8 / 90.5 | 93.5 / 86.3 | 93.8 / 90.3 |
| roberta-base-conll2003     |     False     | 94.1 / 90.7 | 93.9 / 85.7 | 94.1 / 90.7 |
| roberta-large-conll2003    |     False     |    - / -    |    - / -    |    - / -    |

_Table 2. Named entity recognition performance of the models. The scores
represent the validation/test scores._

## 🔎 Extracting Named Entities

Once the `xlm-roberta-base-conll2003` is trained one can use the model described
as in the [./scripts/inference.py][inference] file. It contains sentence examples
in different languages as shown in Table 3.

| Language | Sentence                                                                                                  |
| -------- | --------------------------------------------------------------------------------------------------------- |
| English  | "Janez Novak is a researcher working at the Jozef Stefan Institute located in Ljubljana, Slovenia."       |
| Slovene  | "Janez Novak je raziskovalec, ki dela na Institutu Jožef Stefan, ki se nahaja v Ljubljani, Sloveniji."    |
| German   | "Janez Novak ist ein Suchender am Jozef Stefan Institut in Ljubljana, Slowenien."                         |
| Swahili  | "Janez Novak ni mtafuta kazi anayefanya kazi katika Taasisi ya Jozef Stefan iliyoko Ljubljana, Slovenia." |
| Croatian | "Janez Novak je pretraživač koji radi na Institutu Jozef Stefan smještenom u Ljubljani, Slovenija."       |
| Polish   | "Janez Novak jest poszukiwaczem pracującym w Instytucie Jozefa Stefana w Lublanie w Słowenii."            |

_Table 3. Sentence examples in different languages. All of the sentences are translations of the English one._

To run it with the trained model run:

```bash
python scripts/inference.py ./models/xlm-roberta-base-conll2003.ckpt
```

The above script contains will return the labels presented in Table 4.

| Language | Named Entity Labels                                                                                             |
| -------- | --------------------------------------------------------------------------------------------------------------- |
| English  | ("Janez Novak", "PER"), ("Jozef Stefan Institute", "ORG"), ("Ljubljana", "LOC"), ("Slovenia", "LOC")            |
| Slovene  | ("Janez Novak", "PER"), ("Institutu Jožef Stefan", "ORG"), ("Ljubljani", "LOC"), ("Sloveniji", "LOC")           |
| German   | ("Janez Novak", "PER"), ("Jozef Stefan Institut", "ORG"), ("Ljubljana", "LOC"), ("Slowenien", "LOC")            |
| Swahili  | ("Janez Novak", "PER"), ("Taasisi", "ORG"), ("Jozef Stefan", "ORG"), ("Ljubljana", "LOC"), ("Slovenia", "LOC")  |
| Croatian | ("Janez Novak", "PER"), ("Institutu Jozef Stefan", "ORG"), ("Ljubljani", "LOC"), ("Slovenija", "LOC")           |
| Polish   | ("Janez Novak", "PER"), ("Instytu", "ORG"), ("Jozefa Stefana", "ORG"), ("Lublanie", "LOC"), ("Słowenii", "LOC") |

_Table 4. The named entity labels for the same sentence in different languages provided by the trained model._

## 💽 Models

To get access to the models with the above performance results contact the
repository maintainer.

# 🏬 Acknowledgments

This work is developed by [Department of Artificial Intelligence][ailab] at [Jozef Stefan Institute][ijs].

The work is supported by the following EU Horizon 2020 projects:

- [Humane AI NET][humaneai] (H2020-ICT-952026)

[git]: https://git-scm.com/
[conda]: https://docs.conda.io/en/latest/
[pytorch]: https://pytorch.org/
[xlm-roberta]: https://huggingface.co/transformers/model_doc/xlmroberta.html
[conll2003]: https://huggingface.co/datasets/conll2003
[inference]: ./scripts/inference.py
[ailab]: http://ailab.ijs.si/
[ijs]: https://www.ijs.si/
[humaneai]: https://www.humane-ai.eu/
