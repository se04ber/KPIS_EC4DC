# Document for collecting research papers interesting for the project and documenting some results they find/discuss

Most entries below were suggested by Claude Code and are not read in full by us yet. To check one by one.

## Encoders

### NeoBERT (Le Breton et al. 2025)
- source: [arXiv 2502.19587](https://arxiv.org/pdf/2502.19587)
- what: new BERT-style encoder, compared to ModernBERT, BERT etc. in architecture and performance
- why for us: overview of current encoders, outlook
- comments: we tried it first on Kaggle, custom model code gave NaN outputs there, switched to BERT-base

### BERT (Devlin et al. 2019)
- source: [arXiv 1810.04805](https://arxiv.org/abs/1810.04805), NAACL 2019
- what: the original encoder, pretraining with masked words, fine-tuning recipe
- why for us: our model, baseline settings (lr 2e-5 to 5e-5, batch 16/32, 2-4 epochs)
- comments: must cite

## Fine-tuning BERT for classification

### How to Fine-Tune BERT for Text Classification? (Sun et al. 2019)
- source: [arXiv 1905.05583](https://arxiv.org/abs/1905.05583), CCL 2019
- what: many fine-tuning tricks tested on 8 text classification datasets: learning rate, which layer to use, long texts
- why for us: long documents: keeping head + tail of the text worked best, layer choice
- comments: closest to our task, read early

### On the Stability of Fine-tuning BERT (Mosbach et al. 2021)
- source: [arXiv 2006.04884](https://arxiv.org/abs/2006.04884), ICLR 2021
- what: why fine-tuning results jump between runs, and a simple stable recipe
- why for us: training length, warmup, why we need several seeds
- comments: for our "settings to vary: training length"

### Fine-Tuning Pretrained Language Models: Weight Initializations, Data Orders, and Early Stopping (Dodge et al. 2020)
- source: [arXiv 2002.06305](https://arxiv.org/abs/2002.06305)
- what: same setup, only random seed changes, results differ a lot
- why for us: argument for 3+ seeds and reporting spread

## LoRA and variants

### LoRA (Hu et al. 2022)
- source: [arXiv 2106.09685](https://arxiv.org/abs/2106.09685), ICLR 2022
- what: freeze the model, train two small matrices per layer instead
- why for us: our method. Their RoBERTa-base GLUE settings: lr 5e-4, r = 8, LoRA on query + value, 5 seeds
- comments: rank study shows even small r works

### AdaLoRA (Zhang et al. 2023)
- source: [arXiv 2303.10512](https://arxiv.org/abs/2303.10512), ICLR 2023
- what: gives different layers different LoRA ranks during training
- why for us: supports testing "LoRA in all layers vs. only top half"
- comments: we only borrow the idea, no need to implement AdaLoRA

### DoRA (Liu et al. 2024)
- source: [arXiv 2402.09353](https://arxiv.org/abs/2402.09353), ICML 2024
- what: splits weights into size and direction, LoRA only on direction
- why for us: one flag in peft (`use_dora=True`), easy extra comparison
- comments: "maybe" setting, only if time

## Long documents and pooling

### Universal Text Representation from BERT (Ma et al. 2019)
- source: [arXiv 1910.07973](https://arxiv.org/abs/1910.07973)
- what: compares ways to get one vector per text from BERT ([CLS], mean, max, layers)
- why for us: mean pooling beat [CLS] for classification with frozen BERT, so worth testing with LoRA
- comments: frozen setup, may differ after fine-tuning

### Revisiting Transformer-based Models for Long Document Classification (Dai et al. 2022)
- source: [arXiv 2204.06683](https://arxiv.org/abs/2204.06683), Findings of EMNLP 2022
- what: long documents: cutting vs. splitting into chunks, longer inputs help
- why for us: max length setting (128 vs. 512), documents longer than 512 tokens
- comments: chunking probably too much work for us, cite for motivation

## Looking inside the model (if time)

### BERT Rediscovers the Classical NLP Pipeline (Tenney et al. 2019)
- source: [arXiv 1905.05950](https://arxiv.org/abs/1905.05950), ACL 2019
- what: small classifiers per layer show which layer holds which kind of information
- why for us: per-layer classifier idea for the "if time" part

### What Happens To BERT Embeddings During Fine-tuning? (Merchant et al. 2020)
- source: [arXiv 2004.14448](https://arxiv.org/abs/2004.14448), BlackboxNLP 2020
- what: fine-tuning changes mostly the top layers, lower layers stay similar
- why for us: reason to test LoRA only in the top layers, and for the "if time" part

## Datasets

### GLUE (Wang et al. 2019)
- source: [arXiv 1804.07461](https://arxiv.org/abs/1804.07461), ICLR 2019
- what: benchmark of 9 sentence tasks, includes SST-2 (sentiment)
- why for us: SST-2 as first dataset, short sentences

### Character-level Convolutional Networks for Text Classification (Zhang et al. 2015)
- source: [arXiv 1509.01626](https://arxiv.org/abs/1509.01626), NeurIPS 2015
- what: introduces AG News and other text classification datasets
- why for us: citation for AG News if we use it as document dataset
