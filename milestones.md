# Document for collecting Milestones for the KPIS Project
Authors: Muhammad and Sabrina
Topic: Document Classification Performance Comparison Study Encoder Architecture

Mainly: how LoRA fine-tuning settings affect a BERT encoder's document classification performance.
If there is time, out of interest: a exploratory look inside the fine-tuned models (vector space, explainable AI).

## Milestones:

### Research review
- how BERT works
  *Devlin et al. 2019*
- how LoRA works, which settings matter
  *Hu et al. 2022*
- how others fine-tune BERT for text classification
  *Sun et al. 2019, Mosbach et al. 2021*
- pooling and long documents
  *Ma et al. 2019, Dai et al. 2022*

### Approaches
1. **Baseline:** BERT as it is, simple classifier on top
2. **LoRA study:** change one setting at a time, look at accuracy and cost
3. **Second dataset:** short sentences vs. longer documents
4. **Fallback:** compare pretrained encoders without training, if training doesn't fit our resources (unlikely, LoRA already ran on Kaggle)
5. **If time:** look inside the trained models to explain the results
   *Tenney et al. 2019, Merchant et al. 2020*

### Setup
- Ressources decided on to use: Kaggle, 2x T4, 30 GPU hours per week
- fixed split, separate texts for training and testing
- SST-2 (GLUE) first, then maybe a more document specific dataset, e.g. AG News
  *Wang et al. 2019, Zhang et al. 2015*

### Settings to possibly vary (will still be from priority sorted, based on research review)
(training maybe:)
- rank r = 4 / 8 / 16 / 64, alpha = r vs. 2r
- learning rate 1e-4 / 5e-4 / 1e-3
  *Hu et al. 2022*
- training length
  *Mosbach et al. 2021*

architecture:
- LoRA on query + value vs. all linear layers
  *Hu et al. 2022*
- LoRA in all 12 layers vs. only the top half
  *Merchant et al. 2020, Zhang et al. 2023 (AdaLoRA)*
- pooling: [CLS] vs. mean
  *Ma et al. 2019*
- max length 128 vs. 512
  *Sun et al. 2019*
- maybe: DoRA instead of LoRA
  *Liu et al. 2024*

for strengthening/certainty estimation of results: every setting atleast 3 times with different seeds
*Dodge et al. 2020*

### Evaluation
- how well it classifies: accuracy, (macro F1, so small classes count too)
  scikit-learn `accuracy_score`, `f1_score`
- which classes get mixed up: confusion matrix
  scikit-learn `confusion_matrix`
- what it costs: trainable parameters, training time, peak GPU memory
  peft `print_trainable_parameters`, `torch.cuda.max_memory_allocated`
- mean and standard deviation over the seeds, plots with matplotlib
- if time (5.): simple classifier per layer, how well classes group (silhouette score), which words decide
  scikit-learn `LogisticRegression`, `silhouette_score`, captum

### Write up
- paper, 4 pages
- talk, 10 + 5 min

## Timeline

| from | what | deadline |
|---|---|---|
| 27.10 | research review, plan settings | 27.10 milestones |
| 03.11 | research review, plan settings | 03.11 paper outline |
| 10.11 | setup | |
| 17.11 | run experiments | |
| 24.11 | run experiments | |
| 01.12 | run experiments | ~02./03.12 interim talk |
| 08.12 | evaluation (+ 5. if time) | |
| 15.12 | evaluation (+ 5. if time) | |
| 21.12 | write up (Christmas break) | |
| 05.01 | write up, review others' papers | 05.01 draft paper |
| 12.01 | revise paper, prepare talk | 12.01 reviews |
| 19.01 | prepare talk | 19.01 final paper |
| 20.01 | talk | 20./21.01 or 27./28.01 |

Sources are mostly suggestions from Claude Code. We will go through them in the next weeks and keep, replace or drop them.
