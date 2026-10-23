# Discussed Ideas

Training/Retraining Encoder for different architectures and compare:
1. Train the full Encoder on the dataset along with the classifier
2. Train some part of the Encoder or train additoinal weights[its called LoRA which is Low Rank Adaptation]
3. The final approach would be to freeze the encoder and only train the classifier.

Just Inference performance comparison:
4. Of different pretrained encoders
5. Of encoder vs. full small or commercial LMs

Further Ideas:
6. Trying to see if we can analyse/understand something in the vector space of Encoder for (document classification)
(We also discussed about understanding how information is actually stored in the vector space. But it would probably be too much for this course :( )

More ideas can be added, if not mentioned......

# Suggestions / Things to Keep In Mind
- Professor also mentioned/emphasized on the preprocessing and the quality of the data on which we are training. TBH, we didn't thought about it during our discussion, but its quite important.
Yes therefore maybe if we do just a inference analysis we need to optimal find encoders trained mostly on the same dataset, if possible.

- Encoder models can take an input of specific size for example 512. (document chunks sizes, rechunking maybe needed)

- We would have to see how much compute power we have and how much is required for our approach. Current recommended environment by claude is Kaggle


# Ressources available on Kaggle:
## GPU compute:
* fixed weekly allowance of 
* guaranteed 30 GPU hours/per week (T4 or P100)
* 20h/week on TPU (v6e-8 (8 cores)?)
* we can use both in the same week
(source: currently only Claude, needs to be checked on Kaggle again)

## Storage:
* 20 GB, persistent
* per session ca. 29–30 GB of system RAM and 4 CPU cores
(source: currently only Claude, needs to be checked on Kaggle again)


# First estimate for needed ressources for encoder experiments/training:
(Source: Claude:)
# Encoder Training Time on Kaggle (per 1 run)

**Assume:** ~20k docs, 512 tok, 3 epochs, fp16. Rough, scaled from BERT-base.
**Format:** `2×T4 / 1×P100`. P100 ≈ 1.5× slower than 2×T4 (P100 ≈ 1 T4, no tensor cores).
**Budget:** 30h GPU/week · 1 session max ~9–12h

| Task (1 run) | BERT-base (original, baseline) | BERT-large (original) | ModernBERT-base | Ettin-150m | DeBERTa-v3-base | NeoBERT | ModernBERT-large | Ettin-400m | Ettin-1B |
|---|---|---|---|---|---|---|---|---|---|
| Params | 110M | 340M | 149M | 150M | 184M | 250M | 395M | 400M | 1B |
| 1. Full fine-tune | 45 / 70 min | 2h / 3h | 55 / 80 min | 55 / 80 min | 75 / 115 min | 80 / 120 min | 2h / 3h | 2h / 3h | 5–6h / 8–9h, may not fit mem |
| 2. LoRA | 30 / 45 min | 80 / 120 min | 35 / 55 min | 35 / 55 min | 50 / 75 min | 50 / 75 min | 80 / 120 min | 80 / 120 min | 3.5–4h / 5–6h |
| 3. Frozen (embed once) | 5 / 8 min | 13 / 20 min | 6 / 9 min | 6 / 9 min | 8 / 12 min | 9 / 14 min | 13 / 20 min | 13 / 20 min | 40 / 60 min |
| Full runs / week | 40 / 26 | 15 / 10 | 32 / 22 | 32 / 22 | 24 / 16 | 22 / 15 | 15 / 10 | 15 / 10 | 5 / 3 (LoRA 8 / 5) |

**Notes**
- 2×T4 needs DDP (2 GPUs). Single-GPU code on T4 ≈ P100 time
- T4 + P100: no FlashAttention 2 → ModernBERT/Ettin/NeoBERT speed advantage smaller here
- P100: 1 GPU, 16 GB → large models need smaller batch + grad accumulation
- Ettin-1B on P100: ~9h = session limit → checkpoint, or use LoRA
- Longer docs (ModernBERT/Ettin/NeoBERT allow 4k–8k) → much slower, ~×(len/512) or worse
- DeBERTa: slow attention, slower than its size suggests
- Classifier after embedding: CPU, ~0 GPU
- Do 1 test run per encoder → measure → fix this table
