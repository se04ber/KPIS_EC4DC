# KPIS Project: Timeline

**Topic:** Document Classification Performance Comparison Study (Encoder Architecture)\
**Focus:** How LoRA fine-tuning settings affect a BERT encoder's document classification performance\
**If there is time:** out of interest, an exploratory look inside the fine-tuned models (vector space, explainable AI)\
**Authors:** Muhammad and Sabrina

## Weekly plan (winter semester 2026/27)

| Week from | Activity | Deadline / event |
|---|---|---|
| 27.10 | Research review, plan settings | 27.10 milestones |
| 03.11 | Research review, plan settings, paper outline | 03.11 paper outline |
| 10.11 | Setup: notebooks on Kaggle, data split, first baseline run (approach 1) | |
| 17.11 | Run experiments: LoRA study, first half of the settings (approach 2) | |
| 24.11 | Run experiments: LoRA study, second half of the settings (approach 2) | |
| 01.12 | Run experiments: second dataset (approach 3), slides for the interim talk | ~02./03.12 interim talk |
| 08.12 | Evaluation: tables and plots (+ approach 5, if time) | |
| 15.12 | Evaluation: what worked, what didn't (+ approach 5, if time) | |
| 21.12 | Write up (Christmas break) | |
| 05.01 | Write up, review others' papers | 05.01 draft paper |
| 12.01 | Revise paper, prepare talk | 12.01 reviews |
| 19.01 | Prepare talk | 19.01 final paper |
| 20.01 | Talk | 20./21.01 or 27./28.01 |

## Approaches

1. **Baseline:** BERT as it is, simple classifier on top
2. **LoRA study:** change one setting at a time, look at accuracy and cost, every setting 3 times
3. **Second dataset:** short sentences vs. longer documents, do the same settings still work?
4. **Fallback:** just inference, compare pretrained encoders without any LoRA/fine-tuning
5. **If time:** exploratory look inside the trained models to explain the results (vector space, explainable AI)

Dates are planned and may shift slightly as the work progresses.\
Table and PDF layout generated with Claude Code. Structure and content about 95% planned by us, with some suggestions from Claude Code (e.g. 3 runs per setting) and help with refactoring grammar and phrasing.
