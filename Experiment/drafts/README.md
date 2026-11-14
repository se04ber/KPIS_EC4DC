# Drafts

First tests by Sabrina, not yet discussed together. Hamza's main notebook is `../Experiment.ipynb`, the grid search next to it is `../Draft_addGridsearch.ipynb`.
Every draft starts with Hamza's cells unchanged, the new cells after them are marked `# NEW`.

| notebook | what | status |
|---|---|---|
| `Draft_addLayerStudy.ipynb` | Step 2 of `next_steps_studies.md`, layers: LoRA in different blocks with the defaults from the grid search, 4 seeds | tested at tiny size on CPU, not run on Kaggle yet |
| `Draft_testVariation.ipynb` | earlier idea: one LoRA setting at a time (layers, modules, rank, lr, text length), plus Step 1 and 2 at the end | tested at tiny size, superseded by the grid search and the layer study |
| `Draft_testXAI.ipynb` | `Draft_testVariation` plus a look inside the trained models (weight change, layer similarity, classifier per layer, word attributions) | for the "if time" part, tested at tiny size |

Run on Kaggle with `STUDY_QUICK = True` first, then `False`. Outputs go to `/kaggle/working/outputs/study` and `study.zip`.
