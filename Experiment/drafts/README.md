# Drafts

First tests by Sabrina, not yet discussed together. Hamza's main notebook is `../Experiment.ipynb`.
Every draft starts with his cells unchanged, the new cells after them are marked `# NEW`.

| notebook | what | status |
|---|---|---|
| `Draft_addGridsearch.ipynb` | Step 1 of `next_steps_studies.md`: rank x number of last LoRA layers, three datasets, 3 seeds, prints the default values | tested at tiny size on CPU, not run on Kaggle yet |
| `Draft_addLayerStudy.ipynb` | Step 2, layers: LoRA in different blocks with the defaults from the grid search, 4 seeds | tested at tiny size on CPU, not run on Kaggle yet |
| `Draft_addStudy.ipynb` | earlier idea: one LoRA setting at a time (layers, modules, rank, lr, text length), plus Step 1 and 2 at the end | tested at tiny size, superseded by the two above |
| `Draft_addXAI.ipynb` | `Draft_addStudy` plus a look inside the trained models (weight change, layer similarity, classifier per layer, word attributions) | for the "if time" part, tested at tiny size |

Run on Kaggle with `STUDY_QUICK = True` first, then `False`. Outputs go to `/kaggle/working/outputs/study` and `study.zip`.
