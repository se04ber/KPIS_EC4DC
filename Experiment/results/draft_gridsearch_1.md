# Grid search 1: results, short

Notebook: `Experiment/Draft_addGridsearch.ipynb`. Two full runs on Kaggle (T4), same code, different seeds.
Data: `grid_results_combined.csv` (1002 runs), per setting over seeds: `grid_results_seed_summary.csv`.
Plots (PNG and SVG): `grid_heatmaps` (mean, variance in the cells), `grid_variance_heatmaps` (seed variance), `grid_main_effects` (mean +- 1 SD).
Script: `plot_grid_seed_stats.py`, makes all three plots and the summary from the combined CSV.

## What ran

| run | seeds | SST-2 | AG News | 20 Newsgroups | output |
|---|---|---|---|---|---|
| Sabrina | 0, 1, 2 | 60 of 60 | 60 of 60 | 34 of 60 (stopped) | local zip |
| Muhammad | 100, 101, 102 | 60 of 60 | 60 of 60 | 60 of 60 | `Output/outputs_backup.zip` |

So SST-2 and AG News have 6 seeds per setting, 20 Newsgroups has 3 (6 for 34 settings, rank 16 only 3).

Both runs agree: 154 settings ran in both, mean difference 0.2 (AG News), 0.4 (SST-2), 1.1 (20 Newsgroups) points, correlation 0.99.

Versions in Muhammad's run: torch 2.11.0, transformers 5.20.2, peft 0.21.2, GPU Tesla T4.

## How the grid works (pseudocode)

```
libraries: transformers (BERT, Trainer), peft (LoRA), datasets, scikit-learn (accuracy, macro F1), pandas, matplotlib

for each dataset in [sst2, ag_news, 20newsgroups]:
    train = 20000 texts (20 Newsgroups: all of it), dev = 2000 texts (SST-2: half of validation)
    test split is never used

for dataset, rank r in [1, 4, 8, 16], last n blocks in [1, 2, 4, 8, 12], lr in [1e-4, 5e-4, 1e-3], seed:
    model = BERT-base + new classifier layer
    put LoRA on query + value, only in the last n blocks, rank r, alpha = 2r
    freeze BERT, train only LoRA + classifier
    train 2 epochs (20 Newsgroups 4), batch 32, warmup 6%, weight decay 0.01, fp16, max 128 tokens
    accuracy + macro F1 on dev
    save one row to grid_results.csv
```

All settings that are not in the grid come from Muhammad's config cell (section 2).

## Best settings (dev accuracy, mean over seeds)

| dataset | best | accuracy | seed std | seeds | 2nd best | gap | clearly better? |
|---|---|---|---|---|---|---|---|
| SST-2 | r 16, all 12 blocks, lr 1e-3 | 0.916 | 0.006 | 6 | r 16, last 2 blocks: 0.912 | 0.5 | no, gap 1.1 x its error |
| AG News | r 16, all 12 blocks, lr 1e-3 | 0.939 | 0.003 | 6 | r 8, all 12 blocks: 0.934 | 0.5 | yes, 3.8 x |
| 20 Newsgroups | r 16, all 12 blocks, lr 1e-3 | 0.721 | 0.004 | 3 | r 16, last 8 blocks: 0.712 | 1.0 | yes, 2.7 x |

Same winner on all three datasets. Next best over all three: r 16 with the last 8 blocks, then r 8 with all 12 blocks.
"Clearly better" here means the gap is more than about 2 times its standard error (from the seed spread of both settings). A rough rule, not a test.

**Defaults for the next studies: lr 1e-3, rank 16, all 12 blocks.**

## Does it still go up, or does it flatten? (accuracy in %, the other two settings at the default)

Learning rate (rank 16, all 12 blocks):

| | 1e-4 | 5e-4 | 1e-3 |
|---|---|---|---|
| SST-2 | 89.1 | 91.1 (+2.0) | 91.6 (+0.5) |
| AG News | 91.6 | 93.2 (+1.6) | 93.9 (+0.7) |
| 20 Newsgroups | 54.5 | 70.0 (+15.5) | 72.1 (+2.1) |

Big jump first, then small. Flattens.

Rank (lr 1e-3, all 12 blocks):

| | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| SST-2 | 90.3 | 90.6 (+0.3) | 90.7 (+0.2) | 91.6 (+0.9) |
| AG News | 92.3 | 93.0 (+0.7) | 93.4 (+0.3) | 93.9 (+0.5) |
| 20 Newsgroups | 67.9 | 70.2 (+2.3) | 70.7 (+0.5) | 72.1 (+1.4) |

Small steps, but no clear stop. 16 is still a bit better than 8.

LoRA in the last n blocks (lr 1e-3, rank 16):

| | 1 | 2 | 4 | 8 | 12 |
|---|---|---|---|---|---|
| SST-2 | 91.1 | 91.2 (+0.1) | 90.4 (-0.8) | 90.9 (+0.5) | 91.6 (+0.7) |
| AG News | 92.0 | 92.3 (+0.3) | 92.8 (+0.5) | 93.2 (+0.4) | 93.9 (+0.7) |
| 20 Newsgroups | 65.9 | 68.7 (+2.8) | 69.9 (+1.2) | 71.2 (+1.3) | 72.1 (+1.0) |

SST-2: flat, last block alone almost enough. AG News: slow steady gain. 20 Newsgroups: big gain first, then smaller.

Seed noise per setting (median SD): about 0.6 points (SST-2), 0.3 (AG News), 1.1 (20 Newsgroups).

## Is a step more than seed noise? (gap in points / how many times its error)

| | rank 8 -> 16 | last 8 -> all 12 blocks | last 1 -> all 12 blocks | lr 5e-4 -> 1e-3 |
|---|---|---|---|---|
| SST-2 | +0.9 / 2.5 x | +0.7 / 1.5 x | +0.5 / 1.6 x | +0.5 / 1.4 x |
| AG News | +0.5 / 3.8 x | +0.7 / 4.4 x | +1.9 / 17 x | +0.7 / 3.5 x |
| 20 Newsgroups | +1.4 / 4.2 x | +1.0 / 2.7 x | +6.3 / 7.9 x | +2.1 / 5.0 x |

SST-2: only the rank step is clearly above noise, blocks and lr steps are not. AG News and 20 Newsgroups: all steps are real, even small ones, because the seeds agree well there.

## Where is the variance high? (`grid_variance_heatmaps`)

| median seed SD in points | lr 1e-4 | 5e-4 | 1e-3 |
|---|---|---|---|
| SST-2 | 0.6 | 0.6 | 0.6 |
| AG News | 0.4 | 0.3 | 0.3 |
| 20 Newsgroups | 2.3 | 0.8 | 0.6 |

- low learning rate (1e-4) and low rank (1) are the most unstable, most clearly on 20 Newsgroups (rank 1, lr 1e-4: SD up to about 4.7 points)
- the best settings (lr 1e-3, rank 8 to 16) are also the most stable
- SST-2 has the same noise everywhere (about 0.6), its small dev set (436 sentences) is the limit there

## Short

- learning rate matters most, 1e-3 best, flattens
- rank and blocks help a bit, more on harder data (20 Newsgroups)
- short texts (SST-2): only the last block is nearly enough, the differences between block settings are within noise
- best settings are also the most stable across seeds
- best values sit at the edge of the grid (lr 1e-3, rank 16, 12 blocks), so higher values were not tried
- dev scores only, test untouched

## Open

- try rank 32 and lr 2e-3 once, to see if it really flattens
- layer study with the defaults (lr 1e-3, rank 16)
