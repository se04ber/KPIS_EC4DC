#!/usr/bin/env python3
"""Summarize a LoRA grid across seeds and plot mean accuracy + seed variance.

Defaults to the original experiment results directory. Run with:
    ~/Work/Project/venvs/kpis-test/bin/python plot_grid_seed_stats.py

Optional:
    python plot_grid_seed_stats.py --input /path/to/grid_results_combined.csv --output-dir /path/to/plots

The unit of replication is the *distinct seed*, not a CSV row. Multiple rows
for the same dataset/hyperparameters/seed are averaged first, so a repeated
seed cannot accidentally receive extra weight.

All accuracies are on the original CSV scale (e.g. 0.91, not 91%). Heatmaps
annotate mean and sample variance; bold text flags high observed variance
(the top quartile of seed variances across ALL configurations). Annotation text
uses one consistent color with a subtle outline to preserve readability.
Line-plot error bars are +/- 1 sample SD (the square root of the sample
variance), which has the same units as accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
import matplotlib.patheffects as path_effects

plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})

RESULTS_DIR = (
    Path.home()
    / "Work/Project/bert-lora-vectorspace/Github/KPIS_EC4DC/Experiment/results"
)
DATASETS = ("sst2", "ag_news", "20newsgroups")
LEARNING_RATES = (1e-4, 5e-4, 1e-3)
RANKS = (1, 4, 8, 16)
LAYERS = (1, 2, 4, 8, 12)
PARAMETERS = ("rank", "last_layers", "lr")
REQUIRED_COLUMNS = ("dataset", *PARAMETERS, "seed", "accuracy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=RESULTS_DIR / "grid_results_combined.csv",
        help="Combined CSV containing one or more experiment rows per seed.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=RESULTS_DIR,
        help="Directory to save plots and the seed summary CSV.",
    )
    return parser.parse_args()


def seed_summary(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    missing = set(REQUIRED_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    for column in (*PARAMETERS, "accuracy"):
        raw[column] = pd.to_numeric(raw[column], errors="raise")

    n_before = len(raw)
    raw = raw.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    if len(raw) < n_before:
        print(f"Warning: skipped {n_before - len(raw)} rows with missing required values.")
    if raw.empty:
        raise ValueError("No valid experiment rows were found.")

    # Average any repeated measurements within the same seed before computing
    # cross-seed statistics; otherwise repeats would bias the means/variances.
    seed_keys = ["dataset", *PARAMETERS, "seed"]
    aggregation = {"accuracy": ("accuracy", "mean")}
    if "trainable" in raw.columns:
        aggregation["trainable"] = ("trainable", "first")
    per_seed = raw.groupby(seed_keys, as_index=False).agg(**aggregation)
    if len(per_seed) != len(raw):
        print(
            f"Note: averaged {len(raw) - len(per_seed)} additional repeated "
            "row(s) within the same seed and configuration."
        )

    config_keys = ["dataset", *PARAMETERS]
    summary_aggregation = {
        "mean": ("accuracy", "mean"),
        "variance": ("accuracy", "var"),  # sample variance (ddof=1)
        "std": ("accuracy", "std"),  # sample standard deviation (ddof=1)
        "n_seeds": ("accuracy", "size"),
    }
    if "trainable" in per_seed.columns:
        summary_aggregation["trainable"] = ("trainable", "first")
    summary = per_seed.groupby(config_keys, as_index=False).agg(**summary_aggregation)
    return summary.sort_values(config_keys).reset_index(drop=True)


def format_var(value: float) -> str:
    if not np.isfinite(value):
        return "n/a"
    if value == 0:
        return "0"
    return f"{value:.1e}" if abs(value) < 1e-3 else f"{value:.4f}"


def grid_for(summary: pd.DataFrame, dataset: str, lr: float, field: str) -> np.ndarray:
    match = summary.loc[
        summary["dataset"].eq(dataset)
        & np.isclose(summary["lr"], lr, rtol=1e-8, atol=1e-12)
    ]
    pivot = match.pivot(index="rank", columns="last_layers", values=field)
    return pivot.reindex(index=RANKS, columns=LAYERS).to_numpy(dtype=float)


def plot_heatmaps(summary: pd.DataFrame, output_dir: Path, field: str) -> None:
    """Shared color scales; bold labels flag relatively high seed variance.

    Color encodes the selected field, NEVER text color. Text color is always
    black (with an identical white outline for contrast on all backgrounds).
    Bold labels identify variance in the highest quartile of all grid settings;
    this indicates seed-to-seed instability, not statistical significance.
    """
    is_variance = field == "variance"
    fig = plt.figure(figsize=(13.8, 9.3))
    # Reserve narrow columns for colorbars so none overlap cells.
    gs = fig.add_gridspec(
        3, 6,
        width_ratios=[1, 0.035, 1, 0.035, 1, 0.035],
        left=0.075, right=0.97, bottom=0.095, top=0.925,
        wspace=0.38, hspace=0.28,
    )
    axes = np.empty((3, 3), dtype=object)
    for i in range(3):
        for j in range(3):
            axes[i, j] = fig.add_subplot(gs[i, 2 * j])
    fig.suptitle(
        "Across-seed sample variance of dev accuracy"
        if is_variance
        else "Mean dev accuracy across seeds (cell labels: mean and sample variance)",
        fontsize=13,
        y=0.99,
    )

    # One color scale for ALL nine heatmaps, not one per dataset.
    all_values = summary[field].to_numpy(dtype=float)
    all_values = all_values[np.isfinite(all_values)]
    if all_values.size:
        vmin, vmax = float(all_values.min()), float(all_values.max())
    else:
        vmin, vmax = 0.0, 1.0
    if np.isclose(vmin, vmax, rtol=0, atol=1e-15):
        vmax = vmin + max(abs(vmin), 1e-4) * 0.1
    norm = Normalize(vmin=vmin, vmax=vmax)

    # One variance threshold across ALL datasets, rates, ranks and layers.
    # Never fabricate a variance for single-seed configurations.
    valid_variances = summary["variance"].to_numpy(dtype=float)
    valid_variances = valid_variances[np.isfinite(valid_variances)]
    if valid_variances.size and valid_variances.max() > valid_variances.min():
        q75 = float(np.quantile(valid_variances, 0.75))
        # When the 75th percentile equals the minimum due to ties (e.g.,
        # many exact zero variances), only strictly higher values are bold.
        bold_above = q75 <= valid_variances.min()
    else:
        q75 = np.nan  # No meaningful high-variance group exists.
        bold_above = False

    # Same annotation appearance in every cell. Background-independent
    # legibility is obtained by a fine outline, not a white/black text switch.
    text_effects = [path_effects.withStroke(linewidth=1.45, foreground="white")]

    for j, dataset in enumerate(DATASETS):
        for i, lr in enumerate(LEARNING_RATES):
            ax = axes[i, j]
            values = grid_for(summary, dataset, lr, field)
            means = grid_for(summary, dataset, lr, "mean")
            variances = grid_for(summary, dataset, lr, "variance")
            n_seeds = grid_for(summary, dataset, lr, "n_seeds")
            image = ax.imshow(
                np.ma.masked_invalid(values),
                cmap="viridis",
                norm=norm,
                aspect="auto",
                interpolation="nearest",
            )
            image.cmap.set_bad("#dedede")

            for a in range(len(RANKS)):
                for b in range(len(LAYERS)):
                    variance = variances[a, b]
                    if not np.isfinite(means[a, b]):
                        label = "—"
                    elif is_variance:
                        label = format_var(variance) if np.isfinite(variance) else "n=1"
                    else:
                        variance_label = (
                            f"var {format_var(variance)}"
                            if n_seeds[a, b] >= 2
                            else "var n/a (1 seed)"
                        )
                        label = f"{means[a, b]:.3f}\n{variance_label}"

                    is_high_variance = (
                        np.isfinite(variance)
                        and np.isfinite(q75)
                        and (variance > q75 if bold_above else variance >= q75)
                    )
                    ax.text(
                        b, a, label, ha="center", va="center",
                        color="black", fontweight="bold" if is_high_variance else "normal",
                        fontsize=7 if not is_variance else 8,
                        path_effects=text_effects,
                    )

            ax.set_xticks(range(len(LAYERS)), labels=LAYERS)
            ax.set_yticks(range(len(RANKS)), labels=RANKS)
            if i == 0:
                ax.set_title(dataset, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"lr {lr:.0e}\nrank r")
            if i == 2:
                ax.set_xlabel("LoRA in the last n blocks")

        cb = fig.colorbar(axes[0, j].images[0], cax=fig.add_subplot(gs[:, 2 * j + 1]))
        cb.set_label("Sample variance" if is_variance else "Mean dev accuracy", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    if np.isfinite(q75):
        comparator = ">" if bold_above else "≥"
        bold_note = f"Bold text = high seed variance ({comparator} global 75th percentile: {q75:.4g})"
    else:
        bold_note = "Bold text = high seed variance (not shown: insufficient variation/data)"
    fig.text(0.5, 0.042, bold_note + "; NOT statistical significance.",
             ha="center", fontsize=8)
    fig.text(
        0.5, 0.022,
        "All labels are black with a white outline for contrast; colorbar limits are shared across all nine panels."
        + (" n/a = only one seed." if not is_variance else " n=1 means variance undefined."),
        ha="center", fontsize=8,
    )
    basename = "grid_variance_heatmaps" if is_variance else "grid_heatmaps"
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"{basename}.{extension}", dpi=220)
    plt.close(fig)


def select_fixed(summary: pd.DataFrame, fixed: dict[str, float]) -> pd.DataFrame:
    selected = summary
    for parameter, target in fixed.items():
        selected = selected[
            np.isclose(selected[parameter], target, rtol=1e-8, atol=1e-12)
        ]
    return selected


def plot_main_effects(summary: pd.DataFrame, output_dir: Path) -> None:
    slices = (
        ("lr", LEARNING_RATES, {"rank": 16, "last_layers": 12}),
        ("rank", RANKS, {"lr": 1e-3, "last_layers": 12}),
        ("last_layers", LAYERS, {"lr": 1e-3, "rank": 16}),
    )
    labels = {
        "lr": "Learning rate",
        "rank": "Rank r",
        "last_layers": "LoRA in the last n blocks",
    }
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9))

    for ax, (variable, values, fixed) in zip(axes, slices):
        for j, dataset in enumerate(DATASETS):
            subset = select_fixed(
                summary.loc[summary["dataset"].eq(dataset)], fixed
            )
            subset = subset.set_index(variable).reindex(values)
            mean = subset["mean"].to_numpy(dtype=float)
            std = subset["std"].to_numpy(dtype=float)
            # A single seed has no estimable variance; draw no bar at that point,
            # rather than pretending the observed SD is an estimated zero.
            errors = np.nan_to_num(std, nan=0.0)
            ax.errorbar(
                np.arange(len(values)), mean, yerr=errors,
                marker="o", linewidth=1.6, elinewidth=1.2, capsize=3,
                color=f"C{j}", label=dataset,
            )
            # Distinguish single-seed points visually from points with SD bars.
            single_seed = subset["n_seeds"].to_numpy(dtype=float) == 1
            ax.scatter(
                np.arange(len(values))[single_seed], mean[single_seed],
                marker="x", s=44, color=f"C{j}", zorder=5,
            )

        formatted_fixed = ", ".join(
            f"lr {value:.0e}" if key == "lr" else
            f"{'blocks' if key == 'last_layers' else key} {value:g}"
            for key, value in fixed.items()
        )
        ticklabels = [f"{x:.0e}" if variable == "lr" else str(x) for x in values]
        ax.set_xticks(range(len(values)), labels=ticklabels)
        ax.set(xlabel=labels[variable], ylabel="Mean dev accuracy", title=f"Fixed: {formatted_fixed}")
        ax.grid(axis="y", alpha=0.22)

    axes[0].legend(fontsize=8, loc="best")
    fig.suptitle("Across-seed mean ± 1 SD (error bars); × marks single-seed points", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    for extension in ("png", "svg"):
        fig.savefig(output_dir / f"grid_main_effects.{extension}", dpi=220)
    plt.close(fig)


def print_report(summary: pd.DataFrame) -> None:
    print("\nStatistics are computed across distinct seeds; variance and SD use ddof=1.")
    print("A configuration with only one seed has undefined variance and SD (NaN).\n")
    for dataset in DATASETS:
        subset = summary.loc[summary["dataset"].eq(dataset)]
        leaders = subset.sort_values("mean", ascending=False).head(3)
        cols = [*PARAMETERS, "mean", "std", "variance", "n_seeds"]
        if "trainable" in summary:
            cols.append("trainable")
        print(f"== {dataset} best (top 3):")
        print(leaders[cols].round(6).to_string(index=False))
        print(f"  Median seed variance: {subset['variance'].median():.6g}\n")

    wide = summary.pivot_table(index=list(PARAMETERS), columns="dataset", values="mean")
    available = [dataset for dataset in DATASETS if dataset in wide.columns]
    wide["mean_across_datasets"] = wide[available].mean(axis=1)
    print("Overall top configurations (unweighted average of dataset seed means):")
    print(wide.sort_values("mean_across_datasets", ascending=False).head(6).round(4))

    slices = (
        ("rank", RANKS, {"lr": 1e-3, "last_layers": 12}),
        ("last_layers", LAYERS, {"lr": 1e-3, "rank": 16}),
        ("lr", LEARNING_RATES, {"rank": 16, "last_layers": 12}),
    )
    print("\nMean dev accuracy in percent, with step-to-step changes in percentage points:")
    for dataset in DATASETS:
        for variable, values, fixed in slices:
            subset = select_fixed(summary[summary["dataset"].eq(dataset)], fixed)
            series = subset.set_index(variable)["mean"].reindex(values) * 100
            print(
                f"  {dataset} {variable}: "
                f"means={series.round(1).tolist()}, "
                f"deltas={series.diff().iloc[1:].round(1).tolist()}"
            )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = seed_summary(args.input)
    summary_path = args.output_dir / "grid_results_seed_summary.csv"
    summary.to_csv(summary_path, index=False)
    print_report(summary)
    plot_heatmaps(summary, args.output_dir, field="mean")
    plot_heatmaps(summary, args.output_dir, field="variance")
    plot_main_effects(summary, args.output_dir)
    print(f"\nSaved seed summary: {summary_path}")
    print(f"Saved grid_heatmaps, grid_variance_heatmaps, and grid_main_effects as PNG and SVG in {args.output_dir}")


if __name__ == "__main__":
    main()
