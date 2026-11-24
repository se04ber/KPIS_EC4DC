// One summary slide: idea, papers, data and process, grid search result, next steps.
// Build: NODE_PATH=~/Work/Project/venvs/node-pptx/node_modules node make_summary_slide.js
const pptxgen = require("pptxgenjs");
const path = require("path");

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.33 x 7.5 in
pres.title = "LoRA on BERT, status";
pres.author = "Sabrina Ebert, Muhammad Hamza";

const INK = "222222";
const HEAD = "222222"; // headers in plain text colour
const MUTED = "666666";
const FONT = "Calibri";

const slide = pres.addSlide();
slide.background = { color: "FFFFFF" };

slide.addText("LoRA on BERT, where we are", {
  x: 0.5, y: 0.3, w: 9.5, h: 0.6, fontFace: FONT, fontSize: 30, bold: true, color: INK, margin: 0, isTextBox: true,
});
slide.addText("KPIS, Ebert and Hamza, 24 Nov 2026", {
  x: 9.8, y: 0.42, w: 3.0, h: 0.4, fontFace: FONT, fontSize: 12, color: MUTED, align: "right", margin: 0, isTextBox: true,
});

function block(title, items, x, y, w, h) {
  const runs = [{ text: title, options: { bold: true, color: HEAD, fontSize: 16, breakLine: true, paraSpaceAfter: 4 } }];
  items.forEach((t, i) => runs.push({
    text: t, options: { bullet: { indent: 12 }, fontSize: 13, color: INK, paraSpaceAfter: 2, breakLine: i < items.length - 1 },
  }));
  slide.addText(runs, { x, y, w, h, fontFace: FONT, valign: "top", margin: 0, isTextBox: true });
}

// top row: three narrow columns
block("Idea", [
  "BERT-base + LoRA for text classification",
  "BERT frozen, train only LoRA + classifier",
  "which LoRA settings actually matter?",
  "guess: lr and layers more than rank",
], 0.5, 1.1, 3.9, 1.9);

block("Papers", [
  "BERT (Devlin 2019)",
  "LoRA (Hu 2022): small rank already works",
  "LoRA on small BERT (Frees 2025, preprint)",
  "top layers beat bottom ones (AdaLoRA 2023)",
], 4.7, 1.1, 4.1, 1.7);

block("Datasets", [
  "all three from Hugging Face",
  "SST-2: movie review sentences, 2 labels",
  "AG News: news, 4 topics",
  "20 Newsgroups (NG): forum posts, 20 groups",
], 9.1, 1.1, 4.1, 1.9);

// bottom: big plot left, results and next right
const PLOT = path.join(__dirname, "..", "Experiment", "results", "grid_main_effects.png");
const pw = 8.4, ph = pw * 858 / 2706;
slide.addImage({ path: PLOT, x: 0.4, y: 2.95, w: pw, h: ph, altText: "Grid search: dev accuracy against learning rate, rank and LoRA blocks for SST-2, AG News and 20 Newsgroups" });
slide.addText("grid search, 1002 runs, mean ± SD over seeds", {
  x: 0.5, y: 2.95 + ph + 0.03, w: pw - 0.1, h: 0.3, fontFace: FONT, fontSize: 11, italic: true, color: MUTED, align: "right", margin: 0, isTextBox: true,
});

// small full heatmap below, the plot above is three slices through it
const HEAT = path.join(__dirname, "..", "Experiment", "results", "grid_heatmaps.png");
const hy = 2.95 + ph + 0.45, hh = 7.25 - hy, hw = hh * 3036 / 2046;
slide.addImage({ path: HEAT, x: 0.5, y: hy, w: hw, h: hh, altText: "Full grid: mean dev accuracy for every learning rate, rank and block setting on all three datasets" });
slide.addShape(pres.shapes.LINE, { x: 0.5 + hw * 0.55, y: 2.95 + ph - 0.05, w: 0, h: hy - (2.95 + ph - 0.05), line: { color: MUTED, width: 1.25, beginArrowType: "triangle" } });
slide.addText([
  { text: "full grid: 3 lr × 4 ranks × 5 layer settings per dataset", options: { bold: true, breakLine: true } },
  { text: "plot above = 3 slices of it, the other two settings at their best", options: { breakLine: true } },
  { text: "all numbers on GitHub (Experiment/results)" },
], { x: 0.65 + hw, y: hy + 0.1, w: pw - hw - 0.4, h: hh - 0.1, fontFace: FONT, fontSize: 12, color: INK, valign: "top", margin: 0, isTextBox: true });

block("Experiment setup", [
  "Jupyter notebooks on Kaggle (T4 GPU)",
  "grid search first, 6 seeds",
  "best values = defaults for the studies",
], 9.2, 2.9, 3.8, 1.45);

block("Grid search (plot left)", [
  "lr has by far the biggest effect",
  "rank and layers: small, larger on 20 NG",
  "SST-2: last layer alone almost enough",
  "best on all 3: lr 1e-3, r 16, all 12 layers",
  "that is the edge of our grid though",
], 9.2, 4.2, 3.8, 1.75);

block("Next", [
  "layer study with the defaults",
  "rank study, incl. r 32",
  "XAI, if time",
], 9.2, 6.05, 3.8, 1.1);

slide.addNotes(
  "We fine-tune BERT-base with LoRA, so BERT itself stays frozen and only the small LoRA matrices and the classifier on top are trained. " +
  "The question is which LoRA settings actually change the accuracy. " +
  "Papers: BERT, the LoRA paper, one preprint that tried LoRA on a small BERT, and AdaLoRA, where LoRA in the top layers did better than in the bottom ones. " +
  "Data: three public datasets from Hugging Face, from short movie review sentences with 2 labels, over news with 4 topics, to forum posts with 20 groups. " +
  "We run everything as Jupyter notebooks on Kaggle on a T4 GPU (16 GB). First a grid search over learning rate, rank and how many of the last layers get LoRA, 6 seeds (3 each), so 1002 runs. " +
  "Result: the learning rate has by far the biggest effect. Rank and layers add a little, more on 20 Newsgroups. On SST-2 the last layer alone is almost enough. " +
  "The best setting is the same on all three datasets, lr 1e-3, rank 16, all 12 layers, but it sits at the edge of our grid. " +
  "Next we use these as defaults for the layer study and the rank study, and try rank 32. XAI if there is time."
);

pres.writeFile({ fileName: path.join(__dirname, "draft_update_slide_2411.pptx") }).then(f => console.log("wrote", f));
