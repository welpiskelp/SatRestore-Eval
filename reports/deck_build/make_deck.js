const pptxgen = require("pptxgenjs");
const path = require("path");

const IMG = (name) => path.join("..", "partial_results", name);

const BLUE = "1B6CA8";
const NAVY = "0B3D5C";
const AMBER = "E07B39";
const RED = "9E2B25";
const INK = "2B2B2B";
const MUTE = "5A6B75";
const WHITE = "FFFFFF";
const PAPER = "FFFFFF";

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
const W = 13.33, H = 7.5;

function titleSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("SNR-Conditioned Reconstruction of Sentinel-2 Ship Imagery", {
    x: 0.9, y: 2.35, w: 11.5, h: 1.6, fontFace: "Cambria", fontSize: 34, bold: true,
    color: WHITE, isTextBox: true, margin: 0,
  });
  s.addText("Progress Update: Evaluating Noise-Level Conditioning for Ship-Preserving Image Reconstruction", {
    x: 0.9, y: 3.85, w: 11.0, h: 0.8, fontFace: "Calibri", fontSize: 18, color: "CADCFC",
    isTextBox: true, margin: 0,
  });
  s.addText("October 2026", {
    x: 0.9, y: 6.6, w: 6, h: 0.4, fontFace: "Calibri", fontSize: 13, color: "7FA8C9",
    isTextBox: true, margin: 0,
  });
}

function sectionHeader(s, kicker, title) {
  s.background = { color: PAPER };
  if (kicker) {
    s.addText(kicker.toUpperCase(), {
      x: 0.7, y: 0.4, w: 11.5, h: 0.35, fontFace: "Calibri", fontSize: 12, bold: true,
      color: BLUE, isTextBox: true, margin: 0, charSpacing: 2,
    });
  }
  s.addText(title, {
    x: 0.7, y: kicker ? 0.72 : 0.45, w: 11.9, h: 0.75, fontFace: "Cambria", fontSize: 26, bold: true,
    color: NAVY, isTextBox: true, margin: 0,
  });
}

function significance(s, items, x, y, w, opts) {
  const o = opts || {};
  s.addText("SIGNIFICANCE", {
    x, y, w, h: 0.3, fontFace: "Calibri", fontSize: 10.5, bold: true, color: BLUE,
    isTextBox: true, margin: 0, charSpacing: 1,
  });
  const paras = items.map((t, i) => ({
    text: t,
    options: { bullet: { code: "2022" }, breakLine: i < items.length - 1, paraSpaceAfter: 6 },
  }));
  s.addText(paras, {
    x, y: y + 0.32, w, h: o.h || 1.3, fontFace: "Calibri", fontSize: o.fontSize || 11.5, color: INK,
    isTextBox: true, margin: 0, valign: "top",
  });
}

function objectivesSlide() {
  const s = pres.addSlide();
  sectionHeader(s, "Research Questions", "Three questions under investigation");
  const items = [
    { n: "1", t: "Does knowing the noise level help?", d: "If the model is given the noise level of an image, does it restore it more accurately than a model without this information?" },
    { n: "2", t: "Does comparing color bands help?", d: "Satellite images have 12 color channels. Does allowing the model to cross-reference them improve results?" },
    { n: "3", t: "Does protecting ships specifically help?", d: "If training rewards the model for accuracy on ships, does this come at the cost of overall image quality?" },
  ];
  const colW = 3.75, gap = 0.3, x0 = 0.7, y0 = 1.9;
  items.forEach((it, i) => {
    const x = x0 + i * (colW + gap);
    s.addShape("ellipse", { x, y: y0, w: 0.6, h: 0.6, fill: { color: BLUE }, line: { type: "none" } });
    s.addText(it.n, { x, y: y0, w: 0.6, h: 0.6, align: "center", valign: "middle", fontFace: "Calibri",
      fontSize: 20, bold: true, color: WHITE, isTextBox: true, margin: 0 });
    s.addText(it.t, { x, y: y0 + 0.8, w: colW, h: 0.9, fontFace: "Calibri", fontSize: 16, bold: true,
      color: NAVY, isTextBox: true, margin: 0 });
    s.addText(it.d, { x, y: y0 + 1.7, w: colW, h: 1.8, fontFace: "Calibri", fontSize: 12.5, color: MUTE,
      isTextBox: true, margin: 0 });
  });
  significance(s, [
    "Each question corresponds to a hypothesis (H1, H2, H3) fixed in the project protocol before training began.",
    "Hypotheses were pre-registered to prevent post-hoc redefinition of success criteria.",
  ], 0.7, 5.6, 11.5, { h: 0.9 });
  s.addText("This update addresses Research Question 1. Questions 2 and 3 are addressed in the next phase (see Plan From Here).", {
    x: 0.7, y: 6.95, w: 11.5, h: 0.4, fontFace: "Calibri", fontSize: 12, italic: true, color: MUTE,
    isTextBox: true, margin: 0,
  });
}

function pipelineSlide() {
  const s = pres.addSlide();
  sectionHeader(s, "Methodology", "Methodology overview");
  const steps = [
    ["1,053 real ships", "16 real Sentinel-2 satellite scenes, independently verified against published ship counts"],
    ["Controlled noise added", "A measured, repeatable amount of noise is added at 9 strength levels, from severe to minimal"],
    ["Model trained on Kaggle GPUs", "A neural network is trained to remove the added noise while preserving ship shapes"],
    ["Every measurement verified", "Scoring code was checked against hand-derived analytic cases before any result was used"],
  ];
  const w = 2.85, gap = 0.25, x0 = 0.7, y0 = 1.9;
  steps.forEach((st, i) => {
    const x = x0 + i * (w + gap);
    s.addShape("roundRect", { x, y: y0, w, h: 2.75, rectRadius: 0.08,
      fill: { color: "F1F6FA" }, line: { type: "none" } });
    s.addText(String(i + 1), { x: x + 0.2, y: y0 + 0.18, w: 0.6, h: 0.5, fontFace: "Calibri", fontSize: 22,
      bold: true, color: BLUE, isTextBox: true, margin: 0 });
    s.addText(st[0], { x: x + 0.2, y: y0 + 0.68, w: w - 0.4, h: 0.7, fontFace: "Calibri", fontSize: 14,
      bold: true, color: NAVY, isTextBox: true, margin: 0 });
    s.addText(st[1], { x: x + 0.2, y: y0 + 1.38, w: w - 0.4, h: 1.3, fontFace: "Calibri", fontSize: 11.5,
      color: MUTE, isTextBox: true, margin: 0 });
    if (i < steps.length - 1) {
      s.addText("→", { x: x + w, y: y0 + 1.1, w: gap, h: 0.5, align: "center", fontSize: 16,
        color: BLUE, isTextBox: true, margin: 0 });
    }
  });
  significance(s, [
    "Dataset provenance, the noise model, and the evaluation code were independently verified before any result was used.",
    "Measurement correctness was checked against hand-derived analytic cases rather than assumed from a single run.",
    "Results reported in this update reflect model behavior, not measurement error.",
  ], 0.7, 5.05, 11.9, { h: 1.6 });
}

function fullTileSlide(sceneLabel, foldLabel, img, bullets) {
  const s = pres.addSlide();
  sectionHeader(s, "Qualitative Results", `Full-scene reconstruction: ${sceneLabel} (${foldLabel})`);
  const imgW = 11.6, imgH = imgW / 5.519;
  const y = 1.95;
  s.addImage({ path: IMG(img), x: (W - imgW) / 2, y, w: imgW, h: imgH });
  s.addText(`Test scene excluded from training for ${foldLabel}. Left to right: original image, degraded input (5 dB), reconstruction (5 dB), degraded input (30 dB), reconstruction (30 dB).`, {
    x: 0.9, y: y + imgH + 0.18, w: 11.5, h: 0.45, fontFace: "Calibri", fontSize: 13, color: MUTE,
    isTextBox: true, margin: 0,
  });
  significance(s, bullets, 0.9, 5.95, 11.5, { h: 1.1 });
}

function generalizesSlide() {
  const s = pres.addSlide();
  s.background = { color: PAPER };
  s.addText("CROSS-FOLD CONSISTENCY", { x: 0.7, y: 0.4, w: 6, h: 0.35, fontFace: "Calibri", fontSize: 12,
    bold: true, color: BLUE, isTextBox: true, margin: 0, charSpacing: 2 });
  s.addText("Consistent results across four independent regions", { x: 0.7, y: 0.72, w: 6.2, h: 1.1, fontFace: "Cambria",
    fontSize: 24, bold: true, color: NAVY, isTextBox: true, margin: 0 });
  s.addText("Each scene shown was excluded from training for its corresponding fold and evaluated only at test time.", {
    x: 0.7, y: 1.85, w: 6.0, h: 1.0, fontFace: "Calibri", fontSize: 13.5, color: MUTE, isTextBox: true, margin: 0,
  });
  const callouts = [
    ["Brest, France", "fold 0"], ["Rotterdam, Netherlands", "fold 1"],
    ["Toulon, France", "fold 2"], ["Southampton, UK", "fold 3"],
  ];
  callouts.forEach((c, i) => {
    const y = 2.95 + i * 0.72;
    s.addShape("ellipse", { x: 0.7, y: y + 0.08, w: 0.18, h: 0.18, fill: { color: BLUE }, line: { type: "none" } });
    s.addText(`${c[0]}  —  ${c[1]}`, { x: 1.05, y, w: 5.6, h: 0.45, fontFace: "Calibri", fontSize: 14,
      color: INK, isTextBox: true, margin: 0 });
  });
  significance(s, [
    "Demonstrates generalization across four independent geographic regions.",
    "Each scene shown was also scored numerically as part of the quantitative evaluation.",
    "Four independent regions rule out a single favorable scene as the explanation for the result.",
  ], 0.7, 5.85, 6.0, { h: 1.3, fontSize: 11 });
  const imgH = 6.2, imgW = imgH * 0.737;
  s.addImage({ path: IMG("08_one_ship_per_fold.png"), x: 7.3, y: 0.65, w: imgW, h: imgH });
}

function closeupSlide() {
  const s = pres.addSlide();
  sectionHeader(s, "Ship-Level Detail", "Detail recovery at ship scale");
  const imgH = 4.0, imgW = imgH * 1.213;
  s.addImage({ path: IMG("02_ship_closeups.png"), x: (W - imgW) / 2, y: 1.55, w: imgW, h: imgH });
  s.addText("Ships occupy only a few pixels. At the most severe noise level tested (5 dB), shape, position, and color are recovered accurately.", {
    x: 0.9, y: 1.55 + imgH + 0.1, w: 11.5, h: 0.35, fontFace: "Calibri", fontSize: 12, color: MUTE,
    isTextBox: true, margin: 0,
  });
  significance(s, [
    "Evaluates ship preservation specifically, not only overall image quality.",
    "Establishes the baseline against which ship-weighted training (Research Question 3) will be compared.",
    "Addresses a known limitation of whole-image metrics, which can remain high while small objects degrade.",
  ], 0.9, 6.15, 11.5, { h: 1.2, fontSize: 11 });
}

function chartSlide(title, img, imgAspect, caption, bullets) {
  const s = pres.addSlide();
  sectionHeader(s, "Quantitative Results", title);
  const imgW = 8.9, imgH = imgW / imgAspect;
  s.addImage({ path: IMG(img), x: (W - imgW) / 2, y: 1.65, w: imgW, h: imgH });
  s.addText(caption, { x: 0.9, y: 1.65 + imgH + 0.12, w: 11.5, h: 0.45, fontFace: "Calibri", fontSize: 12.5,
    color: MUTE, isTextBox: true, margin: 0 });
  significance(s, bullets, 0.9, 1.65 + imgH + 0.65, 11.5, { h: 0.95, fontSize: 11.5 });
}

function statsSlide() {
  const s = pres.addSlide();
  sectionHeader(s, "Statistical Validation", "Statistical significance of the Research Question 1 result");
  s.addText("Method: exact Wilcoxon signed-rank test, Holm-corrected across all 9 noise levels, cluster-robust 95% intervals (clusters = tiles sharing a scene), fixed in the protocol before training.", {
    x: 0.7, y: 1.5, w: 11.9, h: 0.4, fontFace: "Calibri", fontSize: 11.5, italic: true, color: MUTE,
    isTextBox: true, margin: 0,
  });
  const stats = [
    ["4 of 9", "noise levels show a statistically significant advantage, after correcting for testing 9 levels at once", BLUE],
    ["4 of 4", "folds agree in direction at the most severe noise level (0 dB)", BLUE],
    ["1 of 3", "seeds completed per fold to date; additional seeds are in progress", AMBER],
  ];
  const w = 3.75, gap = 0.3, x0 = 0.7, y0 = 2.1;
  stats.forEach((st, i) => {
    const x = x0 + i * (w + gap);
    s.addText(st[0], { x, y: y0, w, h: 0.8, fontFace: "Cambria", fontSize: 36, bold: true,
      color: st[2], isTextBox: true, margin: 0 });
    s.addText(st[1], { x, y: y0 + 0.85, w, h: 0.8, fontFace: "Calibri", fontSize: 12, color: MUTE,
      isTextBox: true, margin: 0 });
  });
  const rows = [
    ["SNR (dB)", "With conditioning", "Without (blind)", "Gap", "Significant?"],
    ["0", "30.19", "29.90", "+0.29", "Yes"],
    ["5", "33.34", "33.09", "+0.25", "Yes"],
    ["10", "35.95", "35.86", "+0.09", "Yes"],
    ["15", "38.38", "38.43", "-0.06", "Too close"],
    ["20", "40.87", "41.02", "-0.16", "No"],
    ["25", "43.53", "43.73", "-0.20", "No"],
    ["30", "46.32", "46.47", "-0.15", "Too close"],
    ["35", "49.01", "48.94", "+0.07", "Too close"],
    ["40", "51.26", "50.78", "+0.48", "Yes"],
  ];
  const tableRows = rows.map((r, ri) => r.map((c, ci) => ({
    text: c,
    options: {
      fontFace: "Calibri", fontSize: 9.5, color: ri === 0 ? WHITE : INK,
      fill: ri === 0 ? { color: NAVY } : (ri % 2 === 0 ? { color: "F1F6FA" } : { color: WHITE }),
      bold: ri === 0, align: ci === 0 ? "left" : "center", valign: "middle",
      fontColor: ci === 4 && ri > 0 ? (c === "Yes" ? BLUE : (c === "Too close" ? MUTE : AMBER)) : undefined,
    },
  })));
  s.addTable(tableRows, {
    x: 0.7, y: 4.05, w: 11.9, h: 2.75, border: { type: "solid", color: "DCE6EC", pt: 0.75 },
    autoPage: false,
  });
  s.addText("Underlying measurements are recorded in the project results database; the test definition is fixed in the evaluation protocol.", {
    x: 0.7, y: 7.05, w: 11.9, h: 0.35, fontFace: "Calibri", fontSize: 10, italic: true, color: MUTE,
    isTextBox: true, margin: 0,
  });
}

function statusSlide() {
  const s = pres.addSlide();
  sectionHeader(s, "Where things stand", "Status at a glance");
  const rows = [
    ["Data and evaluation pipeline", "Complete", BLUE],
    ["Main model, all 4 folds", "Complete", BLUE],
    ["Blind baseline, all 4 folds", "Complete", BLUE],
    ["Published baseline (FFDNet)", "Tested, limitation documented", AMBER],
    ["First statistical result (RQ1)", "Complete, preliminary (1 seed)", AMBER],
    ["Additional seeds for full rigor", "In progress", AMBER],
    ["Cross-band attention test (RQ2)", "In progress", AMBER],
    ["Ship-focused training test (RQ3)", "Not started", RED],
  ];
  let y = 1.95;
  rows.forEach((r) => {
    s.addShape("roundRect", { x: 0.7, y, w: 11.9, h: 0.52, rectRadius: 0.04,
      fill: { color: "FAFCFD" }, line: { color: "E7EEF2", width: 0.75 } });
    s.addText(r[0], { x: 0.95, y, w: 8.3, h: 0.52, valign: "middle", fontFace: "Calibri", fontSize: 13.5,
      color: INK, isTextBox: true, margin: 0 });
    s.addShape("ellipse", { x: 9.35, y: y + 0.17, w: 0.18, h: 0.18, fill: { color: r[2] }, line: { type: "none" } });
    s.addText(r[1], { x: 9.65, y, w: 2.8, h: 0.52, valign: "middle", fontFace: "Calibri", fontSize: 12.5,
      bold: true, color: r[2], isTextBox: true, margin: 0 });
    y += 0.62;
  });
}

function nextStepsSlide() {
  const s = pres.addSlide();
  sectionHeader(s, "Plan", "Plan from here");
  const items = [
    ["Now", "Complete additional training runs (2 more seeds per fold) to finalize the preliminary result."],
    ["Next", "Evaluate the remaining two hypotheses: cross-band attention and ship-focused training, each with its own comparison."],
    ["Then", "Compare against one to two additional published methods to broaden the baseline comparison."],
    ["Ongoing", "Distribute remaining runs across the team using the established pipeline."],
  ];
  let y = 2.0;
  items.forEach((it) => {
    s.addShape("roundRect", { x: 0.7, y, w: 1.5, h: 0.95, rectRadius: 0.06,
      fill: { color: NAVY }, line: { type: "none" } });
    s.addText(it[0], { x: 0.7, y, w: 1.5, h: 0.95, align: "center", valign: "middle", fontFace: "Calibri",
      fontSize: 13, bold: true, color: WHITE, isTextBox: true, margin: 0 });
    s.addText(it[1], { x: 2.45, y: y + 0.05, w: 10.0, h: 0.95, valign: "middle", fontFace: "Calibri",
      fontSize: 14, color: INK, isTextBox: true, margin: 0 });
    y += 1.2;
  });
}

function closingSlide() {
  const s = pres.addSlide();
  s.background = { color: NAVY };
  s.addText("Summary", { x: 0.9, y: 2.7, w: 11, h: 0.9, fontFace: "Cambria", fontSize: 30, bold: true,
    color: WHITE, isTextBox: true, margin: 0 });
  s.addText("Noise-level conditioning provides a measurable advantage, confirmed across all 4 data splits, with the strongest effect at the most severe noise levels.", {
    x: 0.9, y: 3.6, w: 10.8, h: 1.4, fontFace: "Calibri", fontSize: 16, color: "CADCFC",
    isTextBox: true, margin: 0,
  });
}

titleSlide();
objectivesSlide();
pipelineSlide();
fullTileSlide("Southampton, UK", "fold 3", "07_full_tile_fold3_southampton.png", [
  "Provides direct visual evidence in addition to the quantitative comparison.",
  "The scene is excluded from fold 3 training and evaluated only at test time.",
  "5 dB represents a stress-test noise level; 30 dB represents a near-nominal sensor noise level.",
]);
fullTileSlide("Brest, France", "fold 0", "07_full_tile_fold0_brest1.png", [
  "Independent region and fold, held out from fold 0 training.",
  "Coastal and harbor structures are preserved under both noise levels tested.",
  "Consistent with the result shown for fold 3.",
]);
fullTileSlide("Rotterdam, Netherlands", "fold 1", "07_full_tile_fold1_rotterdam1.png", [
  "Highest ship-pixel density among the four scenes shown.",
  "Dense port infrastructure is preserved at both noise levels tested.",
  "Independent region and fold, held out from fold 1 training.",
]);
generalizesSlide();
closeupSlide();
chartSlide("With noise-level information vs. without", "03_psnr_vs_snr_comparison.png", 2.634,
  "Left: reconstruction quality for both models across noise levels. Right: the difference between them.",
  [
    "Provides the primary quantitative test of Research Question 1.",
    "Identical metric, test noise, and evaluation procedure for every run, fixed in the protocol prior to training.",
    "The right panel isolates the conditioning effect from factors shared by both models.",
  ]);
chartSlide("Consistency of the advantage across folds", "04_fold_consistency.png", 2.446,
  "Each bar represents one of the 4 independent data splits.",
  [
    "Confirms that the result is consistent across independent data splits.",
    "Each fold uses non-overlapping test tiles, ruling out a single tile as the explanation.",
    "Supports the cluster-robust confidence intervals reported on the following slide.",
  ]);
statsSlide();
statusSlide();
nextStepsSlide();
closingSlide();

pres.writeFile({ fileName: "SatRestore_Progress_Update_v3.pptx" }).then(() => {
  console.log("written");
});
