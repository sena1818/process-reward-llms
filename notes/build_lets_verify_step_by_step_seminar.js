const PptxGenJS = require("pptxgenjs");
const path = require("path");

const pptx = new PptxGenJS();
pptx.layout = "LAYOUT_WIDE";
pptx.author = "葛婉宁";
pptx.company = "Heidelberg University";
pptx.subject = "Process Reward in Large Language Models Seminar";
pptx.title = "Let's Verify Step by Step Seminar";
pptx.lang = "en-US";
pptx.theme = {
  headFontFace: "Aptos",
  bodyFontFace: "Aptos",
  lang: "en-US",
};

const W = 13.333;
const H = 7.5;

const COLORS = {
  blue: "142D69",
  blue2: "1D3B82",
  lightBlue: "AAB9DC",
  paleBlue: "EEF3FB",
  paleBlue2: "E2EAF7",
  line: "D6DFEF",
  text: "20345B",
  muted: "5D6E92",
  soft: "7A88A8",
  white: "FFFFFF",
  green: "DCEEE3",
  greenText: "2E6A49",
  orange: "FCE8D1",
  orangeText: "A85A10",
  red: "F9D8D8",
  redText: "9B2F2F",
};

const DIR = __dirname;
const assets = {
  logo: path.join(__dirname, "assets", "heidelberg_logo.png"),
};

function addBackground(slide) {
  slide.background = { color: COLORS.white };
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 0,
    w: W,
    h: 0.45,
    line: { color: COLORS.paleBlue },
    fill: { color: COLORS.paleBlue2 },
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0,
    y: 0.46,
    w: W,
    h: 0,
    line: { color: COLORS.line, pt: 1.2 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 0,
    y: 6.92,
    w: 4.4,
    h: 0.58,
    line: { color: COLORS.blue },
    fill: { color: COLORS.blue },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 4.4,
    y: 6.92,
    w: 2.25,
    h: 0.58,
    line: { color: COLORS.blue2 },
    fill: { color: COLORS.blue2 },
  });
  slide.addShape(pptx.ShapeType.rect, {
    x: 6.65,
    y: 6.92,
    w: W - 6.65,
    h: 0.58,
    line: { color: COLORS.lightBlue },
    fill: { color: COLORS.lightBlue },
  });
  slide.addImage({
    path: assets.logo,
    x: 12.42,
    y: 0.03,
    h: 0.35,
    w: 0.66,
  });
  slide.addText("Process Reward in LLMs", {
    x: 0.25,
    y: 7.05,
    w: 3.9,
    h: 0.18,
    fontFace: "Aptos",
    fontSize: 10,
    color: COLORS.white,
    align: "center",
    margin: 0,
  });
  slide.addText("YanGang", {
    x: 4.55,
    y: 7.05,
    w: 1.95,
    h: 0.18,
    fontFace: "Aptos",
    fontSize: 10,
    color: COLORS.white,
    align: "center",
    margin: 0,
  });
  slide.addText("Heidelberg, 27.04.2026", {
    x: 7.05,
    y: 7.05,
    w: 5.7,
    h: 0.18,
    fontFace: "Aptos",
    fontSize: 10,
    color: COLORS.blue,
    align: "center",
    margin: 0,
  });
}

function addTitle(slide, title) {
  slide.addText(title, {
    x: 0.58,
    y: 0.62,
    w: 11.6,
    h: 0.35,
    fontFace: "Aptos",
    fontSize: 25,
    bold: true,
    color: COLORS.blue,
    margin: 0,
  });
}

function addSectionLabel(slide, label) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.62,
    y: 1.05,
    w: 1.7,
    h: 0.3,
    rectRadius: 0.04,
    line: { color: COLORS.lightBlue, pt: 1 },
    fill: { color: COLORS.paleBlue },
  });
  slide.addText(label, {
    x: 0.72,
    y: 1.11,
    w: 1.5,
    h: 0.14,
    fontSize: 9,
    color: COLORS.muted,
    bold: true,
    margin: 0,
    align: "center",
  });
}

function addBulletBlock(slide, x, y, w, title, bullets, options = {}) {
  slide.addText(title, {
    x,
    y,
    w,
    h: 0.22,
    fontSize: options.titleSize ?? 16,
    bold: true,
    color: COLORS.blue,
    margin: 0,
  });
  const lines = bullets.map((b) => `- ${b}`).join("\n");
  slide.addText(lines, {
    x,
    y: y + 0.28,
    w,
    h: options.h ?? 1.45,
    fontSize: options.fontSize ?? 14,
    color: COLORS.text,
    breakLine: false,
    valign: "top",
    margin: 0.02,
    paraSpaceAfterPt: 9,
  });
}

function addCallout(slide, text, options = {}) {
  const x = options.x ?? 0.75;
  const y = options.y ?? 5.95;
  const w = options.w ?? 11.8;
  const h = options.h ?? 0.72;
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.05,
    line: { color: options.line ?? COLORS.blue, pt: 1.3 },
    fill: { color: options.fill ?? COLORS.white },
  });
  slide.addText(text, {
    x: x + 0.15,
    y: y + 0.14,
    w: w - 0.3,
    h: h - 0.2,
    fontSize: options.fontSize ?? 16,
    bold: options.bold ?? true,
    color: options.color ?? COLORS.blue,
    align: "center",
    valign: "mid",
    margin: 0,
  });
}

function addMiniCard(slide, x, y, w, h, title, body, options = {}) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.05,
    line: { color: options.line ?? COLORS.line, pt: 1 },
    fill: { color: options.fill ?? COLORS.white },
  });
  slide.addText(title, {
    x: x + 0.12,
    y: y + 0.12,
    w: w - 0.24,
    h: 0.25,
    fontSize: options.titleSize ?? 14,
    bold: true,
    color: options.titleColor ?? COLORS.blue,
    margin: 0,
    align: options.align ?? "left",
  });
  slide.addText(body, {
    x: x + 0.12,
    y: y + 0.42,
    w: w - 0.24,
    h: h - 0.48,
    fontSize: options.fontSize ?? 11,
    color: options.bodyColor ?? COLORS.text,
    valign: options.valign ?? "mid",
    margin: 0,
    align: options.align ?? "left",
  });
}

function addNumberCard(slide, x, y, w, h, big, label, sublabel) {
  slide.addShape(pptx.ShapeType.roundRect, {
    x,
    y,
    w,
    h,
    rectRadius: 0.06,
    line: { color: COLORS.line, pt: 1 },
    fill: { color: COLORS.white },
  });
  slide.addText(big, {
    x: x + 0.12,
    y: y + 0.12,
    w: w - 0.24,
    h: 0.38,
    fontSize: 22,
    bold: true,
    color: COLORS.blue,
    align: "center",
    margin: 0,
  });
  slide.addText(label, {
    x: x + 0.12,
    y: y + 0.52,
    w: w - 0.24,
    h: 0.22,
    fontSize: 12.5,
    bold: true,
    color: COLORS.text,
    align: "center",
    margin: 0,
  });
  slide.addText(sublabel, {
    x: x + 0.12,
    y: y + 0.78,
    w: w - 0.24,
    h: 0.32,
    fontSize: 9.8,
    color: COLORS.muted,
    align: "center",
    valign: "mid",
    margin: 0,
  });
}

function addArrow(slide, x, y, w, h) {
  slide.addShape(pptx.ShapeType.chevron, {
    x,
    y,
    w,
    h,
    line: { color: COLORS.lightBlue, pt: 1 },
    fill: { color: COLORS.paleBlue },
  });
}

// Slide 1
{
  const slide = pptx.addSlide();
  addBackground(slide);
  slide.addText("Let's Verify Step by Step", {
    x: 0.62,
    y: 1.55,
    w: 7.6,
    h: 0.52,
    fontSize: 28,
    bold: true,
    color: COLORS.blue,
    margin: 0,
  });
  slide.addText("Process supervision for mathematical reasoning", {
    x: 0.64,
    y: 2.2,
    w: 6.7,
    h: 0.32,
    fontSize: 20,
    color: COLORS.muted,
    margin: 0,
  });
  slide.addText("Lightman et al. · OpenAI · 2023 / ICLR 2024", {
    x: 0.66,
    y: 2.8,
    w: 6.4,
    h: 0.24,
    fontSize: 14,
    color: COLORS.soft,
    margin: 0,
  });
  slide.addText("Process Reward in Large Language Models", {
    x: 0.68,
    y: 3.35,
    w: 5.6,
    h: 0.22,
    fontSize: 14,
    bold: true,
    color: COLORS.blue,
    margin: 0,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 0.66,
    y: 4.0,
    w: 5.3,
    h: 0,
    line: { color: COLORS.lightBlue, pt: 1.5 },
  });
  slide.addText(
    "From rewarding correct answers\n to rewarding correct reasoning steps.",
    {
      x: 0.7,
      y: 4.18,
      w: 5.8,
      h: 0.85,
      fontSize: 18,
      color: COLORS.text,
      margin: 0,
      breakLine: false,
    }
  );
}

// Slide 2
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Why This Paper Matters");
  addSectionLabel(slide, "Motivation");
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.72,
    y: 1.55,
    w: 11.9,
    h: 0.95,
    rectRadius: 0.05,
    line: { color: COLORS.blue, pt: 1.3 },
    fill: { color: COLORS.white },
  });
  slide.addText("Step-by-step reasoning can look persuasive while still being wrong.", {
    x: 1.05,
    y: 1.88,
    w: 11.2,
    h: 0.28,
    fontSize: 23,
    bold: true,
    color: COLORS.blue,
    align: "center",
    margin: 0,
  });
  addMiniCard(slide, 0.8, 3.0, 3.65, 2.05, "Failure mode 1", "A wrong intermediate step breaks the whole chain, but outcome-only feedback cannot localize the error.");
  addMiniCard(slide, 4.83, 3.0, 3.65, 2.05, "Failure mode 2", "A flawed derivation can still land on the correct final answer by accident.");
  addMiniCard(slide, 8.86, 3.0, 3.65, 2.05, "Core implication", "If we care about reliable reasoning, the reward signal should say something about the process.");
  addCallout(slide, "The paper asks whether supervising reasoning steps leads to better reward models than supervising final answers alone.");
}

// Slide 3
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Outcome Supervision vs Process Supervision");
  addSectionLabel(slide, "Setup");
  addMiniCard(slide, 0.82, 1.65, 5.7, 3.85, "Outcome supervision", "Reward model sees the full solution and learns only from final answer correctness.\n\nStrength: cheap and easy to scale.\n\nWeakness: coarse signal, poor credit assignment, may reward spurious reasoning.", {
    titleSize: 19,
    fontSize: 15,
  });
  addMiniCard(slide, 6.8, 1.65, 5.7, 3.85, "Process supervision", "Reward model receives labels on intermediate reasoning steps.\n\nStrength: denser feedback and direct localization of mistakes.\n\nWeakness: human annotation is much more expensive.", {
    titleSize: 19,
    fontSize: 15,
    line: COLORS.blue,
  });
  slide.addShape(pptx.ShapeType.line, {
    x: 6.58,
    y: 1.95,
    w: 0,
    h: 3.2,
    line: { color: COLORS.line, pt: 1.2, dash: "dash" },
  });
  addCallout(slide, "The comparison is about supervision granularity: what kind of signal teaches a verifier what good reasoning looks like?", { y: 5.8, h: 0.6, fontSize: 15 });
}

// Slide 4
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Method Overview");
  addSectionLabel(slide, "Pipeline");
  addMiniCard(slide, 0.8, 2.15, 2.1, 1.55, "1. Generator", "Produces many candidate solutions for each problem.", {
    titleSize: 16,
    fontSize: 12.5,
    align: "center",
  });
  addArrow(slide, 2.98, 2.55, 0.55, 0.55);
  addMiniCard(slide, 3.55, 2.15, 2.25, 1.55, "2. Labels", "Final-answer labels for ORM; step labels for PRM.", {
    titleSize: 16,
    fontSize: 12.5,
    align: "center",
  });
  addArrow(slide, 5.9, 2.55, 0.55, 0.55);
  addMiniCard(slide, 6.47, 2.15, 2.25, 1.55, "3. Reward model", "Train ORM and PRM separately.", {
    titleSize: 16,
    fontSize: 12.5,
    align: "center",
  });
  addArrow(slide, 8.82, 2.55, 0.55, 0.55);
  addMiniCard(slide, 9.4, 2.15, 2.95, 1.55, "4. Best-of-N search", "Reward model ranks candidates and selects the best one.", {
    titleSize: 16,
    fontSize: 12.5,
    align: "center",
  });
  addBulletBlock(slide, 0.95, 4.45, 3.45, "What stays fixed", [
    "same generator family",
    "same evaluation problems",
    "same search-style selection setting",
  ], { h: 1.25, fontSize: 13.5 });
  addBulletBlock(slide, 4.65, 4.45, 3.1, "What changes", [
    "how labels are defined",
    "what the reward model learns",
    "how solution quality is recognized",
  ], { h: 1.25, fontSize: 13.5 });
  addBulletBlock(slide, 8.05, 4.45, 4.0, "Main evaluation question", [
    "Which supervision method is better at picking truly correct solutions from many candidates?",
  ], { h: 1.25, fontSize: 13.5 });
}

// Slide 5
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "PRM800K and the Labeling Design");
  addSectionLabel(slide, "Data");
  addNumberCard(slide, 0.9, 1.8, 2.45, 1.55, "800K", "step-level labels", "human feedback on intermediate reasoning");
  addNumberCard(slide, 3.6, 1.8, 2.45, 1.55, "75K", "solutions", "annotated generator outputs");
  addNumberCard(slide, 6.3, 1.8, 2.45, 1.55, "12K+", "problems", "coverage across MATH-style questions");
  addMiniCard(slide, 9.0, 1.8, 3.2, 1.55, "Conservative rule", "For wrong solutions, annotation stops at the first incorrect step.", {
    titleSize: 16,
    fontSize: 12.5,
  });
  addMiniCard(slide, 0.95, 4.05, 3.55, 1.75, "Why stop at the first error?", "It makes the ORM vs PRM comparison fairer: both know the solution is wrong overall, but PRM knows where it first goes wrong.", {
    fontSize: 12.8,
  });
  addMiniCard(slide, 4.85, 4.05, 3.55, 1.75, "How PRM scores a solution", "The model predicts step correctness and combines those probabilities into a score for the whole reasoning chain.", {
    fontSize: 12.8,
  });
  addMiniCard(slide, 8.75, 4.05, 3.55, 1.75, "Why this matters", "Even with this conservative design, process supervision still wins clearly in the experiments.", {
    fontSize: 12.8,
  });
}

// Slide 6
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Main Result: PRM Beats ORM");
  addSectionLabel(slide, "Results");
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.8,
    y: 1.65,
    w: 5.35,
    h: 4.95,
    rectRadius: 0.05,
    line: { color: COLORS.line, pt: 1 },
    fill: { color: COLORS.white },
  });
  slide.addText("Representative MATH subset\nBest-of-1860", {
    x: 1.08,
    y: 1.95,
    w: 4.8,
    h: 0.55,
    fontSize: 20,
    bold: true,
    color: COLORS.blue,
    align: "center",
    margin: 0,
  });
  const bars = [
    { label: "Majority voting", val: 69.6, color: COLORS.lightBlue, y: 3.05 },
    { label: "ORM", val: 72.4, color: COLORS.soft, y: 4.0 },
    { label: "PRM", val: 78.2, color: COLORS.blue, y: 4.95 },
  ];
  bars.forEach((bar) => {
    slide.addText(bar.label, {
      x: 1.05,
      y: bar.y + 0.1,
      w: 1.65,
      h: 0.18,
      fontSize: 12,
      color: COLORS.text,
      margin: 0,
    });
    slide.addShape(pptx.ShapeType.rect, {
      x: 2.45,
      y: bar.y,
      w: (bar.val - 60) * 0.18,
      h: 0.45,
      line: { color: bar.color, pt: 1 },
      fill: { color: bar.color },
    });
    slide.addText(`${bar.val}%`, {
      x: 5.0,
      y: bar.y + 0.09,
      w: 0.65,
      h: 0.16,
      fontSize: 13,
      bold: true,
      color: COLORS.blue,
      margin: 0,
      align: "right",
    });
  });
  addMiniCard(slide, 6.55, 1.7, 5.85, 1.4, "Interpretation", "PRM is substantially better at identifying the truly correct solution among many generated candidates.", {
    titleSize: 18,
    fontSize: 13.5,
  });
  addMiniCard(slide, 6.55, 3.3, 5.85, 1.4, "Scaling pattern", "As N grows, the gap between PRM and ORM widens. Process supervision becomes even more useful in large search spaces.", {
    titleSize: 18,
    fontSize: 13.5,
  });
  addMiniCard(slide, 6.55, 4.9, 5.85, 1.4, "Takeaway", "The gain is not just better final accuracy. It is better ranking of candidate chains of thought.", {
    titleSize: 18,
    fontSize: 13.5,
  });
}

// Slide 7
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Why Process Supervision Helps");
  addSectionLabel(slide, "Explanation");
  addMiniCard(slide, 0.85, 1.8, 3.75, 3.55, "1. Better credit assignment", "Outcome supervision only says whether the whole trajectory worked.\n\nProcess supervision can identify the first bad step and preserve information about earlier good steps.", {
    titleSize: 18,
    fontSize: 14,
  });
  addMiniCard(slide, 4.8, 1.8, 3.75, 3.55, "2. Denser supervision signal", "One full solution yields many training signals instead of a single binary label.\n\nThat gives the verifier a much more detailed view of reasoning quality.", {
    titleSize: 18,
    fontSize: 14,
  });
  addMiniCard(slide, 8.75, 1.8, 3.75, 3.55, "3. Less reward for spurious reasoning", "A lucky correct answer can still come from a flawed derivation.\n\nProcess supervision is more likely to reject that solution as unreliable.", {
    titleSize: 18,
    fontSize: 14,
  });
  addCallout(slide, "Supervision granularity changes what the reward model can actually learn about reasoning quality.", { y: 5.85, h: 0.65 });
}

// Slide 8
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Active Learning Makes Process Labels More Efficient");
  addSectionLabel(slide, "Practicality");
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 0.85,
    y: 1.65,
    w: 3.1,
    h: 4.8,
    rectRadius: 0.05,
    line: { color: COLORS.blue, pt: 1.2 },
    fill: { color: COLORS.paleBlue },
  });
  slide.addText("2.6x", {
    x: 1.3,
    y: 2.25,
    w: 2.2,
    h: 0.7,
    fontSize: 34,
    bold: true,
    color: COLORS.blue,
    align: "center",
    margin: 0,
  });
  slide.addText("label efficiency gain", {
    x: 1.2,
    y: 3.05,
    w: 2.4,
    h: 0.24,
    fontSize: 16,
    bold: true,
    color: COLORS.text,
    align: "center",
    margin: 0,
  });
  slide.addText("with active learning", {
    x: 1.2,
    y: 3.45,
    w: 2.4,
    h: 0.22,
    fontSize: 13,
    color: COLORS.muted,
    align: "center",
    margin: 0,
  });
  addMiniCard(slide, 4.35, 1.85, 3.65, 1.35, "Uniform annotation", "Spend effort on many easy, low-information solutions.", {
    titleSize: 17,
    fontSize: 12.8,
  });
  addMiniCard(slide, 4.35, 3.45, 3.65, 1.35, "Active learning", "Prioritize examples that are ambiguous, difficult, or most informative for the current reward model.", {
    titleSize: 17,
    fontSize: 12.8,
    line: COLORS.blue,
  });
  addMiniCard(slide, 8.35, 1.85, 3.8, 2.95, "Why this matters", "Process supervision is often criticized as too expensive.\n\nThis result does not remove the cost, but it shows that targeted annotation can make step-level supervision much more realistic in practice.", {
    titleSize: 18,
    fontSize: 13.5,
  });
  addCallout(slide, "The paper's practical message is not that process labels are cheap, but that they are worth spending strategically.", { y: 5.65, h: 0.7, fontSize: 15 });
}

// Slide 9
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Beyond One Benchmark: Generalization and Alignment");
  addSectionLabel(slide, "Meaning");
  addNumberCard(slide, 0.95, 1.95, 2.55, 1.45, "72.9%", "PRM on OOD STEM tasks", "AP Calculus, Physics, Chemistry, AMC");
  addNumberCard(slide, 3.8, 1.95, 2.55, 1.45, "63.8%", "ORM on OOD STEM tasks", "weaker transfer beyond the main subset");
  addMiniCard(slide, 6.7, 1.8, 5.6, 1.75, "Interpretation", "The advantage of process supervision is not limited to one exact MATH evaluation split. It also helps under moderate distribution shift.", {
    titleSize: 18,
    fontSize: 13.5,
  });
  addMiniCard(slide, 0.95, 4.05, 3.55, 1.85, "Auditability", "Humans can inspect which step the reward model found suspicious, making the system easier to analyze.", {
    titleSize: 18,
    fontSize: 13.5,
  });
  addMiniCard(slide, 4.9, 4.05, 3.55, 1.85, "Alignment intuition", "The supervision target becomes closer to the behavior we actually want: sound intermediate reasoning.", {
    titleSize: 18,
    fontSize: 13.5,
  });
  addMiniCard(slide, 8.85, 4.05, 3.55, 1.85, "Negative alignment tax", "Here, the more aligned strategy is not weaker. It also improves empirical performance.", {
    titleSize: 18,
    fontSize: 13.5,
  });
}

// Slide 10
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Limits and Open Questions");
  addSectionLabel(slide, "Critique");
  const items = [
    ["Mostly mathematics", "The strongest evidence is in a domain with structured solutions and often checkable final answers."],
    ["High annotation cost", "Step-level human labels are still expensive even after active learning."],
    ["Outcome metric remains final", "The paper proves process supervision helps optimize solved-rate, but not that it fully captures all notions of good reasoning."],
    ["Harder domains remain open", "Law, medicine, open-ended planning, and creative reasoning may require different kinds of process labels."],
  ];
  let y = 1.75;
  items.forEach(([title, body]) => {
    addMiniCard(slide, 0.9, y, 11.55, 1.0, title, body, {
      titleSize: 16,
      fontSize: 12.7,
    });
    y += 1.15;
  });
  addCallout(slide, "The paper is strongest as a proof of concept: process reward can be both more aligned and more effective on difficult reasoning tasks.", { y: 6.45, h: 0.35, fontSize: 14 });
}

// Slide 11
{
  const slide = pptx.addSlide();
  addBackground(slide);
  addTitle(slide, "Takeaway and Discussion");
  addSectionLabel(slide, "Closing");
  addMiniCard(slide, 0.95, 1.85, 3.65, 1.45, "Takeaway 1", "Only supervising final answers throws away crucial information about reasoning quality.", {
    titleSize: 17,
    fontSize: 13.2,
  });
  addMiniCard(slide, 4.83, 1.85, 3.65, 1.45, "Takeaway 2", "Process-supervised reward models select better mathematical solutions than outcome-supervised ones.", {
    titleSize: 17,
    fontSize: 13.2,
  });
  addMiniCard(slide, 8.7, 1.85, 3.65, 1.45, "Takeaway 3", "Process reward is not only interpretable. In this paper, it is also the stronger empirical strategy.", {
    titleSize: 17,
    fontSize: 13.2,
  });
  slide.addShape(pptx.ShapeType.roundRect, {
    x: 1.05,
    y: 3.75,
    w: 10.95,
    h: 1.35,
    rectRadius: 0.05,
    line: { color: COLORS.blue, pt: 1.4 },
    fill: { color: COLORS.white },
  });
  slide.addText(
    "This paper shifts reward learning from “Was the answer correct?”\n to “Was the reasoning correct step by step?”",
    {
      x: 1.45,
      y: 4.12,
      w: 10.15,
      h: 0.55,
      fontSize: 21,
      bold: true,
      color: COLORS.blue,
      align: "center",
      margin: 0,
    }
  );
  slide.addText("Discussion prompts:", {
    x: 1.05,
    y: 5.55,
    w: 2.1,
    h: 0.2,
    fontSize: 15,
    bold: true,
    color: COLORS.blue,
    margin: 0,
  });
  slide.addText(
    "1. How should process labels look outside mathematics?\n2. Is human step annotation a long-term method or mainly a bootstrap?\n3. If final evaluation is still outcome-based, how much better reasoning are we really measuring?",
    {
      x: 1.25,
      y: 5.83,
      w: 10.3,
      h: 0.62,
      fontSize: 13.2,
      color: COLORS.text,
      margin: 0,
    }
  );
  slide.addText("Thank you", {
    x: 10.65,
    y: 6.42,
    w: 1.2,
    h: 0.18,
    fontSize: 18,
    bold: true,
    color: COLORS.blue,
    margin: 0,
    align: "right",
  });
}

const out = path.join(DIR, "2026-04-27_lets_verify_step_by_step_seminar.pptx");

(async () => {
  await pptx.writeFile({ fileName: out });
  console.log(`Wrote ${out}`);
})().catch((err) => {
  console.error(err);
  process.exit(1);
});
