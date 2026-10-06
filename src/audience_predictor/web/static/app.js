const $ = (id) => document.getElementById(id);

function logGamma(z) {
  const g = 7;
  const p = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (z < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * z)) - logGamma(1 - z);
  }
  z -= 1;
  let x = p[0];
  for (let i = 1; i < g + 2; i += 1) x += p[i] / (z + i);
  const t = z + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(x);
}

function betaPdf(x, a, b) {
  if (x <= 0 || x >= 1 || a <= 0 || b <= 0) return 0;
  return Math.exp(
    logGamma(a + b) - logGamma(a) - logGamma(b) + (a - 1) * Math.log(x) + (b - 1) * Math.log(1 - x),
  );
}

function pct(n) {
  if (n == null || Number.isNaN(n)) return "—";
  return String(Math.round(n)).padStart(3, " ");
}

function setStatus(msg) {
  $("status").textContent = msg || "";
}

function renderDemos(items) {
  const root = $("demos");
  root.innerHTML = "";
  items.forEach((item) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "demo";
    btn.innerHTML = `<span class="demo-title">${item.title}</span><span class="demo-slug">${item.slugline}<br />${item.logline}</span>`;
    btn.addEventListener("click", () => run({ demo_id: item.id }));
    root.appendChild(btn);
  });
}

function drawRt(rt) {
  const svg = $("rt-plot");
  svg.innerHTML = "";
  if (!rt || !rt.alpha) {
    svg.innerHTML = `<text x="8" y="60" fill="#6e6658" font-size="12" font-family="Courier New, monospace">no RT head on this checkpoint</text>`;
    return;
  }
  const w = 320;
  const h = 120;
  const pad = 16;
  const aC = rt.alpha[0];
  const bC = rt.beta[0];
  const aA = rt.alpha[1];
  const bA = rt.beta[1];
  const xs = [];
  let peak = 0;
  for (let i = 1; i < 80; i += 1) {
    const x = i / 80;
    const yc = betaPdf(x, aC, bC);
    const ya = betaPdf(x, aA, bA);
    xs.push([x, yc, ya]);
    peak = Math.max(peak, yc, ya);
  }
  const path = (col) =>
    xs
      .map(([x, yc, ya], i) => {
        const yv = col === 0 ? yc : ya;
        const px = pad + (x * (w - 2 * pad));
        const py = h - pad - (yv / peak) * (h - 2 * pad);
        return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
      })
      .join(" ");
  const mark = (value, color) => {
    const x = pad + (value / 100) * (w - 2 * pad);
    return `<line x1="${x}" y1="${pad}" x2="${x}" y2="${h - pad}" stroke="${color}" stroke-dasharray="2 3" />`;
  };
  svg.innerHTML = `
    <path d="${path(0)}" fill="none" stroke="#8a5a12" stroke-width="1.6" />
    <path d="${path(1)}" fill="none" stroke="#1d4e4a" stroke-width="1.6" />
    ${mark(rt.critics, "#8a5a12")}
    ${mark(rt.audience, "#1d4e4a")}
    <text x="${pad}" y="${h - 2}" fill="#6e6658" font-size="10">0</text>
    <text x="${w - pad - 18}" y="${h - 2}" fill="#6e6658" font-size="10">100</text>
    <text x="${pad}" y="12" fill="#8a5a12" font-size="10">CRITICS</text>
    <text x="${pad + 64}" y="12" fill="#1d4e4a" font-size="10">AUDIENCE</text>
  `;
}

function fillBar(id, value) {
  $(id).style.width = `${Math.max(0, Math.min(1, Number(value) || 0)) * 100}%`;
}

function two(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return Number(n).toFixed(2);
}

const JEV_KEYS = [
  { key: "turns", color: "#161410", label: "TURNS" },
  { key: "want", color: "#1d4e4a", label: "WANT" },
  { key: "specific", color: "#8a5a12", label: "SPECIFIC" },
  { key: "on_the_nose", color: "#8f1d18", label: "ON THE NOSE" },
  { key: "dead_air", color: "#6e6658", label: "DEAD AIR" },
];

function drawJev(cov) {
  const svg = $("jev-plot");
  const readout = $("jev-readout");
  const scenes = (cov && cov.scenes) || [];
  const n = Math.max((cov && cov.stats && cov.stats.n_scenes) || 0, scenes.length, 1);
  if (!svg || !readout) return;
  if (!scenes.length) {
    svg.innerHTML = `<text x="12" y="88" fill="#6e6658" font-size="12" font-family="Courier New, monospace">no scenes judged</text>`;
    readout.textContent = "softmax of the five noul heads, left to right through the pages";
    return;
  }

  const w = 640;
  const h = 168;
  const pad = { l: 28, r: 12, t: 12, b: 22 };
  const innerW = w - pad.l - pad.r;
  const innerH = h - pad.t - pad.b;
  const denom = Math.max(n - 1, 1);
  const xAt = (index) => pad.l + (index / denom) * innerW;
  const yAt = (v) => pad.t + (1 - Math.max(0, Math.min(1, v))) * innerH;
  const pts = scenes.map((row) => ({
    i: row.index,
    x: xAt(row.index),
    heading: row.heading,
    scores: row.scores || {},
    softmax: row.softmax || {},
    craft: row.craft,
  }));
  const opacities = [0.16, 0.26, 0.3, 0.2, 0.14];

  const bandPaths = [];
  let acc = pts.map(() => 0);
  JEV_KEYS.forEach((series, si) => {
    const lo = acc.slice();
    acc = pts.map((p, j) => lo[j] + Number(p.softmax[series.key] || 0));
    const hi = acc;
    const top = pts.map((p, j) => `${p.x.toFixed(1)},${yAt(hi[j]).toFixed(1)}`);
    const bot = [...pts].reverse().map((p, j) => {
      const idx = pts.length - 1 - j;
      return `${p.x.toFixed(1)},${yAt(lo[idx]).toFixed(1)}`;
    });
    let d;
    if (pts.length === 1) {
      const x0 = pts[0].x;
      const x1 = Math.min(w - pad.r, x0 + 14);
      d = `M${x0},${yAt(hi[0]).toFixed(1)} L${x1},${yAt(hi[0]).toFixed(1)} L${x1},${yAt(lo[0]).toFixed(1)} L${x0},${yAt(lo[0]).toFixed(1)} Z`;
    } else {
      d = `M${top.join(" L")} L${bot.join(" L")} Z`;
    }
    bandPaths.push(
      `<path d="${d}" fill="${series.color}" fill-opacity="${opacities[si]}" stroke="none" />`,
    );
  });

  const line = (values, color, dash) => {
    const d = pts
      .map((p, j) => `${j === 0 ? "M" : "L"}${p.x.toFixed(1)},${yAt(values[j]).toFixed(1)}`)
      .join(" ");
    const extra = dash ? ` stroke-dasharray="4 3" stroke-width="1.7"` : ` stroke-width="1.25"`;
    return `<path d="${d}" fill="none" stroke="${color}"${extra} />`;
  };
  const scoreLines = JEV_KEYS.map((series) =>
    line(
      pts.map((p) => Number(p.scores[series.key] || 0)),
      series.color,
      false,
    ),
  ).join("");
  const craftLine = line(
    pts.map((p) => Number(p.craft || 0)),
    "#161410",
    true,
  );
  const dots = pts
    .map(
      (p) =>
        `<circle cx="${p.x.toFixed(1)}" cy="${yAt(Number(p.craft || 0)).toFixed(1)}" r="2.3" fill="#161410" />`,
    )
    .join("");
  const face = `fill="#6e6658" font-size="10" font-family="Courier New, monospace"`;
  svg.innerHTML =
    bandPaths.join("") +
    scoreLines +
    craftLine +
    dots +
    `<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${h - pad.b}" stroke="#c9bba3" />` +
    `<line x1="${pad.l}" y1="${h - pad.b}" x2="${w - pad.r}" y2="${h - pad.b}" stroke="#c9bba3" />` +
    `<text x="${pad.l}" y="${pad.t + 8}" ${face}>1</text>` +
    `<text x="${pad.l - 6}" y="${h - pad.b + 3}" ${face} text-anchor="end">0</text>` +
    `<text x="${pad.l}" y="${h - 6}" ${face}>FADE IN</text>` +
    `<text x="${w - pad.r}" y="${h - 6}" ${face} text-anchor="end">FADE OUT</text>` +
    `<line id="jev-hover" x1="${pts[0].x}" y1="${pad.t}" x2="${pts[0].x}" y2="${h - pad.b}" stroke="#161410" stroke-dasharray="2 3" opacity="0.35" />` +
    `<rect x="${pad.l}" y="${pad.t}" width="${innerW}" height="${innerH}" fill="transparent" />`;

  const setReadout = (p) => {
    const hoverLine = document.getElementById("jev-hover");
    if (hoverLine) {
      hoverLine.setAttribute("x1", p.x.toFixed(1));
      hoverLine.setAttribute("x2", p.x.toFixed(1));
    }
    const s = p.softmax;
    readout.textContent =
      `sc ${String(p.i + 1).padStart(2, "0")}  ${p.heading}  ·  craft ${two(p.craft)}  ·  ` +
      `T ${two(s.turns)}  W ${two(s.want)}  S ${two(s.specific)}  N ${two(s.on_the_nose)}  D ${two(s.dead_air)}`;
  };
  setReadout(pts[Math.floor(pts.length / 2)] || pts[0]);

  svg.onmousemove = (ev) => {
    const box = svg.getBoundingClientRect();
    const x = ((ev.clientX - box.left) / Math.max(box.width, 1)) * w;
    let best = pts[0];
    let dist = Infinity;
    pts.forEach((p) => {
      const d = Math.abs(p.x - x);
      if (d < dist) {
        dist = d;
        best = p;
      }
    });
    setReadout(best);
  };
  svg.onmouseleave = () => setReadout(pts[Math.floor(pts.length / 2)] || pts[0]);
}

function showCoverage(cov) {
  if (!cov) return;
  const stamp = cov.stamp || "consider";
  $("verdict").textContent = stamp.toUpperCase();
  $("verdict").className = `verdict ${stamp}`;
  const judge = cov.judge || {};
  $("verdict-sub").textContent =
    `craft ${two(cov.craft)}  ·  ${judge.kind || "heuristic"}  ·  ${cov.n_judged || 0} scenes`;
  $("cov-genre").textContent = (cov.genre || "—").toUpperCase();
  const stats = cov.stats || {};
  $("cov-scenes").textContent = String(stats.n_scenes ?? "—");
  $("cov-chars").textContent = String(stats.n_characters ?? "—");
  $("cov-generic").textContent = two(cov.genericness);
  fillBar("bar-generic", cov.genericness);
  $("cov-people").textContent = (stats.characters || []).join("  ·  ");
  const means = cov.scene_means || {};
  $("m-turns").textContent = two(means.turns);
  $("m-want").textContent = two(means.want);
  $("m-specific").textContent = two(means.specific);
  $("m-nose").textContent = two(means.on_the_nose);
  $("m-dead").textContent = two(means.dead_air);
  fillBar("bar-turns", means.turns);
  fillBar("bar-want", means.want);
  fillBar("bar-specific", means.specific);
  fillBar("bar-nose", means.on_the_nose);
  fillBar("bar-dead", means.dead_air);
  drawJev(cov);
  const list = $("scene-rows");
  list.innerHTML = "";
  (cov.scenes || []).forEach((row) => {
    const s = row.scores || {};
    const li = document.createElement("li");
    li.innerHTML =
      `<span>${String(row.index + 1).padStart(2, "0")}</span>` +
      `<span class="heading">${row.heading}</span>` +
      `<span class="ticks">T ${two(s.turns)}  W ${two(s.want)}  S ${two(s.specific)}</span>`;
    list.appendChild(li);
  });
}

function show(result) {
  $("cut").hidden = false;
  $("desk").hidden = false;
  const demo = result.demo || {};
  const title = demo.title || "UNTITLED";
  const sigma = result.sigma == null ? "—" : Number(result.sigma).toFixed(2);
  const head = result.path || "coverage";
  $("meta").textContent =
    `${title}  ·  ${result.document_type}  ·  ${head}  ·  ${result.n_blocks} blocks  ·  σ ${sigma}`;
  showCoverage(result.coverage);

  const sidecar = $("sidecar");
  if (!result.audience_available) {
    $("audience-note").textContent =
      result.audience_error || "Audience map is not trained on this machine.";
    $("mean-rating").textContent = "—";
    $("p-love").textContent = "—";
    $("p-hate").textContent = "—";
    fillBar("bar-love", 0);
    fillBar("bar-hate", 0);
    $("rating-scale").innerHTML = "";
    $("rt-critics").textContent = "—";
    $("rt-audience").textContent = "—";
    $("rt-gap").textContent = "—";
    $("rt-note").textContent = "";
    $("comps").innerHTML = "";
    drawRt(null);
    sidecar.classList.add("dim");
    $("desk").scrollIntoView({ behavior: "smooth", block: "start" });
    $("excerpt").textContent = result.excerpt || "";
    return;
  }
  sidecar.classList.remove("dim");
  $("audience-note").textContent = "Neighborhood in MovieLens z. Not a quality grade.";
  const pol = result.polarization || {};
  $("mean-rating").textContent = two(pol.mean_rating);
  $("p-love").textContent = two(pol.p_love);
  $("p-hate").textContent = two(pol.p_hate);
  fillBar("bar-love", pol.p_love);
  fillBar("bar-hate", pol.p_hate);
  const ratingPct = Math.max(0, Math.min(1, ((pol.mean_rating || 0.5) - 0.5) / 4.5)) * 100;
  $("rating-scale").innerHTML = `<i class="tick" style="left:${ratingPct}%"></i>`;

  const rt = result.rt;
  if (rt && rt.critics != null) {
    const gap = rt.audience - rt.critics;
    const lean =
      gap >= 8 ? "crowd-pleaser" : gap <= -8 ? "critics' movie" : "the two crowds agree";
    $("rt-critics").textContent = `${pct(rt.critics)}   90% [${pct(rt.critics_lo)}, ${pct(rt.critics_hi)}]`;
    $("rt-audience").textContent = `${pct(rt.audience)}   90% [${pct(rt.audience_lo)}, ${pct(rt.audience_hi)}]`;
    $("rt-gap").textContent = `${gap >= 0 ? "+" : ""}${Math.round(gap)}   ${lean}`;
    $("rt-note").textContent = rt.source || "";
    drawRt(rt);
  } else {
    $("rt-critics").textContent = "—";
    $("rt-audience").textContent = "—";
    $("rt-gap").textContent = "—";
    $("rt-note").textContent = "Train rt_scores to print a posterior.";
    drawRt(null);
  }

  const comps = $("comps");
  comps.innerHTML = "";
  (result.comps || []).forEach((row, i) => {
    const li = document.createElement("li");
    li.innerHTML = `<span>${String(i + 1).padStart(2, "0")}</span><span>${row.title}</span><span class="cos">${Number(row.cosine).toFixed(3)}</span>`;
    comps.appendChild(li);
  });
  $("excerpt").textContent = result.excerpt || "";
  $("desk").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function readBody(res) {
  const raw = await res.text();
  try {
    return JSON.parse(raw);
  } catch {
    throw new Error("the desk is not answering. Run: uv run audience-predictor serve");
  }
}

function failMessage(err) {
  const msg = String(err && err.message ? err.message : err);
  if (msg === "Failed to fetch" || msg === "Load failed" || msg === "NetworkError when attempting to fetch resource.") {
    return "could not reach the desk. Keep `audience-predictor serve` running and open http://127.0.0.1:8000 — not the HTML file.";
  }
  return msg;
}

async function run(payload) {
  setStatus("reading the pages…");
  $("drop").classList.add("busy");
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await readBody(res);
    if (!res.ok) throw new Error(data.detail || res.statusText);
    show(data);
    setStatus("end of coverage.");
  } catch (err) {
    setStatus(failMessage(err));
  } finally {
    $("drop").classList.remove("busy");
  }
}

async function runFile(file) {
  setStatus(`reading ${file.name}…`);
  $("drop").classList.add("busy");
  const body = new FormData();
  body.append("file", file);
  try {
    const res = await fetch("/api/predict-file", { method: "POST", body });
    const data = await readBody(res);
    if (!res.ok) throw new Error(data.detail || res.statusText);
    show(data);
    setStatus("end of coverage.");
  } catch (err) {
    setStatus(failMessage(err));
  } finally {
    $("drop").classList.remove("busy");
  }
}

$("drop").addEventListener("click", () => $("file").click());
$("drop").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") $("file").click();
});
$("file").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (file) runFile(file);
});
["dragenter", "dragover"].forEach((ev) => {
  $("drop").addEventListener(ev, (e) => {
    e.preventDefault();
    $("drop").style.background = "rgba(22,20,16,0.06)";
  });
});
["dragleave", "drop"].forEach((ev) => {
  $("drop").addEventListener(ev, (e) => {
    e.preventDefault();
    $("drop").style.background = "";
  });
});
$("drop").addEventListener("drop", (e) => {
  const file = e.dataTransfer.files && e.dataTransfer.files[0];
  if (file) runFile(file);
});

fetch("/api/demos")
  .then((r) => r.json())
  .then((items) => {
    renderDemos(items);
    const wanted = new URLSearchParams(window.location.search).get("demo");
    if (wanted && items.some((item) => item.id === wanted)) run({ demo_id: wanted });
  })
  .catch((err) => setStatus(failMessage(err)));
