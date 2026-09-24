"""Replayable demo: a fleet of 40 models monitored step by step, driftfdr against river's defaults.

The fleet's errors share a common factor (rho = 0.6). Five models drift on their
own; at step 2500 a short common spike hits every model and passes (not a
drift); from step 4200 every model is worse (a fleet-wide event). Both monitors
see the same values one step at a time:

* river's Page-Hinkley at its default threshold, one per model;
* driftfdr's StreamingMonitor with MeanShift(3), BH within each window and
  ``split_common=True``.

Writes ``demo.html`` (self-contained, open it in a browser) with a heat map of the
errors, the alarms of both monitors and running counts of right and wasted retrains.

Usage: python examples/live_demo.py [output.html]
"""

import json
import sys

import numpy as np
from river import drift

from driftfdr import CalibrationConfig, MeanShift, StreamingMonitor

K, T, RHO, PHI = 40, 6000, 0.6, 0.5
SPIKE, SPIKE_LEN, FLEET_EVENT = 2500, 150, 4200
rng = np.random.default_rng(7)

# errors of the fleet: AR(1) with a common factor, then the events
common = np.zeros(T)
own = np.zeros((K, T))
e_c, e_o = rng.normal(size=T), rng.normal(size=(K, T))
for t in range(1, T):
    common[t] = PHI * common[t - 1] + np.sqrt(1 - PHI**2) * e_c[t]
    own[:, t] = PHI * own[:, t - 1] + np.sqrt(1 - PHI**2) * e_o[:, t]
values = np.sqrt(RHO) * common + np.sqrt(1 - RHO) * own
drifting = rng.choice(K, size=5, replace=False)
onsets = {int(k): int(rng.integers(900, 3800)) for k in drifting}
for k, t0 in onsets.items():
    values[k, t0:] += 1.0
values[:, SPIKE : SPIKE + SPIKE_LEN] += 1.5  # transient, passes by itself
values[:, FLEET_EVENT:] += 1.0


def changes(k):
    return sorted([onsets[k]] if k in onsets else []) + [FLEET_EVENT]


def classify(k, t, caught):
    """An alarm is right if it is the model's first alarm after its latest change."""
    past = [c for c in changes(k) if c < t]
    if past and (k, past[-1]) not in caught:
        caught.add((k, past[-1]))
        return True
    return False


river_ph = [drift.PageHinkley(mode="up") for _ in range(K)]
monitor = StreamingMonitor(n_models=K, detector_factory=lambda: MeanShift(3), procedure="bh_window", alpha=0.05,
                           n_ref=300, window=100, horizon=5, calibration=CalibrationConfig(n_boot=500),
                           split_common=True)
events, caught_river, caught_fdr = [], set(), set()
for t in range(T):
    for k in range(K):
        river_ph[k].update(values[k, t])
        if river_ph[k].drift_detected:
            events.append({"t": t, "k": k, "who": "river", "ok": classify(k, t, caught_river)})
    for k in monitor.update(values[:, t]):
        events.append({"t": t, "k": int(k), "who": "driftfdr", "ok": classify(int(k), t, caught_fdr)})
    if monitor.fleet_alarm:
        events.append({"t": t, "k": -1, "who": "fleet", "ok": t >= FLEET_EVENT})

BUCKET = 20
heat = values[:, : T // BUCKET * BUCKET].reshape(K, -1, BUCKET).mean(axis=2)
data = {"K": K, "T": T, "bucket": BUCKET, "heat": np.round(heat, 2).tolist(), "events": events,
        "spike": SPIKE, "spike_len": SPIKE_LEN, "fleet": FLEET_EVENT, "onsets": onsets}

for who in ("river", "driftfdr"):
    ev = [e for e in events if e["who"] == who]
    print(f"{who}: {len(ev)} retrains, {sum(not e['ok'] for e in ev)} wasted")
print("fleet alarms at", [e["t"] for e in events if e["who"] == "fleet"])

HTML = """<!doctype html>
<html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>driftfdr live demo</title>
<style>
:root { --bg:#fbfaf7; --fg:#1f1e1c; --muted:#6b6862; --grid:#e4e1da; --ok:#2e7d4f; --bad:#c2410c; --river:#7c7a75; --fleet:#1d4ed8; }
@media (prefers-color-scheme: dark) { :root { --bg:#191817; --fg:#ecebe8; --muted:#a3a09a; --grid:#34322f; --ok:#5fbf86; --bad:#fb923c; --river:#9c9a95; --fleet:#7aa2ff; } }
html, body { overflow-x:hidden; } body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.5 system-ui, sans-serif; }
main { max-width:1100px; margin:0 auto; padding:24px 16px; }
h1 { font-size:22px; margin:0 0 4px; } p { color:var(--muted); margin:4px 0 16px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(220px,100%),1fr)); gap:12px; margin:16px 0; }
.card { border:1px solid var(--grid); border-radius:10px; padding:12px 14px; }
.card b { font-size:26px; font-variant-numeric:tabular-nums; } .card span { color:var(--muted); font-size:13px; display:block; }
canvas { width:100%; border:1px solid var(--grid); border-radius:8px; display:block; }
.controls { display:flex; gap:12px; align-items:center; margin:12px 0; flex-wrap:wrap; }
button { font:inherit; padding:6px 14px; border-radius:8px; border:1px solid var(--grid); background:var(--bg); color:var(--fg); cursor:pointer; }
input[type=range] { flex:1; min-width:160px; }
.legend { display:flex; gap:16px; flex-wrap:wrap; color:var(--muted); font-size:13px; }
.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; vertical-align:middle; }
</style></head><body><main>
<h1>Парк из __K__ моделей: driftfdr против river по умолчанию</h1>
<p>Ошибки моделей связаны общим фактором. У пяти моделей свой дрейф; на шаге __SPIKE__ короткий общий всплеск, который проходит сам;
с шага __FLEET__ хуже становятся все модели. Оба монитора видят одни и те же значения, шаг за шагом.</p>
<div class="controls"><button id="play">▶ Пуск</button><input id="t" type="range" min="0" max="__T__" value="0"><span id="tl">шаг 0</span></div>
<div class="cards">
 <div class="card"><span>river Page-Hinkley: переобучений</span><b id="rn">0</b><span id="rw">из них впустую 0</span></div>
 <div class="card"><span>driftfdr: переобучений</span><b id="fn">0</b><span id="fw">из них впустую 0</span></div>
 <div class="card"><span>driftfdr: тревога по всему парку</span><b id="fl">—</b><span>«сломалось общее»: инцидент, а не 40 переобучений</span></div>
</div>
<canvas id="c" height="460"></canvas>
<div class="legend" style="margin-top:8px">
 <span><span class="dot" style="background:var(--river)"></span>river: тревога</span>
 <span><span class="dot" style="background:var(--ok)"></span>driftfdr: верное переобучение</span>
 <span><span class="dot" style="background:var(--bad)"></span>driftfdr: впустую</span>
 <span><span class="dot" style="background:var(--fleet)"></span>driftfdr: тревога парка</span>
</div>
</main><script>
const D = __DATA__;
const cv = document.getElementById('c'), ctx = cv.getContext('2d');
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
let t = 0, timer = null;
function draw() {
  const W = cv.clientWidth, H = cv.height; cv.width = W;
  const cols = D.heat[0].length, cw = W / cols, rh = (H - 30) / D.K, upto = Math.floor(t / D.bucket);
  ctx.fillStyle = css('--bg'); ctx.fillRect(0, 0, W, H);
  for (let k = 0; k < D.K; k++) for (let j = 0; j <= Math.min(upto, cols - 1); j++) {
    const v = Math.max(-2, Math.min(3, D.heat[k][j])), a = (v + 2) / 5;
    ctx.fillStyle = `rgba(194,65,12,${(a * a).toFixed(3)})`; ctx.fillRect(j * cw, k * rh, cw + 0.5, rh + 0.5);
  }
  ctx.fillStyle = css('--muted'); ctx.font = '12px system-ui';
  [[D.spike, 'всплеск'], [D.fleet, 'весь парк хуже']].forEach(([s, l]) => { const x = s / D.T * W;
    ctx.fillRect(x, 0, 1, H - 30); ctx.fillText(l, x + 4, H - 12); });
  let rn = 0, rw = 0, fn = 0, fw = 0, fl = null;
  for (const e of D.events) { if (e.t > t) break; const x = e.t / D.T * W;
    if (e.who === 'fleet') { if (fl === null) fl = e.t; ctx.fillStyle = css('--fleet'); ctx.globalAlpha = 0.18; ctx.fillRect(x, 0, W - x, H - 30);
      ctx.globalAlpha = 1; ctx.fillRect(x - 1.5, 0, 3, H - 30); ctx.fillText('тревога парка', x + 4, 14); continue; }
    const y = e.k * rh + rh / 2;
    if (e.who === 'river') { rn++; rw += !e.ok; ctx.fillStyle = css('--river'); ctx.fillRect(x - 1.5, y - 1.5, 3, 3); }
    else { fn++; fw += !e.ok; ctx.fillStyle = css(e.ok ? '--ok' : '--bad'); ctx.beginPath(); ctx.arc(x, y, 4.5, 0, 7); ctx.fill(); }
  }
  for (const [k, s] of Object.entries(D.onsets)) { ctx.strokeStyle = css('--fg'); ctx.strokeRect(s / D.T * W, k * rh, 2, rh); }
  document.getElementById('rn').textContent = rn; document.getElementById('rw').textContent = 'из них впустую ' + rw;
  document.getElementById('fn').textContent = fn; document.getElementById('fw').textContent = 'из них впустую ' + fw;
  document.getElementById('fl').textContent = fl === null ? '—' : 'шаг ' + fl;
  document.getElementById('tl').textContent = 'шаг ' + t; document.getElementById('t').value = t;
}
document.getElementById('t').oninput = e => { t = +e.target.value; draw(); };
document.getElementById('play').onclick = () => {
  if (timer) { clearInterval(timer); timer = null; return document.getElementById('play').textContent = '▶ Пуск'; }
  if (t >= D.T) t = 0; document.getElementById('play').textContent = '❚❚ Пауза';
  timer = setInterval(() => { t = Math.min(D.T, t + 20); draw(); if (t >= D.T) { clearInterval(timer); timer = null; document.getElementById('play').textContent = '▶ Пуск'; } }, 30);
};
window.onresize = draw; t = D.T; draw();
</script></body></html>
"""

out = sys.argv[1] if len(sys.argv) > 1 else "demo.html"
page = HTML.replace("__DATA__", json.dumps(data)).replace("__K__", str(K)).replace("__T__", str(T))
page = page.replace("__SPIKE__", str(SPIKE)).replace("__FLEET__", str(FLEET_EVENT))
with open(out, "w") as f:
    f.write(page)
print("wrote", out)
