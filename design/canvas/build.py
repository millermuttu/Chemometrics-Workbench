#!/usr/bin/env python3
"""Assemble the .dc.html artboards from a shared chrome + per-screen content."""
import pathlib

BASE = pathlib.Path(__file__).parent
CSS = (BASE / "_base.css").read_text()

ICON = {
 "data": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><ellipse cx="12" cy="6" rx="8" ry="3"></ellipse><path d="M4 6v12c0 1.7 3.6 3 8 3s8-1.3 8-3V6"></path><path d="M4 12c0 1.7 3.6 3 8 3s8-1.3 8-3"></path></svg>',
 "node": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="9" width="7" height="6" rx="1"></rect><rect x="14" y="9" width="7" height="6" rx="1"></rect><path d="M10 12h4"></path></svg>',
 "flask": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 3v6L4 19a1.6 1.6 0 0 0 1.4 2h13.2A1.6 1.6 0 0 0 20 19l-5-10V3"></path><path d="M8 3h8"></path><path d="M7 14h10"></path></svg>',
 "model": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3l8 4.5v9L12 21l-8-4.5v-9z"></path><path d="M12 12l8-4.5M12 12v9M12 12L4 7.5"></path></svg>',
 "chev": '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M6 9l6 6 6-6"></path></svg>',
 "chart": '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 20h18"></path><path d="M4 16l5-6 4 3 6-8"></path></svg>',
 "stop": '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="6" y="6" width="12" height="12" rx="1.5"></rect></svg>',
 "lock": '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="4" y="10" width="16" height="10" rx="2"></rect><path d="M8 10V7a4 4 0 0 1 8 0v3"></path></svg>',
}

def sidebar(sel):
    def row(key, label, dim="", indent=16, icon=None):
        c = "srow sel" if key == sel else "srow"
        ic = ICON[icon] if icon else ""
        d = f'<span class="sdim mono">{dim}</span>' if dim else ""
        return (f'<div class="{c}" style="padding-left:{indent}px">{ic}'
                f'<span>{label}</span>{d}</div>')
    return f'''<aside class="side">
<div class="shead"><span>Datasets</span><span style="color:var(--ink3)">+</span></div>
{row("corn_raw","corn_raw","v2 · 80×700",16,"data")}
{row("corn_v1","corn_v1","v1 · 80×700",16,"data")}
{row("valid","validation_set","24×700",16,"data")}
<div class="shead"><span>Pipeline</span><span class="mono" style="font-size:10px">11 nodes</span></div>
{row("n_src","corn_raw v2","",26,"node")}
{row("n_snv","SNV","",34,"node")}
{row("n_msc","MSC","",34,"node")}
{row("n_sg","SG d1 w11","",34,"node")}
{row("n_snvsg","SNV + SG d1","",34,"node")}
{row("n_base","Baseline ALS","",34,"node")}
{row("n_cv","K-fold 10 · seed 42","",34,"node")}
{row("n_pls","PLS 6 LV","",34,"node")}
<div class="shead"><span>Experiments</span><span class="mono" style="font-size:10px">12</span></div>
{row("exp","Runs","",26,"flask")}
<div class="shead"><span>Models</span><span class="mono" style="font-size:10px">4</span></div>
{row("mA","Model A","0.412",26,"model")}
{row("mB","Model B","0.418",26,"model")}
{row("mC","Model C","0.389",26,"model")}
{row("mD","Model D","0.381",26,"model")}
</aside>'''

def tabs(active):
    items = [("corn_raw v2","data"),("Pipeline","node"),("SG d1 w11","chart"),
             ("PCA scores","chart"),("Model C","model"),("Runs","flask")]
    out = []
    for name, ic in items:
        cls = "tab act" if name == active else ("tab tr" if name == "PCA scores" else "tab")
        x = '<span class="tabx">×</span>' if name == active else ""
        out.append(f'<div class="{cls}">{ICON[ic]}<span>{name}</span>{x}</div>')
    out.append('<div class="tab mono" style="color:var(--ink3);border-right:none">+3</div>')
    return '<div class="tabs">' + "".join(out) + '</div>'

def status(job=None):
    if job:
        left = (f'<div style="display:flex;align-items:center;gap:9px">'
                f'<span style="color:var(--accent);font-weight:500">{job[0]}</span>'
                f'<div class="prog"><i style="width:{job[1]}"></i></div>'
                f'<span class="mono">{job[2]}</span>'
                f'<span style="display:flex;align-items:center;gap:4px;color:var(--ink3)">{ICON["stop"]}Cancel</span></div>')
    else:
        left = '<span>Ready</span>'
    return (f'<div class="status">{left}'
            f'<div style="display:flex;align-items:center;gap:6px;color:var(--ink3)">{ICON["lock"]}'
            f'<span>Local · nothing leaves this machine</span></div></div>')

def titlebar(right):
    return f'''<div class="tbar">
<div class="tb-l">{ICON["flask"]}<span class="proj">Corn NIR study</span>
<span class="crumb mono">~/lab/corn-nir</span></div>
<div class="tb-r">{right}</div></div>'''

def page(name, title, body, script, theme="light", preview=(1440, 900)):
    props = ('{"theme":{"editor":"enum","options":["light","dark"],"default":"%s","section":"Theme"},'
             '"$preview":{"width":%d,"height":%d}}') % (theme, preview[0], preview[1])
    sc = script or "class Component extends DCLogic {\n  renderVals() { return { t: this.props.theme === 'dark' ? 't-dark' : 't-light' }; }\n}"
    html = f'''<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <script src="./support.js"></script>
</head>
<body>
<x-dc>
<helmet>
  <style>
{CSS}
  </style>
</helmet>
<div class="app {{{{t}}}}">
{body}
</div>
</x-dc>
<script data-dc-script data-props='{props}'>
{sc}
</script>
</body>
</html>
'''
    (BASE / name).write_text(html)
    print("wrote", name, len(html), "bytes")

import math, random

# ---------------------------------------------------------------- plot helpers
def poly(pts, prec=1):
    return "M" + "L".join(f"{x:.{prec}f},{y:.{prec}f}" for x, y in pts)

def axes(W, H, xt, yt, xlab, ylab, pad=(46, 12, 30, 12)):
    """pad = left, right, bottom, top. Returns (svg_prefix, mapx, mapy, plotbox)."""
    l, r, b, t = pad
    x0, x1 = l, W - r
    y0, y1 = H - b, t
    g = [f'<rect x="{x0}" y="{y1}" width="{x1-x0}" height="{y0-y1}" fill="var(--surface)"></rect>']
    for v, lab in yt:
        y = y1 + (y0 - y1) * (1 - v)
        g.append(f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" stroke="var(--grid)" stroke-width="1"></line>')
        g.append(f'<text x="{x0-7}" y="{y+3.5:.1f}" text-anchor="end" font-family="IBM Plex Mono" font-size="9" fill="var(--ink3)">{lab}</text>')
    for v, lab in xt:
        x = x0 + (x1 - x0) * v
        g.append(f'<line x1="{x:.1f}" y1="{y1}" x2="{x:.1f}" y2="{y0}" stroke="var(--grid)" stroke-width="1"></line>')
        g.append(f'<text x="{x:.1f}" y="{y0+13}" text-anchor="middle" font-family="IBM Plex Mono" font-size="9" fill="var(--ink3)">{lab}</text>')
    g.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="var(--rule)" stroke-width="1"></line>')
    g.append(f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="var(--rule)" stroke-width="1"></line>')
    if xlab:
        g.append(f'<text x="{(x0+x1)/2:.0f}" y="{H-3}" text-anchor="middle" font-family="IBM Plex Sans" font-size="9.5" fill="var(--ink3)">{xlab}</text>')
    if ylab:
        g.append(f'<text transform="translate(11,{(y0+y1)/2:.0f}) rotate(-90)" text-anchor="middle" font-family="IBM Plex Sans" font-size="9.5" fill="var(--ink3)">{ylab}</text>')
    mx = lambda u: x0 + (x1 - x0) * u
    my = lambda u: y1 + (y0 - y1) * (1 - u)
    return "".join(g), mx, my

BANDS = [(0.10,.030,.30),(0.21,.022,.55),(0.32,.045,.40),(0.45,.018,.88),
         (0.51,.030,.52),(0.63,.055,.46),(0.74,.016,.70),(0.81,.026,.36),
         (0.91,.040,.60),(0.97,.020,.28)]

def raw_curve(rnd, n=170):
    off = rnd.uniform(-.10, .16); scale = rnd.uniform(.86, 1.16); tilt = rnd.uniform(.10, .26)
    jit = [(c + rnd.uniform(-.006, .006), w * rnd.uniform(.92, 1.08), a * rnd.uniform(.86, 1.14))
           for c, w, a in BANDS]
    out = []
    for i in range(n + 1):
        u = i / n
        y = .16 + tilt * u + off
        for c, w, a in jit:
            d = (u - c) / w
            y += a * math.exp(-.5 * d * d)
        out.append((u, y * scale))
    return out

def proc_curve(rnd, n=170):
    """SNV + first derivative: oscillates about zero."""
    jit = [(c + rnd.uniform(-.008, .008), w * rnd.uniform(.9, 1.1), a * rnd.uniform(.8, 1.2))
           for c, w, a in BANDS]
    out = []
    for i in range(n + 1):
        u = i / n
        y = 0.0
        for c, w, a in jit:
            d = (u - c) / w
            y += -a * d * math.exp(-.5 * d * d) * .42
        out.append((u, y))
    return out

# ================================================================ 1. Shell
rnd = random.Random(7)
SAMPLES = []
for i in range(1, 19):
    SAMPLES.append(dict(
        n=i, id=f"C{i:03d}",
        set="cal" if i % 5 else "val",
        moist=round(rnd.uniform(9.2, 11.4), 2),
        prot=round(rnd.uniform(7.1, 9.6), 2),
        oil=round(rnd.uniform(3.1, 3.9), 2),
        starch=round(rnd.uniform(62.0, 65.2), 2),
        excl=i in (6, 13)))

rows = []
for s in SAMPLES:
    style = ' style="opacity:.45"' if s["excl"] else ""
    flag = ('<span class="mono" style="color:var(--stale);font-size:10px">excluded</span>'
            if s["excl"] else "")
    setchip = (f'<span class="mono" style="font-size:10px;padding:1px 5px;border-radius:2px;'
               f'background:{"var(--accentSoft)" if s["set"]=="cal" else "var(--sunken)"};'
               f'color:{"var(--accentInk)" if s["set"]=="cal" else "var(--ink3)"}">{s["set"]}</span>')
    rows.append(f'<tr{style}><td class="n" style="color:var(--ink3)">{s["n"]}</td>'
                f'<td class="mono" style="color:var(--ink)">{s["id"]}</td><td>{setchip}</td>'
                f'<td class="n">{s["moist"]:.2f}</td><td class="n">{s["prot"]:.2f}</td>'
                f'<td class="n">{s["oil"]:.2f}</td><td class="n">{s["starch"]:.2f}</td>'
                f'<td style="text-align:right">{flag}</td></tr>')

shell_body = f'''{titlebar('<span class="pill mono">80 × 700</span><span class="btn">Import…</span><span class="btn btn-p">Run pipeline</span>')}
<div class="body">
{sidebar("corn_raw")}
<main class="doc">
{tabs("corn_raw v2")}
<div class="pane">
  <div style="height:40px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 14px;gap:12px;border-bottom:1px solid var(--rule2)">
    <div style="display:flex;align-items:center;gap:9px">
      <span style="font-weight:600;font-size:13.5px">corn_raw</span>
      <span class="pill mono">v2 {ICON["chev"]}</span>
      <span class="mono" style="font-size:10.5px;color:var(--ink3)">sha256:9f3c…a71b</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px">
      <span class="pill mono">3 targets</span>
      <span class="pill mono">1100–2498 nm</span>
      <span class="pill mono" style="color:var(--stale);border-color:var(--stale)">2 excluded</span>
    </div>
  </div>
  <div style="flex:1;min-height:0;overflow:hidden">
    <table>
      <thead><tr><th class="n" style="width:38px">#</th><th style="width:74px">Sample</th>
      <th style="width:60px">Set</th><th class="n">Moisture %</th><th class="n">Protein %</th>
      <th class="n">Oil %</th><th class="n">Starch %</th><th style="text-align:right">Status</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
  </div>
</div>
</main>
<aside class="insp">
  <div class="ihead"><span class="ilabel">Dataset version</span>
    <span style="font-weight:600;font-size:13px">corn_raw · v2</span></div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="kv"><b>Samples</b><span>80</span></div>
    <div class="kv"><b>Variables</b><span>700</span></div>
    <div class="kv"><b>Content hash</b><span>9f3c…a71b</span></div>
    <div class="kv"><b>Derived from</b><span>v1</span></div>
    <div class="kv"><b>Created</b><span>2026-08-19</span></div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Variable axis</div>
    <div class="kv"><b>Kind</b><span>wavelength</span></div>
    <div class="kv"><b>Range</b><span>1100–2498 nm</span></div>
    <div class="kv"><b>Step</b><span>2 nm</span></div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Source</div>
    <div class="kv"><b>File</b><span>corn_raw.jdx</span></div>
    <div class="kv"><b>Reader</b><span>jcamp_dx 0.3.1</span></div>
    <div class="kv"><b>File hash</b><span>4b81…02de</span></div>
    <div class="kv"><b>Imported</b><span>14:22</span></div>
  </div>
  <div style="padding:9px 12px;display:flex;align-items:center;justify-content:space-between;color:var(--ink2)">
    <span>Provenance record</span><span style="transform:rotate(-90deg);display:inline-flex">{ICON["chev"]}</span>
  </div>
</aside>
</div>
{status(("PLS 5 LV · fold 7 of 10", "70%", "0:42"))}'''

page("Main.dc.html", "Workbench shell", shell_body, None, "light")

# ================================================================ 2. Pipeline canvas
NW = 132
COLS = [22, 180, 338, 496, 654]
ROWC = [84, 214, 344, 474, 604]
H = {"source": 74, "preprocess": 70, "split": 70, "estimator": 94, "model": 62}

def node(col, rowc, kind, title, sub, state, foot=None, extra=""):
    h = H[kind]; x = COLS[col]; y = rowc - h / 2
    border = "1px solid var(--rule)"; bg = "var(--surface)"; hbg = "var(--sunken)"
    hcol = "var(--ink3)"; op = "1"; tcol = "var(--ink)"
    stripe = ""
    if state == "running":
        border = "1.5px solid var(--accent)"; hbg = "var(--accentSoft)"; hcol = "var(--accentInk)"
    elif state == "stale":
        border = "1px dashed var(--stale)"; hcol = "var(--stale)"; op = ".72"
        bg = ("repeating-linear-gradient(135deg,var(--surface),var(--surface) 5px,"
              "var(--staleSoft) 5px,var(--staleSoft) 10px)")
        hbg = "var(--staleSoft)"
    elif state == "failed":
        border = "1px solid var(--fail)"; hbg = "var(--failSoft)"; hcol = "var(--fail)"
        stripe = ('<div style="position:absolute;left:0;top:0;bottom:0;width:3px;'
                  'background:var(--fail);border-radius:3px 0 0 3px"></div>')
    elif state == "queued":
        border = "1px dashed var(--rule)"; bg = "transparent"; hbg = "transparent"
        op = ".62"; tcol = "var(--ink3)"
    f = (f'<div style="padding:4px 8px 6px;font-family:\'IBM Plex Mono\',monospace;font-size:10px;'
         f'font-variant-numeric:tabular-nums;border-top:1px solid var(--rule2)">{foot}</div>') if foot else ""
    return f'''<div style="position:absolute;left:{x}px;top:{y:.0f}px;width:{NW}px;height:{h}px;
border:{border};border-radius:4px;background:{bg};opacity:{op};overflow:hidden;display:flex;flex-direction:column">{stripe}
<div style="height:17px;flex:none;display:flex;align-items:center;padding:0 8px;background:{hbg};
font-family:'IBM Plex Mono',monospace;font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:{hcol}">{kind}</div>
<div style="flex:1;padding:5px 8px 0;min-height:0">
<div style="font-size:12px;font-weight:500;color:{tcol};line-height:1.25">{title}</div>
<div style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;color:var(--ink3);margin-top:2px">{sub}</div>
{extra}</div>{f}</div>'''

def edge(c1, r1, k1, c2, r2, k2, colour="var(--rule)", w=1.4, dash=""):
    x1 = COLS[c1] + NW; y1 = ROWC[r1]; x2 = COLS[c2]; y2 = ROWC[r2]
    m = 42
    d = f"M{x1},{y1} C{x1+m},{y1} {x2-m},{y2} {x2},{y2}"
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="{w}"{da}></path>'
            f'<circle cx="{x2-3}" cy="{y2}" r="2.4" fill="{colour}"></circle>')

METRIC = lambda v: f'<span style="color:var(--ink3)">RMSECV</span> <b style="font-weight:600;color:var(--ink)">{v}</b>'

nodes = [
  node(0, 2, "source", "corn_raw", "v2 · 80 × 700", "ok", 'sha256:9f3c…a71b'),
  node(1, 0, "preprocess", "SNV", "—", "ok"),
  node(1, 1, "preprocess", "MSC", "reference: mean", "ok"),
  node(1, 2, "preprocess", "SG d1 w11", "window 11 · poly 2 · deriv 1", "ok"),
  node(1, 3, "preprocess", "SNV + SG d1", "window 15 · poly 3 · deriv 1", "stale",
       '<span style="color:var(--stale)">edited · downstream stale</span>'),
  node(1, 4, "preprocess", "Baseline ALS", "lam 1e5 · p 0.01", "ok"),
]
for r in range(5):
    st = "stale" if r == 3 else "ok"
    nodes.append(node(2, r, "split", "K-fold", "10 folds · shuffle · seed 42", st))
nodes += [
  node(3, 0, "estimator", "PLS", "6 LV · NIPALS · moisture", "ok", METRIC("0.412")),
  node(3, 1, "estimator", "PLS", "6 LV · NIPALS · moisture", "ok", METRIC("0.418")),
  node(3, 2, "estimator", "PLS", "5 LV · NIPALS · moisture", "running",
       '<span style="color:var(--accentInk)">fold 7 of 10 · 0:42</span>',
       '<div style="margin-top:5px;height:3px;border-radius:2px;background:var(--sunken);overflow:hidden">'
       '<div style="width:70%;height:100%;background:var(--accent)"></div></div>'),
  node(3, 3, "estimator", "PLS", "5 LV · NIPALS · moisture", "stale", METRIC("0.381")),
  node(3, 4, "estimator", "PLS", "6 LV · NIPALS · moisture", "failed",
       '<span style="color:var(--fail)">did not converge</span>'),
  node(4, 0, "model", "Model A", "saved 09:14", "ok"),
  node(4, 1, "model", "Model B", "saved 09:16", "ok"),
  node(4, 2, "model", "Model C", "awaiting fit", "queued"),
  node(4, 3, "model", "Model D", "saved 09:41", "stale"),
]
edges = []
for r in range(5):
    col = "var(--accent)" if r == 2 else ("var(--stale)" if r == 3 else "var(--rule)")
    dash = "4 3" if r == 3 else ""
    w = 1.8 if r == 2 else 1.4
    edges.append(edge(0, 2, "source", 1, r, "preprocess", col, w, dash))
    edges.append(edge(1, r, "preprocess", 2, r, "split", col, w, dash))
    edges.append(edge(2, r, "split", 3, r, "estimator", col, w, dash))
    if r != 4:
        c2 = "var(--rule)" if r not in (2, 3) else col
        edges.append(edge(3, r, "estimator", 4, r, "model", c2, w, dash))

LEG = [("Complete", "1px solid var(--rule)", "var(--surface)", "var(--ink2)"),
       ("Running", "1.5px solid var(--accent)", "var(--accentSoft)", "var(--accentInk)"),
       ("Stale", "1px dashed var(--stale)", "var(--staleSoft)", "var(--stale)"),
       ("Failed", "1px solid var(--fail)", "var(--failSoft)", "var(--fail)"),
       ("Not run", "1px dashed var(--rule)", "transparent", "var(--ink3)")]
legend = "".join(f'<div style="display:flex;align-items:center;gap:6px;font-size:10.5px;color:var(--ink2)">'
                 f'<span style="width:14px;height:10px;border:{b};background:{bg};border-radius:2px"></span>{n}</div>'
                 for n, b, bg, c in LEG)

pipe_body = f'''{titlebar('<span class="pill mono">11 nodes · 5 branches</span><span class="btn">Add node</span><span class="btn btn-p">Run all</span>')}
<div class="body">
{sidebar("n_pls")}
<main class="doc">
{tabs("Pipeline")}
<div class="pane" style="position:relative;background:var(--surface)">
  <div style="position:absolute;inset:0;background-image:radial-gradient(var(--grid) 1px,transparent 1px);background-size:22px 22px;opacity:.9"></div>
  <svg width="900" height="798" style="position:absolute;left:0;top:0;overflow:visible">{"".join(edges)}</svg>
  {"".join(nodes)}
  <div style="position:absolute;left:14px;bottom:14px;display:flex;align-items:center;gap:2px;
    background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:3px">
    <span style="width:22px;height:20px;display:flex;align-items:center;justify-content:center;color:var(--ink2)">−</span>
    <span class="mono" style="font-size:10.5px;color:var(--ink2);padding:0 4px">100%</span>
    <span style="width:22px;height:20px;display:flex;align-items:center;justify-content:center;color:var(--ink2)">+</span>
    <span style="width:1px;height:14px;background:var(--rule);margin:0 3px"></span>
    <span style="font-size:11px;color:var(--ink2);padding:0 6px">Fit</span>
  </div>
  <div style="position:absolute;right:14px;bottom:14px;display:flex;flex-direction:column;gap:5px;
    background:var(--panel);border:1px solid var(--rule);border-radius:4px;padding:8px 10px">{legend}</div>
</div>
</main>
<aside class="insp">
  <div class="ihead"><span class="ilabel">Node · estimator</span>
    <span style="font-weight:600;font-size:13px">PLS · 6 LV</span></div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Parameters</div>
    <div class="kv"><b>Latent variables</b><span>6</span></div>
    <div class="kv"><b>Algorithm</b><span>NIPALS</span></div>
    <div class="kv"><b>Target</b><span>moisture</span></div>
    <div class="kv"><b>Input</b><span>K-fold 10</span></div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Metrics</div>
    <div class="kv"><b>RMSECV</b><span style="color:var(--accentInk);font-weight:600">0.412</span></div>
    <div class="kv"><b>RMSEC</b><span>0.371</span></div>
    <div class="kv"><b>R²</b><span>0.974</span></div>
    <div class="kv"><b>Bias</b><span>−0.006</span></div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Provenance</div>
    <div class="kv"><b>Pipeline hash</b><span>c41d…8e0f</span></div>
    <div class="kv"><b>Seed</b><span>42</span></div>
    <div class="kv"><b>scikit-learn</b><span>1.7.2</span></div>
    <div class="kv"><b>numpy</b><span>2.1.0</span></div>
    <div class="kv"><b>Run at</b><span>09:14:22</span></div>
  </div>
  <div style="padding:10px 12px;display:flex;gap:6px">
    <span class="btn" style="flex:1;justify-content:center">Duplicate branch</span>
    <span class="btn" style="flex:1;justify-content:center">Compare…</span>
  </div>
</aside>
</div>
{status(("PLS 5 LV · fold 7 of 10", "70%", "0:42"))}'''

page("PipelineCanvas.dc.html", "Pipeline canvas", pipe_body, None, "light")

# ================================================================ 3. Spectra view
SW, SH = 868, 320
PAD = (48, 14, 30, 14)

rr = random.Random(11)
allraw = [raw_curve(rr) for _ in range(80)]
lo = [min(c[i][1] for c in allraw) for i in range(len(allraw[0]))]
hi = [max(c[i][1] for c in allraw) for i in range(len(allraw[0]))]
ymax = max(hi) * 1.04
xt = [(0.0, "1100"), (0.215, "1400"), (0.43, "1700"), (0.645, "2000"), (0.86, "2300"), (1.0, "2498")]
ytr = [(v, f"{v*ymax:.2f}") for v in (0, .25, .5, .75, 1.0)]
gr, mx, my = axes(SW, SH, xt, ytr, "Wavelength (nm)", "Absorbance", PAD)

env = ([(mx(i / (len(lo) - 1)), my(lo[i] / ymax)) for i in range(len(lo))] +
       [(mx(i / (len(hi) - 1)), my(hi[i] / ymax)) for i in reversed(range(len(hi)))])
band = (f'<path d="{poly(env)}Z" fill="var(--band)" opacity=".45" stroke="none"></path>')

DCOL = ["var(--d1)", "var(--d2)", "var(--d3)", "var(--d4)", "var(--d5)", "var(--d6)"]
SIDS = ["C004", "C011", "C023", "C038", "C052", "C067"]
hi_raw = []
rr2 = random.Random(3)
for k in range(6):
    c = raw_curve(rr2)
    pts = [(mx(u), my(v / ymax)) for u, v in c]
    hi_raw.append(f'<path d="{poly(pts)}" fill="none" stroke="{DCOL[k]}" stroke-width="1.35"></path>')
raw_svg = f'<svg width="{SW}" height="{SH}" viewBox="0 0 {SW} {SH}">{gr}{band}{"".join(hi_raw)}</svg>'

pr = [proc_curve(random.Random(20 + k)) for k in range(6)]
pmax = max(abs(v) for c in pr for _, v in c) * 1.15
ytp = [(v, f"{(v*2-1)*pmax:+.2f}".replace("+0.00", " 0.00")) for v in (0, .25, .5, .75, 1.0)]
gr2, mx2, my2 = axes(SW, SH, xt, ytp, "Wavelength (nm)", "SNV + 1st derivative", PAD)
zero = f'<line x1="{mx2(0)}" y1="{my2(.5):.1f}" x2="{mx2(1)}" y2="{my2(.5):.1f}" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 3"></line>'
hi_pr = []
for k, c in enumerate(pr):
    pts = [(mx2(u), my2((v / pmax + 1) / 2)) for u, v in c]
    hi_pr.append(f'<path d="{poly(pts)}" fill="none" stroke="{DCOL[k]}" stroke-width="1.35"></path>')
pro_svg = f'<svg width="{SW}" height="{SH}" viewBox="0 0 {SW} {SH}">{gr2}{zero}{"".join(hi_pr)}</svg>'

legend_s = "".join(
    f'<div style="display:flex;align-items:center;gap:5px;font-family:\'IBM Plex Mono\',monospace;'
    f'font-size:10.5px;color:var(--ink2)"><span style="width:13px;height:2px;background:{DCOL[k]}"></span>{SIDS[k]}</div>'
    for k in range(6))

def seg(label, on):
    return (f'<span style="padding:0 10px;height:24px;display:flex;align-items:center;font-size:11.5px;'
            f'background:{"var(--surface)" if on else "transparent"};'
            f'color:{"var(--ink)" if on else "var(--ink3)"};font-weight:{600 if on else 400}">{label}</span>')

spec_body = f'''{titlebar('<span class="pill mono">SG · d1 · window 11</span><span class="btn">Add step</span><span class="btn btn-p">Apply to branch</span>')}
<div class="body">
{sidebar("n_sg")}
<main class="doc">
{tabs("SG d1 w11")}
<div class="pane">
  <div style="height:42px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 14px;gap:12px;border-bottom:1px solid var(--rule2)">
    <div style="display:flex;align-items:center;gap:10px">
      <div style="display:flex;border:1px solid var(--rule);border-radius:3px;overflow:hidden;background:var(--sunken)">
        {seg("Raw", False)}{seg("Processed", False)}{seg("Both", True)}</div>
      <span class="mono" style="font-size:11px;color:var(--ink3)">80 spectra · 6 highlighted · 700 variables</span>
    </div>
    <div style="display:flex;align-items:center;gap:7px">
      <span class="pill mono">decimated to 700 px</span>
      <span class="pill mono">colour by: sample</span>
    </div>
  </div>
  <div style="flex:1;min-height:0;padding:8px 14px 0;display:flex;flex-direction:column;gap:6px">
    <div style="display:flex;align-items:baseline;justify-content:space-between">
      <span style="font-size:11.5px;font-weight:600">Raw absorbance</span>
      <span class="mono" style="font-size:10.5px;color:var(--ink3)">shaded band = full set (80) · lines drawn at full resolution</span>
    </div>
    {raw_svg}
    <div style="display:flex;align-items:baseline;justify-content:space-between;margin-top:2px">
      <span style="font-size:11.5px;font-weight:600">After SNV → Savitzky–Golay (d1, window 11, poly 2)</span>
      <span class="mono" style="font-size:10.5px;color:var(--ink3)">live preview · 0.3 s</span>
    </div>
    {pro_svg}
    <div style="display:flex;align-items:center;gap:14px;padding:6px 0 0">{legend_s}
      <span style="margin-left:auto;font-size:10.5px;color:var(--ink3)">click a trace to open its sample</span></div>
  </div>
</div>
</main>
<aside class="insp">
  <div class="ihead"><span class="ilabel">Node · preprocess</span>
    <span style="font-weight:600;font-size:13px">Savitzky–Golay</span></div>
  <div style="padding:9px 12px;border-bottom:1px solid var(--rule);display:flex;flex-direction:column;gap:8px">
    <div style="display:flex;flex-direction:column;gap:3px">
      <span class="ilabel">Window length</span>
      <div style="display:flex;align-items:center;gap:8px">
        <div style="flex:1;height:3px;background:var(--sunken);border-radius:2px;position:relative">
          <div style="width:34%;height:100%;background:var(--accent);border-radius:2px"></div>
          <div style="position:absolute;left:34%;top:-4px;width:11px;height:11px;border-radius:50%;background:var(--surface);border:1.5px solid var(--accent)"></div>
        </div><span class="mono" style="font-size:11px">11</span></div>
      <span style="font-size:10px;color:var(--ink3)">must be odd</span>
    </div>
    <div style="display:flex;gap:8px">
      <div style="flex:1;display:flex;flex-direction:column;gap:3px"><span class="ilabel">Poly order</span>
        <div class="pill mono" style="justify-content:space-between">2 {ICON["chev"]}</div></div>
      <div style="flex:1;display:flex;flex-direction:column;gap:3px"><span class="ilabel">Derivative</span>
        <div class="pill mono" style="justify-content:space-between;border-color:var(--accent);color:var(--accentInk)">1 {ICON["chev"]}</div></div>
    </div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Input</div>
    <div class="kv"><b>From</b><span>SNV</span></div>
    <div class="kv"><b>Shape</b><span>80 × 700</span></div>
    <div class="kv"><b>Range kept</b><span>1100–2498 nm</span></div>
  </div>
  <div style="padding:9px 12px;background:var(--staleSoft);border-bottom:1px solid var(--rule);
    display:flex;gap:8px;align-items:flex-start">
    <span style="color:var(--stale);font-weight:600;font-size:14px;line-height:1">!</span>
    <span style="font-size:11.5px;color:var(--ink2);line-height:1.4">Changing this marks
      <b style="font-weight:600">2 downstream models</b> stale. Their results stay visible until re-run.</span>
  </div>
  <div style="padding:10px 12px;display:flex;gap:6px">
    <span class="btn" style="flex:1;justify-content:center">Revert</span>
    <span class="btn btn-p" style="flex:1;justify-content:center">Re-run downstream</span>
  </div>
</aside>
</div>
{status(("PLS 5 LV · fold 7 of 10", "70%", "0:42"))}'''

page("SpectraView.dc.html", "Spectra view", spec_body, None, "light")

# ================================================================ 4. Analysis results
def card(title, note, svg, w, h):
    return f'''<div style="width:{w}px;height:{h}px;border:1px solid var(--rule2);border-radius:4px;
background:var(--surface);display:flex;flex-direction:column;overflow:hidden">
<div style="height:24px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 9px;border-bottom:1px solid var(--rule2)">
<span style="font-size:11px;font-weight:600">{title}</span>
<span class="mono" style="font-size:9.5px;color:var(--ink3)">{note}</span></div>{svg}</div>'''

# -- scores + Hotelling T2
W1, H1 = 428, 312
xs = [(v, f"{(v*2-1)*4:+.0f}".replace("+0", "0")) for v in (0, .25, .5, .75, 1)]
ys = [(v, f"{(v*2-1)*2.4:+.1f}") for v in (0, .25, .5, .75, 1)]
g1, m1x, m1y = axes(W1, H1 - 24, xs, ys, "PC 1  (74.2%)", "PC 2  (16.8%)", (44, 14, 30, 12))
rs = random.Random(5); pts = []
for i in range(80):
    grp = 0 if i % 5 else 1
    x = rs.gauss(-1.1 if grp == 0 else 1.4, 1.15)
    y = rs.gauss(0.15 if grp == 0 else -0.2, 0.78)
    pts.append((x, y, grp))
pts[7] = (3.55, 1.85, 0); pts[41] = (-3.35, -1.72, 0)
dots = "".join(
    f'<circle cx="{m1x((x/4+1)/2):.1f}" cy="{m1y((y/2.4+1)/2):.1f}" r="3" '
    f'fill="{"var(--d1)" if g==0 else "var(--d2)"}" fill-opacity=".82"></circle>' for x, y, g in pts)
ell = (f'<ellipse cx="{m1x(.5+0.03):.1f}" cy="{m1y(.5):.1f}" rx="{(m1x(1)-m1x(0))*0.335:.1f}" '
       f'ry="{(m1y(0)-m1y(1))*0.335:.1f}" fill="none" stroke="var(--ink3)" stroke-width="1" stroke-dasharray="4 3"></ellipse>')
outl = "".join(
    f'<circle cx="{m1x((x/4+1)/2):.1f}" cy="{m1y((y/2.4+1)/2):.1f}" r="6.5" fill="none" stroke="var(--fail)" stroke-width="1.2"></circle>'
    f'<text x="{m1x((x/4+1)/2)+9:.1f}" y="{m1y((y/2.4+1)/2)+3:.1f}" font-family="IBM Plex Mono" font-size="9" fill="var(--fail)">{lab}</text>'
    for (x, y, _), lab in ((pts[7], "C038"), (pts[41], "C013")))
scores = card("Scores", "Hotelling T² 95%",
              f'<svg width="{W1}" height="{H1-24}" viewBox="0 0 {W1} {H1-24}">{g1}{ell}{dots}{outl}</svg>', W1, H1)

# -- loadings
g2, m2x, m2y = axes(W1, H1 - 24, xt, [(v, f"{(v*2-1)*0.14:+.2f}") for v in (0, .25, .5, .75, 1)],
                    "Wavelength (nm)", "Loading", (44, 14, 30, 12))
lp = []
for k, sd in enumerate((31, 47)):
    c = proc_curve(random.Random(sd))
    mxv = max(abs(v) for _, v in c)
    pp = [(m2x(u), m2y((v / mxv * (0.85 if k == 0 else 0.6) + 1) / 2)) for u, v in c]
    lp.append(f'<path d="{poly(pp)}" fill="none" stroke="{"var(--d1)" if k==0 else "var(--d2)"}" stroke-width="1.5"></path>')
lz = f'<line x1="{m2x(0)}" y1="{m2y(.5):.1f}" x2="{m2x(1)}" y2="{m2y(.5):.1f}" stroke="var(--rule)" stroke-width="1" stroke-dasharray="3 3"></line>'
lleg = ('<g><rect x="52" y="18" width="86" height="30" rx="3" fill="var(--surface)" stroke="var(--rule2)"></rect>'
        '<line x1="60" y1="28" x2="74" y2="28" stroke="var(--d1)" stroke-width="1.6"></line>'
        '<text x="79" y="31" font-family="IBM Plex Mono" font-size="9" fill="var(--ink2)">PC 1</text>'
        '<line x1="60" y1="40" x2="74" y2="40" stroke="var(--d2)" stroke-width="1.6"></line>'
        '<text x="79" y="43" font-family="IBM Plex Mono" font-size="9" fill="var(--ink2)">PC 2</text></g>')
loadings = card("Loadings", "PC 1–2",
                f'<svg width="{W1}" height="{H1-24}" viewBox="0 0 {W1} {H1-24}">{g2}{lz}{"".join(lp)}{lleg}</svg>', W1, H1)

# -- explained variance
W2, H2 = 280, 308
ev = [74.2, 16.8, 4.1, 2.2, 1.1, 0.8]
cum = [sum(ev[:i + 1]) for i in range(6)]
g3, m3x, m3y = axes(W2, H2 - 24, [(i / 5, f"{i+1}") for i in range(6)],
                    [(v, f"{v*100:.0f}") for v in (0, .25, .5, .75, 1)], "Component", "% variance", (36, 14, 30, 12))
bw = (m3x(1) - m3x(0)) / 8
bars = "".join(
    f'<rect x="{m3x(i/5)-bw/2:.1f}" y="{m3y(ev[i]/100):.1f}" width="{bw:.1f}" '
    f'height="{m3y(0)-m3y(ev[i]/100):.1f}" fill="var(--d1)" fill-opacity=".85"></rect>' for i in range(6))
cline = poly([(m3x(i / 5), m3y(cum[i] / 100)) for i in range(6)])
cumsvg = (f'<path d="{cline}" fill="none" stroke="var(--d2)" stroke-width="1.5"></path>' +
          "".join(f'<circle cx="{m3x(i/5):.1f}" cy="{m3y(cum[i]/100):.1f}" r="2.6" fill="var(--d2)"></circle>' for i in range(6)))
explvar = card("Explained variance", "cumulative 99.2%",
               f'<svg width="{W2}" height="{H2-24}" viewBox="0 0 {W2} {H2-24}">{g3}{bars}{cumsvg}</svg>', W2, H2)

# -- predicted vs actual
W3 = 284
g4, m4x, m4y = axes(W3, H2 - 24, [(v, f"{9.0+v*2.6:.1f}") for v in (0, .5, 1)],
                    [(v, f"{9.0+v*2.6:.1f}") for v in (0, .5, 1)], "Measured moisture %", "Predicted %", (40, 14, 30, 12))
rp = random.Random(9); pv = []
for i in range(80):
    a = rp.uniform(0.04, 0.96); p = a + rp.gauss(0, 0.055)
    pv.append((a, min(max(p, 0), 1), 0 if i % 5 else 1))
idl = f'<line x1="{m4x(0)}" y1="{m4y(0):.1f}" x2="{m4x(1)}" y2="{m4y(1):.1f}" stroke="var(--ink3)" stroke-width="1" stroke-dasharray="4 3"></line>'
pdots = "".join(f'<circle cx="{m4x(a):.1f}" cy="{m4y(p):.1f}" r="2.6" fill="{"var(--d1)" if g==0 else "var(--d2)"}" fill-opacity=".8"></circle>' for a, p, g in pv)
predact = card("Predicted vs measured", "cal · val",
               f'<svg width="{W3}" height="{H2-24}" viewBox="0 0 {W3} {H2-24}">{g4}{idl}{pdots}</svg>', W3, H2)

# -- residuals
g5, m5x, m5y = axes(W3, H2 - 24, [(v, f"{9.0+v*2.6:.1f}") for v in (0, .5, 1)],
                    [(v, f"{(v*2-1)*0.9:+.1f}") for v in (0, .25, .5, .75, 1)], "Measured moisture %", "Residual", (40, 14, 30, 12))
sig = 0.30
bandr = (f'<rect x="{m5x(0)}" y="{m5y(.5+sig):.1f}" width="{m5x(1)-m5x(0):.1f}" '
         f'height="{m5y(.5-sig)-m5y(.5+sig):.1f}" fill="var(--accent)" fill-opacity=".07"></rect>')
zl = f'<line x1="{m5x(0)}" y1="{m5y(.5):.1f}" x2="{m5x(1)}" y2="{m5y(.5):.1f}" stroke="var(--rule)" stroke-width="1"></line>'
rdots = "".join(f'<circle cx="{m5x(a):.1f}" cy="{m5y(.5+(p-a)*3.2):.1f}" r="2.6" fill="{"var(--d1)" if g==0 else "var(--d2)"}" fill-opacity=".8"></circle>' for a, p, g in pv)
resid = card("Residuals", "±2σ band",
             f'<svg width="{W3}" height="{H2-24}" viewBox="0 0 {W3} {H2-24}">{g5}{bandr}{zl}{rdots}</svg>', W3, H2)

chip = lambda k, v, hi=False: (
    f'<div style="display:flex;flex-direction:column;gap:1px;padding:0 11px;border-left:1px solid var(--rule2)">'
    f'<span class="ilabel" style="font-size:9px">{k}</span>'
    f'<span class="mono" style="font-size:14px;font-weight:600;'
    f'color:{"var(--accentInk)" if hi else "var(--ink)"}">{v}</span></div>')

ana_body = f'''{titlebar('<span class="pill mono">Model C</span><span class="btn">Export model…</span><span class="btn btn-p">Compare</span>')}
<div class="body">
{sidebar("mC")}
<main class="doc">
{tabs("Model C")}
<div class="pane">
  <div style="height:52px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 14px 0 0;border-bottom:1px solid var(--rule2)">
    <div style="display:flex;align-items:center;gap:9px;padding-left:14px">
      <span style="font-weight:600;font-size:13.5px">Model C</span>
      <span class="mono" style="font-size:11px;color:var(--ink3)">SG d1 w11 → PLS 5 LV → moisture</span>
    </div>
    <div style="display:flex;align-items:stretch">
      {chip("RMSECV","0.389",True)}{chip("RMSEC","0.352")}{chip("R²","0.981")}{chip("Bias","−0.004")}{chip("LV","5")}
    </div>
  </div>
  <div style="flex:1;min-height:0;padding:12px 14px;display:flex;flex-direction:column;gap:12px">
    <div style="display:flex;gap:12px">{scores}{loadings}</div>
    <div style="display:flex;gap:12px">{explvar}{predact}{resid}</div>
  </div>
</div>
</main>
<aside class="insp">
  <div class="ihead"><span class="ilabel">Model</span>
    <span style="font-weight:600;font-size:13px">Model C</span></div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Validation</div>
    <div class="kv"><b>Protocol</b><span>K-fold 10</span></div>
    <div class="kv"><b>Seed</b><span>42</span></div>
    <div class="kv"><b>RMSECV</b><span style="color:var(--accentInk);font-weight:600">0.389</span></div>
    <div class="kv"><b>RMSEP (val)</b><span>0.401</span></div>
    <div class="kv"><b>Q²</b><span>0.977</span></div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Diagnostics</div>
    <div class="kv"><b>T² outliers</b><span style="color:var(--fail)">2</span></div>
    <div class="kv"><b>Max SPE</b><span>1.84</span></div>
    <div class="kv"><b>VIP &gt; 1</b><span>184 vars</span></div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Provenance</div>
    <div class="kv"><b>Dataset</b><span>corn_raw v2</span></div>
    <div class="kv"><b>Content hash</b><span>9f3c…a71b</span></div>
    <div class="kv"><b>Pipeline hash</b><span>7a20…dd41</span></div>
    <div class="kv"><b>scikit-learn</b><span>1.7.2</span></div>
    <div class="kv"><b>App version</b><span>0.4.1</span></div>
  </div>
  <div style="padding:10px 12px;display:flex;flex-direction:column;gap:6px">
    <span class="btn" style="justify-content:center">Export JSON model</span>
    <span class="btn" style="justify-content:center">Export Python snippet</span>
    <span class="btn" style="justify-content:center">Export report (PDF)</span>
  </div>
</aside>
</div>
{status(("PLS 5 LV · fold 7 of 10", "70%", "0:42"))}'''

page("AnalysisResults.dc.html", "Analysis results", ana_body, None, "dark")

# ================================================================ 5. Experiment comparison
RUNS = [
 ("Model A", "SNV",                   "K-fold 10", "PLS 6 LV", "0.412", "0.974", "−0.006", "ok",     False),
 ("Model B", "MSC",                   "K-fold 10", "PLS 6 LV", "0.418", "0.972", "+0.011", "ok",     False),
 ("Model C", "SG d1 w11",             "K-fold 10", "PLS 5 LV", "0.389", "0.981", "−0.004", "ok",     True),
 ("Model D", "SNV → SG d1 w11",       "K-fold 10", "PLS 5 LV", "0.381", "0.983", "−0.002", "stale",  True),
 ("run-08",  "SNV → SG d2 w15",       "K-fold 10", "PLS 7 LV", "0.436", "0.968", "+0.014", "ok",     False),
 ("run-07",  "Baseline ALS",          "K-fold 10", "PLS 6 LV", "—",     "—",     "—",      "failed", False),
 ("run-06",  "SNV",                   "LOOCV",     "PLS 6 LV", "0.407", "0.975", "−0.005", "ok",     False),
 ("run-05",  "MSC → SG d1 w11",       "K-fold 10", "PLS 5 LV", "0.394", "0.980", "+0.003", "ok",     False),
 ("run-04",  "Range 1300–2200 → SNV", "K-fold 10", "PLS 6 LV", "0.421", "0.971", "−0.009", "ok",     False),
 ("run-03",  "Autoscale",             "K-fold 10", "PLS 8 LV", "0.512", "0.951", "+0.021", "ok",     False),
 ("run-02",  "SNV",                   "Train/test","PLS 6 LV", "0.399", "0.977", "−0.007", "ok",     False),
 ("run-01",  "none",                  "K-fold 10", "PLS 6 LV", "0.688", "0.902", "+0.038", "ok",     False),
]

def spark(rmse, seed):
    if rmse == "—":
        return '<span style="font-size:10px;color:var(--ink3)">—</span>'
    s = float(rmse); rr = random.Random(seed); w, h = 88, 15
    dots = "".join(f'<circle cx="{w/2 + rr.gauss(0,1)*s*w/3.4:.1f}" cy="{h/2 + rr.uniform(-3.4,3.4):.1f}" '
                   f'r="1.4" fill="var(--d1)" fill-opacity=".65"></circle>' for _ in range(26))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
            f'<line x1="{w/2}" y1="1" x2="{w/2}" y2="{h-1}" stroke="var(--rule)" stroke-width="1"></line>{dots}</svg>')

def badge(st):
    m = {"ok": ("var(--accentSoft)", "var(--accentInk)", "complete"),
         "stale": ("var(--staleSoft)", "var(--stale)", "stale"),
         "failed": ("var(--failSoft)", "var(--fail)", "failed")}
    bg, c, lab = m[st]
    return (f'<span class="mono" style="font-size:9.5px;padding:1.5px 6px;border-radius:2px;'
            f'background:{bg};color:{c};letter-spacing:.04em">{lab}</span>')

trows = []
for i, (name, prep, split, mod, rm, r2, bias, st, sel) in enumerate(RUNS):
    base = 'background:var(--accentSoft)' if sel else ''
    left = ('box-shadow:inset 3px 0 0 var(--accent)' if sel else '')
    box = ('<span style="width:12px;height:12px;border-radius:2px;display:inline-flex;align-items:center;'
           'justify-content:center;background:var(--accent);color:#fff;font-size:9px">✓</span>' if sel
           else '<span style="width:12px;height:12px;border-radius:2px;border:1px solid var(--rule);display:inline-block"></span>')
    ph = (' style="background:var(--surface);border-radius:3px;box-shadow:0 0 0 1px var(--accent);'
          'font-weight:600;color:var(--ink)"' if sel else '')
    dim = ' style="opacity:.55"' if st == "failed" else ''
    trows.append(
        f'<tr style="{base};{left}"{dim}><td style="text-align:center">{box}</td>'
        f'<td class="mono" style="color:var(--ink);font-weight:500">{name}</td>'
        f'<td><span class="mono"{ph}>{prep}</span></td>'
        f'<td class="mono">{split}</td><td class="mono">{mod}</td>'
        f'<td class="n" style="font-weight:600">{rm}</td><td class="n">{r2}</td><td class="n">{bias}</td>'
        f'<td>{spark(rm, 100+i)}</td><td>{badge(st)}</td></tr>')

diff_row = lambda name, prep, rm, tone: (
    f'<div style="display:flex;align-items:center;gap:10px;font-size:12px">'
    f'<span class="mono" style="width:66px;font-weight:600;color:var(--ink)">{name}</span>'
    f'<span class="mono" style="flex:1;padding:2px 8px;border-radius:3px;background:{tone};color:var(--ink)">{prep}</span>'
    f'<span class="mono" style="color:var(--ink3)">PLS 5 LV · K-fold 10 · seed 42</span>'
    f'<span class="mono" style="width:52px;text-align:right;font-weight:600">{rm}</span></div>')

cmp_body = f'''{titlebar('<span class="pill mono">12 runs · 2 selected</span><span class="btn">Export table</span><span class="btn btn-p">Open comparison</span>')}
<div class="body">
{sidebar("exp")}
<main class="doc">
{tabs("Runs")}
<div class="pane">
  <div style="flex:none;padding:12px 14px;border-bottom:1px solid var(--rule2);display:flex;flex-direction:column;gap:8px;background:var(--sunken)">
    <div style="display:flex;align-items:center;justify-content:space-between">
      <span style="font-size:11.5px;font-weight:600">Comparing 2 runs</span>
      <span class="mono" style="font-size:10.5px;color:var(--ink3)">same dataset · corn_raw v2 · sha256:9f3c…a71b</span>
    </div>
    {diff_row("Model C", "SG d1 w11", "0.389", "var(--surface)")}
    {diff_row("Model D", "SNV → SG d1 w11", "0.381", "var(--accentSoft)")}
    <div style="display:flex;align-items:center;gap:8px;font-size:11.5px;color:var(--ink2);padding-top:2px">
      <span style="font-family:'IBM Plex Mono',monospace;font-size:9.5px;letter-spacing:.1em;
        text-transform:uppercase;color:var(--accentInk);background:var(--accentSoft);padding:1.5px 6px;border-radius:2px">only difference</span>
      <span>Preprocessing — <b style="font-weight:600">one step added</b>: SNV before Savitzky–Golay.
        Everything else identical, including split indices.</span>
      <span class="mono" style="margin-left:auto;color:var(--accentInk);font-weight:600">RMSECV −0.008</span>
    </div>
  </div>
  <div style="flex:1;min-height:0;overflow:hidden">
    <table>
      <thead><tr><th style="width:30px"></th><th style="width:78px">Run</th><th>Preprocessing</th>
      <th style="width:96px">Split</th><th style="width:84px">Model</th>
      <th class="n" style="width:72px">RMSECV</th><th class="n" style="width:58px">R²</th>
      <th class="n" style="width:64px">Bias</th><th style="width:100px">Residuals</th>
      <th style="width:84px">Status</th></tr></thead>
      <tbody>{"".join(trows)}</tbody>
    </table>
  </div>
</div>
</main>
<aside class="insp">
  <div class="ihead"><span class="ilabel">Comparison</span>
    <span style="font-weight:600;font-size:13px">Model C ↔ Model D</span></div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Shared</div>
    <div class="kv"><b>Dataset</b><span>corn_raw v2</span></div>
    <div class="kv"><b>Split</b><span>K-fold 10</span></div>
    <div class="kv"><b>Seed</b><span>42</span></div>
    <div class="kv"><b>Fold indices</b><span>identical</span></div>
    <div class="kv"><b>Estimator</b><span>PLS 5 LV</span></div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Differs</div>
    <div style="padding:4px 12px;display:flex;flex-direction:column;gap:5px">
      <div style="display:flex;gap:7px;align-items:baseline">
        <span class="mono" style="font-size:10px;color:var(--fail)">−</span>
        <span class="mono" style="font-size:11px;color:var(--ink2)">SG d1 w11</span></div>
      <div style="display:flex;gap:7px;align-items:baseline">
        <span class="mono" style="font-size:10px;color:var(--accentInk)">+</span>
        <span class="mono" style="font-size:11px;color:var(--ink)">SNV</span></div>
      <div style="display:flex;gap:7px;align-items:baseline">
        <span class="mono" style="font-size:10px;color:var(--ink3)">=</span>
        <span class="mono" style="font-size:11px;color:var(--ink2)">SG d1 w11</span></div>
    </div>
  </div>
  <div style="padding:8px 0;border-bottom:1px solid var(--rule)">
    <div class="ilabel" style="padding:2px 12px 4px">Delta</div>
    <div class="kv"><b>RMSECV</b><span style="color:var(--accentInk);font-weight:600">−0.008</span></div>
    <div class="kv"><b>R²</b><span style="color:var(--accentInk)">+0.002</span></div>
    <div class="kv"><b>Bias</b><span>+0.002</span></div>
  </div>
  <div style="padding:9px 12px;background:var(--staleSoft);border-bottom:1px solid var(--rule);display:flex;gap:8px;align-items:flex-start">
    <span style="color:var(--stale);font-weight:600;font-size:14px;line-height:1">!</span>
    <span style="font-size:11.5px;color:var(--ink2);line-height:1.4">Model D is
      <b style="font-weight:600">stale</b> — its preprocessing changed after the fit. Re-run before trusting the delta.</span>
  </div>
  <div style="padding:10px 12px;display:flex;gap:6px">
    <span class="btn" style="flex:1;justify-content:center">Open side by side</span>
  </div>
</aside>
</div>
{status()}'''

page("ExperimentComparison.dc.html", "Experiment comparison", cmp_body, None, "light")

# ---------------------------------------------------------------- canvas.json
import json
GAP, RY = 120, 1020
lay = {
  "artboards": [
    {"file": "Main.dc.html",                 "x": 0,    "y": 0,    "w": 1440, "h": 900, "title": "1 · Shell"},
    {"file": "PipelineCanvas.dc.html",       "x": 1560, "y": 0,    "w": 1440, "h": 900, "title": "2 · Pipeline canvas"},
    {"file": "SpectraView.dc.html",          "x": 3120, "y": 0,    "w": 1440, "h": 900, "title": "3 · Spectra"},
    {"file": "AnalysisResults.dc.html",      "x": 0,    "y": RY,   "w": 1440, "h": 900, "title": "4 · Analysis results (dark)"},
    {"file": "ExperimentComparison.dc.html", "x": 1560, "y": RY,   "w": 1440, "h": 900, "title": "5 · Experiment comparison"},
  ],
  "annotations": [
    {"id": "brief", "x": 3120, "y": RY, "w": 420,
     "text": "Chemometrics Workbench — core screens\n\nOne window, three regions: project tree left, "
             "tabbed documents centre, contextual inspector right. Every preprocessing step and analysis "
             "opens as its own tab; selecting a node in the canvas focuses its tab.\n\n"
             "Type: IBM Plex Sans / Plex Mono, tabular numerals throughout.\n"
             "Data series: Okabe-Ito, colourblind-safe, kept separate from the teal UI accent so a failing "
             "node can never be mistaken for a red spectrum.\n\n"
             "Artboard 4 is set to the dark palette; every artboard carries a light/dark tweak."},
    {"id": "states", "x": 3120, "y": RY + 300, "w": 420,
     "text": "Node states are encoded in form as well as colour: solid (complete), accent border with "
             "progress (running), dashed + hatched (stale), left stripe (failed), dashed outline (not run).\n\n"
             "Stale is the state a branching graph forces — edit a parameter upstream and downstream "
             "results are invalid but must not vanish."},
  ],
  "launch": {"view": "canvas"},
}
(BASE / "canvas.json").write_text(json.dumps(lay, indent=2))
print("wrote canvas.json")
