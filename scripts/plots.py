"""Charts for the README, drawn from results/*.json as plain SVG (stdlib only).

  results/charts/modulation_top1.svg  blind S5 top-1 accuracy per modulation,
                                      standard CFO vs zero CFO
  results/charts/bitlayer_grid.svg    interleaver x FEC: frames recovered,
                                      CRC pass written in each cell
  results/charts/snr_curves.svg       blind top-1 accuracy vs SNR per
                                      modulation family (--snr-sweep runs)

Colours follow the dataviz reference palette (categorical slots 1-2 and the
blue sequential ramp, validated light + dark); every value is also written
as text, and the README tables are the table view. Each mark carries an SVG
<title> (hover tooltip when the file is opened directly).

Run:  python scripts/plots.py   (after the result scripts)
"""
import json
import os
from xml.sax.saxutils import escape

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(RESULTS, "charts")

STYLE = """<style>
  svg { --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
        --grid:#e1e0d9; --axis:#c3c2b7; --s1:#2a78d6; --s2:#eb6834;
        --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
        font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }
  @media (prefers-color-scheme: dark) {
    svg { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
          --grid:#2c2c2a; --axis:#383835; --s1:#3987e5; --s2:#d95926;
          --s3:#199e70; --s4:#c98500; --s5:#d55181; } }
  .bg { fill: var(--surface); } .title { fill: var(--ink); font-size: 15px; font-weight: 600; }
  .sub { fill: var(--ink2); font-size: 12px; } .lab { fill: var(--ink2); font-size: 12px; }
  .tick { fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }
  .val { fill: var(--ink2); font-size: 11px; font-variant-numeric: tabular-nums; }
  .grid { stroke: var(--grid); stroke-width: 1; } .axis { stroke: var(--axis); stroke-width: 1; }
  .s1 { fill: var(--s1); } .s2 { fill: var(--s2); }
  .s3 { fill: var(--s3); } .s4 { fill: var(--s4); } .s5 { fill: var(--s5); }
  .l1 { stroke: var(--s1); } .l2 { stroke: var(--s2); } .l3 { stroke: var(--s3); }
  .l4 { stroke: var(--s4); } .l5 { stroke: var(--s5); }
  .line { fill: none; stroke-width: 2; stroke-linejoin: round; }
  .dot { stroke: var(--surface); stroke-width: 2; }
  .cell-dark { fill: #ffffff; font-size: 12px; } .cell-light { fill: #0b0b0b; font-size: 12px; }
</style>"""

# blue sequential ramp, light -> dark (reference palette steps 100..650)
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"]


def _bar(x, y, w, h, cls, tip, r=4):
    """Horizontal bar anchored at x, rounded only at its data end."""
    if w <= 0:
        return ""
    r = min(r, w, h / 2)
    d = (f"M{x},{y}h{w - r}a{r},{r} 0 0 1 {r},{r}v{h - 2 * r}"
         f"a{r},{r} 0 0 1 {-r},{r}h{-(w - r)}z")
    return f'<path class="{cls}" d="{d}"><title>{escape(tip)}</title></path>'


def _svg(w, h, body, label):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}">'
            f'{STYLE}<rect class="bg" width="{w}" height="{h}" rx="6"/>'
            f'{body}</svg>\n')


def modulation_chart(d):
    pc = d["per_class"]
    mods = [m for m in pc if "(" not in m]
    rows = [(m, pc[m]["top1_accuracy"], pc.get(f"{m} (CFO 0)", {})
             .get("top1_accuracy")) for m in mods]
    left, top, plot_w, bar_h, gap, row_gap = 96, 76, 440, 9, 2, 9
    row_h = 2 * bar_h + gap + row_gap
    h = top + len(rows) * row_h + 34
    w = left + plot_w + 70
    out = ['<text class="title" x="16" y="26">Blind modulation ID, top-1 accuracy</text>',
           f'<text class="sub" x="16" y="44">synthetic, '
           f'{escape(d["provenance"]["git_commit"])}; one seed per case</text>']
    # legend (>= 2 series)
    for i, (cls, name) in enumerate((("s1", "CFO 0.004"), ("s2", "CFO 0 (stress)"))):
        lx = 16 + i * 150
        out.append(f'<rect class="{cls}" x="{lx}" y="54" width="12" height="9" rx="2"/>'
                   f'<text class="lab" x="{lx + 18}" y="63">{name}</text>')
    bottom = top + len(rows) * row_h - row_gap
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        gx = left + frac * plot_w
        out.append(f'<line class="{"axis" if frac == 0 else "grid"}" '
                   f'x1="{gx}" y1="{top - 4}" x2="{gx}" y2="{bottom + 4}"/>'
                   f'<text class="tick" x="{gx}" y="{bottom + 20}" '
                   f'text-anchor="middle">{int(frac * 100)}%</text>')
    for i, (m, a, b) in enumerate(rows):
        y = top + i * row_h
        out.append(f'<text class="lab" x="{left - 8}" y="{y + bar_h + 5}" '
                   f'text-anchor="end">{escape(m)}</text>')
        for j, (v, cls, tag) in enumerate(((a, "s1", "CFO 0.004"),
                                           (b, "s2", "CFO 0"))):
            if v is None:
                continue
            by = y + j * (bar_h + gap)
            out.append(_bar(left, by, v * plot_w, bar_h, cls,
                            f"{m}, {tag}: {v:.0%}"))
            if v < 1:                       # label only the misses
                out.append(f'<text class="val" x="{left + v * plot_w + 6}" '
                           f'y="{by + bar_h - 1}">{v:.0%}</text>')
    return _svg(w, h, "".join(out), "Blind modulation ID top-1 accuracy per "
                "modulation at CFO 0.004 and CFO 0")


def bitlayer_chart(d):
    # one cell per (interleaver, FEC) at the highest SNR, averaged over seeds
    top_snr = max(c["snr_db"] for c in d["cases"])
    cases = [c for c in d["cases"] if c["snr_db"] == top_snr]
    ils = list(dict.fromkeys(c["interleaver"] for c in cases))
    fecs = list(dict.fromkeys(c["fec"] for c in cases))
    cell = {}
    for (il, f) in {(c["interleaver"], c["fec"]) for c in cases}:
        cs = [c for c in cases if (c["interleaver"], c["fec"]) == (il, f)]
        cell[(il, f)] = {
            "frames_recovered": sum(c["frames_recovered"] for c in cs) / len(cs),
            "crc_pass": sum(c["crc_pass"] for c in cs) / len(cs) >= 0.5,
            "crc_rate": sum(c["crc_pass"] for c in cs) / len(cs),
            "interleaver_correct": all(c["interleaver_correct"] for c in cs),
            "fec_correct": all(c["fec_correct"] for c in cs), "n": len(cs)}
    left, top, cw, ch, g = 118, 92, 104, 46, 2
    w = left + len(fecs) * (cw + g) + 16
    h = top + len(ils) * (ch + g) + 24
    out = ['<text class="title" x="16" y="26">Bit layer: frames recovered '
           '(interleaver x FEC)</text>',
           f'<text class="sub" x="16" y="44">synthetic, PN9-whitened QPSK, '
           f'{top_snr} dB, '
           f'{escape(d["provenance"]["git_commit"])}; darker = more frames</text>']
    for j, f in enumerate(fecs):
        out.append(f'<text class="lab" x="{left + j * (cw + g) + cw / 2}" '
                   f'y="{top - 10}" text-anchor="middle">{escape(f)}</text>')
    for i, il in enumerate(ils):
        y = top + i * (ch + g)
        out.append(f'<text class="lab" x="{left - 8}" y="{y + ch / 2 + 4}" '
                   f'text-anchor="end">{escape(il)}</text>')
        for j, f in enumerate(fecs):
            c = cell.get((il, f))
            if not c:
                continue
            v = c["frames_recovered"]
            fill = RAMP[min(len(RAMP) - 1, int(round(v * (len(RAMP) - 1))))]
            txt = "cell-dark" if v >= 0.5 else "cell-light"
            line1 = "CRC ok" if c["crc_pass"] else "no frames"
            line2 = f"{v:.0%} frames" if c["crc_pass"] else ""
            x = left + j * (cw + g)
            tip = (f"{il} x {f}: CRC pass in {c['crc_rate']:.0%} of "
                   f"{c['n']} runs, {v:.0%} of frames, interleaver ID "
                   f"{'right' if c['interleaver_correct'] else 'wrong'}, FEC ID "
                   f"{'right' if c['fec_correct'] else 'wrong'}")
            y1 = y + 20 if line2 else y + ch / 2 + 4      # centre one line
            out.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" '
                       f'fill="{fill}"><title>{escape(tip)}</title></rect>'
                       f'<text class="{txt}" x="{x + cw / 2}" y="{y1}" '
                       f'text-anchor="middle">{line1}</text>'
                       f'<text class="{txt}" x="{x + cw / 2}" y="{y + 35}" '
                       f'text-anchor="middle">{line2}</text>')
    return _svg(w, h, "".join(out), "Bit-layer frames recovered per "
                "interleaver and FEC family")


def snr_chart(d):
    curves = d["snr_curves"]
    fams = [f for f in ("PSK", "QAM", "APSK", "ASK", "FSK") if f in curves]
    snrs = sorted({p["snr_db"] for f in fams for p in curves[f]})
    left, top, pw, ph = 56, 70, 460, 240
    w, h = left + pw + 70, top + ph + 48
    lo, hi = snrs[0], snrs[-1]
    X = lambda s: left + (s - lo) / max(1, hi - lo) * pw
    Y = lambda a: top + (1 - a) * ph
    out = ['<text class="title" x="16" y="26">Blind modulation ID vs SNR</text>',
           f'<text class="sub" x="16" y="44">synthetic, top-1 accuracy per '
           f'family, {escape(d["provenance"]["git_commit"])}; hover a point '
           f'for its 95% interval</text>']
    for a in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line class="{"axis" if a == 0 else "grid"}" x1="{left}" '
                   f'y1="{Y(a)}" x2="{left + pw}" y2="{Y(a)}"/>'
                   f'<text class="tick" x="{left - 8}" y="{Y(a) + 4}" '
                   f'text-anchor="end">{int(a * 100)}%</text>')
    for s in snrs:
        out.append(f'<text class="tick" x="{X(s)}" y="{top + ph + 18}" '
                   f'text-anchor="middle">{s}</text>')
    out.append(f'<text class="tick" x="{left + pw / 2}" y="{top + ph + 36}" '
               f'text-anchor="middle">SNR (dB)</text>')
    for i, f in enumerate(fams, start=1):
        pts = curves[f]
        path = " ".join(f"{'M' if j == 0 else 'L'}{X(p['snr_db']):.1f},"
                        f"{Y(p['top1_accuracy']):.1f}" for j, p in enumerate(pts))
        out.append(f'<path class="line l{i}" d="{path}"/>')
        for p in pts:
            ci = p["top1_ci95"]
            out.append(f'<circle class="s{i} dot" cx="{X(p["snr_db"]):.1f}" '
                       f'cy="{Y(p["top1_accuracy"]):.1f}" r="4"><title>'
                       f'{f} at {p["snr_db"]} dB: {p["top1_accuracy"]:.0%} '
                       f'(95% CI {ci[0]:.0%}-{ci[1]:.0%}, n={p["n"]})'
                       f'</title></circle>')
        end = pts[-1]                       # direct label (contrast relief)
        out.append(f'<text class="lab" x="{X(end["snr_db"]) + 8}" '
                   f'y="{Y(end["top1_accuracy"]) + 4 + (i - 3) * 11}">{f}</text>')
    for i, f in enumerate(fams, start=1):   # legend
        lx = 16 + (i - 1) * 90
        out.append(f'<rect class="s{i}" x="{lx}" y="52" width="12" height="9" rx="2"/>'
                   f'<text class="lab" x="{lx + 18}" y="61">{f}</text>')
    return _svg(w, h, "".join(out), "Blind modulation ID accuracy versus SNR "
                "per modulation family")


def main():
    os.makedirs(OUT, exist_ok=True)
    written = []
    for name, fn, src in (("modulation_top1", modulation_chart,
                           "validation_report"),
                          ("snr_curves", snr_chart, "validation_report"),
                          ("bitlayer_grid", bitlayer_chart, "bitlayer_report")):
        p = os.path.join(RESULTS, src + ".json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            data = json.load(f)
        if name == "snr_curves" and "snr_curves" not in data:
            continue
        svg = fn(data)
        path = os.path.join(OUT, name + ".svg")
        with open(path, "w") as f:
            f.write(svg)
        written.append(path)
    print("\n".join(written) or "no results to plot")


if __name__ == "__main__":
    main()
