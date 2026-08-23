"""Render paper sections (same markdown subset as assemble.py) to a single self-contained HTML (base64 images) for pasting into Google Docs."""
import re, os, glob, base64, html as H
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "stix"; plt.rcParams["font.family"] = "STIXGeneral"
from PIL import Image

P = os.path.dirname(os.path.abspath(__file__))
SEC = sorted(glob.glob(os.environ.get("SEC_DIR", f"{P}/sections") + "/S*.md"))
OUT = os.environ.get("OUT", f"{P}/paper.html")
MAXW = int(os.environ.get("MAXW", "1400"))   # downscale figures for clipboard size
TITLE = "Few-Step Distillation of Multi-View RGB-D World Models: Does Geometry Survive?"
AUTHOR = "Tae Yang Hong"; AFFIL = "NAIS Lab"; EMAIL = "hongtaeyang1231@gmail.com"

SYM = {r"\rho": "ρ", r"\Omega": "Ω", r"\omega": "ω", r"\gamma": "γ", r"\Gamma": "Γ", r"\kappa": "κ", r"\tau": "τ", r"\pi": "π", r"\xi": "ξ", r"\eta": "η", r"\zeta": "ζ", r"\psi": "ψ", r"\Psi": "Ψ", r"\Sigma": "Σ", r"\Theta": "Θ", r"\Lambda": "Λ", r"\Phi": "Φ", r"\nu": "ν", r"\chi": "χ", r"\iota": "ι", r"\upsilon": "υ", r"\Pi": "Π", r"\ell": "ℓ", r"\partial": "∂", r"\bar{z}": "z̄", r"\hat{x}": "x̂", r"\tilde{z}": "z̃", r"\sigma": "σ", r"\epsilon": "ε", r"\varepsilon": "ε", r"\theta": "θ", r"\phi": "φ", r"\lambda": "λ", r"\beta": "β", r"\alpha": "α",
       r"\mu": "μ", r"\delta": "δ", r"\Delta": "Δ", r"\nabla": "∇", r"\times": "×", r"\cdot": "·", r"\approx": "≈", r"\leq": "≤", r"\geq": "≥",
       r"\rightarrow": "→", r"\to": "→", r"\in": "∈", r"\sim": "~", r"\infty": "∞", r"\pm": "±", r"\mathbb{E}": "𝔼", r"\mathbb{R}": "ℝ",
       r"\ldots": "…", r"\dots": "…", r"\|": "‖", r"\langle": "⟨", r"\rangle": "⟩", r"\log": "log", r"\exp": "exp", r"\min": "min", r"\max": "max",
       r"\operatorname{median}": "median", r"\mathrm{median}": "median", r"\text{median}": "median", r"\circ": "∘", r"\star": "★", r"\ast": "∗",
       r"\hat": "^", r"\bar": "‾", r"\tilde": "~", r"\prime": "′", r"\partial": "∂", r"\sum": "Σ", r"\int": "∫", r"\sqrt": "√", r"\propto": "∝", r"\ne": "≠", r"\neq": "≠"}
SUB = str.maketrans("0123456789+-=()aeijklmnoprstuvx", "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
SUP = str.maketrans("0123456789+-=()n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾ⁿ")

def tex_to_text(t):
    t = t.strip()
    for k, v in sorted(SYM.items(), key=lambda kv: -len(kv[0])): t = t.replace(k, v)
    t = re.sub(r"\\(mathbf|mathrm|text|boldsymbol|mathcal|textrm|operatorname)\{([^{}]*)\}", r"\2", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    def sub(m):
        s = m.group(1); return s.translate(SUB) if all(c in "0123456789+-=()aeijklmnoprstuvx" for c in s) else "_" + s
    def sup(m):
        s = m.group(1); return s.translate(SUP) if all(c in "0123456789+-=()n" for c in s) else "^" + s
    t = re.sub(r"_\{([^{}]*)\}", sub, t); t = re.sub(r"_(\w)", sub, t)
    t = re.sub(r"\^\{([^{}]*)\}", sup, t); t = re.sub(r"\^(\w)", sup, t)
    t = re.sub(r"\\left|\\right|\\,|\\;|\\!|\\ ", "", t)
    return t.replace("{", "").replace("}", "").replace("\\", "")

def img_b64(path, maxw=MAXW):
    im = Image.open(path).convert("RGB")
    if im.width > maxw: im = im.resize((maxw, int(im.height * maxw / im.width)), Image.LANCZOS)
    import io; b = io.BytesIO(); im.save(b, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode(), im.width, im.height

def render_eq(tex, idx):
    path = f"{P}/figures/eqh_{idx}.png"
    try:
        fig = plt.figure(figsize=(0.1, 0.1)); fig.text(0, 0, f"${tex}$", fontsize=13)
        fig.savefig(path, dpi=220, bbox_inches="tight", pad_inches=0.04, facecolor="white"); plt.close(fig); return path
    except Exception as e:
        plt.close("all"); print(f"  [eq {idx}] mathtext failed: {str(e)[:80]}"); return None

def inline(text):
    text = text.replace('\\*', '\u2217')
    out, pos = [], 0
    for m in re.finditer(r"(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`|\$[^$]+\$)", text):
        out.append(H.escape(text[pos:m.start()])); tok = m.group(0)
        if tok.startswith("**"): out.append(f"<b>{H.escape(tok[2:-2])}</b>")
        elif tok.startswith("`"): out.append(f"<code>{H.escape(tok[1:-1])}</code>")
        elif tok.startswith("$"): out.append(f"<i>{H.escape(tex_to_text(tok[1:-1]))}</i>")
        else: out.append(f"<i>{H.escape(tok[1:-1])}</i>")
        pos = m.end()
    out.append(H.escape(text[pos:])); return "".join(out)

CSS_P = 'style="font-family:Times New Roman;font-size:10.5pt;text-align:justify;margin:0 0 6pt 0;line-height:1.15"'
CSS_CAP = 'style="font-family:Times New Roman;font-size:9.5pt;margin:2pt 0 10pt 0"'
body = [f'<p style="text-align:center;font-family:Times New Roman;font-size:16pt;font-weight:bold;margin:0 0 8pt 0">{H.escape(TITLE)}</p>',
        f'<p style="text-align:center;font-family:Times New Roman;font-size:11pt;margin:0 0 14pt 0"><b>{AUTHOR}</b><br>{AFFIL}<br>{EMAIL}</p>']
eq_idx = 0
for f in SEC:
    lines = open(f, encoding="utf-8").read().split("\n"); i = 0; pending = None
    is_ref = os.path.basename(f).startswith("S6")
    while i < len(lines):
        ln = lines[i].rstrip()
        if not ln.strip(): i += 1; continue
        if ln.startswith("# "):
            h = ln[2:].strip()
            if h.lower() == "abstract": body.append(f'<p style="text-align:center;font-family:Times New Roman;font-size:12pt;font-weight:bold;margin:10pt 0 4pt 0">Abstract</p>')
            else: body.append(f'<h1 style="font-family:Times New Roman;font-size:13pt;font-weight:bold;margin:14pt 0 4pt 0">{H.escape(h)}</h1>')
            i += 1; continue
        if ln.startswith("## "): body.append(f'<h2 style="font-family:Times New Roman;font-size:11.5pt;font-weight:bold;margin:10pt 0 4pt 0">{H.escape(ln[3:].strip())}</h2>'); i += 1; continue
        if ln.startswith("### "): body.append(f'<h3 style="font-family:Times New Roman;font-size:10.5pt;font-weight:bold;margin:8pt 0 3pt 0">{H.escape(ln[4:].strip())}</h3>'); i += 1; continue
        m = re.match(r"!\[(.*?)\]\((.*?)\)", ln.strip())
        if m:
            cap, path = m.group(1), os.path.join(P, m.group(2))
            if os.path.exists(path):
                src, w, h_ = img_b64(path); dw = 624; dh = int(h_ * dw / w)
                body.append(f'<p style="text-align:center;margin:6pt 0 2pt 0"><img src="{src}" width="{dw}" height="{dh}"></p>')
            else:
                body.append(f'<p style="color:red">[missing figure: {H.escape(m.group(2))}]</p>')
            mm = re.match(r"(Figure \d+:)\s*(.*)", cap)
            body.append(f'<p {CSS_CAP}><b>{H.escape(mm.group(1))}</b> {inline(mm.group(2))}</p>' if mm else f'<p {CSS_CAP}>{inline(cap)}</p>')
            i += 1; continue
        if re.match(r"^Table \d+:", ln.strip()) and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            pending = ln.strip(); i += 1; continue
        if ln.strip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.match(r"^:?-+:?$", c) for c in cells if c): rows.append(cells)
                i += 1
            if pending:
                mm = re.match(r"(Table \d+:)\s*(.*)", pending)
                body.append(f'<p style="text-align:center;font-family:Times New Roman;font-size:9.5pt;margin:8pt 0 3pt 0"><b>{H.escape(mm.group(1))}</b> {inline(mm.group(2))}</p>' if mm else f'<p {CSS_CAP}>{inline(pending)}</p>')
            ncol = max(len(r) for r in rows)
            maxlen = [max(len(re.sub(r"[*`$]", "", r[j])) if j < len(r) else 0 for r in rows) for j in range(ncol)]
            widths = [min(max(5.2 * ml + 14, 34), 230) for ml in maxlen]
            tot = sum(widths); limit = 620
            if tot > limit: widths = [w * limit / tot for w in widths]
            fs = 8.5 if tot <= limit else (8 if tot < 1.25 * limit else 7.5)
            t = [f'<table width="{int(sum(widths))}" style="border-collapse:collapse;margin:0 auto 8pt auto;font-family:Times New Roman;font-size:{fs}pt">']
            for ri, r in enumerate(rows):
                t.append("<tr>")
                for j in range(ncol):
                    c = r[j] if j < len(r) else ""
                    bt = "border-top:1.2pt solid #000;" if ri == 0 else ""
                    bb = "border-bottom:0.6pt solid #000;" if ri == 0 else ("border-bottom:1.2pt solid #000;" if ri == len(rows) - 1 else "")
                    al = "left" if j == 0 else "center"
                    content = f"<b>{inline(c)}</b>" if ri == 0 else f'<span style="font-weight:400">{inline(c)}</span>'
                    t.append(f'<td width="{int(widths[j])}" style="{bt}{bb}padding:2pt 4pt;text-align:{al};font-size:{fs}pt;font-weight:{"bold" if ri == 0 else "normal"}">{content}</td>')
                t.append("</tr>")
            t.append("</table>"); body.append("".join(t)); pending = None; continue
        if ln.strip().startswith("$$"):
            buf = ln.strip()
            while not re.search(r"\$\$\s*(\(\d+\))?\s*$", buf) and i + 1 < len(lines): i += 1; buf += " " + lines[i].strip()
            mm = re.match(r"\$\$(.*)\$\$\s*(\(\d+\))?\s*$", buf, re.S)
            tex, tag = (mm.group(1).strip(), mm.group(2) or "") if mm else (buf.strip("$ "), "")
            eq_idx += 1; img = render_eq(tex, eq_idx)
            if img:
                src, w, h_ = img_b64(img, 1600); sc = 0.36; dw = int(w * sc); dh = int(h_ * sc)
                if dw > 540: dh = int(dh * 540 / dw); dw = 540
                body.append(f'<table width="624" style="border-collapse:collapse;margin:2pt 0 6pt 0"><tr><td width="570" style="text-align:center;padding:0"><img src="{src}" width="{dw}" height="{dh}"></td><td width="54" style="text-align:right;padding:0;font-family:Times New Roman;font-size:10.5pt"><span style="font-weight:400">{tag}</span></td></tr></table>')
            else:
                body.append(f'<p style="text-align:center;font-family:Times New Roman;font-size:10.5pt"><i>{H.escape(tex_to_text(tex))}</i> &nbsp;&nbsp;{tag}</p>')
            i += 1; continue
        if ln.strip().startswith("```"):
            i += 1; code = []
            while i < len(lines) and not lines[i].strip().startswith("```"): code.append(lines[i]); i += 1
            i += 1; body.append(f'<pre style="font-family:Courier New;font-size:8.5pt;margin:4pt 0 8pt 20pt">{H.escape(chr(10).join(code))}</pre>'); continue
        if ln.strip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].strip().startswith("- "): items.append(f'<li style="font-family:Times New Roman;font-size:10.5pt;margin:0 0 3pt 0">{inline(lines[i].strip()[2:])}</li>'); i += 1
            body.append("<ul>" + "".join(items) + "</ul>"); continue
        buf = [ln.strip()]
        while i + 1 < len(lines) and lines[i + 1].strip() and not re.match(r"^(#|\||!\[|\$\$|```|- |Table \d+:)", lines[i + 1].strip()):
            i += 1; buf.append(lines[i].strip())
        text = " ".join(buf)
        if is_ref: body.append(f'<p style="font-family:Times New Roman;font-size:9.5pt;margin:0 0 3pt 0;padding-left:22pt;text-indent:-22pt">{inline(text)}</p>')
        else: body.append(f"<p {CSS_P}>{inline(text)}</p>")
        i += 1

html = "<html><head><meta charset=\"utf-8\"></head><body>" + "\n".join(body) + "</body></html>"
open(OUT, "w", encoding="utf-8").write(html)
print("saved", OUT, f"{len(html)/1e6:.1f} MB, equations {eq_idx}")
