"""Figure 4: qualitative comparison composite (PIL only).

Source grids are 3200 x 1024 PNGs: 10 frames (320 px wide each) x 4 rows (256 px tall each).
Row order (results/qualitative/README.md):
  dmd6a_qual/*_left.png : GT / teacher25 / student3 / student1
  t3r_qual/*_left.png   : GT / teacher25 / T3r3 / T3r1   (T3r3 = training-free 3-step re-noising teacher)

Composite layout:
  rows    = GT, Teacher 25 steps, Teacher 3 steps (no training), DMD student 3 steps, DMD student 1 step
  columns = frames 1, 4, 7, 10 (grid indices 0, 3, 6, 9)
  blocks  = RGB (left) and depth (right), same rows.

Each source tile carries a small yellow text label in its top-left corner ("GT t=0", "student3 t=0", ...).
We remove it by cropping the top LABEL_CROP px of every tile (identically for all tiles), so the composite
does not show two sets of conflicting labels.
"""
import os
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
QUAL = os.path.normpath(os.path.join(HERE, "..", "..", "repo", "results", "qualitative"))
OUT = os.path.join(HERE, "fig4_qualitative.png")

TILE_W, TILE_H = 320, 256          # native tile size in the source grids
LABEL_CROP = 14                    # px removed from the top of every tile (embedded text label)
FRAME_IDX = [0, 3, 6, 9]           # grid column indices -> frames 1, 4, 7, 10 (1-indexed)
FRAME_NAMES = ["t=1", "t=4", "t=7", "t=10"]
SCALE = 0.85                       # tile downscale so that the composite is ~2600 px wide

# (row label lines, source folder, source row index)
ROWS = [
    (["Ground truth"],                         "dmd6a_qual", 0),
    (["Teacher", "25 steps"],                  "dmd6a_qual", 1),
    (["Teacher", "3 steps", "(no training)"],  "t3r_qual",   2),
    (["DMD student", "3 steps"],               "dmd6a_qual", 2),
    (["DMD student", "1 step"],                "dmd6a_qual", 3),
]
BLOCKS = [("RGB", "rgb_left.png"), ("Depth (brighter = farther)", "depth_left.png")]

DPI = 200
BG = (255, 255, 255)
FG = (0, 0, 0)


def load_font(size):
    candidates = [
        ("/System/Library/Fonts/Helvetica.ttc", 0),
        ("/System/Library/Fonts/HelveticaNeue.ttc", 0),
        ("/Library/Fonts/Arial.ttf", 0),
        ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 0),
    ]
    for path, idx in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size, index=idx)
            except Exception:
                continue
    return ImageFont.load_default()


def text_size(draw, text, font):
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def main():
    grids = {}
    for folder in {"dmd6a_qual", "t3r_qual"}:
        for _, fname in BLOCKS:
            p = os.path.join(QUAL, folder, fname)
            im = Image.open(p).convert("RGB")
            assert im.size == (3200, 1024), (p, im.size)
            grids[(folder, fname)] = im

    tw = int(round(TILE_W * SCALE))
    th = int(round((TILE_H - LABEL_CROP) * SCALE))

    def tile(folder, fname, row, col):
        im = grids[(folder, fname)]
        x0, y0 = col * TILE_W, row * TILE_H + LABEL_CROP
        return im.crop((x0, y0, x0 + TILE_W, y0 + TILE_H - LABEL_CROP)).resize((tw, th), Image.LANCZOS)

    # typography (px at 200 dpi; 9 pt at 200 dpi = 25 px, we stay well above)
    f_row = load_font(44)
    f_col = load_font(42)
    f_blk = load_font(48)

    probe = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    row_line_h = int(text_size(probe, "Hg", f_row)[1] * 1.35)
    max_label_w = max(text_size(probe, line, f_row)[0] for lines, _, _ in ROWS for line in lines)

    margin = 24
    gap = 8                     # between tiles
    block_gap = 56              # between RGB and depth blocks
    label_col_w = max_label_w + 40
    blk_h = text_size(probe, "Hg", f_blk)[1]
    col_h = text_size(probe, "Hg", f_col)[1]
    header_h = blk_h + 18 + col_h + 16

    block_w = 4 * tw + 3 * gap
    W = margin + label_col_w + block_w + block_gap + block_w + margin
    H = margin + header_h + 5 * th + 4 * gap + margin

    canvas = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(canvas)

    block_x = [margin + label_col_w, margin + label_col_w + block_w + block_gap]
    grid_y = margin + header_h

    # headers
    for bi, (bname, _) in enumerate(BLOCKS):
        bw, _ = text_size(draw, bname, f_blk)
        draw.text((block_x[bi] + (block_w - bw) // 2, margin), bname, fill=FG, font=f_blk)
        for ci, cname in enumerate(FRAME_NAMES):
            cw, _ = text_size(draw, cname, f_col)
            cx = block_x[bi] + ci * (tw + gap) + (tw - cw) // 2
            draw.text((cx, margin + blk_h + 18), cname, fill=FG, font=f_col)

    # rows
    for ri, (lines, folder, srow) in enumerate(ROWS):
        y = grid_y + ri * (th + gap)
        # label (vertically centred, left aligned within the label column)
        total_h = len(lines) * row_line_h
        ty = y + (th - total_h) // 2
        for li, line in enumerate(lines):
            draw.text((margin, ty + li * row_line_h), line, fill=FG, font=f_row)
        for bi, (_, fname) in enumerate(BLOCKS):
            for ci, fidx in enumerate(FRAME_IDX):
                x = block_x[bi] + ci * (tw + gap)
                canvas.paste(tile(folder, fname, srow, fidx), (x, y))

    # thin separator between the two blocks and between label column and grid (visual only)
    sep_x = block_x[1] - block_gap // 2
    draw.line([(sep_x, grid_y), (sep_x, grid_y + 5 * th + 4 * gap)], fill=(160, 160, 160), width=2)

    canvas.save(OUT, dpi=(DPI, DPI))
    print("saved", OUT, canvas.size, "tile", (tw, th), "label col", label_col_w)


if __name__ == "__main__":
    main()
