# -*- coding: utf-8 -*-
"""
Re-embed regenerated figures into the report docx (replacing stale images).

Maps each docx media file (word/media/imageN.png) to the corresponding figure PNG
in our_work/figures/, converts RGBA -> RGB (composited on white), and rewrites the
docx zip in place. Only the listed images are touched; everything else is preserved
byte-for-byte.

Usage:
  python reembed_figures_docx.py [report.docx]

  Defaults to the authoritative report in our_work/reports/.
"""
import os
import sys
import shutil
import zipfile
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, '..', 'figures')
REPORTS = os.path.join(HERE, '..', 'reports')

REPORT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPORTS, 'Energy-Ecology Trade-off of Wind Farm Array Orientation Methods and Results.docx')

# media filename -> figure PNG (regenerated figures only; fig1/2/3 unchanged)
MAPPING = {
    'word/media/image4.png':  os.path.join(FIGS, 'fig5_tradeoff_scatter.png'),
    'word/media/image5.png':  os.path.join(FIGS, 'fig20_onshore_aep_sensitivity.png'),
    'word/media/image6.png':  os.path.join(FIGS, 'fig14_onshore_tradeoff_scatter.png'),
    'word/media/image7.png':  os.path.join(FIGS, 'fig15_onshore_distribution_sensitivity.png'),
    'word/media/image8.png':  os.path.join(FIGS, 'fig16_onshore_theta_comparison.png'),
    'word/media/image9.png':  os.path.join(FIGS, 'fig17_onshore_stratification.png'),
    'word/media/image10.png': os.path.join(FIGS, 'fig18_onshore_spatial_map.png'),
    'word/media/image11.png': os.path.join(FIGS, 'fig19_wake_efficiency_polar.png'),
    'word/media/image12.png': os.path.join(FIGS, 'fig21_conclusion_distribution.png'),
}


def to_rgb_png(src_path):
    """Load a figure and return RGB PNG bytes (composited on white if RGBA)."""
    im = Image.open(src_path)
    if im.mode == 'RGBA':
        bg = Image.new('RGB', im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[3])
        im = bg
    elif im.mode != 'RGB':
        im = im.convert('RGB')
    import io
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    return buf.getvalue()


def main():
    if not os.path.exists(REPORT):
        print(f'NOT FOUND: {REPORT}')
        sys.exit(1)

    # Preload replacement bytes (fail early if a figure is missing)
    replacements = {}
    for media, fig in MAPPING.items():
        if not os.path.exists(fig):
            print(f'MISSING FIGURE: {fig}')
            sys.exit(1)
        replacements[media] = to_rgb_png(fig)
        print(f'  {media} <- {os.path.basename(fig)} ({len(replacements[media])} bytes RGB)')

    tmp = REPORT + '.tmp'
    with zipfile.ZipFile(REPORT, 'r') as zin:
        with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                if item.filename in replacements:
                    data = replacements[item.filename]
                zout.writestr(item, data)

    shutil.move(tmp, REPORT)
    print(f'Re-embedded {len(replacements)} figures into:\n  {REPORT}')


if __name__ == '__main__':
    main()