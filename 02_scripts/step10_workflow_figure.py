"""
Step 10 — Methodology Workflow Diagram
=======================================
Generates fig_workflow.png: a top-to-bottom flowchart of the full pipeline.

Outputs saved to:
  - 03_outputs/step10/
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

matplotlib.rcParams.update({'font.family': 'sans-serif', 'font.size': 9})

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR  = os.path.join(BASE_DIR, '03_outputs', 'step10')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
C_DATA   = '#E3F2FD'   # light blue  — data sources
C_PROC   = '#E8F5E9'   # light green — processing steps
C_MODEL  = '#FFF3E0'   # light orange — modelling
C_OUT    = '#F3E5F5'   # light purple — outputs
C_BORDER = '#546E7A'   # dark grey border
C_ARROW  = '#37474F'

def box(ax, x, y, w, h, text, facecolor, fontsize=8.5, bold=False,
        subtext=None):
    """Draw a rounded rectangle with centred text (and optional subtext)."""
    rect = mpatches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle='round,pad=0.02',
        facecolor=facecolor, edgecolor=C_BORDER, linewidth=1.2,
        zorder=3
    )
    ax.add_patch(rect)
    weight = 'bold' if bold else 'normal'
    if subtext:
        ax.text(x, y + h * 0.18, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=4)
        ax.text(x, y - h * 0.22, subtext, ha='center', va='center',
                fontsize=fontsize - 1.5, color='#546E7A', zorder=4,
                style='italic')
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=C_ARROW, lw=1.4),
                zorder=5)

def bracket_arrow(ax, x_src, y_src, x_dst, y_dst, label=''):
    """Elbow arrow for side branches."""
    ax.annotate('', xy=(x_dst, y_dst), xytext=(x_src, y_src),
                arrowprops=dict(
                    arrowstyle='->',
                    color=C_ARROW, lw=1.2,
                    connectionstyle='arc3,rad=0.0'
                ), zorder=5)
    if label:
        mx = (x_src + x_dst) / 2
        my = (y_src + y_dst) / 2
        ax.text(mx + 0.02, my, label, ha='left', va='center',
                fontsize=7.5, color='#546E7A')

# ── Canvas ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 12))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')

# ── Layout constants ───────────────────────────────────────────────────────
cx   = 0.50   # main column centre x
BW   = 0.48   # main box width
BH   = 0.048  # main box height
SBW  = 0.24   # side box width
SBH  = 0.040  # side box height
GAP  = 0.030  # vertical gap between boxes

# y positions (top → bottom)
Y = {}
Y['proact']   = 0.955
Y['feat_eng'] = Y['proact']   - BH - GAP
Y['exclusion']= Y['feat_eng'] - BH - GAP
Y['split']    = Y['exclusion']- BH - GAP
Y['cv_label'] = Y['split']    - BH - GAP * 0.7   # section label
Y['cv_box']   = Y['cv_label'] - 0.005             # CV loop dashed border top

# Inside CV loop
Y['resamp']   = Y['cv_label'] - 0.058
Y['train_clf']= Y['resamp']   - SBH - GAP * 0.9
Y['optuna']   = Y['train_clf']- SBH - GAP * 0.9
Y['cv_metric']= Y['optuna']   - SBH - GAP * 0.9

Y['cv_bottom']= Y['cv_metric']- SBH * 0.8         # CV loop dashed border bottom

Y['retrain']  = Y['cv_bottom'] - GAP * 1.6
Y['test']     = Y['retrain']   - BH - GAP
Y['shap']     = Y['test']      - BH - GAP
Y['output']   = Y['shap']      - BH - GAP * 1.2

# ── 1. PRO-ACT Database ────────────────────────────────────────────────────
box(ax, cx, Y['proact'], BW, BH,
    'PRO-ACT Database',
    subtext='12,504 identifiers / 9,483 starting set\n16 clinical tables',
    facecolor=C_DATA, bold=True)

# ── 2. Feature Engineering ────────────────────────────────────────────────
arrow(ax, cx, Y['proact'] - BH/2, cx, Y['feat_eng'] + BH/2)
box(ax, cx, Y['feat_eng'], BW, BH,
    'Feature Engineering',
    subtext='23 ordinal/binary trial-record features',
    facecolor=C_PROC)

# side note — feature types
box(ax, 0.85, Y['feat_eng'], SBW, SBH,
    'Demographics · ALSFRS-R slopes\nFVC · BMI · Riluzole · Regions',
    facecolor='#ECEFF1', fontsize=7.5)
bracket_arrow(ax, cx + BW/2, Y['feat_eng'],
              0.85 - SBW/2, Y['feat_eng'])

# ── 3. Filtering & Exclusions ─────────────────────────────────────────────
arrow(ax, cx, Y['feat_eng'] - BH/2, cx, Y['exclusion'] + BH/2)
box(ax, cx, Y['exclusion'], BW, BH,
    'Filtering & Exclusions',
    subtext='Censoring · Missing predictors · Onset-site exclusions',
    facecolor=C_PROC)

# side note — result
box(ax, 0.85, Y['exclusion'], SBW, SBH,
    '1,502 patients remain\n(174 Short · 1,328 Standard)',
    facecolor='#ECEFF1', fontsize=7.5)
bracket_arrow(ax, cx + BW/2, Y['exclusion'],
              0.85 - SBW/2, Y['exclusion'])

# ── 4. Stratified Split ───────────────────────────────────────────────────
arrow(ax, cx, Y['exclusion'] - BH/2, cx, Y['split'] + BH/2)
box(ax, cx, Y['split'], BW, BH,
    'Stratified Train / Test Split  (80 / 20)',
    subtext='Train: 1,201 pts · Test: 301 pts  (sealed)\nMinMax fitted on training data only',
    facecolor=C_PROC)

# ── 5. CV Loop dashed border ──────────────────────────────────────────────
loop_left  = cx - BW/2 - 0.01
loop_right = cx + BW/2 + 0.01
loop_top   = Y['split'] - BH/2 - GAP * 0.3
loop_bot   = Y['cv_bottom']
loop_h     = loop_top - loop_bot
loop_w     = loop_right - loop_left

cv_rect = mpatches.FancyBboxPatch(
    (loop_left, loop_bot), loop_w, loop_h,
    boxstyle='round,pad=0.01',
    facecolor='#FAFAFA', edgecolor='#90A4AE',
    linewidth=1.2, linestyle='dashed', zorder=1
)
ax.add_patch(cv_rect)
ax.text(loop_left + 0.01, loop_top - 0.006,
        'Repeated Stratified 5-Fold CV  (×3 repeats)',
        fontsize=7.5, color='#546E7A', style='italic', va='top', zorder=4)

# Arrow into CV loop
arrow(ax, cx, Y['split'] - BH/2, cx, Y['resamp'] + SBH/2)

# Boxes inside CV loop
box(ax, cx, Y['resamp'], BW - 0.06, SBH,
    'Fold-Specific Preprocessing & Resampling',
    subtext='MinMax fitted within fold · 10 imbalance strategies',
    facecolor=C_PROC, fontsize=8)

arrow(ax, cx, Y['resamp'] - SBH/2, cx, Y['train_clf'] + SBH/2)
box(ax, cx, Y['train_clf'], BW - 0.06, SBH,
    '7 Classifiers  (KNN · NB · DT · RF · SVM · MLP · LightGBM)',
    facecolor=C_MODEL, fontsize=8)

arrow(ax, cx, Y['train_clf'] - SBH/2, cx, Y['optuna'] + SBH/2)
box(ax, cx, Y['optuna'], BW - 0.06, SBH,
    'Bayesian Hyperparameter Optimisation  (Optuna · 50 trials)',
    facecolor=C_MODEL, fontsize=8)

arrow(ax, cx, Y['optuna'] - SBH/2, cx, Y['cv_metric'] + SBH/2)
box(ax, cx, Y['cv_metric'], BW - 0.06, SBH,
    'CV Metric: ROC-AUC  →  Best config per classifier',
    facecolor=C_MODEL, fontsize=8)

# ── 6. Retrain on Full Training Set ───────────────────────────────────────
arrow(ax, cx, Y['cv_bottom'], cx, Y['retrain'] + BH/2)
box(ax, cx, Y['retrain'], BW, BH,
    'Retrain Best Config on Full Training Set',
    facecolor=C_PROC)

# ── 7. Held-Out Test Evaluation ───────────────────────────────────────────
arrow(ax, cx, Y['retrain'] - BH/2, cx, Y['test'] + BH/2)
box(ax, cx, Y['test'], BW, BH,
    'Held-Out Test Evaluation  (n = 301)',
    subtext='AUC · Balanced Accuracy · Sensitivity · Specificity · G-Mean',
    facecolor=C_MODEL, bold=False)

# ── 8. SHAP Interpretability ──────────────────────────────────────────────
arrow(ax, cx, Y['test'] - BH/2, cx, Y['shap'] + BH/2)
box(ax, cx, Y['shap'], BW, BH,
    'SHAP Interpretability',
    subtext='TreeSHAP (LightGBM)  ·  KernelSHAP (MLP)',
    facecolor=C_MODEL)

# ── 9. Outputs ────────────────────────────────────────────────────────────
arrow(ax, cx, Y['shap'] - BH/2, cx, Y['output'] + BH/2)
box(ax, cx, Y['output'], BW, BH,
    'Outputs',
    subtext='Performance tables · Confusion matrices · PR curves · SHAP plots',
    facecolor=C_OUT, bold=True)

# ── Legend ────────────────────────────────────────────────────────────────
legend_items = [
    mpatches.Patch(facecolor=C_DATA,   edgecolor=C_BORDER, label='Data source'),
    mpatches.Patch(facecolor=C_PROC,   edgecolor=C_BORDER, label='Processing / Filtering'),
    mpatches.Patch(facecolor=C_MODEL,  edgecolor=C_BORDER, label='Modelling / Evaluation'),
    mpatches.Patch(facecolor=C_OUT,    edgecolor=C_BORDER, label='Outputs'),
]
ax.legend(handles=legend_items, loc='upper center', fontsize=7.5,
          framealpha=0.9, edgecolor='#B0BEC5', ncol=2,
          bbox_to_anchor=(0.5, -0.005))

plt.tight_layout(pad=0.3)

path = os.path.join(OUT_DIR, 'fig_workflow.png')
fig.savefig(path, dpi=200, bbox_inches='tight')
print(f'Saved: {path}')

plt.close(fig)
print('Done.')
