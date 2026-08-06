"""
03_visualization_fig1_10.py
===========================
Generate publication figures 1-10 for the Migratory Bird-Wind Farm Ecology project.

Figures: 1(global flyways), 2(radar coverage), 3a-d(direction roses+exposure),
         4(data readiness), 5(trade-off scatter), 6(top20 farms),
         7(budget sensitivity), 8(by country), 9(regional comparison),
         10(BIRDBASE comparison)

Run: python 03_visualization_fig1_10.py
Requires: processed CSVs in our_work/data/processed/
"""

import os as _os, sys as _sys
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _SCRIPT_DIR
while not _os.path.exists(_os.path.join(REPO_ROOT, 'pyproject.toml')):
    parent = _os.path.dirname(REPO_ROOT)
    if parent == REPO_ROOT:
        REPPO_ROOT = _os.path.dirname(_SCRIPT_DIR)
        break
    REPO_ROOT = parent
ENG_REPO = _os.path.join(REPO_ROOT, '..', 'wind-direction-to-electricity-transition-main')
FIG_DIR = _os.path.join(REPO_ROOT, 'our_work', 'figures')
PROC_DATA = _os.path.join(REPO_ROOT, 'our_work', 'data', 'processed')
_os.makedirs(FIG_DIR, exist_ok=True)
import sys,io,os,math
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
import pandas as pd,numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch,Circle
from matplotlib.lines import Line2D

for fp in fm.findSystemFonts():
    if 'simhei' in fp.lower(): fm.fontManager.addfont(fp)
plt.rcParams['font.sans-serif']=['SimHei','DejaVu Sans']; plt.rcParams['font.family']='sans-serif'; plt.rcParams['axes.unicode_minus']=False

out_dir=r'FIG_DIR'
os.makedirs(out_dir,exist_ok=True)

# Load data once
td=pd.read_csv(r'os.path.join(PROC_DATA, 'tradeoff_all_171.csv')')
groups=pd.read_csv(r'os.path.join(PROC_DATA, 'farm_groups_updated.csv')')
summary=pd.read_csv(r'os.path.join(PROC_DATA, 'all_171_risk_summary.csv')')

# ============================================================
# FIG 3a: nlhrw Direction Rose (standalone, clean)
# ============================================================
fig,ax=plt.subplots(figsize=(6,6))
for r in [0.3,0.6,0.9]:
    ax.add_patch(plt.Circle((0,0),r,fill=False,color='#ddd',lw=0.5))
for season,angle,clr in [('Spring',32.6,'#E74C3C'),('Autumn',233.2,'#3498DB')]:
    spread=45;spread_rad=np.linspace(math.radians(angle-spread),math.radians(angle+spread),50);rv=0.85
    ax.fill_between([0]+list(np.sin(spread_rad)*rv),[0]+list(np.cos(spread_rad)*rv),[0]+[0]*len(spread_rad),color=clr,alpha=0.15)
    ax.plot(np.sin(spread_rad)*rv,np.cos(spread_rad)*rv,color=clr,lw=2.5)
    ax.annotate('',xy=(math.sin(math.radians(angle))*rv*1.15,math.cos(math.radians(angle))*rv*1.15),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=clr,lw=4))
    ax.text(math.sin(math.radians(angle))*1.6,math.cos(math.radians(angle))*1.6,season,fontsize=11,ha='center',color=clr,fontweight='bold')
ax.text(0,1.8,'N',ha='center',fontsize=14,fontweight='bold'); ax.text(0,-1.8,'S',ha='center',fontsize=10,color='gray')
ax.text(1.8,0,'E',ha='center',fontsize=10,color='gray'); ax.text(-1.8,0,'W',ha='center',fontsize=10,color='gray')
ax.text(0,0,'nlhrw\nDen Helder, NL\nn=14,258',ha='center',va='center',fontsize=9,fontweight='bold')
ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_aspect('equal');ax.axis('off')
ax.set_title('nlhrw Radar (Den Helder, Netherlands)\nBird Migration Direction Signatures',fontsize=12,fontweight='bold')
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig3a_nlhrw_rose.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 3b: deess Rose (standalone, small + warning label)
# ============================================================
fig,ax=plt.subplots(figsize=(6,6))
for r in [0.3,0.6,0.9]:
    ax.add_patch(plt.Circle((0,0),r,fill=False,color='#ddd',lw=0.5))
angle,clr=21.0,'#E74C3C'
spread=40;sr=np.linspace(math.radians(angle-spread),math.radians(angle+spread),50);rv=0.55
ax.fill_between([0]+list(np.sin(sr)*rv),[0]+list(np.cos(sr)*rv),[0]+[0]*len(sr),color=clr,alpha=0.15)
ax.plot(np.sin(sr)*rv,np.cos(sr)*rv,color=clr,lw=2.5)
ax.annotate('',xy=(math.sin(math.radians(angle))*rv*1.15,math.cos(math.radians(angle))*rv*1.15),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=clr,lw=4))
ax.text(math.sin(math.radians(angle))*1.6,math.cos(math.radians(angle))*1.6,'Spring',fontsize=11,ha='center',color=clr,fontweight='bold')
ax.text(0,1.8,'N',ha='center',fontsize=14,fontweight='bold')
ax.text(0,0,'deess\nEssen, DE\nn=16',ha='center',va='center',fontsize=9,fontweight='bold')
ax.text(0,-1.5,'Small sample!',ha='center',fontsize=10,color='#E74C3C',fontweight='bold',style='italic')
ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_aspect('equal');ax.axis('off')
ax.set_title('deess Radar (Essen, Germany)\nLimited sample (16 windows)',fontsize=12,fontweight='bold')
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig3b_deess_rose.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 3c: bejab Rose (new station, standalone)
# ============================================================
fig,ax=plt.subplots(figsize=(6,6))
for r in [0.3,0.6,0.9]:
    ax.add_patch(plt.Circle((0,0),r,fill=False,color='#ddd',lw=0.5))
for season,angle,clr in [('Spring',25.7,'#E74C3C'),('Autumn',217.2,'#3498DB')]:
    spread=45;sr=np.linspace(math.radians(angle-spread),math.radians(angle+spread),50);rv=0.85
    ax.fill_between([0]+list(np.sin(sr)*rv),[0]+list(np.cos(sr)*rv),[0]+[0]*len(sr),color=clr,alpha=0.15)
    ax.plot(np.sin(sr)*rv,np.cos(sr)*rv,color=clr,lw=2.5)
    ax.annotate('',xy=(math.sin(math.radians(angle))*rv*1.15,math.cos(math.radians(angle))*rv*1.15),xytext=(0,0),arrowprops=dict(arrowstyle='->',color=clr,lw=4))
    ax.text(math.sin(math.radians(angle))*1.6,math.cos(math.radians(angle))*1.6,season,fontsize=11,ha='center',color=clr,fontweight='bold')
ax.text(0,1.8,'N',ha='center',fontsize=14,fontweight='bold');ax.text(0,-1.8,'S',ha='center',fontsize=10,color='gray')
ax.text(0,0,'bejab\nJabbeke, BE\nn=13,866',ha='center',va='center',fontsize=9,fontweight='bold')
ax.set_xlim(-2,2);ax.set_ylim(-2,2);ax.set_aspect('equal');ax.axis('off')
ax.set_title('bejab Radar (Jabbeke, Belgium)\nBird Migration Direction Signatures',fontsize=12,fontweight='bold')
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig3c_bejab_rose.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 3d: Exposure Curve (standalone, clean labels)
# ============================================================
fig,ax=plt.subplots(figsize=(8,5))
theta=np.linspace(0,180,500)
# Spring (birds heading 32.6deg)
d1=np.minimum(np.abs(theta-32.6),np.minimum(np.abs(theta-212.6),np.abs(theta+147.4)))
e1=np.sin(np.radians(d1))**2
# Autumn (birds heading 233.2 = 53.2 axial)
d2=np.minimum(np.abs(theta-53.2),np.minimum(np.abs(theta-233.2),np.abs(theta+126.8)))
e2=np.sin(np.radians(d2))**2
# Average
e_avg=(e1+e2)/2

ax.plot(theta,e1,color='#E74C3C',lw=3,label='Spring (32.6deg NNE)')
ax.plot(theta,e2,color='#3498DB',lw=3,label='Autumn (233.2deg SW)')
ax.plot(theta,e_avg,color='#2C3E50',lw=2,linestyle='--',alpha=0.6,label='Annual average')
ax.axvline(x=33,color='#27AE60',linestyle='--',alpha=0.4,lw=2)
ax.axvline(x=53,color='#27AE60',linestyle='--',alpha=0.4,lw=2)
ax.axvline(x=123,color='#E74C3C',linestyle=':',alpha=0.4,lw=2)
ax.fill_between([33,53],0,1,color='#27AE60',alpha=0.08)
ax.annotate('Best: 33-53deg',xy=(43,0.05),fontsize=11,ha='center',color='#27AE60',fontweight='bold')
ax.annotate('Worst: 123deg',xy=(123,0.95),fontsize=11,ha='center',color='#E74C3C',fontweight='bold')
ax.set_xlabel('Farm Row Orientation (degrees)',fontsize=11)
ax.set_ylabel('Relative Exposure',fontsize=11)
ax.set_title('Exposure Curve (nlhrw, spring+autumn)',fontsize=13,fontweight='bold')
ax.set_xlim(0,180);ax.set_ylim(0,1.05)
ax.legend(fontsize=10);ax.grid(alpha=0.2)
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig3d_exposure_curve.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 5: Trade-off Scatter (clean, no textbox, just legend+labels)
# ============================================================
b01=td[td['budget']==0.01].copy()
radar=b01[b01['group']=='Europe (direction data)']
era5=b01[b01['group']!='Europe (direction data)']

fig,ax=plt.subplots(figsize=(10,7))
ax.scatter(era5['aep_cost_pct'],era5['risk_reduction']*100,c='#D4A017',s=30,alpha=0.5,label='ERA5 wind proxy (130 farms)')
ax.scatter(radar['aep_cost_pct'],radar['risk_reduction']*100,c='#27AE60',s=55,alpha=0.85,edgecolors='#333',linewidth=0.3,label='Radar-measured (41 farms)')
for _,r in radar.nlargest(5,'risk_reduction').iterrows():
    ax.annotate(f"F{int(r['farm_id'])}",(r['aep_cost_pct'],r['risk_reduction']*100),fontsize=7,ha='left',xytext=(5,3),textcoords='offset points')
ax.axhline(y=50,color='#555',linestyle=':',alpha=0.4);ax.axvline(x=1.0,color='#555',linestyle=':',alpha=0.4)
ax.fill_between([0,1.0],50,100,color='#27AE60',alpha=0.04)
ax.set_xlabel('AEP Cost (%)',fontsize=11);ax.set_ylabel('Risk Reduction (%)',fontsize=11)
ax.set_title('Energy-Ecology Trade-off (1% AEP Budget)',fontsize=13,fontweight='bold')
ax.set_xlim(0,1.05);ax.set_ylim(-5,105)
ax.legend(fontsize=10);ax.grid(alpha=0.2)
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig5_tradeoff_scatter.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 6: Top Radar Farms (horizontal bar, clean)
# ============================================================
fig,ax=plt.subplots(figsize=(10,8))
top20=radar.nlargest(20,'risk_reduction').sort_values('risk_reduction')
ylabels=[f'Farm {int(r["farm_id"])} ({r["country"]})' for _,r in top20.iterrows()]
bars=ax.barh(range(20),top20['risk_reduction']*100,color='#27AE60',alpha=0.85)
for i,(_,r) in enumerate(top20.iterrows()):
    ax.text(r['risk_reduction']*100+1,i,f'-{r["aep_cost_pct"]:.2f}%',fontsize=8,va='center')
ax.set_yticks(range(20));ax.set_yticklabels(ylabels,fontsize=9)
ax.set_xlabel('Risk Reduction (%)',fontsize=11)
ax.set_title('Top 20 Radar-Measured Farms (1% AEP Budget)',fontsize=13,fontweight='bold')
ax.set_xlim(0,105);ax.grid(axis='x',alpha=0.2);ax.axvline(x=50,color='#555',linestyle=':',alpha=0.4)
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig6_radar_top20.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 7: Budget Sensitivity (clean)
# ============================================================
fig,ax=plt.subplots(figsize=(8,5))
budgets=[0.005,0.01,0.02,0.05]
for grp,clr,lb in [('Europe (direction data)','#27AE60','Radar (41 farms)'),
                     ('East Asia','#E74C3C','East Asia (88 farms, ERA5)'),
                     ('Europe (no direction data)','#F39C12','Europe no-data (37 farms, ERA5)')]:
    sub=td[td['group']==grp]
    means=[sub[sub['budget']==b]['risk_reduction'].mean()*100 for b in budgets]
    stds=[sub[sub['budget']==b]['risk_reduction'].std()*100 for b in budgets]
    ax.plot([b*100 for b in budgets],means,'o-',color=clr,lw=2.5,label=lb,markersize=8)
ax.set_xlabel('AEP Budget (%)',fontsize=11);ax.set_ylabel('Mean Risk Reduction (%)',fontsize=11)
ax.set_title('Diminishing Returns: Risk Reduction vs Budget',fontsize=13,fontweight='bold')
ax.legend(fontsize=10);ax.grid(alpha=0.2);ax.set_xlim(0.2,5.5);ax.set_ylim(0,105)
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig7_budget_sensitivity.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 8: By Country (clean horizontal bar)
# ============================================================
fig,ax=plt.subplots(figsize=(9,6))
cnt=radar.groupby('country').agg(n=('farm_id','nunique'),rr=('risk_reduction','mean'),cost=('aep_cost_pct','mean')).reset_index()
cnt=cnt.sort_values('rr')
ylab=[f'{r["country"]} ({int(r["n"])})' for _,r in cnt.iterrows()]
ax.barh(range(len(cnt)),cnt['rr']*100,color='#27AE60',alpha=0.85)
for i,(_,r) in enumerate(cnt.iterrows()):
    ax.text(r['rr']*100+1,i,f'-{r["cost"]:.3f}%',fontsize=8,va='center')
ax.set_yticks(range(len(cnt)));ax.set_yticklabels(ylab,fontsize=9)
ax.set_xlabel('Mean Risk Reduction (%)',fontsize=11)
ax.set_title('Radar-Measured Farms: Trade-off by Country',fontsize=13,fontweight='bold')
ax.grid(axis='x',alpha=0.2);ax.axvline(x=50,color='#555',linestyle=':',alpha=0.4)
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig8_radar_countries.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 9: Regional Comparison (3 panels side by side, clean)
# ============================================================
fig,axes=plt.subplots(1,3,figsize=(18,6))
for idx,(gn,clr,lb) in enumerate([('Europe (direction data)','#27AE60','EU Radar (41 farms)'),
                                    ('Europe (no direction data)','#E67E22','EU No-data (37 farms)'),
                                    ('East Asia','#E74C3C','East Asia (88 farms)')]):
    ax=axes[idx];sub=td[(td['group']==gn)&(td['budget']==0.01)]
    ax.scatter(sub['aep_cost_pct'],sub['risk_reduction']*100,c=clr,s=40,alpha=0.7,edgecolors='#333',linewidth=0.3)
    for c in sub['country'].unique():
        cs=sub[sub['country']==c]
        if len(cs)>=2:
            ax.annotate(c,(cs['aep_cost_pct'].mean(),cs['risk_reduction'].mean()*100),fontsize=7,ha='center',fontweight='bold')
    ax.axhline(y=50,color='#555',linestyle=':',alpha=0.4);ax.axvline(x=1.0,color='#555',linestyle=':',alpha=0.4)
    ax.fill_between([0,1.0],50,100,color=clr,alpha=0.04)
    ax.set_title(lb,fontsize=11,fontweight='bold');ax.set_xlabel('AEP Cost (%)',fontsize=10)
    ax.set_xlim(0,1.05);ax.set_ylim(-5,105);ax.grid(alpha=0.2)
axes[0].set_ylabel('Risk Reduction (%)',fontsize=10)
fig.suptitle('Trade-off by Region (1% AEP Budget)',fontsize=14,fontweight='bold')
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig9_regional.png'),dpi=200,bbox_inches='tight');plt.close()

# ============================================================
# FIG 10: Evidence Pyramid (clean)
# ============================================================
fig,ax=plt.subplots(figsize=(12,6))
ax.set_xlim(0,14);ax.set_ylim(0,7);ax.axis('off')
tiers=[
    (1.5,4.5,11,1.5,'TIER A — Radar-Measured (41 farms)','VPTS 200m nighttime obs | nlhrw+bejab 28,124 windows | <1% AEP for 84% risk reduction','#27AE60'),
    (1.5,2.5,11,1.7,'TIER B — ERA5 Wind Proxy (130 farms)','11yr reanalysis direction | 16 distinct best orientations East Asia | Screening only — NOT a scientific claim','#D4A017'),
    (1.5,0.8,11,1.4,'TIER C — Known Gaps','U1: no field-level overlap (CAFF 99.9%, AVISTEP 9 countries only) | U4: no behavioral validation | East Asia: no GPS','#95A5A6'),
]
for x,y,w,h,title,detail,clr in tiers:
    rect=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.1',facecolor=clr,edgecolor='white',alpha=0.9)
    ax.add_patch(rect)
    ax.text(x+w/2,y+h-0.2,title,ha='center',va='top',fontsize=13,fontweight='bold',color='white')
    ax.text(x+w/2,y+0.3,detail,ha='center',va='bottom',fontsize=9,color='white',alpha=0.9)

ax.set_title('Evidence Hierarchy & Scientific Claim Boundaries',fontsize=15,fontweight='bold',y=0.98)
fig.tight_layout();fig.savefig(os.path.join(out_dir,'fig10_evidence_pyramid.png'),dpi=200,bbox_inches='tight');plt.close()

print('Done! Generated:')
for f in sorted(os.listdir(out_dir)):
    if f.startswith('fig3') or f.startswith('fig5') or f.startswith('fig6') or f.startswith('fig7') or f.startswith('fig8') or f.startswith('fig9') or f.startswith('fig10'):
        print(f'  {f} ({os.path.getsize(os.path.join(out_dir,f))/1e3:.0f} KB)')
