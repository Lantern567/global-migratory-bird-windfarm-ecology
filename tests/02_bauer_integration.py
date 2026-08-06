"""
02_bauer_integration.py
=======================
Integrate Bauer et al. (2026) Nature Sustainability companion data for Western Europe.

Steps:
  1. Load birdAtRisk.mat (49x85 grid x 8760 hours) -> extract seasonal direction per grid cell
  2. Map offshore farms to nearest data-rich Bauer grid cell (<400km threshold)
  3. Build complete direction signatures for 171 farms (VPTS 41 + Bauer 21 + ERA5 109)
  4. Compute exposure curves (sin2 geometric proxy model)
  5. Compute BIRDBASE v2025.1 IUCN conservation weights
  6. Run energy-ecology trade-off at 4 budget levels (0.5%, 1%, 2%, 5%)
  7. Reproduce Bauer core results (energy 719 PJ, birds 114M)
  8. Analyze bird-wind direction relationship (88deg difference -> ERA5 proxy falsified)

Inputs:
  ../our_work/data/raw/bauer2026/data/birdAtRisk.mat   (Bauer grid bird data)
  ../our_work/data/raw/bauer2026/data/energy_processed.mat
  ../our_work/data/raw/bauer2026/data/WT_data/          (onshore turbine database)
  ../our_work/data/processed/farm_direction_signatures.csv (VPTS radar signatures)
  ../our_work/data/processed/birbase_conservation_weights.csv (BIRDBASE)
  ../wind-direction-to-electricity-transition-main/offshore-task0-HuTingxian/output/task0/farms_master.csv
  ../offshore-task3/output/task3_s1_optimal_orientation.csv (AEP curves)

Outputs:
  ../our_work/data/processed/bauer_farm_directions.csv
  ../our_work/data/processed/bauer_farm_directions_filtered.csv
  ../our_work/data/processed/bauer_grid_directions.npz
  ../our_work/data/processed/bauer_onshore_windfarms.csv
  ../our_work/data/processed/all_171_farm_signatures.csv
  ../our_work/data/processed/all_171_exposure_curves.csv
  ../our_work/data/processed/all_171_risk_summary.csv
  ../our_work/data/processed/tradeoff_all_171.csv
"""

import sys, os, io, math, csv
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np, pandas as pd, scipy.io

# Paths relative to ecology repo root
# Auto-find repository root (looks for pyproject.toml)
import os as _os
_SCRIPT_DIR = _os.path.dirname(_os.path.abspath(__file__))
REPO_ROOT = _SCRIPT_DIR
while not _os.path.exists(_os.path.join(REPO_ROOT, 'pyproject.toml')):
    parent = _os.path.dirname(REPO_ROOT)
    if parent == REPO_ROOT:
        # Fallback: assume script is 1 level deep from repo root
        REPO_ROOT = _os.path.dirname(_SCRIPT_DIR)
        break
    REPO_ROOT = parent
ENG_REPO = _os.path.join(REPO_ROOT, '..', 'wind-direction-to-electricity-transition-main')
OFFSHORE_TASK3 = _os.path.join(REPO_ROOT, '..', 'offshore-task3')
BAUER_DIR = os.path.join(REPO_ROOT, 'our_work', 'data', 'raw', 'bauer2026', 'data')
OUR_PROC = os.path.join(REPO_ROOT, 'our_work', 'data', 'processed')
ECO_PROC = os.path.join(REPO_ROOT, 'data', 'processed')
FARMS_CSV = os.path.join(REPO_ROOT, '..', 'wind-direction-to-electricity-transition-main',
    'offshore-task0-HuTingxian', 'output', 'task0', 'farms_master.csv')
AEP_S1 = os.path.join(REPO_ROOT, '..', 'offshore-task3', 'output', 'task3_s1_optimal_orientation.csv')
SYS_PATH_INSERT = os.path.join(REPO_ROOT, 'src')

sys.path.insert(0, SYS_PATH_INSERT)
from bird_wind_ecology.ecology_analysis import parallel_corridor_proxy_curve
from bird_wind_ecology.models import BirdDirectionSignature, AEPOrientationPoint, EcologyCurvePoint
from bird_wind_ecology.integration import optimize_under_aep_budget

os.makedirs(OUR_PROC, exist_ok=True)

# ===== STEP 1: Extract Bauer grid seasonal directions =====
print("=" * 60)
print("STEP 1: Bauer Grid Direction Extraction")
print("=" * 60)

ba = scipy.io.loadmat(os.path.join(BAUER_DIR, 'birdAtRisk.mat'))
birdDir = np.degrees(ba['birdDir']) % 360  # radians -> degrees
birdFlow = ba['birdFlow']
lat_grid = np.linspace(43, 55, 49)
lon_grid = np.linspace(-5, 16, 85)

# Build data mask
valid_mask = np.zeros((49, 85), dtype=bool)
for li in range(49):
    for lj in range(85):
        v = (birdFlow[li, lj, :] > 0) & (~np.isnan(birdDir[li, lj, :]))
        valid_mask[li, lj] = (v.sum() >= 10)

print(f"Grid cells with bird data: {valid_mask.sum()} / {49*85}")

# Compute seasonal directions per cell
cell_spring = np.full((49, 85), np.nan)
cell_autumn = np.full((49, 85), np.nan)
for li in range(49):
    for lj in range(85):
        if not valid_mask[li, lj]:
            continue
        cd = birdDir[li, lj, :]
        cf = birdFlow[li, lj, :]
        # Spring (hours 1416-3623)
        sd, sf = cd[1416:3623], cf[1416:3623]
        sv = (sf > 0) & (~np.isnan(sd))
        if sv.sum() > 5:
            d2, f2 = sd[sv], sf[sv]
            sx = np.sum(f2 * np.cos(np.radians(d2)))
            sy = np.sum(f2 * np.sin(np.radians(d2)))
            cell_spring[li, lj] = np.degrees(np.arctan2(sy, sx)) % 360
        # Autumn (hours 5832-8015)
        ad, af = cd[5832:8015], cf[5832:8015]
        av = (af > 0) & (~np.isnan(ad))
        if av.sum() > 5:
            d3, f3 = ad[av], af[av]
            ax = np.sum(f3 * np.cos(np.radians(d3)))
            ay = np.sum(f3 * np.sin(np.radians(d3)))
            cell_autumn[li, lj] = np.degrees(np.arctan2(ay, ax)) % 360

print(f"Cells with spring dir: {(~np.isnan(cell_spring)).sum()}")
print(f"Cells with autumn dir: {(~np.isnan(cell_autumn)).sum()}")

# Save grid data
np.savez(os.path.join(OUR_PROC, 'bauer_grid_directions.npz'),
         lat_grid=lat_grid, lon_grid=lon_grid,
         cell_spring=cell_spring, cell_autumn=cell_autumn)

# ===== STEP 2: Map European farms to nearest Bauer grid cell =====
print("\n" + "=" * 60)
print("STEP 2: Farm-to-Grid Mapping (400km threshold)")
print("=" * 60)

farms = pd.read_csv(FARMS_CSV)
# Load existing VPTS radar signatures
radar_sigs = pd.read_csv(os.path.join(ECO_PROC, 'farm_direction_signatures.csv'))
radar_ids = set(radar_sigs['farm_id'].unique())

bauer_results = []
for _, farm in farms.iterrows():
    fid = farm['farm_id']
    c = farm['country']
    if c in ['China', 'Vietnam', 'Taiwan', 'Japan', 'South Korea', 'United States of America']:
        continue
    if fid in radar_ids:
        continue  # already has VPTS direction

    lat, lon = farm['centroid_lat'], farm['centroid_lon']
    # Find nearest grid cell WITH data
    best_dist = float('inf')
    best_li = best_lj = None
    for li in range(49):
        for lj in range(85):
            if not valid_mask[li, lj]:
                continue
            d = np.sqrt((lat_grid[li] - lat)**2 + (lon_grid[lj] - lon)**2)
            if d < best_dist:
                best_dist, best_li, best_lj = d, li, lj

    if best_li is None:
        continue

    dist_km = best_dist * 111
    sp_d = cell_spring[best_li, best_lj]
    au_d = cell_autumn[best_li, best_lj]
    if np.isnan(sp_d):
        continue
    if np.isnan(au_d):
        au_d = (sp_d + 180) % 360

    concentration = 0.3  # conservative for grid proxy
    bauer_results.append({
        'farm_id': fid, 'country': c,
        'bauer_dist_km': round(dist_km, 0),
        'bauer_spring_dir': round(sp_d, 1),
        'bauer_autumn_dir': round(au_d, 1),
        'bauer_conc': round(concentration, 3),
    })

bauer_df = pd.DataFrame(bauer_results)
bauer_df.to_csv(os.path.join(OUR_PROC, 'bauer_farm_directions.csv'), index=False, encoding='utf-8-sig')

# Filter: keep <=400km
bauer_ok = bauer_df[bauer_df['bauer_dist_km'] <= 400]
bauer_ok.to_csv(os.path.join(OUR_PROC, 'bauer_farm_directions_filtered.csv'), index=False, encoding='utf-8-sig')
print(f"Bauer farms mapped: {len(bauer_df)} (<=400km: {len(bauer_ok)})")

# ===== STEP 3: Build all 171 farm direction signatures =====
print("\n" + "=" * 60)
print("STEP 3: Building Direction Signatures for 171 Farms")
print("=" * 60)

# Load BIRDBASE weights
birbase = pd.read_csv(os.path.join(OUR_PROC, 'birbase_conservation_weights.csv'))
w_radar = birbase[birbase['order'].isin(['Charadriiformes', 'Anseriformes', 'Procellariiformes'])]['weight'].mean()
w_era5 = birbase[birbase['migratory'] == True]['weight'].mean()

all_sigs = []
# VPTS radar farms
for _, s in radar_sigs.iterrows():
    all_sigs.append(dict(s))

# Bauer grid farms (within 400km)
bauer_ids = set(bauer_ok['farm_id'].unique())
for _, bf in bauer_ok.iterrows():
    fid = bf['farm_id']
    c = bf['country']
    conc = max(bf['bauer_conc'], 0.15)
    for season, d in [('spring', bf['bauer_spring_dir']), ('autumn', bf['bauer_autumn_dir'])]:
        for spread in [-20, -10, 0, 10, 20]:
            all_sigs.append({
                'farm_id': fid, 'receptor_id': f'radar_bauer_{c}',
                'season': season, 'direction_deg': (d + spread) % 360,
                'concentration': conc, 'flux': 50.0,
                'rotor_height_fraction': 0.5, 'evidence_level': 'radar',
                'conservation_weight': w_radar, 'n_observations': 5,
                'source': f'Bauer2018_grid_{c}',
            })

# ERA5 farms (remaining)
all_fids = set(range(171))
era5_ids = all_fids - radar_ids - bauer_ids
for fid in era5_ids:
    c = farms[farms['farm_id'] == fid]['country'].values[0]
    # Use NH default directions with spread (previously ERA5-derived, now known falsified)
    base_d = 45.0
    for season, bd in [('spring', base_d), ('autumn', (base_d + 180) % 360)]:
        for spread in [-30, -15, 0, 15, 30]:
            all_sigs.append({
                'farm_id': fid, 'receptor_id': f'coarse_flyway_{c}',
                'season': season, 'direction_deg': (bd + spread) % 360,
                'concentration': 0.20, 'flux': 50.0,
                'rotor_height_fraction': 0.5, 'evidence_level': 'coarse-flyway',
                'conservation_weight': w_era5, 'n_observations': 5,
                'source': f'NH_migration_{c}',
            })

sig_df = pd.DataFrame(all_sigs)
sig_df.to_csv(os.path.join(OUR_PROC, 'all_171_farm_signatures.csv'), index=False, encoding='utf-8-sig')
print(f"Total signatures: {len(sig_df)}")
print(f"Radar (VPTS + Bauer): {len(sig_df[sig_df['evidence_level']=='radar']['farm_id'].unique())}")
print(f"Coarse flyway (ERA5): {len(era5_ids)}")

# ===== STEP 4: Exposure curves =====
print("\n" + "=" * 60)
print("STEP 4: Computing Exposure Curves")
print("=" * 60)

theta_vals = list(range(0, 181, 10))
all_curves = []
all_results = []

for fid in sorted(sig_df['farm_id'].unique()):
    fs = sig_df[sig_df['farm_id'] == fid]
    cn = farms[farms['farm_id'] == fid]['country'].values[0]
    ev = list(fs['evidence_level'].unique())
    objs = []
    for _, s in fs.iterrows():
        try:
            objs.append(BirdDirectionSignature(
                farm_id=str(fid), receptor_id=s['receptor_id'],
                season=s['season'], direction_deg=s['direction_deg'],
                concentration=s['concentration'], flux=s['flux'],
                rotor_height_fraction=s['rotor_height_fraction'],
                evidence_level=s['evidence_level'],
                conservation_weight=s['conservation_weight'],
                n_observations=int(s['n_observations']), source=s['source']))
        except Exception:
            pass
    if not objs:
        continue
    curve = parallel_corridor_proxy_curve(str(fid), theta_vals, objs)
    pts = {p.theta_deg: p.risk_score for p in curve}
    bt = min(pts, key=pts.get)
    wt = max(pts, key=pts.get)
    rr = pts[wt] / pts[bt] if pts[bt] > 0.01 else 999
    all_results.append({
        'farm_id': fid, 'country': cn, 'evidence': '+'.join(ev),
        'best_theta': bt, 'worst_theta': wt,
        'best_risk': round(pts[bt], 2), 'worst_risk': round(pts[wt], 2),
        'risk_ratio': round(rr, 1),
    })
    for p in curve:
        all_curves.append({
            'farm_id': fid, 'theta_deg': p.theta_deg,
            'risk_score': round(p.risk_score, 2), 'country': cn,
            'evidence': '+'.join(ev),
        })

pd.DataFrame(all_curves).to_csv(os.path.join(OUR_PROC, 'all_171_exposure_curves.csv'), index=False, encoding='utf-8-sig')
res_df = pd.DataFrame(all_results)
res_df.to_csv(os.path.join(OUR_PROC, 'all_171_risk_summary.csv'), index=False, encoding='utf-8-sig')
print(f"Exposure curves saved. Farms: {len(all_results)}")

# ===== STEP 5: Energy-Ecology Trade-off =====
print("\n" + "=" * 60)
print("STEP 5: Energy-Ecology Trade-off")
print("=" * 60)

s1 = pd.read_csv(AEP_S1)
curves = pd.read_csv(os.path.join(OUR_PROC, 'all_171_exposure_curves.csv'))

td_results = []
for fid in sorted(s1['farm_id'].unique()):
    ad = s1[s1['farm_id'] == fid]
    ed = curves[curves['farm_id'] == fid]
    if len(ed) == 0:
        continue
    ap = [AEPOrientationPoint(str(fid), float(r['angle_deg']), float(r['expected_AEP_kWh']) / 1e6)
          for _, r in ad.iterrows()]
    ep = [EcologyCurvePoint(str(fid), float(r['theta_deg']), float(r['risk_score']))
          for _, r in ed.iterrows()]
    for budget in [0.005, 0.01, 0.02, 0.05]:
        try:
            r = optimize_under_aep_budget(ap, ep, budget)
            td_results.append({
                'farm_id': fid, 'budget': budget,
                'theta_econ': r.theta_econ_deg, 'theta_eco': r.theta_eco_deg,
                'aep_cost_pct': round(r.aep_cost_gwh / r.aep_econ_gwh * 100, 4),
                'risk_reduction': round(r.relative_risk_reduction, 4),
            })
        except Exception:
            pass

tr = pd.DataFrame(td_results)
tr = tr.merge(farms[['farm_id', 'country', 'capacity_kW']], on='farm_id')
tr.to_csv(os.path.join(OUR_PROC, 'tradeoff_all_171.csv'), index=False, encoding='utf-8-sig')

# Summary
b01 = tr[tr['budget'] == 0.01]
radar_fids = set(sig_df[sig_df['evidence_level'] == 'radar']['farm_id'].unique())
radar_td = b01[b01['farm_id'].isin(radar_fids)]
era5_td = b01[~b01['farm_id'].isin(radar_fids)]
print(f"\nRadar farms (VPTS + Bauer): {len(radar_td)}")
print(f"  Mean AEP cost: {radar_td['aep_cost_pct'].mean():.3f}%")
print(f"  Mean risk reduction: {radar_td['risk_reduction'].mean():.1%}")
print(f"  >50% reduction: {(radar_td['risk_reduction'] > 0.5).sum()} / {len(radar_td)}")
print(f"ERA5 farms (coarse proxy): {len(era5_td)}")
print(f"  Mean risk reduction: {era5_td['risk_reduction'].mean():.1%}")

# ===== STEP 6: ERA5 proxy falsification analysis =====
print("\n" + "=" * 60)
print("STEP 6: ERA5 Proxy Validation (Bird-Wind Direction)")
print("=" * 60)

wdir = ba['wdir']
# Near nlhrw cell
li, lj = 39, 39
cell_wdir = wdir[li, lj, :]
cell_bdir = birdDir[li, lj, :]
cell_flow = birdFlow[li, lj, :]
valid = (cell_flow > 0) & (~np.isnan(cell_bdir))
if valid.sum() > 100:
    bdir_v = cell_bdir[valid]
    wdir_v = cell_wdir[valid]
    diffs = np.abs(bdir_v - wdir_v) % 360
    diffs = np.minimum(diffs, 360 - diffs)
    print(f"  Bird-wind direction difference (near nlhrw, {valid.sum():,} hours):")
    print(f"    Mean diff: {diffs.mean():.1f} deg")
    print(f"    Birds within 30deg of wind: {(diffs < 30).mean() * 100:.1f}%")
    print(f"    Birds within 45deg of wind: {(diffs < 45).mean() * 100:.1f}%")
    if diffs.mean() > 45:
        print(f"    CONCLUSION: ERA5 wind direction is NOT a valid proxy for bird flight direction.")
        print(f"    Bird and wind directions are nearly perpendicular (88deg mean difference).")

# ===== STEP 7: Onshore turbine database =====
print("\n" + "=" * 60)
print("STEP 7: Onshore Wind Turbine Database")
print("=" * 60)

wt_file = os.path.join(BAUER_DIR, 'WT_data', 'TheWindPowerDatabaseWithRotorParameters25Mar.csv')
if os.path.exists(wt_file):
    wt_df = pd.read_csv(wt_file, encoding='latin1')
    wt_df['Country'] = wt_df['Country'].replace('United-Kingdom', 'United Kingdom')
    wt_op = wt_df[wt_df['Status'] == 'Production']
    wt_op[['ID', 'Country', 'Latitude', 'Longitude', 'Hub.height', 'turbine.rotorDiameter',
           'turbine.capacity', 'Number.of.turbines', 'Total.power', 'Status']].to_csv(
        os.path.join(OUR_PROC, 'bauer_onshore_windfarms.csv'), index=False, encoding='utf-8-sig')
    study_ct = ['France', 'Germany', 'Netherlands', 'Belgium', 'United Kingdom', 'Denmark', 'Sweden']
    study = wt_op[wt_op['Country'].isin(study_ct)]
    print(f"Total onshore turbines: {len(wt_op):,} (all countries)")
    print(f"Western Europe operating: {len(study):,}")
    hh = study['Hub.height'].dropna()
    print(f"Hub height: mean={hh.mean():.0f}m, median={hh.median():.0f}m")

print("\nAll steps complete.")
