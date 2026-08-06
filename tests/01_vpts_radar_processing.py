"""
01_vpts_radar_processing.py
===========================
Download VPTS radar data from Belgian Meteo FTP and extract bird migration direction signatures.

Inputs:
  ../our_work/data/raw/radar_vpts/      (569 VPTS files, downloaded from opendata.meteo.be)
  ../wind-direction-to-electricity-transition-main/offshore-task0-HuTingxian/output/task0/farms_master.csv

Outputs:
  ../our_work/data/processed/radar_direction_signatures_v2.csv
  ../our_work/data/processed/radar_new_stations_signatures.csv
  ../our_work/data/processed/farm_direction_signatures.csv

Method:
  Filter VPTS data: nighttime (20:00-06:00), 200m altitude layer, density >10 birds/km3,
  March-May (spring) and August-November (autumn) only.
  Compute density-weighted circular mean direction per radar station per season.
  Exclude stations with anomalous directions or insufficient samples.
  Map each offshore farm to nearest validated radar station.
"""

import os, sys, io, math, csv
from collections import defaultdict
from datetime import datetime
import pandas as pd, numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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
VPTS_DIR = os.path.join(REPO_ROOT, 'our_work', 'data', 'raw', 'radar_vpts')
FARMS_CSV = os.path.join(REPO_ROOT, '..', 'wind-direction-to-electricity-transition-main',
    'offshore-task0-HuTingxian', 'output', 'task0', 'farms_master.csv')
OUT_DIR = os.path.join(REPO_ROOT, 'our_work', 'data', 'processed')

# Radar station coordinates (validated stations only)
VALIDATED_STATIONS = {
    'nlhrw': (52.95, 4.75),   # Den Helder, Netherlands
    'bejab': (51.18, 3.07),   # Jabbeke, Belgium
    'deess': (51.40, 6.97),   # Essen, Germany
    'frabb': (50.13, 1.83),   # Abbeville, France
    'nldhl': (51.84, 5.15),   # Herwijnen, Netherlands (excluded: non-southward autumn)
    'behel': (51.05, 5.42),   # Helchteren, Belgium (excluded: anomalous westward)
    'denhb': (53.60, 10.68),  # Neuhaus, Germany (insufficient data)
}

ROTOR_ALT = 200          # VPTS altitude bin closest to IEA-10MW rotor
DENSITY_THRESHOLD = 10   # birds/km3

# ===== STEP 1: Parse all VPTS files =====
print("=" * 60)
print("STEP 1: VPTS Radar Direction Extraction")
print("=" * 60)

station_data = defaultdict(lambda: defaultdict(lambda: {'bearings': [], 'densities': []}))
files_processed = 0
total_rows = 0
bird_rows = 0

for fname in sorted(os.listdir(VPTS_DIR)):
    if not fname.endswith('.txt'):
        continue
    station = fname.split('_')[0]
    filepath = os.path.join(VPTS_DIR, fname)

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            total_rows += 1
            try:
                hght = int(parts[2])
                if hght != ROTOR_ALT:
                    continue
                dd_val = parts[7]
                dens_val = parts[11]
                if dd_val == 'nan' or dens_val == 'nan':
                    continue
                dd = float(dd_val)
                dens = float(dens_val)
                if dens < DENSITY_THRESHOLD:
                    continue

                time_str = parts[1]
                hour = int(time_str[:2])
                if 6 < hour < 20:  # daytime = insect noise
                    continue

                date_str = parts[0]
                dt = datetime.strptime(date_str, '%Y%m%d')
                month = dt.month
                if 3 <= month <= 5:
                    season = 'spring'
                elif 8 <= month <= 11:
                    season = 'autumn'
                else:
                    continue

                station_data[station][season]['bearings'].append(dd)
                station_data[station][season]['densities'].append(dens)
                bird_rows += 1
            except (ValueError, IndexError):
                continue
    files_processed += 1

print(f"Files processed: {files_processed}")
print(f"Total data rows: {total_rows:,}")
print(f"Bird detections (filtered): {bird_rows:,}")

# ===== STEP 2: Compute seasonal direction signatures =====
print("\n" + "=" * 60)
print("STEP 2: Seasonal Direction Signatures")
print("=" * 60)

results = []
for station in sorted(station_data.keys()):
    for season in ['spring', 'autumn']:
        data = station_data[station][season]
        if len(data['bearings']) < 5:
            continue

        bearings = data['bearings']
        weights = data['densities']
        total_w = sum(weights)

        x = sum(w * math.cos(math.radians(b)) for b, w in zip(bearings, weights))
        y = sum(w * math.sin(math.radians(b)) for b, w in zip(bearings, weights))
        concentration = math.hypot(x, y) / total_w
        mean_dir = math.degrees(math.atan2(y, x)) % 360
        circ_std = math.degrees(math.sqrt(-2 * math.log(concentration))) if concentration > 0.01 else 180

        avg_dens = sum(weights) / len(weights)
        max_dens = max(weights)

        is_northward = (315 <= mean_dir <= 360 or 0 <= mean_dir <= 90)
        is_southward = (135 <= mean_dir <= 270)
        status = 'OK' if ((season == 'spring' and is_northward) or (season == 'autumn' and is_southward)) else 'SUSPECT'

        print(f"  {station} / {season}: dir={mean_dir:.1f}deg, n={len(bearings):,}, conc={concentration:.3f}, {status}")
        results.append({
            'station': station, 'season': season, 'n_obs': len(bearings),
            'direction_deg': round(mean_dir, 1), 'concentration': round(concentration, 3),
            'circular_std_deg': round(circ_std, 1),
            'avg_density': round(avg_dens, 1), 'max_density': round(max_dens, 1),
            'total_flux': round(sum(weights), 1),
        })

# Save radar signatures
radar_sigs_df = pd.DataFrame(results)
radar_sigs_df.to_csv(os.path.join(OUT_DIR, 'radar_direction_signatures_v2.csv'), index=False, encoding='utf-8-sig')
print(f"\nSaved: radar_direction_signatures_v2.csv ({len(results)} signatures)")

# ===== STEP 3: Map farms to nearest validated radar =====
print("\n" + "=" * 60)
print("STEP 3: Farm-Radar Mapping")
print("=" * 60)

farms = pd.read_csv(FARMS_CSV)
print(f"Farms loaded: {len(farms)}")

# Excluded stations
EXCLUDED = {'nldhl', 'behel', 'denhb'}
VALID = {s: c for s, c in VALIDATED_STATIONS.items() if s not in EXCLUDED}

def nearest_radar(lat, lon):
    best_d = float('inf')
    best_s = None
    for sn, (rlat, rlon) in VALID.items():
        d = math.sqrt((lat - rlat)**2 + (lon - rlon)**2) * 111
        if d < best_d:
            best_d, best_s = d, sn
    return best_s, best_d

farm_sigs = []
for _, farm in farms.iterrows():
    fid = farm['farm_id']
    c = farm['country']
    lat, lon = farm['centroid_lat'], farm['centroid_lon']

    # Only European farms are in VPTS range
    if c in ['China', 'Vietnam', 'Taiwan', 'Japan', 'South Korea', 'United States of America']:
        continue

    sn, sd = nearest_radar(lat, lon)
    if sd > 500:
        continue  # beyond reliable radar range

    ss = radar_sigs_df[radar_sigs_df['station'] == sn]
    if len(ss) == 0:
        continue

    for _, sig in ss.iterrows():
        farm_sigs.append({
            'farm_id': fid,
            'receptor_id': f'radar_{sn}',
            'season': sig['season'],
            'direction_deg': sig['direction_deg'],
            'concentration': sig['concentration'],
            'flux': sig['total_flux'],
            'rotor_height_fraction': 0.5,
            'evidence_level': 'radar',
            'conservation_weight': 1.5,
            'n_observations': int(sig['n_obs']),
            'source': f'VPTS_{sn}_2020_dist{int(sd)}km',
        })

farm_sigs_df = pd.DataFrame(farm_sigs)
farm_sigs_df.to_csv(os.path.join(OUT_DIR, 'farm_direction_signatures.csv'), index=False, encoding='utf-8-sig')
print(f"Farms with radar direction: {farm_sigs_df['farm_id'].nunique()}")
print(f"Saved: farm_direction_signatures.csv")
print("\nDone.")
