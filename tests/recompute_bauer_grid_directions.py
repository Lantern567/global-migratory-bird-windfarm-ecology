"""
Recompute Bauer grid cell-level bird migration directions from raw MAT data.

Fixes a bug in the original processing where spring directions were incorrectly
computed (stored as ~0 degrees in the NPZ instead of ~30-50 degrees NNE-NE).

Algorithm:
  For each 49×85 grid cell:
    1. Extract hourly birdDir and daynight from birdAtRisk.mat
    2. Filter: night hours only (daynight == 0, consistent with VPTS analysis)
    3. Split: spring (months 3-5, hours 1416-3623) vs autumn (months 8-11, hours 5088-8015)
    4. Compute circular mean direction for each season
    5. Output as clean CSV for use in onshore exposure computation
"""
import numpy as np
import scipy.io as sio
import pandas as pd
import os

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RAW_FILE = os.path.join(DATA_DIR, 'raw', 'bauer2026', 'data', 'birdAtRisk.mat')
OUT_FILE = os.path.join(DATA_DIR, 'processed', 'bauer_grid_cell_directions.csv')

# Hour indices for months (8760 hours starting Jan 1, non-leap year)
# Jan: 0-743, Feb: 744-1415, Mar: 1416-2159, Apr: 2160-2879,
# May: 2880-3623, Jun: 3624-4343, Jul: 4344-5087, Aug: 5088-5831,
# Sep: 5832-6551, Oct: 6552-7295, Nov: 7296-8015, Dec: 8016-8759
SPRING_START = 1416   # March 1
SPRING_END = 3623     # May 31
AUTUMN_START = 5088   # August 1
AUTUMN_END = 8015     # November 30

# CRITICAL: birdDir values are in RADIANS [-pi to pi], not degrees.
# daynight: 0 = night, 1 = day (verified: hour 0 UTC=midnight->0, hour 12 UTC=noon->1)

print("Loading birdAtRisk.mat...")
f = sio.loadmat(RAW_FILE)
birdDir_rad = f['birdDir']    # (49, 85, 8760) — bird flight direction [RADIANS]
daynight = f['daynight']      # (49, 85, 8760) — 0=night, 1=day

# Convert to degrees
birdDir = np.degrees(birdDir_rad) % 360  # 0-360 degrees
print(f"Direction range after conversion: {np.nanmin(birdDir):.0f} - {np.nanmax(birdDir):.0f} deg")

n_lat, n_lon, n_hours = birdDir.shape
print(f"Grid: {n_lat}×{n_lon}, {n_hours} hours")

# Generate lat/lon grids
lat_grid = np.linspace(43.0, 55.0, 49)  # 43 to 55 N
lon_grid = np.linspace(-5.0, 16.0, 85)  # 5W to 16E

results = []

for ri in range(n_lat):
    for ci in range(n_lon):
        # Get hourly data for this cell
        directions = birdDir[ri, ci, :]
        night_mask = daynight[ri, ci, :] == 0  # 0 = night

        # Filter valid directions (night + finite)
        valid_mask = np.isfinite(directions) & night_mask

        # Spring analysis
        spring_mask = (np.arange(n_hours) >= SPRING_START) & (np.arange(n_hours) <= SPRING_END)
        spring_valid = valid_mask & spring_mask
        n_spring = spring_valid.sum()

        spring_dir = np.nan
        spring_conc = np.nan
        if n_spring >= 10:
            spring_angles = np.radians(directions[spring_valid])
            # Circular mean
            sin_sum = np.sin(spring_angles).sum()
            cos_sum = np.cos(spring_angles).sum()
            spring_dir = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
            spring_conc = np.sqrt(sin_sum**2 + cos_sum**2) / n_spring

        # Autumn analysis
        autumn_mask = (np.arange(n_hours) >= AUTUMN_START) & (np.arange(n_hours) <= AUTUMN_END)
        autumn_valid = valid_mask & autumn_mask
        n_autumn = autumn_valid.sum()

        autumn_dir = np.nan
        autumn_conc = np.nan
        if n_autumn >= 10:
            autumn_angles = np.radians(directions[autumn_valid])
            sin_sum = np.sin(autumn_angles).sum()
            cos_sum = np.cos(autumn_angles).sum()
            autumn_dir = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360
            autumn_conc = np.sqrt(sin_sum**2 + cos_sum**2) / n_autumn

        # Annual (all night hours)
        n_all = valid_mask.sum()
        annual_dir = np.nan
        if n_all >= 10:
            all_angles = np.radians(directions[valid_mask])
            sin_sum = np.sin(all_angles).sum()
            cos_sum = np.cos(all_angles).sum()
            annual_dir = np.degrees(np.arctan2(sin_sum, cos_sum)) % 360

        if n_spring >= 10 or n_autumn >= 10:
            results.append({
                'lat': round(lat_grid[ri], 2),
                'lon': round(lon_grid[ci], 2),
                'row': ri,
                'col': ci,
                'spring_dir': round(spring_dir, 1) if n_spring >= 10 else np.nan,
                'spring_conc': round(spring_conc, 4) if n_spring >= 10 else np.nan,
                'spring_n': n_spring,
                'autumn_dir': round(autumn_dir, 1) if n_autumn >= 10 else np.nan,
                'autumn_conc': round(autumn_conc, 4) if n_autumn >= 10 else np.nan,
                'autumn_n': n_autumn,
                'annual_dir': round(annual_dir, 1) if n_all >= 10 else np.nan,
                'total_n': n_all,
            })

df = pd.DataFrame(results)
print(f"\nCells with valid data: {len(df)}")

# Quality check
spring_valid = df['spring_n'] >= 10
autumn_valid = df['autumn_n'] >= 10
print(f"Spring valid (>=10 night obs): {spring_valid.sum()}")
print(f"Autumn valid (>=10 night obs): {autumn_valid.sum()}")
print(f"Both valid: {(spring_valid & autumn_valid).sum()}")

print(f"\nSpring direction distribution (cells with >=10 obs):")
sd = df.loc[spring_valid, 'spring_dir']
print(f"  Min: {sd.min():.1f}°, Max: {sd.max():.1f}°")
print(f"  5th: {sd.quantile(0.05):.1f}°, 50th: {sd.quantile(0.5):.1f}°, 95th: {sd.quantile(0.95):.1f}°")

print(f"\nAutumn direction distribution (cells with >=10 obs):")
ad = df.loc[autumn_valid, 'autumn_dir']
print(f"  Min: {ad.min():.1f}°, Max: {ad.max():.1f}°")
print(f"  5th: {ad.quantile(0.05):.1f}°, 50th: {ad.quantile(0.5):.1f}°, 95th: {ad.quantile(0.95):.1f}°")

df.to_csv(OUT_FILE, index=False)
print(f"\nSaved to {OUT_FILE} ({len(df)} cells)")
