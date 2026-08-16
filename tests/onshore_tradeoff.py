"""
Onshore Energy-Ecology Trade-off Computation
=============================================
Aligns AEP(θ) curves with exposure E(θ) curves to compute the energy-ecology
exchange rate for each onshore wind farm.

Algorithm (same as offshore tradeoff.py):
  Given AEP budget (e.g., 99% of max AEP), find θ_eco that minimizes
  geometric exposure risk within the budget constraint.

  θ_econ = argmax AEP(θ)
  θ_eco  = argmin E(θ)  subject to AEP(θ) ≥ (1 - budget) × AEP(θ_econ)

Exposure model (from exposure.py):
  E(θ) = [c_s·sin²(θ - φ_spring) + c_a·sin²(θ - φ_autumn)] / (c_s + c_a)
  c_s, c_a = spring/autumn directional concentration (A-4: seasons weighted by
  concentration; falls back to equal 0.5/0.5 when concentration missing).
"""
import pandas as pd
import numpy as np
import os
import sys

# ---- Config ----
ORIENTATIONS = np.arange(0, 180, 10)  # 18 angles
BUDGETS = [0.005, 0.01, 0.02, 0.05]  # AEP budget levels

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PROC_DIR = os.path.join(DATA_DIR, 'processed')


def compute_exposure_curve(bird_dir_spring, bird_dir_autumn, spring_conc=None, autumn_conc=None):
    """Compute geometric exposure E(θ) for all 18 orientations.

    E(θ) = [c_s·sin²(θ - spring_dir) + c_a·sin²(θ - autumn_dir)] / (c_s + c_a)

    Seasons are weighted by their directional concentration (c_s, c_a) when
    provided (A-4 optional improvement): a season with more tightly clustered
    migration contributes more to the directional exposure. Falls back to equal
    weighting when concentrations are missing/invalid, i.e.
    E(θ) = 0.5·[sin²(θ - spring_dir) + sin²(θ - autumn_dir)].
    sin² captures the geometric alignment: parallel=0 risk, perpendicular=1 risk.
    """
    try:
        c_s = float(spring_conc)
        c_a = float(autumn_conc)
    except (TypeError, ValueError):
        c_s = c_a = np.nan
    if np.isnan(c_s) or np.isnan(c_a) or (c_s + c_a) <= 0:
        w_s = w_a = 0.5
    else:
        w_s = c_s / (c_s + c_a)
        w_a = c_a / (c_s + c_a)

    E = np.zeros(18)
    for i, theta in enumerate(ORIENTATIONS):
        spring_term = np.sin(np.radians(theta - bird_dir_spring)) ** 2
        autumn_term = np.sin(np.radians(theta - bird_dir_autumn)) ** 2
        E[i] = w_s * spring_term + w_a * autumn_term
    return E


def compute_tradeoff(aep_kwh, exposure, budgets=None):
    """Compute trade-off for multiple budget levels.

    Parameters:
      aep_kwh: AEP values [kWh] for 18 orientations
      exposure: exposure values [-] for 18 orientations
      budgets: list of AEP budget fractions (e.g., [0.01, 0.02, 0.05])

    Returns:
      List of dicts with {budget, theta_econ, theta_eco, aep_cost_pct, risk_reduction}
    """
    if budgets is None:
        budgets = BUDGETS

    idx_econ = np.argmax(aep_kwh)
    aep_max = aep_kwh[idx_econ]
    theta_econ = ORIENTATIONS[idx_econ]

    results = []
    for budget in budgets:
        aep_threshold = (1 - budget) * aep_max
        allowed = np.where(aep_kwh >= aep_threshold)[0]

        if len(allowed) == 0:
            continue

        # Find orientation within allowed set that minimizes exposure
        exposure_allowed = exposure[allowed]
        best_in_allowed = allowed[np.argmin(exposure_allowed)]

        theta_eco = ORIENTATIONS[best_in_allowed]
        aep_eco = aep_kwh[best_in_allowed]
        aep_cost_pct = (aep_max - aep_eco) / aep_max * 100

        # Risk reduction from economic optimum to ecological optimum
        risk_econ = exposure[idx_econ]
        risk_eco = exposure[best_in_allowed]
        risk_reduction = (risk_econ - risk_eco) / max(risk_econ, 1e-10) * 100
        risk_reduction_abs = risk_econ - risk_eco  # B-9: absolute reduction ΔE

        results.append({
            'budget': budget,
            'theta_econ': theta_econ,
            'theta_eco': theta_eco,
            'aep_cost_pct': round(aep_cost_pct, 4),
            'risk_reduction': round(risk_reduction, 2),
            'risk_reduction_abs': round(risk_reduction_abs, 6),
        })

    return results


def main():
    # Command-line overrides (backward compatible)
    aep_file = 'onshore_aep_curves.csv'
    out_file = 'onshore_tradeoff_results.csv'
    argv = sys.argv[1:]
    for i, a in enumerate(argv):
        if a == '--aep' and i + 1 < len(argv):
            aep_file = argv[i + 1]
        if a == '--out' and i + 1 < len(argv):
            out_file = argv[i + 1]

    print("Onshore Energy-Ecology Trade-off Computation")
    print(f"  AEP input = {aep_file}, output = {out_file}")
    print("=" * 50)

    # ---- Load data ----
    farms = pd.read_csv(os.path.join(PROC_DIR, 'osm_farm_pca_orientations.csv'))
    aep_df = pd.read_csv(os.path.join(PROC_DIR, aep_file))

    # Load Bauer GRID CELL bird direction data (newly recomputed, correct spring dirs)
    bauer_grid = pd.read_csv(os.path.join(PROC_DIR, 'bauer_grid_cell_directions.csv'))
    # Filter to cells with valid spring AND autumn directions
    bauer_grid_valid = bauer_grid.dropna(subset=['spring_dir', 'autumn_dir']).copy()
    print(f"Loaded: {len(farms)} OSM farms, {len(aep_df)} AEP curves, {len(bauer_grid_valid)} valid grid cells")

    # ---- Match OSM farms to nearest Bauer grid cell ----
    # C-12: project lat/lon to approximate equal-distance coordinates (km) before
    # KDTree, so Euclidean distance ≈ great-circle distance (avoids the ~1.2x
    # overestimate from treating degrees as km).
    from scipy.spatial import cKDTree

    KM_PER_DEG_LAT = 111.0
    LAT_REF = 50.0  # reference latitude for longitude scaling (study region ~43-55N)
    COS_LAT_REF = np.cos(np.radians(LAT_REF))

    grid_coords_km = np.column_stack([
        bauer_grid_valid['lat'].values * KM_PER_DEG_LAT,
        bauer_grid_valid['lon'].values * KM_PER_DEG_LAT * COS_LAT_REF
    ])
    tree = cKDTree(grid_coords_km)

    farm_coords_km = np.column_stack([
        farms['centroid_lat'].values * KM_PER_DEG_LAT,
        farms['centroid_lon'].values * KM_PER_DEG_LAT * COS_LAT_REF
    ])
    max_dist_km = 200.0

    dist, idx = tree.query(farm_coords_km, k=1, distance_upper_bound=max_dist_km)
    valid = np.isfinite(dist)

    # Map matched farms to grid cell directions
    farms = farms.copy()
    farms['grid_idx'] = np.where(valid, idx, -1)
    farms['grid_dist_km'] = np.where(valid, dist, np.nan)

    # Add bird directions from matched grid cell
    grid_match = bauer_grid_valid.iloc[idx[valid]]
    farms.loc[valid, 'spring_dir'] = grid_match['spring_dir'].values
    farms.loc[valid, 'autumn_dir'] = grid_match['autumn_dir'].values
    farms.loc[valid, 'spring_conc'] = grid_match['spring_conc'].values
    farms.loc[valid, 'autumn_conc'] = grid_match['autumn_conc'].values

    print(f"OSM farms matched to Bauer grid cell: {valid.sum()} ({valid.sum()/len(farms)*100:.1f}%)")
    print(f"  Median distance: {np.median(dist[valid]):.0f} km")

    # ---- Merge AEP with exposure directions ----
    farm_results = []
    for _, farm in farms.iterrows():
        fid = farm['farm_id']

        # Skip farms without bird direction match
        if pd.isna(farm.get('spring_dir')) or pd.isna(farm.get('autumn_dir')):
            continue

        # Get AEP curve
        aep_row = aep_df[aep_df['farm_id'] == fid]
        if len(aep_row) == 0:
            continue
        aep_row = aep_row.iloc[0]

        aep_kwh = np.array([aep_row[f'aep_{int(theta):03d}'] for theta in ORIENTATIONS])
        if aep_kwh.max() <= 0:
            continue

        # Get bird direction from matched grid cell
        spring_dir = farm['spring_dir']
        autumn_dir = farm['autumn_dir']

        # Compute exposure curve (seasons weighted by directional concentration, A-4)
        exposure = compute_exposure_curve(
            spring_dir, autumn_dir,
            farm.get('spring_conc'), farm.get('autumn_conc'))

        # Compute trade-off
        tradeoffs = compute_tradeoff(aep_kwh, exposure)

        for t in tradeoffs:
            farm_results.append({
                'farm_id': fid,
                'n_turbines': farm['n_turbines'],
                'centroid_lat': farm['centroid_lat'],
                'centroid_lon': farm['centroid_lon'],
                'hub_height': aep_row['hub_height'],
                'rotor_diam': aep_row['rotor_diam'],
                'capacity_kw': aep_row['capacity_kw'],
                'hub_height_imputed': aep_row['hub_height_imputed'],
                'rotor_diam_imputed': aep_row['rotor_diam_imputed'],
                'capacity_imputed': aep_row['capacity_imputed'],
                'pc1_angle': farm['pc1_angle'],
                'explained_var_ratio': farm['explained_var_ratio'],
                'grid_dist_km': farm['grid_dist_km'],
                'spring_dir': spring_dir,
                'autumn_dir': autumn_dir,
                'spring_conc': farm['spring_conc'],
                'autumn_conc': farm['autumn_conc'],
                **t,
            })

    tradeoff_df = pd.DataFrame(farm_results)

    # ---- Save ----
    out_path = os.path.join(PROC_DIR, out_file)
    tradeoff_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(tradeoff_df)} trade-off results to {out_path}")
    print(f"  Unique farms: {tradeoff_df['farm_id'].nunique()}")
    print(f"  Budget levels: {len(BUDGETS)}")

    # ---- Summary ----
    budget_1pct = tradeoff_df[tradeoff_df['budget'] == 0.01]
    if len(budget_1pct) > 0:
        print(f"\n=== Results at 1% AEP Budget ===")
        print(f"Farms: {len(budget_1pct)}")
        print(f"Mean AEP cost: {budget_1pct['aep_cost_pct'].mean():.4f}%")
        print(f"Mean risk reduction: {budget_1pct['risk_reduction'].mean():.1f}%")
        print(f"Median risk reduction: {budget_1pct['risk_reduction'].median():.1f}%")

        # Farms with significant trade-off (risk reduction > 50%)
        high = budget_1pct[budget_1pct['risk_reduction'] > 50]
        print(f"Risk reduction >50%: {len(high)} farms ({len(high)/len(budget_1pct)*100:.1f}%)")
        print(f"Risk reduction >80%: {(budget_1pct['risk_reduction'] > 80).sum()} farms")

        # Risk reduction by region (C-11: removed undefined bauer_farm)
        # Match to known regions
        tradeoff_df['region'] = 'Unknown'
        for i, row in tradeoff_df.iterrows():
            lat = row['centroid_lat']
            lon = row['centroid_lon']
            if 50 < lat < 60 and -5 < lon < 2:
                tradeoff_df.at[i, 'region'] = 'UK'
            elif 47 < lat < 55 and 2 < lon < 15:
                tradeoff_df.at[i, 'region'] = 'Central Europe'
            elif 55 < lat < 70 and 5 < lon < 30:
                tradeoff_df.at[i, 'region'] = 'Nordic'
            elif 36 < lat < 44 and -10 < lon < 5:
                tradeoff_df.at[i, 'region'] = 'Iberia'
            elif 43 < lat < 47 and -2 < lon < 8:
                tradeoff_df.at[i, 'region'] = 'France'

        print(f"\nBy region (1% budget):")
        for region, group in tradeoff_df[tradeoff_df['budget'] == 0.01].groupby('region'):
            print(f"  {region}: {len(group)} farms, mean risk reduction={group['risk_reduction'].mean():.1f}%, mean AEP cost={group['aep_cost_pct'].mean():.4f}%")

    return tradeoff_df


if __name__ == '__main__':
    tradeoff_df = main()
