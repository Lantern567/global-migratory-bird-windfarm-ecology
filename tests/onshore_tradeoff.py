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
  E(θ) = sin²(θ - bird_direction)  [simplified: weight×flux×height×conc = constant per farm]
"""
import pandas as pd
import numpy as np
import os

# ---- Config ----
ORIENTATIONS = np.arange(0, 180, 10)  # 18 angles
BUDGETS = [0.005, 0.01, 0.02, 0.05]  # AEP budget levels

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
PROC_DIR = os.path.join(DATA_DIR, 'processed')


def compute_exposure_curve(bird_dir_spring, bird_dir_autumn):
    """Compute geometric exposure E(θ) for all 18 orientations.

    E(θ) = 0.5 × [sin²(θ - spring_dir) + sin²(θ - autumn_dir)]

    Uses equal weighting of spring and autumn migration seasons.
    sin² captures the geometric alignment: parallel=0 risk, perpendicular=1 risk.
    """
    E = np.zeros(18)
    for i, theta in enumerate(ORIENTATIONS):
        spring_term = np.sin(np.radians(theta - bird_dir_spring)) ** 2
        autumn_term = np.sin(np.radians(theta - bird_dir_autumn)) ** 2
        E[i] = 0.5 * (spring_term + autumn_term)
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

        results.append({
            'budget': budget,
            'theta_econ': theta_econ,
            'theta_eco': theta_eco,
            'aep_cost_pct': round(aep_cost_pct, 4),
            'risk_reduction': round(risk_reduction, 2),
        })

    return results


def main():
    print("Onshore Energy-Ecology Trade-off Computation")
    print("=" * 50)

    # ---- Load data ----
    farms = pd.read_csv(os.path.join(PROC_DIR, 'osm_farm_pca_orientations.csv'))
    aep_df = pd.read_csv(os.path.join(PROC_DIR, 'onshore_aep_curves.csv'))

    # Load Bauer GRID CELL bird direction data (newly recomputed, correct spring dirs)
    bauer_grid = pd.read_csv(os.path.join(PROC_DIR, 'bauer_grid_cell_directions.csv'))
    # Filter to cells with valid spring AND autumn directions
    bauer_grid_valid = bauer_grid.dropna(subset=['spring_dir', 'autumn_dir']).copy()
    print(f"Loaded: {len(farms)} OSM farms, {len(aep_df)} AEP curves, {len(bauer_grid_valid)} valid grid cells")

    # ---- Match OSM farms to nearest Bauer grid cell ----
    # Grid resolution is 0.25°, max distance 200 km (~1.8°)
    from scipy.spatial import cKDTree

    grid_coords = np.column_stack([
        bauer_grid_valid['lat'].values,
        bauer_grid_valid['lon'].values
    ])
    tree = cKDTree(grid_coords)

    farm_coords = np.column_stack([
        farms['centroid_lat'].values,
        farms['centroid_lon'].values
    ])
    max_dist_deg = 200.0 / 111.0  # 200 km ~ 1.8 degrees

    dist, idx = tree.query(farm_coords, k=1, distance_upper_bound=max_dist_deg)
    valid = np.isfinite(dist)

    # Map matched farms to grid cell directions
    farms = farms.copy()
    farms['grid_idx'] = np.where(valid, idx, -1)
    farms['grid_dist_km'] = np.where(valid, dist * 111, np.nan)

    # Add bird directions from matched grid cell
    grid_match = bauer_grid_valid.iloc[idx[valid]]
    farms.loc[valid, 'spring_dir'] = grid_match['spring_dir'].values
    farms.loc[valid, 'autumn_dir'] = grid_match['autumn_dir'].values
    farms.loc[valid, 'spring_conc'] = grid_match['spring_conc'].values
    farms.loc[valid, 'autumn_conc'] = grid_match['autumn_conc'].values

    print(f"OSM farms matched to Bauer grid cell: {valid.sum()} ({valid.sum()/len(farms)*100:.1f}%)")
    print(f"  Median distance: {np.median(dist[valid]*111):.0f} km")

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

        # Compute exposure curve
        exposure = compute_exposure_curve(spring_dir, autumn_dir)

        # Compute trade-off
        tradeoffs = compute_tradeoff(aep_kwh, exposure)

        for t in tradeoffs:
            farm_results.append({
                'farm_id': fid,
                'n_turbines': farm['n_turbines'],
                'centroid_lat': farm['centroid_lat'],
                'centroid_lon': farm['centroid_lon'],
                'hub_height': farm['med_hub_height'],
                'rotor_diam': farm['med_rotor_diam'],
                'capacity_kw': farm['med_capacity_kw'],
                'pc1_angle': farm['pc1_angle'],
                'explained_var_ratio': farm['explained_var_ratio'],
                'grid_dist_km': farm['grid_dist_km'],
                'spring_dir': spring_dir,
                'autumn_dir': autumn_dir,
                **t,
            })

    tradeoff_df = pd.DataFrame(farm_results)

    # ---- Save ----
    out_path = os.path.join(PROC_DIR, 'onshore_tradeoff_results.csv')
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

        # Risk reduction by country (nearest Bauer data country)
        bauer_countries = bauer_farm[['bauer_lat', 'bauer_lon']]
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
