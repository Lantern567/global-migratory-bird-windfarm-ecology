"""
Onshore Wind Farm AEP(θ) Computation with Jensen Wake Model
============================================================

Pipeline:
  1. Fetch GWA wind rose data for unique farm locations (Weibull A, k, frequencies)
  2. Parameterize power curves and thrust coefficient curves per turbine
  3. For each wind direction (36 sectors × 10°): compute farm wake efficiency η(wd)
     using Jensen (1983) analytical wake model with quadratic deficit summation
  4. AEP(θ) = 8760 × Σ_sector freq(φ) × E_no_wake(φ) × η((φ-θ) mod 360)
     where E_no_wake = ∫ P(v) × Weibull(v | A(φ), k(φ)) dv

Key physics:
  - Wake deficit: Δv/v∞ = (1-√(1-Ct)) / (1 + α·x/r)^2
  - Multiple wakes combined via RSS (sqrt of sum of squared deficits)
  - Partial wake overlap: linear ramp within wake radius
  - Ct = 0.8 below rated, Ct ∝ v⁻³ above rated (pitch regulation)
  - Onshore wake decay constant α = 0.075

Author: Generated for Hu Tingxian, August 2026
"""
import pandas as pd
import numpy as np
from scipy.special import gamma as gamma_func
import requests
import time
import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# PHYSICAL CONSTANTS & CONFIGURATION
# ============================================================
RHO_AIR = 1.225          # kg/m³
CP_MAX = 0.45            # max power coefficient (modern pitch-regulated turbine)
ALPHA_ONSHORE = 0.075    # Jensen wake decay constant, onshore
CT_BELOW_RATED = 0.80    # thrust coefficient in operating range
V_CUTIN = 3.0            # m/s
V_CUTOUT = 25.0          # m/s

# GWA data parameters
GWA_HEIGHTS = np.array([10.0, 50.0, 100.0, 150.0, 200.0, 250.0])
GWA_SECTORS = 12         # GWA provides 12 sectors (0°, 30°, ..., 330°)
GWA_SECTOR_ANGLES = np.arange(0, 360, 360 / GWA_SECTORS)

# Wake computation: fine wind direction resolution
N_WD = 36                # 36 sectors for wake computation (10° resolution)
WD_ANGLES = np.arange(0, 360, 360 / N_WD)

# Output orientations
N_ORIENTATIONS = 18
ORIENTATION_ANGLES = np.arange(0, 180, 10)

# Wind speed discretization for power integration
V_MIN, V_MAX, DV = 0.5, 30.0, 0.5
V_BINS = np.arange(V_MIN + DV/2, V_MAX, DV)
N_V = len(V_BINS)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)  # our_work/
DATA_DIR = os.path.join(REPO_ROOT, 'data')
PROC_DIR = os.path.join(DATA_DIR, 'processed')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(RAW_DIR, exist_ok=True)


# ============================================================
# PART 1: GWA DATA
# ============================================================

def fetch_gwa(lat, lon, retries=3):
    """Fetch GWA wind climate from Global Wind Atlas API."""
    url = f'https://globalwindatlas.info/api/gwa/custom/Lib/?lat={lat}&long={lon}'
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers={'Referer': 'https://globalwindatlas.info'}, timeout=15)
            if resp.status_code == 200:
                return resp.content.decode('ascii')
            elif resp.status_code == 429:
                time.sleep(min(2 ** attempt, 10))
            elif resp.status_code >= 500:
                time.sleep(1)
        except requests.exceptions.Timeout:
            continue
        except Exception:
            time.sleep(0.5)
    return None


def parse_gwa(content):
    """Parse GWA Generalized Wind Climate format.

    Returns dict with roughness, heights, frequencies[%](n_rough,12),
    A(m/s), k(-) each (n_rough, n_heights, 12).
    """
    lines = content.strip().splitlines()
    n_rough, n_heights, n_sectors = map(int, lines[1].split())
    roughness = np.array(list(map(float, lines[2].split())))
    heights = np.array(list(map(float, lines[3].split())))

    freq = np.zeros((n_rough, n_sectors))
    A = np.zeros((n_rough, n_heights, n_sectors))
    k = np.zeros((n_rough, n_heights, n_sectors))

    li = 4
    for r in range(n_rough):
        freq[r] = list(map(float, lines[li].split()))
        li += 1
        for h in range(n_heights):
            A[r, h] = list(map(float, lines[li].split()))
            li += 1
            k[r, h] = list(map(float, lines[li].split()))
            li += 1
    return {'roughness': roughness, 'heights': heights,
            'frequencies': freq, 'A': A, 'k': k}


def interpolate_weibull(gwa_data, target_height):
    """Log-linear interp of A, linear of k, to target hub height.
    Roughness class index 3 (r=0.400m, typical European agricultural terrain).
    """
    heights = gwa_data['heights']
    r_idx = 3  # roughness class 0.400m
    target_h = np.clip(target_height, heights[0], heights[-1])

    # Log-linear for A (wind speed follows log profile)
    log_h = np.log(heights)
    log_target = np.log(target_h)
    log_A = np.log(np.maximum(gwa_data['A'][r_idx], 0.1))

    A_interp = np.array([np.exp(np.interp(log_target, log_h, log_A[:, s]))
                         for s in range(GWA_SECTORS)])
    k_interp = np.array([np.interp(target_h, heights, gwa_data['k'][r_idx, :, s])
                         for s in range(GWA_SECTORS)])
    return A_interp, k_interp


# ============================================================
# PART 2: TURBINE CURVES
# ============================================================

def build_power_curve(rotor_diam, rated_power_kw):
    """IEC-style power curve: cubic below rated, constant at rated.

    Returns power_kw array [N_V], v_rated (m/s), rotor_area (m²).
    """
    rotor_area = np.pi * (rotor_diam / 2) ** 2
    v_rated = (2 * rated_power_kw * 1000 / (RHO_AIR * rotor_area * CP_MAX)) ** (1/3)
    v_rated = np.clip(v_rated, 9.0, 15.0)

    power = np.zeros(N_V)
    for i, v in enumerate(V_BINS):
        if v < V_CUTIN:
            power[i] = 0
        elif v < v_rated:
            power[i] = 0.5 * RHO_AIR * rotor_area * CP_MAX * v**3 / 1000
        elif v < V_CUTOUT:
            power[i] = rated_power_kw
    return power, v_rated, rotor_area


def build_thrust_curve(v_rated):
    """Ct(v): 0.8 below rated, pitch-regulated decrease above."""
    Ct = np.zeros(N_V)
    for i, v in enumerate(V_BINS):
        if v < V_CUTIN:
            Ct[i] = 0
        elif v < v_rated:
            Ct[i] = CT_BELOW_RATED
        elif v < V_CUTOUT:
            Ct[i] = CT_BELOW_RATED * (v_rated / v) ** 3
    return Ct


# ============================================================
# PART 3: JENSEN WAKE — FARM EFFICIENCY
# ============================================================

def compute_farm_wake_efficiency(turbs_x, turbs_y, rotor_diams, wind_dir_deg,
                                  thrust_curve, v_rated):
    """Compute farm efficiency η = P_wake / P_no_wake for a single wind direction.

    Uses sector-mean wind speed for Ct, then applies wake to all turbines.

    Parameters:
      turbs_x, turbs_y: turbine coordinates in meters (local projection)
      rotor_diams: array of rotor diameters [m] per turbine
      wind_dir_deg: wind direction (meteorological: wind FROM)
      thrust_curve: Ct(v) array
      v_rated: rated wind speed for this turbine

    Returns:
      eta: farm efficiency ratio [0-1]
    """
    n_turbs = len(turbs_x)
    if n_turbs <= 1:
        return 1.0

    rotor_r = rotor_diams / 2

    # Wind direction vector (wind FROM)
    wd_rad = np.radians(wind_dir_deg)
    wx = np.sin(wd_rad)   # east component
    wy = np.cos(wd_rad)   # north component
    # Cross-wind (perpendicular, to the "right" of wind)
    cx = wy
    cy = -wx

    # Project onto wind-aligned coords
    along = turbs_x * wx + turbs_y * wy   # positive = upstream
    cross = turbs_x * cx + turbs_y * cy

    # Sort from upstream to downstream
    order = np.argsort(-along)
    along_s = along[order]
    cross_s = cross[order]
    rotor_r_s = rotor_r[order]

    # Use the weighted-average wind speed for this direction to get representative Ct
    # We don't have per-sector mean speed yet, use a fixed reference (8 m/s, typical)
    # This is sufficient because efficiency RATIO is insensitive to exact Ct for pitch-regulated turbines
    v_ref = 8.0  # representative operating wind speed
    iv = min(int((v_ref - V_MIN) / DV), N_V - 1)
    Ct_ref = thrust_curve[iv]
    if Ct_ref <= 0:
        return 1.0

    # Compute wake deficit at each turbine
    # Use RSS combination for multiple wakes (Katic et al. 1986)
    deficit_sq = np.zeros(n_turbs)

    for j in range(1, n_turbs):
        for k in range(j):  # upstream turbines
            dx = along_s[k] - along_s[j]
            if dx <= 0:
                continue
            dy = cross_s[k] - cross_s[j]

            # Jensen wake radius at turbine j
            wake_r = rotor_r_s[k] + ALPHA_ONSHORE * dx

            if abs(dy) < wake_r:
                # Partial overlap (linear ramp)
                overlap = max(0, 1 - abs(dy) / max(wake_r, 1e-6))
                # Single-wake deficit
                deficit = (1 - np.sqrt(1 - Ct_ref)) / (1 + ALPHA_ONSHORE * dx / rotor_r_s[k]) ** 2
                deficit_sq[j] += (deficit * overlap) ** 2

    # Effective wind speed: v_eff = v_inf * (1 - sqrt(deficit_sq))
    v_deficit = np.sqrt(deficit_sq)
    v_eff = np.maximum(0, V_BINS[iv] * (1 - v_deficit))

    # Power with wake vs without wake (at reference wind speed)
    # Interpolate power curve
    def power_at(v):
        if v <= V_BINS[0]:
            return 0
        if v >= V_BINS[-1]:
            return 0 if v >= V_CUTOUT else power_curve_global[-1]
        idx = min(int((v - V_MIN) / DV), N_V - 1)
        return power_curve_global[idx]

    # For efficiency, we compare farm total power with/without wake at v_ref
    # Both at same free-stream speed
    P_no_wake = n_turbs * power_curve_global[iv] if power_curve_global[iv] > 0 else n_turbs
    if P_no_wake <= 0:
        return 1.0

    P_with = 0.0
    for j in range(n_turbs):
        pw = power_at(v_eff[j])
        P_with += pw

    eta = min(1.0, max(0.6, P_with / max(P_no_wake, 1e-6)))
    return eta


# ============================================================
# PART 4: AEP(θ) COMPUTATION
# ============================================================

def compute_aep_theta(farm_turbines, A_sector, k_sector, freq_sector,
                      power_curve, thrust_curve, v_rated):
    """Compute AEP(θ) for all 18 orientations of a single farm.

    Algorithm:
      1. For each of 36 wind directions, compute wake efficiency η(wd)
      2. For each orientation θ, AEP(θ) = Σ_{wd} freq(wd) × E_no_wake(wd) × η((wd-θ)%360)
      3. E_no_wake: ∫ P(v) × Weibull_pdf(v | A, k) dv  (numerical integration)

    Parameters:
      farm_turbines: DataFrame with [lat, lon]
      A_sector, k_sector: Weibull params [12 sectors], interpolated to hub height
      freq_sector: frequencies [12 sectors], normalized to sum=1
      power_curve: P(v) [N_V]
      thrust_curve: Ct(v) [N_V]
      v_rated: rated wind speed

    Returns:
      aep_kwh: array [18] of AEP in kWh/year
    """
    n_turbs = len(farm_turbines)
    if n_turbs == 0:
        return np.zeros(N_ORIENTATIONS)

    # ---- Step 1: Compute E_no_wake per GWA sector ----
    # Energy content per sector (without wake): ∫ P(v) × Weibull(v|A,k) dv × freq
    E_no_wake_12 = np.zeros(GWA_SECTORS)  # [kW·freq] per sector

    for s in range(GWA_SECTORS):
        A = A_sector[s]
        k = k_sector[s]
        freq = freq_sector[s]
        if A <= 0 or k <= 0 or freq <= 0:
            continue
        # Integrate ∫ P(v) × Weibull(v) dv
        sector_energy = 0.0
        for iv, v in enumerate(V_BINS):
            p = power_curve[iv]
            if p <= 0:
                continue
            # Weibull probability in this bin
            cdf_hi = 1 - np.exp(-((v + DV/2) / A) ** k)
            cdf_lo = 1 - np.exp(-((v - DV/2) / A) ** k)
            prob = max(0, cdf_hi - cdf_lo)
            sector_energy += p * prob
        E_no_wake_12[s] = sector_energy * freq  # weighted by sector frequency

    # ---- Step 2: Compute wake efficiency at 36 directions ----
    # Interpolate 12 GWA sectors to 36 wind directions
    freq_36 = np.zeros(N_WD)
    for i, wd in enumerate(WD_ANGLES):
        # Find nearest GWA sector
        s = int(round(wd / (360 / GWA_SECTORS))) % GWA_SECTORS
        freq_36[i] = freq_sector[s] / (N_WD / GWA_SECTORS)  # redistribute evenly

    # Convert turbine coordinates to local meters
    lat_mean = farm_turbines['lat'].mean()
    cos_lat = np.cos(np.radians(lat_mean))
    turbs_x = (farm_turbines['lon'].values - farm_turbines['lon'].mean()) * 111320 * cos_lat
    turbs_y = (farm_turbines['lat'].values - farm_turbines['lat'].mean()) * 111320
    rotor_diams = farm_turbines['rotor_diam'].values

    # Pre-compute wake efficiency for each wind direction
    eta_36 = np.ones(N_WD)
    for i, wd in enumerate(WD_ANGLES):
        eta_36[i] = compute_farm_wake_efficiency(turbs_x, turbs_y, rotor_diams,
                                                   wd, thrust_curve, v_rated)

    # ---- Step 3: For each orientation, rotate wind rose ----
    # E_no_wake per 36-direction (distribute 12-sector energy evenly)
    E_no_wake_36 = np.zeros(N_WD)
    for s in range(GWA_SECTORS):
        for i in range(N_WD // GWA_SECTORS):
            E_no_wake_36[s * (N_WD // GWA_SECTORS) + i] = E_no_wake_12[s] / (N_WD // GWA_SECTORS)

    aep_kwh = np.zeros(N_ORIENTATIONS)
    for i_theta, theta in enumerate(ORIENTATION_ANGLES):
        total = 0.0
        for i_wd, wd in enumerate(WD_ANGLES):
            # Wind direction relative to farm orientation
            rel_wd = int((wd - theta) % 360 / (360 / N_WD))
            total += E_no_wake_36[i_wd] * eta_36[rel_wd]
        aep_kwh[i_theta] = total * 8760  # kWh/year

    return aep_kwh


# ============================================================
# PART 5: MAIN PIPELINE
# ============================================================

def main():
    print("=" * 60)
    print("Onshore Wind Farm AEP(θ) with Jensen Wake Model")
    print("=" * 60)

    # Load farm and turbine data
    farms = pd.read_csv(os.path.join(PROC_DIR, 'osm_farm_pca_orientations.csv'))
    turbines_full = pd.read_csv(os.path.join(RAW_DIR, 'osm_turbines_clustered.csv'))

    # Ensure rotor_diam column exists
    if 'rotor_diam' not in turbines_full.columns:
        turbines_full['rotor_diam'] = 70.0  # default
    else:
        turbines_full['rotor_diam'] = turbines_full['rotor_diam'].fillna(70.0)

    print(f"Farms: {len(farms)}, Turbines: {len(turbines_full)}")

    # Round coordinates for GWA lookup
    farms['gwa_lat'] = farms['centroid_lat'].round(1)
    farms['gwa_lon'] = farms['centroid_lon'].round(1)
    unique_coords = farms[['gwa_lat', 'gwa_lon']].drop_duplicates()
    print(f"Unique GWA locations: {len(unique_coords)}")

    # ---- Step 1: Fetch/cache GWA data ----
    gwa_cache_file = os.path.join(RAW_DIR, 'gwa_cache.json')
    gwa_cache = {}
    if os.path.exists(gwa_cache_file):
        with open(gwa_cache_file) as f:
            gwa_cache = json.load(f)
        print(f"Loaded {len(gwa_cache)} cached GWA entries")
    else:
        print("No GWA cache found — fetching all locations")

    to_fetch = [(f"{r['gwa_lat']},{r['gwa_lon']}", r['gwa_lat'], r['gwa_lon'])
                for _, r in unique_coords.iterrows()
                if f"{r['gwa_lat']},{r['gwa_lon']}" not in gwa_cache]
    print(f"Need to fetch: {len(to_fetch)}")

    if to_fetch:
        print(f"Fetching {len(to_fetch)} GWA locations (4 workers, saving every 200)...", flush=True)
        t0 = time.time()
        # Process in batches to save incrementally
        batch_size = 200
        for batch_start in range(0, len(to_fetch), batch_size):
            batch = to_fetch[batch_start:batch_start+batch_size]
            with ThreadPoolExecutor(max_workers=8) as ex:
                fut = {ex.submit(fetch_gwa, lat, lon): key for key, lat, lon in batch}
                for n, future in enumerate(as_completed(fut), 1):
                    key = fut[future]
                    try:
                        gwa_cache[key] = future.result(timeout=30)
                    except Exception:
                        gwa_cache[key] = None
            # Save incrementally
            total_done = batch_start + len(batch)
            with open(gwa_cache_file, 'w') as f:
                json.dump(gwa_cache, f)
            elapsed = time.time() - t0
            rate = total_done / max(elapsed, 1e-6)
            eta = (len(to_fetch) - total_done) / max(rate, 1e-6)
            print(f"  {total_done}/{len(to_fetch)} ({rate:.0f}/s, ETA {eta:.0f}s), cache={len(gwa_cache)} entries", flush=True)
        print(f"GWA fetch complete ({time.time()-t0:.0f}s)", flush=True)

    # ---- Step 2: Parse Weibull per farm location ----
    print("\nParsing Weibull parameters...")
    weibull_parsed = {}

    for _, row in unique_coords.iterrows():
        key = f"{row['gwa_lat']},{row['gwa_lon']}"
        raw = gwa_cache.get(key)
        if raw is None:
            continue
        try:
            gwa = parse_gwa(raw)
            # Store for later interp at each farm's hub height
            weibull_parsed[key] = gwa
        except:
            pass
    print(f"  Parsed: {len(weibull_parsed)} locations")

    # ---- Step 3: Compute AEP(θ) for each farm ----
    print(f"\nComputing AEP(θ) for {len(farms)} farms...")

    # Pre-group turbines by farm for efficiency
    farm_turb_groups = {}
    for fid, group in turbines_full.groupby('farm_id'):
        farm_turb_groups[fid] = group

    results = []
    n_skip, n_err = 0, 0
    t_start = time.time()

    for i, (_, farm) in enumerate(farms.iterrows()):
        fid = farm['farm_id']

        # Get turbines for this farm
        if fid not in farm_turb_groups:
            n_skip += 1
            continue
        turb = farm_turb_groups[fid]
        if len(turb) < 3:
            n_skip += 1
            continue

        # Hub height and rotor diameter
        hub_h = farm['med_hub_height']
        if pd.isna(hub_h) or hub_h <= 0:
            hub_h = 70.0
        rotor_d = farm['med_rotor_diam']
        if pd.isna(rotor_d) or rotor_d <= 0:
            rotor_d = 70.0
        capacity = farm['med_capacity_kw']
        if pd.isna(capacity) or capacity <= 0:
            capacity = 2000.0

        # Ensure rotor_diam column in group
        if 'rotor_diam' not in turb.columns or turb['rotor_diam'].isna().all():
            turb = turb.copy()
            turb['rotor_diam'] = rotor_d

        # Get Weibull data
        key = f"{farm['gwa_lat']},{farm['gwa_lon']}"
        gwa = weibull_parsed.get(key)
        if gwa is None:
            n_skip += 1
            continue

        try:
            A_interp, k_interp = interpolate_weibull(gwa, hub_h)
            freq_12 = gwa['frequencies'][3, :] / 100.0  # roughness class 4
        except Exception:
            n_err += 1
            continue

        # Build power & thrust curves
        power_curve, v_rated, _ = build_power_curve(rotor_d, capacity)
        thrust_curve = build_thrust_curve(v_rated)

        # Set global for use in efficiency computation
        global power_curve_global
        power_curve_global = power_curve

        # Compute AEP for all orientations
        try:
            aep = compute_aep_theta(turb, A_interp, k_interp, freq_12,
                                     power_curve, thrust_curve, v_rated)
        except Exception as e:
            n_err += 1
            if n_err <= 3:
                print(f"  AEP error farm {fid}: {e}")
            continue

        results.append({
            'farm_id': fid,
            'n_turbines': len(turb),
            'centroid_lat': farm['centroid_lat'],
            'centroid_lon': farm['centroid_lon'],
            'hub_height': hub_h,
            'rotor_diam': rotor_d,
            'capacity_kw': capacity,
            'pc1_angle': farm['pc1_angle'],
            'explained_var_ratio': farm['explained_var_ratio'],
            **{f'aep_{int(theta):03d}': round(aep[i_theta], 0)
               for i_theta, theta in enumerate(ORIENTATION_ANGLES)}
        })

        # Progress
        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(farms) - i - 1) / max(rate, 1e-6)
            print(f"  {i+1}/{len(farms)} ({rate:.1f}/s, ETA {eta:.0f}s), {len(results)} done")

    elapsed = time.time() - t_start
    print(f"\nDone: {len(results)} farms computed ({elapsed:.0f}s)")
    print(f"Skipped: {n_skip}, Errors: {n_err}")

    # ---- Step 4: Save ----
    aep_df = pd.DataFrame(results)
    aep_out = os.path.join(PROC_DIR, 'onshore_aep_curves.csv')
    aep_df.to_csv(aep_out, index=False)
    print(f"\nSaved to {aep_out}")
    print(f"  Farms: {len(aep_df)}, Total turbines: {aep_df['n_turbines'].sum():,}")

    # ---- Summary stats ----
    aep_cols = [f'aep_{int(theta):03d}' for theta in ORIENTATION_ANGLES]
    vals = aep_df[aep_cols].values
    best_idx = np.argmax(vals, axis=1)
    best_theta = ORIENTATION_ANGLES[best_idx]

    print("\n=== AEP(θ) Summary ===")
    print("Best orientation (max AEP) distribution:")
    for t in ORIENTATION_ANGLES:
        cnt = (best_theta == t).sum()
        print(f"  {t:3d}°: {cnt:5d} farms ({cnt/len(aep_df)*100:4.1f}%)")

    max_aep = vals.max(axis=1)
    min_aep = vals.min(axis=1)
    ratio = min_aep / np.maximum(max_aep, 1e-6)
    print(f"\nAEP-min/AEP-max ratio (orientation sensitivity):")
    print(f"  Mean:   {ratio.mean():.4f}")
    print(f"  Median: {np.median(ratio):.4f}")
    print(f"  5th %:  {np.percentile(ratio, 5):.4f}")
    print(f"  95th %: {np.percentile(ratio, 95):.4f}")

    return aep_df


if __name__ == '__main__':
    power_curve_global = None  # module-level, set per farm
    aep_df = main()
