import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy
import shapefile
from shapely.geometry import shape, Point
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

EMB_DIR = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\gpr_alphaearth\embeddings"
BOUNDARIES = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SpeirsBoundaries\boundaries\boundaries.shp"
OUT_DIR = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\clhs_sampling"

N_SAMPLES = 50
MAX_PER_PADDOCK = 5000
PCA_COMPONENTS = 6
MAX_ITER = 20000
RESTARTS = 3
CORR_WEIGHT = 0.5
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# Load paddock boundaries
reader = shapefile.Reader(BOUNDARIES)
fields = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
name_idx = fields.index("FIELD_NAME")

paddock_geom = {}
for sr in reader.shapeRecords():
    name = str(sr.record[name_idx]).strip()
    paddock_geom[name] = shape(sr.shape.__geo_interface__)


def point_in_any(pt):
    for geom in paddock_geom.values():
        if geom.contains(pt) or geom.touches(pt):
            return True
    return False


# Collect candidate pixels from all paddock embeddings (inside boundaries)
X_list = []
coords_list = []
paddock_list = []

for fname in os.listdir(EMB_DIR):
    if not fname.endswith("_alpha_5yr.tif"):
        continue
    paddock = fname.replace("_alpha_5yr.tif", "")
    path = os.path.join(EMB_DIR, fname)
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        bands, height, width = data.shape
        flat = data.reshape(bands, -1).T
        n_pix = flat.shape[0]
        if n_pix == 0:
            continue
        n_pick = min(MAX_PER_PADDOCK, n_pix)
        idx = rng.choice(n_pix, size=n_pick, replace=False)
        X = flat[idx]
        # drop rows with NaN
        valid = ~np.any(~np.isfinite(X), axis=1)
        X = X[valid]
        if X.shape[0] == 0:
            continue
        idx = idx[valid]
        rows = idx // width
        cols = idx % width
        xs, ys = xy(src.transform, rows, cols)
        coords = np.column_stack([xs, ys])

        # keep only points inside boundaries
        keep_mask = []
        for x, y in coords:
            keep_mask.append(point_in_any(Point(x, y)))
        keep_mask = np.array(keep_mask, dtype=bool)
        X = X[keep_mask]
        coords = coords[keep_mask]

        if X.shape[0] == 0:
            continue

        X_list.append(X)
        coords_list.append(coords)
        paddock_list.extend([paddock] * len(X))

if not X_list:
    raise RuntimeError("No candidate pixels found inside boundaries.")

X_all = np.vstack(X_list)
coords_all = np.vstack(coords_list)
paddock_all = np.array(paddock_list)

# PCA to reduce dimensionality
scaler = StandardScaler()
X_std = scaler.fit_transform(X_all)

pca = PCA(n_components=PCA_COMPONENTS, random_state=SEED)
X_pca = pca.fit_transform(X_std)

# Precompute ranks and population correlation
n = X_pca.shape[0]
P = X_pca.shape[1]

rank_norm = np.zeros_like(X_pca, dtype=np.float32)
for j in range(P):
    ranks = rankdata(X_pca[:, j], method="average")
    rank_norm[:, j] = (ranks - 0.5) / n

z = (X_pca - X_pca.mean(axis=0)) / X_pca.std(axis=0, ddof=0)
if P > 1:
    corr_pop = np.corrcoef(z, rowvar=False)
else:
    corr_pop = np.array([[1.0]])

expected = (np.arange(N_SAMPLES) + 0.5) / N_SAMPLES


def cost(sample_idx):
    sample_ranks = rank_norm[sample_idx]
    total = 0.0
    for j in range(P):
        r = np.sort(sample_ranks[:, j])
        total += np.sum((r - expected) ** 2)
    if CORR_WEIGHT > 0 and P > 1:
        zs = z[sample_idx]
        corr_s = np.corrcoef(zs, rowvar=False)
        total += CORR_WEIGHT * np.sum((corr_s - corr_pop) ** 2)
    return float(total)


def clhs_run(seed_offset=0):
    local_rng = np.random.default_rng(SEED + seed_offset)
    N = n
    if N < N_SAMPLES:
        raise RuntimeError("Not enough candidates for the requested sample size.")
    sample_idx = local_rng.choice(N, size=N_SAMPLES, replace=False)
    unselected = np.setdiff1d(np.arange(N), sample_idx)
    best_idx = sample_idx.copy()
    best_cost = cost(best_idx)
    curr = sample_idx.copy()
    curr_cost = best_cost

    T0 = 1.0
    Tend = 1e-3

    for it in range(MAX_ITER):
        T = T0 * (Tend / T0) ** (it / MAX_ITER)
        out_pos = local_rng.integers(0, N_SAMPLES)
        in_pos = local_rng.integers(0, len(unselected))
        out_idx = curr[out_pos]
        in_idx = unselected[in_pos]

        new_sample = curr.copy()
        new_sample[out_pos] = in_idx
        new_cost = cost(new_sample)
        delta = new_cost - curr_cost

        if delta < 0 or local_rng.random() < np.exp(-delta / T):
            curr[out_pos] = in_idx
            unselected[in_pos] = out_idx
            curr_cost = new_cost
            if curr_cost < best_cost:
                best_cost = curr_cost
                best_idx = curr.copy()

    return best_idx, best_cost


best_idx = None
best_cost = None
for r in range(RESTARTS):
    idx, c = clhs_run(seed_offset=1000 * r)
    if best_cost is None or c < best_cost:
        best_cost = c
        best_idx = idx

best_idx = np.array(best_idx, dtype=int)

# Build output table
out = pd.DataFrame({
    "id": np.arange(1, len(best_idx) + 1),
    "paddock": paddock_all[best_idx],
    "lon": coords_all[best_idx][:, 0],
    "lat": coords_all[best_idx][:, 1],
})

csv_path = os.path.join(OUT_DIR, "clhs_sampling_points_50_inside_boundaries.csv")
out.to_csv(csv_path, index=False)

kml_path = os.path.join(OUT_DIR, "clhs_sampling_points_50_inside_boundaries.kml")
with open(kml_path, "w", encoding="ascii") as f:
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
    f.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n")
    f.write("  <Document>\n")
    f.write("    <name>CLHS Sampling Points (n=50, inside boundaries)</name>\n")
    for row in out.itertuples(index=False):
        f.write("    <Placemark>\n")
        f.write(f"      <name>CLHS_{row.id:02d}</name>\n")
        f.write(f"      <description>Paddock: {row.paddock}</description>\n")
        f.write("      <Point>\n")
        f.write(f"        <coordinates>{row.lon},{row.lat},0</coordinates>\n")
        f.write("      </Point>\n")
        f.write("    </Placemark>\n")
    f.write("  </Document>\n")
    f.write("</kml>\n")

print(f"Saved {csv_path}")
print(f"Saved {kml_path}")
