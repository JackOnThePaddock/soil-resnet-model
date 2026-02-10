import numpy as np
import pandas as pd
import rasterio
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor


def predict_raster(in_path, out_path, model, band_cols, selected_features):
    nodata = -9999.0
    idx = [band_cols.index(f) for f in selected_features]
    with rasterio.open(in_path) as src:
        profile = src.profile
        profile.update(count=1, dtype="float32", nodata=nodata, compress="LZW")
        with rasterio.open(out_path, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                data = src.read(window=window, masked=True)
                if np.isscalar(data.mask) or data.mask.shape == ():
                    rows, cols = data.shape[1], data.shape[2]
                    mask = np.zeros((rows, cols), dtype=bool)
                else:
                    mask = np.any(data.mask, axis=0)
                    rows, cols = mask.shape
                flat = data.filled(np.nan).reshape(data.shape[0], -1).T
                valid = ~mask.reshape(-1)
                preds = np.full(flat.shape[0], nodata, dtype=np.float32)
                if np.any(valid):
                    Xv = flat[valid][:, idx]
                    preds[valid] = model.predict(Xv).astype(np.float32)
                preds = preds.reshape(rows, cols)
                dst.write(preds, 1, window=window)


def main():
    base_dir = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests")
    out_dir = base_dir / "exports" / "rf_combined_ph_cec_esp"
    train_path = out_dir / "training_data_combined.csv"
    best_path = out_dir / "best_features.csv"
    emb_path = base_dir / "exports" / "gpr_alphaearth" / "embeddings" / "TURKEY_YARD_alpha_5yr.tif"

    if not emb_path.exists():
        raise FileNotFoundError(emb_path)

    df = pd.read_csv(train_path)
    best = pd.read_csv(best_path)
    band_cols = [c for c in df.columns if c.startswith("A")]

    for target in ["pH", "CEC", "ESP"]:
        sub = df.dropna(subset=[target])
        feat_row = best[best["target"] == target].iloc[0]
        feats = [f.strip() for f in str(feat_row["features"]).split(",") if f.strip()]

        X = sub[feats].values.astype(np.float32)
        y = sub[target].values.astype(np.float32)

        model = RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_features="sqrt",
        )
        model.fit(X, y)

        out_path = out_dir / f"TURKEY_YARD_{target}_rf_bestbands.tif"
        predict_raster(emb_path, out_path, model, band_cols, feats)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
