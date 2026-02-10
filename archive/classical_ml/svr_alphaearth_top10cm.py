import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median.csv"
OUT_DIR = BASE_DIR / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "svr_alphaearth_top10cm_metrics.csv"


def main():
    df = pd.read_csv(IN_CSV)
    band_cols = [c for c in df.columns if c.startswith("A")]

    results = []
    targets = ["ph", "cec_cmolkg", "esp_pct", "na_cmolkg"]

    for target in targets:
        if target not in df.columns:
            continue
        dft = df[band_cols + [target]].copy()
        dft = dft.dropna(subset=[target])
        dft = dft.dropna(subset=band_cols)
        if len(dft) < 10:
            continue

        X = dft[band_cols].values
        y = dft[target].values

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf"))
        ])

        param_grid = {
            "svr__C": [1, 10, 100],
            "svr__gamma": ["scale", 0.1, 0.01],
            "svr__epsilon": [0.05, 0.1, 0.2],
        }

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(pipe, param_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X, y)

        best = grid.best_estimator_
        preds = cross_val_predict(best, X, y, cv=cv, n_jobs=-1)

        rmse = np.sqrt(mean_squared_error(y, preds))
        mae = mean_absolute_error(y, preds)
        r2 = r2_score(y, preds)

        results.append({
            "target": target,
            "n": len(y),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "best_params": grid.best_params_,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
