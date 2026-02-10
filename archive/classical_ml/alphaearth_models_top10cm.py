import pandas as pd
from pathlib import Path
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR, LinearSVR
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import cubist
    HAVE_CUBIST = True
    try:
        import re
        from cubist import _make_names_string as _mns
        from cubist import _make_data_string as _mds

        def _safe_escapes(x):
            chars = [":", ";", "|"]
            out = []
            for c in x:
                s = "" if c is None else str(c)
                for ch in chars:
                    s = s.replace(ch, f"\\{ch}")
                out.append(re.escape(s))
            return out

        _mns._escapes = _safe_escapes
        _mds._escapes = _safe_escapes
    except Exception:
        pass
except Exception:
    HAVE_CUBIST = False

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm_alphaearth_5yr_median.csv"
OUT_DIR = BASE_DIR / "outputs" / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "alphaearth_top10cm_model_metrics.csv"
OUT_RFE = OUT_DIR / "alphaearth_top10cm_svr_rfe_features.csv"


def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def main():
    df = pd.read_csv(IN_CSV)
    band_cols = [c for c in df.columns if c.startswith("A")]

    targets = ["ph", "cec_cmolkg", "esp_pct", "na_cmolkg"]
    results = []
    rfe_rows = []

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    for target in targets:
        if target not in df.columns:
            continue
        dft = df[band_cols + [target]].copy()
        dft = dft.dropna(subset=[target])
        dft = dft.dropna(subset=band_cols)
        if len(dft) < 10:
            continue

        X_df = dft[band_cols].copy()
        X = X_df.values
        y = dft[target].values

        # SVR baseline (all bands)
        svr_pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR(kernel="rbf"))
        ])
        svr_grid = {
            "svr__C": [1, 10],
            "svr__gamma": ["scale", 0.1],
            "svr__epsilon": [0.1],
        }
        grid = GridSearchCV(svr_pipe, svr_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        grid.fit(X, y)
        best = grid.best_estimator_
        preds = cross_val_predict(best, X, y, cv=cv, n_jobs=-1)
        rmse, mae, r2 = eval_metrics(y, preds)
        results.append({
            "model": "SVR_all64",
            "target": target,
            "n": len(y),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "params": grid.best_params_,
            "n_features": len(band_cols),
        })

        # SVR + RFE (linear SVR for selection)
        rfe_sizes = [12, 24]
        best_rfe = None
        best_rfe_rmse = None
        for k in rfe_sizes:
            rfe = RFE(
                estimator=LinearSVR(
                    C=1.0,
                    epsilon=0.1,
                    dual=False,
                    loss="squared_epsilon_insensitive",
                    max_iter=5000,
                ),
                n_features_to_select=k,
            )
            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("rfe", rfe),
                ("svr", SVR(kernel="rbf"))
            ])
            grid = GridSearchCV(pipe, svr_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            grid.fit(X, y)
            best = grid.best_estimator_
            preds = cross_val_predict(best, X, y, cv=cv, n_jobs=-1)
            rmse, mae, r2 = eval_metrics(y, preds)

            results.append({
                "model": "SVR_RFE",
                "target": target,
                "n": len(y),
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "params": grid.best_params_,
                "n_features": k,
            })

            if best_rfe_rmse is None or rmse < best_rfe_rmse:
                best_rfe_rmse = rmse
                best_rfe = best

        if best_rfe is not None:
            support = best_rfe.named_steps["rfe"].support_
            selected = [b for b, keep in zip(band_cols, support) if keep]
            rfe_rows.append({
                "target": target,
                "n_features": len(selected),
                "bands": ";".join(selected),
            })

        # Random Forest
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        rf_grid = {
            "n_estimators": [300],
            "max_depth": [None, 20],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        }
        rf_grid_search = GridSearchCV(rf, rf_grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        rf_grid_search.fit(X, y)
        rf_best = rf_grid_search.best_estimator_
        rf_preds = cross_val_predict(rf_best, X, y, cv=cv, n_jobs=-1)
        rmse, mae, r2 = eval_metrics(y, rf_preds)
        results.append({
            "model": "RandomForest",
            "target": target,
            "n": len(y),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "params": rf_grid_search.best_params_,
            "n_features": len(band_cols),
        })

        # Cubist (if available)
        if HAVE_CUBIST:
            committees = [10, 20]
            neighbors = [None, 3]
            best_score = None
            best_params = None
            best_preds = None

            for c in committees:
                for n in neighbors:
                    preds = np.zeros_like(y, dtype=float)
                    for train_idx, test_idx in cv.split(X):
                        model = cubist.Cubist(n_committees=c, neighbors=n, unbiased=False, auto=False)
                        model.fit(X_df.iloc[train_idx], y[train_idx])
                        preds[test_idx] = model.predict(X_df.iloc[test_idx])
                    rmse, mae, r2 = eval_metrics(y, preds)
                    score = rmse
                    if best_score is None or score < best_score:
                        best_score = score
                        best_params = {"committees": c, "neighbors": n}
                        best_preds = preds

            if best_preds is not None:
                rmse, mae, r2 = eval_metrics(y, best_preds)
                results.append({
                    "model": "Cubist",
                    "target": target,
                    "n": len(y),
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "params": best_params,
                    "n_features": len(band_cols),
                })

    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    if rfe_rows:
        pd.DataFrame(rfe_rows).to_csv(OUT_RFE, index=False)

    print(f"Wrote {OUT_CSV}")
    if rfe_rows:
        print(f"Wrote {OUT_RFE}")


if __name__ == "__main__":
    main()
