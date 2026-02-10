import numpy as np
import pandas as pd
from pathlib import Path
import re

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from catboost import CatBoostRegressor

try:
    import cubist
    HAVE_CUBIST = True
    try:
        import re as _re
        from cubist import _make_names_string as _mns
        from cubist import _make_data_string as _mds

        def _safe_escapes(x):
            chars = [":", ";", "|"]
            out = []
            for c in x:
                s = "" if c is None else str(c)
                for ch in chars:
                    s = s.replace(ch, f"\\{ch}")
                out.append(_re.escape(s))
            return out

        _mns._escapes = _safe_escapes
        _mds._escapes = _safe_escapes
    except Exception:
        pass
except Exception:
    HAVE_CUBIST = False


BASE_DIR = Path(r"C:\Users\jackc\Documents\National Soil Data Standardised")
MERGED_DIR = BASE_DIR / "by_year_cleaned_top10cm_metrics_alphaearth" / "merged"
OUT_DIR = BASE_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_METRICS = OUT_DIR / "alphaearth_merged_model_metrics.csv"


ATTRS = {
    "ph": MERGED_DIR / "top10cm_ph_alphaearth_merged.csv",
    "cec_cmolkg": MERGED_DIR / "top10cm_cec_cmolkg_alphaearth_merged.csv",
    "esp_pct": MERGED_DIR / "top10cm_esp_pct_alphaearth_merged.csv",
}


def band_cols(df):
    return [c for c in df.columns if re.fullmatch(r"A\d{2}", c)]


def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


def select_best_catboost(X, y, cv):
    grid = [
        {"depth": 6, "learning_rate": 0.1, "l2_leaf_reg": 3},
        {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 5},
        {"depth": 8, "learning_rate": 0.1, "l2_leaf_reg": 3},
    ]
    best_params = None
    best_rmse = None
    for params in grid:
        model = CatBoostRegressor(
            iterations=500,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
            thread_count=4,
            **params,
        )
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X.iloc[test_idx])
        rmse = np.sqrt(mean_squared_error(y, preds))
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    return best_params


def select_best_cubist(X, y, cv):
    if not HAVE_CUBIST:
        return None
    grid = [
        {"n_committees": 10, "neighbors": None},
        {"n_committees": 10, "neighbors": 3},
        {"n_committees": 20, "neighbors": None},
        {"n_committees": 20, "neighbors": 3},
    ]
    best_params = None
    best_rmse = None
    for params in grid:
        model = cubist.Cubist(unbiased=False, auto=False, **params)
        preds = np.zeros_like(y, dtype=float)
        for train_idx, test_idx in cv.split(X):
            model.fit(X.iloc[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X.iloc[test_idx])
        rmse = np.sqrt(mean_squared_error(y, preds))
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_params = params
    return best_params


def run_for_attr(attr, path):
    df = pd.read_csv(path)
    bands = band_cols(df)
    if not bands:
        raise RuntimeError(f"No bands found in {path}")

    df = df[bands + [attr]].dropna()
    X = df[bands]
    y = df[attr].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rows = []

    # SVR RBF
    svr_pipe = Pipeline([("scaler", StandardScaler()), ("svr", SVR(kernel="rbf"))])
    svr_grid = {
        "svr__C": [0.1, 1, 10, 100],
        "svr__gamma": ["scale", 0.1, 0.01],
        "svr__epsilon": [0.05, 0.1, 0.2],
    }
    svr_search = GridSearchCV(
        svr_pipe,
        svr_grid,
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    svr_search.fit(X_train, y_train)
    svr_best = svr_search.best_estimator_
    pred = svr_best.predict(X_test)
    rmse, mae, r2 = metrics(y_test, pred)
    rows.append(
        {
            "target": attr,
            "model": "SVR_RBF",
            "params": svr_search.best_params_,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": len(df),
        }
    )

    # Random Forest
    rf = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, max_features="sqrt"
    )
    rf.fit(X_train, y_train)
    pred = rf.predict(X_test)
    rmse, mae, r2 = metrics(y_test, pred)
    rows.append(
        {
            "target": attr,
            "model": "RandomForest",
            "params": {"n_estimators": 300, "max_features": "sqrt"},
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": len(df),
        }
    )

    # Extra Trees
    et = ExtraTreesRegressor(
        n_estimators=300, random_state=42, n_jobs=-1, max_features="sqrt"
    )
    et.fit(X_train, y_train)
    pred = et.predict(X_test)
    rmse, mae, r2 = metrics(y_test, pred)
    rows.append(
        {
            "target": attr,
            "model": "ExtraTrees",
            "params": {"n_estimators": 300, "max_features": "sqrt"},
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": len(df),
        }
    )

    # CatBoost
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cb_params = select_best_catboost(X_train, y_train, cv)
    cb = CatBoostRegressor(
        iterations=500,
        loss_function="RMSE",
        random_seed=42,
        verbose=False,
        thread_count=4,
        **cb_params,
    )
    cb.fit(X_train, y_train)
    pred = cb.predict(X_test)
    rmse, mae, r2 = metrics(y_test, pred)
    rows.append(
        {
            "target": attr,
            "model": "CatBoost",
            "params": cb_params,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "n_samples": len(df),
        }
    )

    # Cubist (if available)
    if HAVE_CUBIST:
        cubist_params = select_best_cubist(X_train, y_train, cv)
        if cubist_params:
            cu = cubist.Cubist(unbiased=False, auto=False, **cubist_params)
            cu.fit(X_train, y_train)
            pred = cu.predict(X_test)
            rmse, mae, r2 = metrics(y_test, pred)
            rows.append(
                {
                    "target": attr,
                    "model": "Cubist",
                    "params": cubist_params,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "n_samples": len(df),
                }
            )
    else:
        rows.append(
            {
                "target": attr,
                "model": "Cubist",
                "params": "not_installed",
                "rmse": None,
                "mae": None,
                "r2": None,
                "n_samples": len(df),
            }
        )

    return rows


def main():
    all_rows = []
    for attr, path in ATTRS.items():
        if not path.exists():
            print(f"Missing {path}")
            continue
        all_rows.extend(run_for_attr(attr, path))

    df_out = pd.DataFrame(all_rows)
    df_out.to_csv(OUT_METRICS, index=False)
    print(f"Wrote {OUT_METRICS}")


if __name__ == "__main__":
    main()
