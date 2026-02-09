"""Point-based predictions from CSV using the trained ensemble."""

from typing import List, Optional

import pandas as pd

from src.models.ensemble import SoilEnsemble


def predict_from_csv(
    input_csv: str, models_dir: str, output_csv: str,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load CSV, run ensemble predictions, save results."""
    ensemble = SoilEnsemble(models_dir)
    df = pd.read_csv(input_csv)
    print(f"Loaded {len(df)} samples from {input_csv}")

    result = ensemble.predict_df(df, feature_cols=feature_cols)
    result.to_csv(output_csv, index=False)
    print(f"Predictions saved to {output_csv}")

    for target in ensemble.target_names:
        pred_col = f"{target}_pred"
        std_col = f"{target}_std"
        if pred_col in result.columns:
            print(f"  {target}: mean={result[pred_col].mean():.3f}, "
                  f"uncertainty={result[std_col].mean():.3f}")
    return result
