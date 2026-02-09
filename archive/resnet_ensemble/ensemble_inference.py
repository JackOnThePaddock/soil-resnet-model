"""
Ensemble Inference Module for ResNet Soil Prediction
=====================================================
Loads trained ensemble models and generates predictions with uncertainty.

Usage:
    # Point predictions from DataFrame
    python ensemble_inference.py --mode points --input data.csv --output predictions.csv

    # Raster predictions from GeoTIFF
    python ensemble_inference.py --mode raster --input alphaearth.tif --output predictions/
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Import model architecture from trainer
from resnet_soil_trainer import NationalSoilNet, CONFIG


# ============================================================================
# Ensemble Loading
# ============================================================================

class SoilEnsemble:
    """
    Loads and manages an ensemble of trained ResNet models.

    Provides methods for:
        - Point predictions with uncertainty
        - Batch predictions from DataFrames
        - Raster predictions from GeoTIFFs
    """

    def __init__(
        self,
        models_dir: Union[str, Path],
        device: Optional[torch.device] = None,
    ):
        """
        Load ensemble from saved models.

        Args:
            models_dir: Directory containing model_1.pth, ..., model_N.pth and scaler.pkl
            device: Torch device (auto-detected if None)
        """
        self.models_dir = Path(models_dir)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Loading ensemble from: {self.models_dir}")
        print(f"Using device: {self.device}")

        # Load scaler
        scaler_path = self.models_dir / 'scaler.pkl'
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found: {scaler_path}")

        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        # Find and load models
        model_files = sorted(self.models_dir.glob('model_*.pth'))
        if not model_files:
            raise FileNotFoundError(f"No model files found in {self.models_dir}")

        self.models = []
        self.target_names = None
        self.config = None

        for model_path in model_files:
            checkpoint = torch.load(model_path, map_location=self.device)

            # Get config and targets from first model
            if self.config is None:
                self.config = checkpoint.get('config', CONFIG)
                self.target_names = checkpoint.get('target_names', CONFIG['targets'])

            # Create and load model
            model = NationalSoilNet(
                input_dim=self.config['input_dim'],
                hidden_dim=self.config['hidden_dim'],
                num_res_blocks=self.config['num_res_blocks'],
                dropout=self.config['dropout'],
                target_names=self.target_names,
            ).to(self.device)

            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            self.models.append(model)

            print(f"  Loaded: {model_path.name}")

        print(f"Loaded {len(self.models)} models")
        print(f"Targets: {self.target_names}")

    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Generate predictions from raw features.

        Args:
            X: Feature array of shape (n_samples, 64)
            return_std: Whether to return standard deviation (uncertainty)

        Returns:
            If return_std=False: predictions array (n_samples, n_targets)
            If return_std=True: (predictions, std) tuple
        """
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)

        # Collect predictions from all models
        all_preds = []

        with torch.no_grad():
            for model in self.models:
                preds = model.forward_stacked(X_tensor)
                all_preds.append(preds.cpu().numpy())

        # Stack: (n_models, n_samples, n_targets)
        all_preds = np.stack(all_preds, axis=0)

        # Ensemble mean and std
        mean_preds = np.mean(all_preds, axis=0)
        std_preds = np.std(all_preds, axis=0)

        if return_std:
            return mean_preds, std_preds
        return mean_preds

    def predict_df(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions from a DataFrame.

        Args:
            df: DataFrame with feature columns
            feature_cols: List of feature column names (default: A00-A63)

        Returns:
            DataFrame with predictions and uncertainties for each target
        """
        if feature_cols is None:
            feature_cols = [f'A{i:02d}' for i in range(64)]

        # Normalize column names
        df_cols = df.columns.str.lower()
        feature_cols_lower = [c.lower() for c in feature_cols]

        # Extract features
        X = df[[c for c in df.columns if c.lower() in feature_cols_lower]].values

        # Predict
        mean_preds, std_preds = self.predict(X, return_std=True)

        # Build output DataFrame
        result = df.copy()

        for i, target in enumerate(self.target_names):
            result[f'{target}_pred'] = mean_preds[:, i]
            result[f'{target}_std'] = std_preds[:, i]

        return result

    def predict_batch(
        self,
        X: np.ndarray,
        batch_size: int = 1024,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Batch prediction for large arrays (memory efficient).

        Args:
            X: Feature array (n_samples, 64)
            batch_size: Number of samples per batch

        Returns:
            (mean_predictions, std_predictions) arrays
        """
        n_samples = X.shape[0]
        n_targets = len(self.target_names)

        mean_preds = np.zeros((n_samples, n_targets), dtype=np.float32)
        std_preds = np.zeros((n_samples, n_targets), dtype=np.float32)

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_X = X[start:end]

            batch_mean, batch_std = self.predict(batch_X, return_std=True)
            mean_preds[start:end] = batch_mean
            std_preds[start:end] = batch_std

        return mean_preds, std_preds


# ============================================================================
# Raster Prediction
# ============================================================================

def predict_raster(
    ensemble: SoilEnsemble,
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    block_size: int = 512,
    nodata: float = -9999.0,
) -> Dict[str, Path]:
    """
    Generate prediction rasters from AlphaEarth embedding GeoTIFF.

    Creates one GeoTIFF per target with prediction and uncertainty bands.

    Args:
        ensemble: Loaded SoilEnsemble instance
        input_path: Path to input GeoTIFF (64-band AlphaEarth embeddings)
        output_dir: Directory for output GeoTIFFs
        block_size: Processing block size in pixels
        nodata: NoData value for output

    Returns:
        Dictionary mapping target names to output file paths
    """
    try:
        import rasterio
        from rasterio.windows import Window
    except ImportError:
        raise ImportError("rasterio is required for raster predictions: pip install rasterio")

    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing raster: {input_path}")

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        n_bands = src.count

        if n_bands != 64:
            print(f"Warning: Expected 64 bands, got {n_bands}")

        print(f"  Dimensions: {width} x {height}")
        print(f"  Bands: {n_bands}")

        # Update profile for output (2 bands: prediction + uncertainty)
        profile.update(
            count=2,
            dtype='float32',
            nodata=nodata,
        )

        output_files = {}

        # Create output files for each target
        writers = {}
        for target in ensemble.target_names:
            output_path = output_dir / f'{input_path.stem}_{target}.tif'
            writers[target] = rasterio.open(output_path, 'w', **profile)
            output_files[target] = output_path

        # Process in blocks
        total_blocks = ((height + block_size - 1) // block_size) * \
                       ((width + block_size - 1) // block_size)
        block_count = 0

        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                # Calculate window
                win_height = min(block_size, height - row)
                win_width = min(block_size, width - col)
                window = Window(col, row, win_width, win_height)

                # Read block (all bands)
                block = src.read(window=window)  # (64, h, w)

                # Reshape to (n_pixels, 64)
                n_pixels = win_height * win_width
                X = block.reshape(n_bands, -1).T  # (n_pixels, 64)

                # Find valid pixels (no NaN)
                valid_mask = ~np.isnan(X).any(axis=1)

                # Initialize outputs
                mean_out = np.full((len(ensemble.target_names), n_pixels), nodata, dtype=np.float32)
                std_out = np.full((len(ensemble.target_names), n_pixels), nodata, dtype=np.float32)

                if valid_mask.sum() > 0:
                    # Predict valid pixels
                    mean_preds, std_preds = ensemble.predict(X[valid_mask])

                    for i in range(len(ensemble.target_names)):
                        mean_out[i, valid_mask] = mean_preds[:, i]
                        std_out[i, valid_mask] = std_preds[:, i]

                # Write to output files
                for i, target in enumerate(ensemble.target_names):
                    # Reshape back to image
                    mean_img = mean_out[i].reshape(win_height, win_width)
                    std_img = std_out[i].reshape(win_height, win_width)

                    writers[target].write(mean_img, 1, window=window)
                    writers[target].write(std_img, 2, window=window)

                block_count += 1
                if block_count % 10 == 0:
                    print(f"  Processed {block_count}/{total_blocks} blocks...")

        # Close output files
        for writer in writers.values():
            writer.close()

    print(f"\nOutput files:")
    for target, path in output_files.items():
        print(f"  {target}: {path}")

    return output_files


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate predictions from trained ResNet ensemble'
    )
    parser.add_argument(
        '--mode', type=str, choices=['points', 'raster'], required=True,
        help='Prediction mode: "points" for CSV, "raster" for GeoTIFF'
    )
    parser.add_argument(
        '--input', type=str, required=True,
        help='Input file (CSV for points, GeoTIFF for raster)'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Output file (CSV) or directory (for raster)'
    )
    parser.add_argument(
        '--models', type=str,
        default=str(CONFIG['output_dir'] / 'models'),
        help='Directory containing trained models'
    )
    parser.add_argument(
        '--block_size', type=int, default=512,
        help='Block size for raster processing (default: 512)'
    )

    args = parser.parse_args()

    print("\n" + "="*60)
    print("ResNet Soil Prediction - Ensemble Inference")
    print("="*60)

    # Load ensemble
    ensemble = SoilEnsemble(args.models)

    if args.mode == 'points':
        # Point predictions from CSV
        print(f"\nLoading data from: {args.input}")
        df = pd.read_csv(args.input)
        print(f"  Samples: {len(df)}")

        print("\nGenerating predictions...")
        result = ensemble.predict_df(df)

        # Save results
        output_path = Path(args.output)
        result.to_csv(output_path, index=False)
        print(f"\nPredictions saved to: {output_path}")

        # Print summary statistics
        print("\nPrediction summary:")
        for target in ensemble.target_names:
            pred_col = f'{target}_pred'
            std_col = f'{target}_std'
            if pred_col in result.columns:
                print(f"  {target}:")
                print(f"    Mean prediction: {result[pred_col].mean():.3f}")
                print(f"    Mean uncertainty: {result[std_col].mean():.3f}")

    else:
        # Raster predictions from GeoTIFF
        output_files = predict_raster(
            ensemble=ensemble,
            input_path=args.input,
            output_dir=args.output,
            block_size=args.block_size,
        )

    print("\nInference complete!")


if __name__ == '__main__':
    main()
