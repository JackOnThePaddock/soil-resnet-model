"""Raster predictions from AlphaEarth GeoTIFF embeddings."""

from pathlib import Path
from typing import Dict, Union

import numpy as np

from src.models.ensemble import SoilEnsemble

NODATA = -9999.0


def predict_raster(
    ensemble: SoilEnsemble, input_path: Union[str, Path],
    output_dir: Union[str, Path], block_size: int = 512, nodata: float = NODATA,
) -> Dict[str, Path]:
    """Generate prediction GeoTIFFs (2 bands: mean + uncertainty) per target."""
    import rasterio
    from rasterio.windows import Window

    input_path, output_dir = Path(input_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        profile = src.profile.copy()
        height, width, n_bands = src.height, src.width, src.count
        profile.update(count=2, dtype="float32", nodata=nodata, compress="deflate")

        output_files, writers = {}, {}
        for target in ensemble.target_names:
            out_path = output_dir / f"{input_path.stem}_{target}.tif"
            writers[target] = rasterio.open(out_path, "w", **profile)
            output_files[target] = out_path

        for row in range(0, height, block_size):
            for col in range(0, width, block_size):
                win_h, win_w = min(block_size, height - row), min(block_size, width - col)
                window = Window(col, row, win_w, win_h)
                block = src.read(window=window)
                n_pixels = win_h * win_w
                X = block.reshape(n_bands, -1).T

                valid_mask = np.isfinite(X).all(axis=1)
                if src.nodata is not None and np.isfinite(src.nodata):
                    valid_mask &= ~(X == src.nodata).any(axis=1)

                mean_out = np.full((len(ensemble.target_names), n_pixels), nodata, dtype=np.float32)
                std_out = np.full((len(ensemble.target_names), n_pixels), nodata, dtype=np.float32)

                if valid_mask.sum() > 0:
                    mean_preds, std_preds = ensemble.predict_batch(X[valid_mask])
                    for i in range(len(ensemble.target_names)):
                        mean_out[i, valid_mask] = mean_preds[:, i]
                        std_out[i, valid_mask] = std_preds[:, i]

                for i, target in enumerate(ensemble.target_names):
                    writers[target].write(mean_out[i].reshape(win_h, win_w), 1, window=window)
                    writers[target].write(std_out[i].reshape(win_h, win_w), 2, window=window)

        for writer in writers.values():
            writer.close()

    return output_files
