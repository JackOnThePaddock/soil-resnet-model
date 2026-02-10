from pathlib import Path
import re

import numpy as np
import rasterio


BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
MOS_DIR = BASE_DIR / "outputs" / "predictions" / "mosaics"
NODATA = -9999.0


def list_yearly(label):
    paths = []
    for p in MOS_DIR.glob(f"farm_mosaic_{label}_*.tif"):
        if re.search(r"_median_|_mean_", p.name):
            continue
        m = re.search(r"_(\d{4})\.tif$", p.name)
        if m:
            paths.append((int(m.group(1)), p))
    paths.sort()
    return paths


def mean_mosaic(label, mosaic_paths, out_path):
    srcs = [rasterio.open(p) for p in mosaic_paths]
    ref = srcs[0]
    for s in srcs[1:]:
        if (
            s.width != ref.width
            or s.height != ref.height
            or s.transform != ref.transform
            or s.crs != ref.crs
        ):
            raise RuntimeError("Mosaic grids do not match; reproject required.")

    profile = ref.profile.copy()
    profile.update(count=1, dtype="float32", nodata=NODATA, compress="deflate")

    with rasterio.open(out_path, "w", **profile) as dst:
        for _, window in ref.block_windows(1):
            stack = []
            for s in srcs:
                arr = s.read(1, window=window).astype("float32")
                arr[arr == NODATA] = np.nan
                stack.append(arr)
            data = np.nanmean(np.stack(stack, axis=0), axis=0)
            data = np.where(np.isnan(data), NODATA, data).astype("float32")
            dst.write(data, 1, window=window)

    for s in srcs:
        s.close()
    print(f"Wrote {out_path}")


def main():
    for label in ["ph", "cec", "esp"]:
        yearly = list_yearly(label)
        if not yearly:
            print(f"No yearly mosaics for {label}")
            continue
        years = [y for y, _ in yearly]
        paths = [p for _, p in yearly]
        span = f"{years[0]}_{years[-1]}" if len(years) > 1 else f"{years[0]}"
        out_path = MOS_DIR / f"farm_mosaic_{label}_mean_{span}.tif"
        mean_mosaic(label, paths, out_path)


if __name__ == "__main__":
    main()
