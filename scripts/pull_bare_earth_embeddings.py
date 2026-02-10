#!/usr/bin/env python
"""Pull DEA Bare Earth values for training points and compute SpectralGPT embeddings."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.bare_earth import (  # noqa: E402
    DEFAULT_BAREST_EARTH_BANDS,
    DEFAULT_COVERAGE_ID,
    DEA_OWS_BASE_URL,
    sample_bare_earth_chips_via_wcs,
    sample_bare_earth_points_via_wcs,
)
from src.features.spectral_gpt import (  # noqa: E402
    OFFICIAL_SPECTRALGPT_PLUS_URL,
    OfficialSpectralGPTConfig,
    OfficialSpectralGPTEncoder,
    SpectralGPTConfig,
    reduce_embeddings_with_pca,
    train_spectral_gpt_embeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample GA Barest Earth at points and produce SpectralGPT embeddings"
    )
    parser.add_argument(
        "--normalized-csv",
        type=str,
        required=True,
        help="Normalized training CSV (may or may not include lon/lat)",
    )
    parser.add_argument(
        "--points-csv",
        type=str,
        default=None,
        help="Optional CSV providing lon/lat for normalized rows",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Output CSV containing normalized data + be_* + sgpt_*",
    )
    parser.add_argument(
        "--output-embeddings-csv",
        type=str,
        default=None,
        help="Optional embeddings-only CSV path",
    )
    parser.add_argument("--lon-col", type=str, default="lon", help="Longitude column name")
    parser.add_argument("--lat-col", type=str, default="lat", help="Latitude column name")
    parser.add_argument("--id-col", type=str, default="id", help="Optional join key for points CSV")
    parser.add_argument("--base-url", type=str, default=DEA_OWS_BASE_URL, help="DEA OWS base URL")
    parser.add_argument("--coverage-id", type=str, default=DEFAULT_COVERAGE_ID, help="DEA WCS coverage ID")
    parser.add_argument(
        "--bands",
        type=str,
        default=",".join(DEFAULT_BAREST_EARTH_BANDS),
        help="Comma-separated band names",
    )
    parser.add_argument(
        "--buffer-m",
        type=float,
        default=6.0,
        help="Half-size of WCS chip around each point in meters",
    )
    parser.add_argument("--workers", type=int, default=12, help="Parallel WCS requests")
    parser.add_argument("--timeout", type=int, default=120, help="HTTP timeout per request (seconds)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per point")
    parser.add_argument(
        "--spectral-backend",
        type=str,
        default="official_pretrained",
        choices=["official_pretrained", "lite"],
        help="Embedding backend: official pretrained SpectralGPT or lightweight approximation",
    )
    parser.add_argument("--spectral-dim", type=int, default=16, help="Final embedding dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--official-checkpoint",
        type=str,
        default=None,
        help="Optional local path to SpectralGPT+.pth checkpoint",
    )
    parser.add_argument(
        "--official-checkpoint-url",
        type=str,
        default=OFFICIAL_SPECTRALGPT_PLUS_URL,
        help="Download URL for official checkpoint if --official-checkpoint is not provided",
    )
    parser.add_argument(
        "--official-repo-dir",
        type=str,
        default=None,
        help="Optional local clone path for official SpectralGPT repo",
    )
    parser.add_argument(
        "--official-model-name",
        type=str,
        default="mae_vit_base_patch8_128",
        help="Official model constructor name",
    )
    parser.add_argument(
        "--official-encode-batch-size",
        type=int,
        default=16,
        help="Batch size for official SpectralGPT encoder inference",
    )
    parser.add_argument(
        "--official-chip-size",
        type=int,
        default=128,
        help="Spatial chip size in pixels for official SpectralGPT input",
    )
    parser.add_argument(
        "--official-pixel-size-m",
        type=float,
        default=10.0,
        help="Chip pixel size in meters for WCS request windows",
    )
    parser.add_argument(
        "--official-request-chunk-size",
        type=int,
        default=64,
        help="Number of points per WCS chip batch when using official backend",
    )
    parser.add_argument(
        "--output-official-raw-csv",
        type=str,
        default=None,
        help="Optional CSV path to save raw official embedding vectors before PCA",
    )
    return parser.parse_args()


def ensure_point_columns(
    normalized_df: pd.DataFrame,
    points_df: pd.DataFrame | None,
    lon_col: str,
    lat_col: str,
    id_col: str,
) -> pd.DataFrame:
    out = normalized_df.copy()
    out.columns = out.columns.str.lower()
    lon_col = lon_col.lower()
    lat_col = lat_col.lower()
    id_col = id_col.lower()

    if lon_col in out.columns and lat_col in out.columns:
        return out

    if points_df is None:
        raise ValueError(
            "Normalized CSV does not include lon/lat. Provide --points-csv with matching rows or an id key."
        )

    pts = points_df.copy()
    pts.columns = pts.columns.str.lower()
    if lon_col not in pts.columns or lat_col not in pts.columns:
        raise ValueError(f"Expected columns '{lon_col}' and '{lat_col}' in points CSV")

    if id_col in out.columns and id_col in pts.columns:
        merged = out.merge(pts[[id_col, lon_col, lat_col]], on=id_col, how="left")
        if merged[lon_col].isna().any() or merged[lat_col].isna().any():
            raise ValueError("Failed to populate lon/lat for all rows using id join")
        return merged

    if len(out) != len(pts):
        raise ValueError(
            f"Row mismatch: normalized has {len(out)} rows but points has {len(pts)} rows"
        )

    return pd.concat(
        [pts[[lon_col, lat_col]].reset_index(drop=True), out.reset_index(drop=True)],
        axis=1,
    )


def main() -> None:
    args = parse_args()

    norm_df = pd.read_csv(args.normalized_csv)
    points_df = pd.read_csv(args.points_csv) if args.points_csv else None

    joined_df = ensure_point_columns(
        normalized_df=norm_df,
        points_df=points_df,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        id_col=args.id_col,
    )

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    if not bands:
        raise ValueError("No bands specified")

    print(f"Sampling Bare Earth for {len(joined_df)} rows ({len(bands)} bands)...")
    be_df = sample_bare_earth_points_via_wcs(
        points_df=joined_df,
        lon_col=args.lon_col.lower(),
        lat_col=args.lat_col.lower(),
        base_url=args.base_url,
        coverage_id=args.coverage_id,
        bands=bands,
        buffer_m=args.buffer_m,
        max_workers=args.workers,
        timeout=args.timeout,
        retries=args.retries,
        output_prefix="be_",
    )

    if args.spectral_backend == "lite":
        print("Training lightweight SpectralGPT-style embeddings...")
        cfg = SpectralGPTConfig(embedding_dim=args.spectral_dim, seed=args.seed)
        embeddings = train_spectral_gpt_embeddings(be_df.values, config=cfg)
        raw_official_embeddings = None
    else:
        print("Extracting embeddings with official pretrained SpectralGPT...")
        official_cfg = OfficialSpectralGPTConfig(
            checkpoint_path=args.official_checkpoint,
            checkpoint_url=args.official_checkpoint_url,
            repo_dir=args.official_repo_dir,
            model_name=args.official_model_name,
            batch_size=args.official_encode_batch_size,
            seed=args.seed,
        )
        encoder = OfficialSpectralGPTEncoder(config=official_cfg)

        raw_chunks: list[np.ndarray] = []
        chunk_size = max(1, int(args.official_request_chunk_size))
        total = len(joined_df)
        for start in range(0, total, chunk_size):
            end = min(start + chunk_size, total)
            print(f"  Official chip batch {start}:{end} / {total}")
            subset = joined_df.iloc[start:end]
            chips = sample_bare_earth_chips_via_wcs(
                points_df=subset,
                lon_col=args.lon_col.lower(),
                lat_col=args.lat_col.lower(),
                base_url=args.base_url,
                coverage_id=args.coverage_id,
                bands=bands,
                patch_size=int(args.official_chip_size),
                pixel_size_m=float(args.official_pixel_size_m),
                max_workers=args.workers,
                timeout=args.timeout,
                retries=args.retries,
            )
            raw_chunks.append(encoder.encode_chips(chips, band_names=bands))

        raw_official_embeddings = (
            np.vstack(raw_chunks) if raw_chunks else np.zeros((0, 0), dtype=np.float32)
        )
        if raw_official_embeddings.shape[0] != len(joined_df):
            raise RuntimeError(
                "Official embedding row mismatch: "
                f"expected {len(joined_df)}, got {raw_official_embeddings.shape[0]}"
            )

        if (
            int(args.spectral_dim) > 0
            and raw_official_embeddings.shape[1] != int(args.spectral_dim)
        ):
            embeddings = reduce_embeddings_with_pca(
                raw_official_embeddings,
                n_components=int(args.spectral_dim),
                random_state=int(args.seed),
            )
        else:
            embeddings = raw_official_embeddings.astype(np.float32)

    emb_cols = [f"sgpt_{i:02d}" for i in range(embeddings.shape[1])]
    emb_df = pd.DataFrame(embeddings, columns=emb_cols, index=joined_df.index)

    out_df = pd.concat([joined_df, be_df, emb_df], axis=1)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    emb_path = Path(args.output_embeddings_csv) if args.output_embeddings_csv else None
    if emb_path is None:
        emb_path = out_path.with_name(f"{out_path.stem}_embeddings.csv")
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    emb_df.to_csv(emb_path, index=False)

    if raw_official_embeddings is not None and args.output_official_raw_csv:
        raw_path = Path(args.output_official_raw_csv)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_cols = [f"sgpt_raw_{i:03d}" for i in range(raw_official_embeddings.shape[1])]
        pd.DataFrame(raw_official_embeddings, columns=raw_cols).to_csv(raw_path, index=False)
        print(
            f"Saved raw official embeddings: {raw_path} "
            f"({raw_official_embeddings.shape[0]} x {raw_official_embeddings.shape[1]})"
        )

    missing_pct = float(be_df.isna().mean().mean() * 100.0)
    print(
        f"Saved fused point table: {out_path} ({len(out_df)} rows, {len(out_df.columns)} cols)"
    )
    print(f"Saved embeddings: {emb_path} ({embeddings.shape[0]} x {embeddings.shape[1]})")
    print(f"Bare Earth missing rate: {missing_pct:.2f}%")


if __name__ == "__main__":
    main()
