import argparse
import math
import os
import shutil

import numpy as np
import shapefile
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, shape
from shapely.geometry.polygon import orient
from shapely.ops import transform
from pyproj import CRS, Transformer


def utm_crs_for_lonlat(lon, lat):
    zone = int(math.floor((lon + 180.0) / 6.0) + 1)
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return CRS.from_epsg(epsg)


def resample_ring(coords, step_m):
    pts = np.asarray(coords, dtype=float)
    if len(pts) < 3:
        return pts
    if not np.allclose(pts[0], pts[-1]):
        pts = np.vstack([pts, pts[0]])

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total == 0.0:
        return pts

    samples = max(int(math.ceil(total / step_m)), 4)
    dist = np.linspace(0.0, total, samples, endpoint=False)
    dist = np.append(dist, total)
    cum = np.concatenate(([0.0], np.cumsum(seg)))
    x = np.interp(dist, cum, pts[:, 0])
    y = np.interp(dist, cum, pts[:, 1])
    resampled = np.column_stack([x, y])
    resampled[-1] = resampled[0]
    return resampled


def taubin_smooth(coords, iterations=8, lam=0.5, mu=-0.53):
    pts = np.asarray(coords, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    if closed:
        pts = pts[:-1]
    if len(pts) < 4:
        return coords

    for _ in range(iterations):
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        avg = (prev + nxt) / 2.0
        pts = pts + lam * (avg - pts)
        prev = np.roll(pts, 1, axis=0)
        nxt = np.roll(pts, -1, axis=0)
        avg = (prev + nxt) / 2.0
        pts = pts + mu * (avg - pts)

    if closed:
        pts = np.vstack([pts, pts[0]])
    return pts


def clamp_deviation(smoothed, original_line, max_dev_m):
    if max_dev_m <= 0:
        return smoothed
    pts = np.asarray(smoothed, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    last = len(pts) - 1 if closed else len(pts)

    for i in range(last):
        p = Point(pts[i])
        dist = p.distance(original_line)
        if dist <= max_dev_m:
            continue
        proj_dist = original_line.project(p)
        proj = original_line.interpolate(proj_dist)
        vx = pts[i, 0] - proj.x
        vy = pts[i, 1] - proj.y
        vlen = math.hypot(vx, vy)
        if vlen == 0.0:
            pts[i, 0] = proj.x
            pts[i, 1] = proj.y
        else:
            scale = max_dev_m / vlen
            pts[i, 0] = proj.x + vx * scale
            pts[i, 1] = proj.y + vy * scale

    if closed:
        pts[-1] = pts[0]
    return pts


def simplify_ring(coords, tol_m):
    if tol_m <= 0:
        return coords
    ring = LineString(coords)
    simp = ring.simplify(tol_m, preserve_topology=False)
    if simp.is_empty:
        return coords
    simp_coords = list(simp.coords)
    if len(simp_coords) < 4:
        return coords
    if simp_coords[0] != simp_coords[-1]:
        simp_coords.append(simp_coords[0])
    return np.asarray(simp_coords, dtype=float)


def process_polygon(poly, ring_fn):
    poly = orient(poly, sign=-1.0)
    exterior = np.asarray(poly.exterior.coords, dtype=float)
    interior_rings = [np.asarray(ring.coords, dtype=float) for ring in poly.interiors]

    new_exterior = ring_fn(exterior)
    if len(new_exterior) < 4:
        new_exterior = exterior

    new_interiors = []
    for ring in interior_rings:
        new_ring = ring_fn(ring)
        if len(new_ring) >= 4:
            new_interiors.append(new_ring)

    new_poly = Polygon(new_exterior, new_interiors)
    if not new_poly.is_valid:
        fixed = new_poly.buffer(0)
        if not fixed.is_empty:
            new_poly = fixed
    return new_poly


def smooth_ring(coords, step_m, iterations, lam, mu, max_dev_m, simplify_m):
    resampled = resample_ring(coords, step_m)
    if len(resampled) < 4:
        return coords
    smoothed = taubin_smooth(resampled, iterations=iterations, lam=lam, mu=mu)
    line = LineString(resampled)
    clamped = clamp_deviation(smoothed, line, max_dev_m)
    simplified = simplify_ring(clamped, simplify_m)
    if len(simplified) < 4:
        simplified = clamped
    if not np.allclose(simplified[0], simplified[-1]):
        simplified = np.vstack([simplified, simplified[0]])
    return simplified


def follow_chain(path, segment_lengths, warmup_loops=2):
    pts = np.asarray(path, dtype=float)
    closed = np.allclose(pts[0], pts[-1])
    if closed:
        pts = pts[:-1]
    if len(pts) < 2:
        return path

    heading = None
    for i in range(1, len(pts)):
        v = pts[i] - pts[0]
        n = float(np.hypot(v[0], v[1]))
        if n > 0:
            heading = v / n
            break
    if heading is None:
        return path

    positions = [pts[0].copy()]
    for L in segment_lengths:
        positions.append(positions[-1] - heading * L)

    outputs = []
    loops = max(1, warmup_loops)
    for loop in range(loops):
        for p in pts:
            positions[0] = p
            for idx, L in enumerate(segment_lengths, start=1):
                prev = positions[idx - 1]
                curr = positions[idx]
                dx = curr[0] - prev[0]
                dy = curr[1] - prev[1]
                dist = math.hypot(dx, dy)
                if dist == 0.0:
                    curr = prev - heading * L
                else:
                    scale = L / dist
                    curr = np.array([prev[0] + dx * scale, prev[1] + dy * scale])
                positions[idx] = curr
            if loop == loops - 1:
                outputs.append(positions[-1].copy())

    if closed and outputs:
        outputs.append(outputs[0])
    return np.asarray(outputs, dtype=float)


def rig_trace_ring(coords, step_m, segment_lengths, simplify_m, warmup_loops=2):
    resampled = resample_ring(coords, step_m)
    if len(resampled) < 4:
        return coords
    traced = follow_chain(resampled, segment_lengths, warmup_loops=warmup_loops)
    if len(traced) < 4:
        return coords
    simplified = simplify_ring(traced, simplify_m)
    if len(simplified) < 4:
        simplified = traced
    if not np.allclose(simplified[0], simplified[-1]):
        simplified = np.vstack([simplified, simplified[0]])
    return simplified


def geometry_to_parts(geom):
    parts = []
    if isinstance(geom, Polygon):
        geom = orient(geom, sign=-1.0)
        parts.append(list(geom.exterior.coords))
        for ring in geom.interiors:
            parts.append(list(ring.coords))
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            poly = orient(poly, sign=-1.0)
            parts.append(list(poly.exterior.coords))
            for ring in poly.interiors:
                parts.append(list(ring.coords))
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom_type}")
    return parts


def main():
    parser = argparse.ArgumentParser(
        description="Smooth polygon boundaries while keeping geometry close to original."
    )
    parser.add_argument("input_shp", help="Input polygon shapefile (.shp)")
    parser.add_argument("output_shp", help="Output shapefile (.shp)")
    parser.add_argument("--step-m", type=float, default=2.0, help="Resample step in meters")
    parser.add_argument(
        "--iterations", type=int, default=8, help="Taubin smoothing iterations"
    )
    parser.add_argument("--lambda", dest="lam", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=-0.53)
    parser.add_argument(
        "--max-dev-m",
        type=float,
        default=2.0,
        help="Maximum deviation from original line in meters",
    )
    parser.add_argument(
        "--simplify-m",
        type=float,
        default=0.5,
        help="Final simplify tolerance in meters",
    )
    parser.add_argument(
        "--mode",
        choices=["smooth", "rig"],
        default="smooth",
        help="Smoothing mode: smooth (Taubin) or rig (tractor/implement trace)",
    )
    parser.add_argument(
        "--rig-length-m",
        type=float,
        default=15.0,
        help="Total rig length in meters (used when rig segments not specified)",
    )
    parser.add_argument(
        "--rig-pivots",
        type=int,
        default=3,
        help="Number of pivot segments (used when rig segments not specified)",
    )
    parser.add_argument(
        "--rig-segments",
        type=str,
        default="",
        help="Comma-separated segment lengths in meters (overrides rig length/pivots)",
    )
    parser.add_argument(
        "--rig-warmup-loops",
        type=int,
        default=2,
        help="Warmup loops for closed rings in rig mode",
    )
    args = parser.parse_args()

    reader = shapefile.Reader(args.input_shp)
    shapes = reader.shapes()
    records = reader.records()

    if not shapes:
        raise SystemExit("No shapes found in input.")

    first_geom = shape(shapes[0].__geo_interface__)
    if first_geom.is_empty:
        raise SystemExit("First geometry is empty; cannot infer CRS.")

    lon, lat = first_geom.centroid.x, first_geom.centroid.y
    utm_crs = utm_crs_for_lonlat(lon, lat)
    to_utm = Transformer.from_crs("EPSG:4326", utm_crs, always_xy=True).transform
    to_wgs = Transformer.from_crs(utm_crs, "EPSG:4326", always_xy=True).transform

    if args.mode == "rig":
        if args.rig_segments:
            segment_lengths = [
                float(x) for x in args.rig_segments.split(",") if x.strip()
            ]
        else:
            pivots = max(1, int(args.rig_pivots))
            segment_lengths = [args.rig_length_m / pivots] * pivots

        def ring_fn(coords):
            return rig_trace_ring(
                coords,
                args.step_m,
                segment_lengths,
                args.simplify_m,
                warmup_loops=args.rig_warmup_loops,
            )

    else:

        def ring_fn(coords):
            return smooth_ring(
                coords,
                args.step_m,
                args.iterations,
                args.lam,
                args.mu,
                args.max_dev_m,
                args.simplify_m,
            )

    writer = shapefile.Writer(args.output_shp, shapeType=shapefile.POLYGON)
    for field in reader.fields[1:]:
        writer.field(field.name, field.field_type, field.size, field.decimal)

    for shp, rec in zip(shapes, records):
        geom = shape(shp.__geo_interface__)
        geom_utm = transform(to_utm, geom)

        if isinstance(geom_utm, Polygon):
            smoothed = process_polygon(geom_utm, ring_fn)
        elif isinstance(geom_utm, MultiPolygon):
            smoothed_polys = [process_polygon(poly, ring_fn) for poly in geom_utm.geoms]
            smoothed = MultiPolygon([p for p in smoothed_polys if not p.is_empty])
        else:
            raise ValueError(f"Unsupported geometry type: {geom_utm.geom_type}")

        smoothed_wgs = transform(to_wgs, smoothed)
        parts = geometry_to_parts(smoothed_wgs)
        writer.poly(parts)
        writer.record(*rec)

    writer.close()

    base, _ = os.path.splitext(args.output_shp)
    for ext in [".prj", ".cpg"]:
        src = os.path.splitext(args.input_shp)[0] + ext
        if os.path.exists(src):
            shutil.copyfile(src, base + ext)


if __name__ == "__main__":
    main()
