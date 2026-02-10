import os
import pandas as pd
import shapefile
from shapely.geometry import shape, Point

BOUNDARIES = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SpeirsBoundaries\boundaries\boundaries.shp"
POINTS_CSV = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\clhs_sampling\clhs_sampling_points_50.csv"
OUT_KML = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\clhs_sampling\clhs_sampling_points_50_with_boundaries.kml"

# Load paddock geometries
reader = shapefile.Reader(BOUNDARIES)
fields = [f[0] for f in reader.fields if f[0] != "DeletionFlag"]
name_idx = fields.index("FIELD_NAME")

paddocks = []
for sr in reader.shapeRecords():
    name = str(sr.record[name_idx]).strip()
    geom = shape(sr.shape.__geo_interface__)
    paddocks.append((name, geom))

# Load points
pts = pd.read_csv(POINTS_CSV)

# Filter points inside any paddock
kept = []
skipped = []
for row in pts.itertuples(index=False):
    pt = Point(float(row.lon), float(row.lat))
    hit = None
    for name, geom in paddocks:
        if geom.contains(pt) or geom.touches(pt):
            hit = name
            break
    if hit is None:
        skipped.append((row.id, row.lon, row.lat))
        continue
    kept.append((row.id, row.paddock, row.lon, row.lat, hit))

# Write KML
with open(OUT_KML, "w", encoding="ascii") as f:
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
    f.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n")
    f.write("  <Document>\n")
    f.write("    <name>CLHS Sampling Points + Paddock Boundaries</name>\n")

    # Boundaries folder
    f.write("    <Folder>\n")
    f.write("      <name>Paddock Boundaries</name>\n")
    for name, geom in paddocks:
        f.write("      <Placemark>\n")
        f.write(f"        <name>{name}</name>\n")
        f.write("        <Polygon>\n")

        # Handle polygons / multipolygons
        geoms = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in geoms:
            exterior = list(poly.exterior.coords)
            f.write("          <outerBoundaryIs><LinearRing><coordinates>\n")
            for x, y in exterior:
                f.write(f"            {x},{y},0\n")
            f.write("          </coordinates></LinearRing></outerBoundaryIs>\n")

            # Holes (if any)
            for interior in poly.interiors:
                f.write("          <innerBoundaryIs><LinearRing><coordinates>\n")
                for x, y in interior.coords:
                    f.write(f"            {x},{y},0\n")
                f.write("          </coordinates></LinearRing></innerBoundaryIs>\n")

        f.write("        </Polygon>\n")
        f.write("      </Placemark>\n")
    f.write("    </Folder>\n")

    # Points folder
    f.write("    <Folder>\n")
    f.write("      <name>CLHS Sampling Points (n=50)</name>\n")
    for pid, paddock, lon, lat, hit in kept:
        f.write("      <Placemark>\n")
        f.write(f"        <name>CLHS_{int(pid):02d}</name>\n")
        f.write(f"        <description>Paddock: {paddock}; Boundary: {hit}</description>\n")
        f.write("        <Point>\n")
        f.write(f"          <coordinates>{lon},{lat},0</coordinates>\n")
        f.write("        </Point>\n")
        f.write("      </Placemark>\n")
    f.write("    </Folder>\n")

    f.write("  </Document>\n")
    f.write("</kml>\n")

print(f"Wrote {OUT_KML}")
if skipped:
    print(f"Skipped {len(skipped)} points outside boundaries")
