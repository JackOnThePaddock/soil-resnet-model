import pandas as pd
from xml.sax.saxutils import escape

CSV = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\clhs_sampling\clhs_sampling_points_50_inside_boundaries_ndvi30.csv"
KML = r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Tests\exports\clhs_sampling\clhs_sampling_points_50_inside_boundaries_ndvi30.kml"

df = pd.read_csv(CSV)
name = escape("CLHS Sampling Points (n=50, NDVI<=0.30)")

with open(KML, "w", encoding="utf-8") as f:
    f.write("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n")
    f.write("<kml xmlns=\"http://www.opengis.net/kml/2.2\">\n")
    f.write("  <Document>\n")
    f.write(f"    <name>{name}</name>\n")
    for row in df.itertuples(index=False):
        pname = escape(str(row.paddock))
        placename = escape(f"CLHS_{int(row.id):02d}")
        f.write("    <Placemark>\n")
        f.write(f"      <name>{placename}</name>\n")
        f.write(f"      <description>Paddock: {pname}</description>\n")
        f.write("      <Point>\n")
        f.write(f"        <coordinates>{row.lon},{row.lat},0</coordinates>\n")
        f.write("      </Point>\n")
        f.write("    </Placemark>\n")
    f.write("  </Document>\n")
    f.write("</kml>\n")

print("Wrote", KML)
