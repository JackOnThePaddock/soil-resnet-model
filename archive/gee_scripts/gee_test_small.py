import ee
import pandas as pd
from pathlib import Path
import requests

ee.Initialize()

BASE_DIR = Path(r"C:\Users\jackc\Documents\EW WH & MG SPEIRS\SOIL Testing model Data")
IN_CSV = BASE_DIR / "external_sources" / "soil_points_all_no400_top10cm.csv"

df = pd.read_csv(IN_CSV).head(50)
prop_cols = ["site_id","date","depth_upper_m","depth_lower_m","ph","cec_cmolkg","esp_pct","na_cmolkg"]
for c in prop_cols:
    if c not in df.columns:
        df[c]=None

def build_feature(row):
    lon=row['lon']; lat=row['lat']
    props={}
    for c in prop_cols:
        v=row.get(c)
        if pd.isna(v):
            continue
        props[c]=v
    return ee.Feature(ee.Geometry.Point([lon,lat]), props)

features=[build_feature(r) for _,r in df.iterrows()]
fc=ee.FeatureCollection(features)
emb=ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL').filterBounds(fc)
emb=emb.map(lambda img: img.set('year', ee.Date(img.get('system:time_start')).get('year')))
last5 = ee.List(emb.aggregate_array('year')).distinct().sort().slice(-5)
median = emb.filter(ee.Filter.inList('year', last5)).median()

sampled=median.sampleRegions(collection=fc, properties=prop_cols, scale=10, geometries=False)
print('sampled', sampled.size().getInfo())

req={'table': sampled, 'format':'CSV'}
print('requesting download id')
did=ee.data.getTableDownloadId(req)
print('download id ok')
url=ee.data.makeTableDownloadUrl(did)
resp=requests.get(url, timeout=300)
resp.raise_for_status()
print('downloaded bytes', len(resp.content))
