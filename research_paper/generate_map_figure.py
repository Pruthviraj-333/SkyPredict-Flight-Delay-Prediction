import os
import json
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES     = os.path.join(ROOT, "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
RAW_DATA = os.path.join(ROOT, "data", "raw", "ontime_2025_10.csv")
COORD_FILE = os.path.join(ROOT, "data", "airport_coordinates.csv")

# US States GeoJSON for the background map
GEOJSON_URL = "https://raw.githubusercontent.com/python-visualization/folium/master/examples/data/us-states.json"

os.makedirs(FIG_DIR, exist_ok=True)

def generate_map():
    print("Generating US Domestic Flights Map with landmass background...")
    
    # 1. Load coordinates
    if not os.path.exists(COORD_FILE):
        print(f"Error: {COORD_FILE} not found. Run airport_coords.py first.")
        return
        
    coords = pd.read_csv(COORD_FILE)
    coords_dict = coords.set_index('iata')[['latitude', 'longitude']].to_dict('index')
    
    # 2. Load routes
    print("  Loading flight data...")
    if not os.path.exists(RAW_DATA):
        print(f"Error: {RAW_DATA} not found.")
        return
        
    df = pd.read_csv(RAW_DATA, usecols=['Origin', 'Dest'])
    routes = df.groupby(['Origin', 'Dest']).size().reset_index(name='count')
    print(f"  Found {len(routes)} unique routes")
    
    # 3. Filter for mainland US
    def is_mainland(iata):
        if iata not in coords_dict: return False
        lat = coords_dict[iata]['latitude']
        lon = coords_dict[iata]['longitude']
        return (24 < lat < 50) and (-125 < lon < -66)

    routes = routes[routes['Origin'].apply(is_mainland) & routes['Dest'].apply(is_mainland)]
    print(f"  {len(routes)} routes remain after mainland US filtering")
    
    # 4. Download GeoJSON for the map background
    print("  Downloading US boundaries...")
    try:
        resp = requests.get(GEOJSON_URL, timeout=15)
        resp.raise_for_status()
        geo_data = resp.json()
    except Exception as e:
        print(f"  Warning: Could not download boundaries ({e}). Falling back to empty map.")
        geo_data = None

    # 5. Prepare lines for plotting
    lines = []
    for _, row in routes.iterrows():
        o = coords_dict[row['Origin']]
        d = coords_dict[row['Dest']]
        lines.append([(o['longitude'], o['latitude']), (d['longitude'], d['latitude'])])
    
    # 6. Plotting
    fig, ax = plt.subplots(figsize=(16, 10), facecolor='#F5F5F0') # Slight off-white background
    ax.set_facecolor('#F5F5F0')
    
    # Set limits for mainland US
    ax.set_xlim(-126, -66)
    ax.set_ylim(23, 51)
    ax.axis('off')
    
    # 7. Draw the Landmass (US Outline)
    if geo_data:
        print("  Plotting US landmass...")
        # Style: Light tan/beige like the user's sample
        for feature in geo_data['features']:
            geom = feature['geometry']
            if geom['type'] == 'Polygon':
                for poly in geom['coordinates']:
                    p = Polygon(poly, fc='#EFEBE0', ec='#D2B48C', lw=0.5, alpha=1.0, zorder=0)
                    ax.add_patch(p)
            elif geom['type'] == 'MultiPolygon':
                for multi in geom['coordinates']:
                    for poly in multi:
                        p = Polygon(poly, fc='#EFEBE0', ec='#D2B48C', lw=0.5, alpha=1.0, zorder=0)
                        ax.add_patch(p)

    # 8. Draw routes with high transparency
    # Warm grey lines
    lc = LineCollection(lines, linewidths=0.25, colors='#8B7D6B', alpha=0.1, zorder=1)
    ax.add_collection(lc)
    
    # 9. Add airports as dark red dots
    mainland_airports = coords[coords['iata'].apply(is_mainland)]
    ax.scatter(mainland_airports['longitude'], mainland_airports['latitude'], 
               s=3, c='#800000', alpha=0.8, edgecolors='none', zorder=2)
    
    # 10. Title - Moved further down to prevent overlap with Florida/Texas
    plt.title("Figure 16. US Domestic Flights Map", fontsize=18, fontweight='bold', 
              fontfamily='serif', y=-0.05, pad=20)
    
    # 11. Save
    output_path = os.path.join(FIG_DIR, "fig_16_us_flights_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.5, facecolor='#F5F5F0')
    plt.close()
    
    print(f"  ✓ Saved to {output_path}")

if __name__ == "__main__":
    generate_map()
