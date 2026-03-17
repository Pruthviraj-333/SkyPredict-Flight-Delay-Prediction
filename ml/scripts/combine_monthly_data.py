"""
Combine all 12 monthly BTS CSVs into a single cleaned dataset for the Fallback Model.

Loads only the columns we need to keep memory usage manageable (~7M rows).
Drops cancelled/diverted flights, creates the IS_DELAYED target.
"""

import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MONTHLY_DIR = os.path.join(BASE_DIR, "..", "..", "data", "monthly")
PROCESSED_DIR = os.path.join(BASE_DIR, "..", "..", "data", "processed")

# Only load columns we actually need (saves ~60% memory)
REQUIRED_COLUMNS = [
    "Year", "Month", "DayofMonth", "DayOfWeek",
    "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "CRSArrTime",
    "ArrDelay",
    "Cancelled", "Diverted",
    "Distance", "CRSElapsedTime",
]


def combine_all_months():
    """Load, clean, and combine all monthly CSVs."""
    print("=" * 70)
    print("  COMBINING 12 MONTHS OF BTS FLIGHT DATA")
    print("=" * 70)
    
    monthly_dir = os.path.abspath(MONTHLY_DIR)
    csv_files = sorted([
        f for f in os.listdir(monthly_dir) if f.endswith(".csv")
    ])
    
    print(f"\n[INFO] Found {len(csv_files)} monthly files in {monthly_dir}")
    for f in csv_files:
        size_mb = os.path.getsize(os.path.join(monthly_dir, f)) / (1024 * 1024)
        print(f"  {f}: {size_mb:.1f} MB")
    
    all_dfs = []
    total_raw = 0
    total_clean = 0
    
    for i, csv_file in enumerate(csv_files, 1):
        filepath = os.path.join(monthly_dir, csv_file)
        print(f"\n[{i}/{len(csv_files)}] Loading {csv_file}...")
        
        # Load only needed columns
        df = pd.read_csv(filepath, usecols=REQUIRED_COLUMNS, low_memory=False)
        raw_count = len(df)
        total_raw += raw_count
        
        # Drop cancelled flights
        if 'Cancelled' in df.columns:
            df = df[df['Cancelled'] == 0.0]
        
        # Drop diverted flights
        if 'Diverted' in df.columns:
            df = df[df['Diverted'] == 0.0]
        
        # Drop rows where ArrDelay is missing
        df = df.dropna(subset=['ArrDelay'])
        
        # Drop Cancelled/Diverted columns (no longer needed)
        df = df.drop(columns=['Cancelled', 'Diverted'], errors='ignore')
        
        clean_count = len(df)
        total_clean += clean_count
        dropped = raw_count - clean_count
        
        print(f"  Raw: {raw_count:,} → Clean: {clean_count:,} (dropped {dropped:,})")
        all_dfs.append(df)
    
    # Combine all months
    print(f"\n[INFO] Concatenating {len(all_dfs)} months...")
    combined = pd.concat(all_dfs, ignore_index=True)
    
    print(f"\n{'=' * 70}")
    print(f"  COMBINED DATASET SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total raw rows:   {total_raw:,}")
    print(f"  Total clean rows: {total_clean:,}")
    print(f"  Dropped:          {total_raw - total_clean:,}")
    print(f"  Shape:            {combined.shape}")
    print(f"  Memory:           {combined.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    
    # Create target variable
    combined['IS_DELAYED'] = (combined['ArrDelay'] >= 15).astype(int)
    delayed = combined['IS_DELAYED'].sum()
    ontime = len(combined) - delayed
    print(f"\n  Target distribution:")
    print(f"    On-Time:  {ontime:,} ({ontime/len(combined)*100:.1f}%)")
    print(f"    Delayed:  {delayed:,} ({delayed/len(combined)*100:.1f}%)")
    
    # Monthly breakdown
    print(f"\n  Monthly breakdown:")
    monthly_stats = combined.groupby(['Year', 'Month']).agg(
        flights=('IS_DELAYED', 'count'),
        delay_rate=('IS_DELAYED', 'mean')
    ).reset_index()
    for _, row in monthly_stats.iterrows():
        print(f"    {int(row['Year'])}-{int(row['Month']):02d}: {int(row['flights']):>8,} flights, {row['delay_rate']*100:.1f}% delayed")
    
    # Save combined dataset
    os.makedirs(os.path.abspath(PROCESSED_DIR), exist_ok=True)
    output_path = os.path.join(os.path.abspath(PROCESSED_DIR), "combined_12month_raw.csv")
    print(f"\n[SAVING] Writing to {output_path}...")
    combined.to_csv(output_path, index=False)
    file_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"[SAVED] {file_size:.1f} MB")
    
    print(f"\n{'=' * 70}")
    print(f"  DONE — Ready for preprocessing and training")
    print(f"{'=' * 70}")
    
    return combined


if __name__ == "__main__":
    combine_all_months()
