"""
Download BTS On-Time Performance Data for October 2025.

Strategy:
1. First try: direct POST to TranStats with proper ASP.NET ViewState handling
2. Second try: Try alternate bulk download URL
3. Fallback: Print manual download instructions
"""

import os
import io
import re
import zipfile
import requests
from html.parser import HTMLParser
import time

# Configuration
YEAR = 2025
MONTH = 10  # October
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

# Fields we need for the fallback model
FIELDS_TO_CHECK = [
    "YEAR", "MONTH", "DAY_OF_MONTH", "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
    "CRS_DEP_TIME", "CRS_ARR_TIME",
    "DEP_DELAY", "ARR_DELAY",
    "CANCELLED", "DIVERTED",
    "DISTANCE", "CRS_ELAPSED_TIME",
]


class ASPNetFormParser(HTMLParser):
    """Extract hidden ASP.NET form fields and checkbox IDs."""
    def __init__(self):
        super().__init__()
        self.hidden_fields = {}
        self.checkboxes = {}
    
    def handle_starttag(self, tag, attrs):
        attrs_dict = dict(attrs)
        if tag == "input" and attrs_dict.get("type") == "hidden":
            name = attrs_dict.get("name", "")
            value = attrs_dict.get("value", "")
            if name:
                self.hidden_fields[name] = value


def download_method_1():
    """Try downloading using the ASP.NET form with proper ViewState handling."""
    output_file = os.path.join(OUTPUT_DIR, f"ontime_{YEAR}_{MONTH:02d}.csv")
    
    if os.path.exists(output_file):
        print(f"[INFO] File already exists: {output_file}")
        return output_file
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    })
    
    base_url = "https://www.transtats.bts.gov/DL_SelectFields.aspx"
    params = {
        "gnoession_VQ": "FIH",
        "QO_fu146_anzr": "Nv4 Pn44vr45"
    }
    
    print("[INFO] Step 1: Fetching the TranStats page to get ViewState...")
    try:
        resp = session.get(base_url, params=params, timeout=60)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch page: {e}")
        return None
    
    # Parse hidden fields
    parser = ASPNetFormParser()
    parser.feed(resp.text)
    
    print(f"[INFO] Found {len(parser.hidden_fields)} hidden fields")
    
    # Build form data with all hidden fields
    form_data = dict(parser.hidden_fields)
    
    # Set year and month
    form_data["cboYear"] = str(YEAR)
    form_data["cboPeriod"] = str(MONTH)
    
    # Check desired field checkboxes
    # The checkbox names follow pattern like "chk_FIELDNAME" or similar
    # We need to find the actual checkbox names from the page
    checkbox_pattern = re.compile(r'name="(chk[^"]*)"', re.IGNORECASE)
    all_checkboxes = checkbox_pattern.findall(resp.text)
    print(f"[INFO] Found checkboxes: {all_checkboxes[:20]}...")
    
    # Map field names to their checkbox names
    for field in FIELDS_TO_CHECK:
        for cb in all_checkboxes:
            # Match checkbox name containing the field name (case insensitive)
            cb_field = cb.replace("chk_", "").replace("chk", "").upper()
            if field.upper() in cb_field or cb_field in field.upper():
                form_data[cb] = "on"
                print(f"  [CHECK] {field} -> {cb}")
                break
    
    # Set download button click
    form_data["btnDownload"] = "Download"
    
    print(f"\n[INFO] Step 2: Submitting download request...")
    try:
        resp2 = session.post(
            base_url,
            params=params,
            data=form_data,
            timeout=180,
            allow_redirects=True
        )
        
        print(f"[INFO] Response status: {resp2.status_code}, size: {len(resp2.content)} bytes")
        
        if len(resp2.content) > 10000:
            if resp2.content[:2] == b'PK':
                print("[INFO] Got ZIP file, extracting...")
                with zipfile.ZipFile(io.BytesIO(resp2.content)) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if csv_files:
                        with z.open(csv_files[0]) as cf:
                            content = cf.read()
                        with open(output_file, 'wb') as f:
                            f.write(content)
                        print(f"[SUCCESS] Saved to {output_file}")
                        return output_file
            
            # Check if it's CSV content
            try:
                text = resp2.content[:1000].decode('utf-8', errors='ignore')
                if 'YEAR' in text or 'MONTH' in text or 'CARRIER' in text:
                    with open(output_file, 'wb') as f:
                        f.write(resp2.content)
                    print(f"[SUCCESS] Saved CSV to {output_file}")
                    return output_file
            except:
                pass
        
        print("[WARN] Response doesn't appear to be valid data.")
        # Save response for debugging
        debug_file = os.path.join(OUTPUT_DIR, "debug_response.html")
        with open(debug_file, 'wb') as f:
            f.write(resp2.content[:5000])
        print(f"[DEBUG] Saved response preview to {debug_file}")
        
    except Exception as e:
        print(f"[ERROR] Download request failed: {e}")
    
    return None


def download_method_2():
    """Try the alternate DownLoad_Table endpoint."""
    output_file = os.path.join(OUTPUT_DIR, f"ontime_{YEAR}_{MONTH:02d}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n[INFO] Trying alternate download endpoint...")
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    
    # Try the DownLoad_Table endpoint
    url = "https://transtats.bts.gov/DownLoad_Table.asp"
    
    fields_str = ",".join(FIELDS_TO_CHECK)
    
    form_data = {
        "Table_ID": "236",
        "Has_Group": "3",
        "Is_Zipped": "0",
        "UserTableName": "Reporting_Carrier_On_Time",
        "DBShortName": "On_Time",
        "RawDataTable": "T_ONTIME_REPORTING",
        "sqlstr": f" SELECT {fields_str} FROM T_ONTIME_REPORTING WHERE Month={MONTH} AND YEAR={YEAR}",
        "varlist": fields_str,
        "grouplist": "",
        "suession": "",
        "FilterType": "Column",
    }
    
    try:
        resp = session.post(url, data=form_data, timeout=180)
        print(f"[INFO] Response status: {resp.status_code}, size: {len(resp.content)} bytes")
        
        if len(resp.content) > 10000:
            if resp.content[:2] == b'PK':
                print("[INFO] Got ZIP file, extracting...")
                with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if csv_files:
                        with z.open(csv_files[0]) as cf:
                            content = cf.read()
                        with open(output_file, 'wb') as f:
                            f.write(content)
                        print(f"[SUCCESS] Saved to {output_file}")
                        return output_file
            
            try:
                text = resp.content[:1000].decode('utf-8', errors='ignore')
                if any(f in text for f in ['YEAR', 'MONTH', 'CARRIER', 'ORIGIN']):
                    with open(output_file, 'wb') as f:
                        f.write(resp.content)
                    print(f"[SUCCESS] Saved CSV to {output_file}")
                    return output_file
            except:
                pass
    
    except Exception as e:
        print(f"[ERROR] Alternate endpoint failed: {e}")
    
    return None


def download_method_3():
    """Try using the dataverse/RITA pre-zipped files."""
    output_file = os.path.join(OUTPUT_DIR, f"ontime_{YEAR}_{MONTH:02d}.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n[INFO] Trying pre-packaged download URLs...")
    
    # Try various known URL patterns for BTS data
    urls = [
        f"https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{YEAR}_{MONTH}.zip",
        f"https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_(1987_present)_{YEAR}_{MONTH}.zip",
    ]
    
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    })
    
    for url in urls:
        try:
            print(f"[INFO] Trying: {url}")
            resp = session.get(url, timeout=120, stream=True)
            print(f"  Status: {resp.status_code}, Content-Type: {resp.headers.get('Content-Type', 'unknown')}")
            
            if resp.status_code == 200 and resp.content[:2] == b'PK':
                content = resp.content
                print(f"  Got ZIP ({len(content)} bytes), extracting...")
                with zipfile.ZipFile(io.BytesIO(content)) as z:
                    csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                    if csv_files:
                        with z.open(csv_files[0]) as cf:
                            data = cf.read()
                        with open(output_file, 'wb') as f:
                            f.write(data)
                        print(f"[SUCCESS] Saved to {output_file}")
                        return output_file
        except Exception as e:
            print(f"  Failed: {e}")
    
    return None


def print_manual_instructions():
    """Print instructions for manual download."""
    print("\n" + "="*70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*70)
    print(f"""
1. Go to: https://www.transtats.bts.gov/DL_SelectFields.aspx?gnoession_VQ=FIH&QO_fu146_anzr=Nv4%20Pn44vr45

2. Select Filter: Year = {YEAR}, Period = October

3. Check these fields:
   {', '.join(FIELDS_TO_CHECK)}

4. Click "Download"

5. Extract the ZIP file and save the CSV as:
   {os.path.join(OUTPUT_DIR, f'ontime_{YEAR}_{MONTH:02d}.csv')}
""")
    print("="*70)


if __name__ == "__main__":
    result = None
    
    # Try method 3 first (pre-zipped, most likely to work)
    result = download_method_3()
    
    if not result:
        result = download_method_1()
    
    if not result:
        result = download_method_2()
    
    if result:
        import pandas as pd
        df = pd.read_csv(result, low_memory=False)
        print(f"\n{'='*50}")
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print(f"First 3 rows:")
        print(df.head(3))
    else:
        print("\n[FAIL] All automatic download methods failed.")
        print_manual_instructions()
