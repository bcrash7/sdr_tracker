import pandas as pd
from datetime import date

INPUT_FILE = r"C:\\Users\\bradl\\Downloads\\SDR_Master.csv"
OUTPUT_FILE = rf"C:\Users\bradl\Box\!INTERNAL\SDR\SDR_exec_summary_{date.today():%Y-%m-%d}.csv"

df = pd.read_csv(INPUT_FILE, dtype=str)

# Normalize column names and cell values
df.columns = df.columns.str.strip()

# Clean cell values a bit
df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

# --- 1. Identify the schema we are working with ---
disposition_col = None
for candidate in ["Call Disposition", "Last Call Disposition"]:
    if candidate in df.columns:
        disposition_col = candidate
        break

has_summary_columns = {"Contacted", "Not_Contacted", "Qualified", "Not_Qualified"}.issubset(df.columns)

# Pick the best available SDR-like column
sdr_col = None
for candidate in ["SDR", "External SDR", "Lead Owner"]:
    if candidate in df.columns:
        sdr_col = candidate
        break

if sdr_col is None:
    raise ValueError(
        "Input CSV does not contain an SDR column. "
        "Expected one of: 'SDR', 'External SDR', or 'Lead Owner'. "
        f"Found columns: {list(df.columns)}"
    )

# --- 2. If this is a detailed file, derive summary flags from disposition ---
if disposition_col:
    # Normalize blanks in grouping key to avoid NaN rows
    df[sdr_col] = df[sdr_col].fillna("").astype(str).str.strip()
    df.loc[df[sdr_col] == "", sdr_col] = "Unknown SDR"

    # Drop blanks in disposition column
    df = df[df[disposition_col].notna() & (df[disposition_col].astype(str).str.strip() != "")]

    # Resolve the "employed" column (new file uses "Employed", old file used "Employed (Verified)")
    employed_col = None
    for candidate in ["Employed (Verified)", "Employed"]:
        if candidate in df.columns:
            employed_col = candidate
            break

    # Clean up columns safely
    bool_cols = ["Qualified", "NOT Qualified"]
    if employed_col:
        bool_cols.append(employed_col)

    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper().isin(["TRUE", "YES", "1"])
        else:
            df[col] = False

    if employed_col is None:
        employed_col = "Employed"
        df[employed_col] = False

    if "Accepts Medicare" in df.columns:
        df["Accepts Medicare"] = df["Accepts Medicare"].astype(str).str.strip().str.upper().eq("YES")
    else:
        df["Accepts Medicare"] = False

    if "Membership" in df.columns:
        df["Membership"] = df["Membership"].astype(str).str.strip().str.upper().eq("YES")
    else:
        df["Membership"] = False

    # Define contacted vs not contacted using the exact categories
    contacted_values = [
        "Contacted",
        "Contacted - ISQ Updated",
        "Contacted - LVM",
        "ISQ Updated",
    ]

    not_contacted_values = [
        "Not Contacted - Bad Number",
        "Not Contacted - Closed",
        "Not Contacted - Other",
    ]

    df["Contacted"] = df[disposition_col].isin(contacted_values)
    df["Not_Contacted"] = df[disposition_col].isin(not_contacted_values)

    # Normalize the qualified column naming
    if "Qualified" not in df.columns:
        df["Qualified"] = False
    if "NOT Qualified" not in df.columns:
        df["NOT Qualified"] = False

    # Build the summary by SDR
    summary = df.groupby(sdr_col, dropna=False).agg(
        Total_Leads=("First Name", "count") if "First Name" in df.columns else (sdr_col, "count"),
        Contacted=("Contacted", "sum"),
        Not_Contacted=("Not_Contacted", "sum"),
        Qualified=("Qualified", "sum"),
        Not_Qualified=("NOT Qualified", "sum"),
        Accepts_Medicare=("Accepts Medicare", "sum"),
        Membership=("Membership", "sum"),
        Verified_Employed=(employed_col, "sum"),
    ).reset_index()

    summary = summary.rename(columns={sdr_col: "SDR"})

# --- 3. If this is already a summary file, use the existing columns directly ---
elif has_summary_columns:
    summary_source = df.copy()

    # Support both spaced and underscored names if needed
    if "Not Contacted" in summary_source.columns and "Not_Contacted" not in summary_source.columns:
        summary_source = summary_source.rename(columns={"Not Contacted": "Not_Contacted"})

    if "NOT Qualified" in summary_source.columns and "Not_Qualified" not in summary_source.columns:
        summary_source = summary_source.rename(columns={"NOT Qualified": "Not_Qualified"})

    required = [
        "SDR",
        "Total_Leads",
        "Contacted",
        "Not_Contacted",
        "Qualified",
        "Not_Qualified",
        "Accepts_Medicare",
        "Membership",
        "Verified_Employed",
    ]

    missing = [col for col in required if col not in summary_source.columns]
    if missing:
        raise ValueError(
            f"CSV looks like a summary file, but it is missing required columns: {missing}. "
            f"Found columns: {list(summary_source.columns)}"
        )

    summary = summary_source[required].copy()
    summary["SDR"] = summary["SDR"].fillna("").astype(str).str.strip()
    summary.loc[summary["SDR"] == "", "SDR"] = "Unknown SDR"

# --- 4. Unsupported schema ---
else:
    raise ValueError(
        "Input CSV does not contain a recognizable disposition column or summary columns. "
        f"Expected 'Call Disposition' or 'Last Call Disposition'. Found columns: {list(df.columns)}"
    )

# --- 5. Print and save ---
print(summary.to_string(index=False))
summary.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved to {OUTPUT_FILE}")
