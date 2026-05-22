import os
import glob
import pandas as pd
from datetime import date

# --- Paths ---
INPUT_DIR = r"C:\Users\bradl\Box\!INTERNAL\SDR"
INPUT_PATTERN = "External SDR Master Table_*.csv"
HISTORY_FILE = r"C:\Users\bradl\Box\!INTERNAL\SDR\SDR_history.csv"
OUTPUT_FILE = rf"C:\Users\bradl\Box\!INTERNAL\SDR\SDR_exec_summary_{date.today():%Y-%m-%d}.csv"
HTML_OUTPUT_FILE = rf"C:\Users\bradl\Box\!INTERNAL\SDR\SDR_exec_summary_{date.today():%Y-%m-%d}.html"

# --- Cumulative baseline (prior-week running totals to seed cumulative table) ---
BASELINE = {
    "Employed": 544,
    "Concierge": 104,
    "Other": 428,
    "Not Qualified": 1076,
    "Qualified": 2574,
}

# --- Disposition categories ---
CONTACTED_VALUES = {
    "Contacted",
    "Contacted - ISQ Updated",
    "Contacted - LVM",
    "ISQ Updated",
}


def find_latest_input(directory: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(directory, pattern))
    if not matches:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}.")
    return max(matches, key=os.path.getmtime)


def _to_bool(series: pd.Series, truthy=("TRUE", "YES", "1", "T", "Y")) -> pd.Series:
    return series.fillna("").astype(str).str.strip().str.upper().isin(truthy)


def load_and_flag(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda c: c.str.strip() if c.dtype == "object" else c)

    # SDR column
    sdr_col = next((c for c in ["SDR", "External SDR", "Lead Owner"] if c in df.columns), None)
    if sdr_col is None:
        raise ValueError(f"No SDR column found. Columns: {list(df.columns)}")
    df["SDR"] = df[sdr_col].fillna("").astype(str).str.strip().replace("", "Unknown SDR")

    # Disposition column
    disp_col = next((c for c in ["Call Disposition", "Last Call Disposition"] if c in df.columns), None)
    if disp_col is None:
        raise ValueError(f"No Call Disposition column found. Columns: {list(df.columns)}")
    df["_Disposition"] = df[disp_col].fillna("").astype(str).str.strip()

    # Date column (for week label)
    df["_Date"] = pd.to_datetime(df["Date"], errors="coerce") if "Date" in df.columns else pd.NaT

    # Contacted / Not Contacted — mutually exclusive, cover every row.
    # Anything not in CONTACTED_VALUES (including blanks and any other disposition) = Not Contacted.
    df["Contacted"] = df["_Disposition"].isin(CONTACTED_VALUES)
    df["Not_Contacted"] = ~df["Contacted"]

    raw_qualified = _to_bool(df["Qualified"]) if "Qualified" in df.columns else pd.Series(False, index=df.index)
    raw_not_qualified = _to_bool(df["NOT Qualified"]) if "NOT Qualified" in df.columns else pd.Series(False, index=df.index)

    # Disqualified: NOT Qualified = TRUE always wins
    df["NotQualified_flag"] = raw_not_qualified

    # Qualified: raw TRUE, OR Contacted with both blank (auto-promotion).
    # NOT Qualified = TRUE always overrides.
    auto_promoted = df["Contacted"] & (~raw_qualified) & (~raw_not_qualified)
    df["Qualified_flag"] = (raw_qualified | auto_promoted) & (~raw_not_qualified)

    # Other flags
    employed_col = next((c for c in ["Employed", "Employed (Verified)"] if c in df.columns), None)
    df["Employed_flag"] = _to_bool(df[employed_col]) if employed_col else pd.Series(False, index=df.index)

    df["Medicare_flag"] = (
        df["Accepts Medicare"].fillna("").astype(str).str.strip().str.upper().eq("YES")
        if "Accepts Medicare" in df.columns else pd.Series(False, index=df.index)
    )
    df["Concierge_flag"] = (
        df["Membership"].fillna("").astype(str).str.strip().str.upper().eq("YES")
        if "Membership" in df.columns else pd.Series(False, index=df.index)
    )

    return df


def build_weekly_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("SDR", dropna=False).agg(
        Calls=("SDR", "count"),
        Contacted=("Contacted", "sum"),
        **{"Not Contacted": ("Not_Contacted", "sum")},
        Qualified=("Qualified_flag", "sum"),
        Disqualified=("NotQualified_flag", "sum"),
        Medicare=("Medicare_flag", "sum"),
        Concierge=("Concierge_flag", "sum"),
        Employed=("Employed_flag", "sum"),
    ).reset_index()

    for c in grouped.columns:
        if c != "SDR":
            grouped[c] = grouped[c].astype(int)

    total_row = {"SDR": "TOTAL"}
    for c in grouped.columns:
        if c != "SDR":
            total_row[c] = int(grouped[c].sum())
    grouped = pd.concat([grouped, pd.DataFrame([total_row])], ignore_index=True)

    return grouped


def append_history(df: pd.DataFrame, history_path: str) -> pd.DataFrame:
    keep = [
        "SDR", "_Date", "_Disposition",
        "Contacted", "Not_Contacted",
        "Qualified_flag", "NotQualified_flag",
        "Employed_flag", "Medicare_flag", "Concierge_flag",
    ]
    new_rows = df[keep].copy()
    today_str = date.today().isoformat()
    new_rows["_RunDate"] = today_str

    if os.path.exists(history_path):
        existing = pd.read_csv(history_path)
        # Skip append if we've already recorded a run for today
        if "_RunDate" in existing.columns and (existing["_RunDate"].astype(str) == today_str).any():
            print(f"History already contains a run dated {today_str}; skipping append.")
            return existing
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    os.makedirs(os.path.dirname(history_path), exist_ok=True)
    combined.to_csv(history_path, index=False)
    return combined


def build_cumulative_table(history: pd.DataFrame) -> pd.DataFrame:
    for col in ["Qualified_flag", "NotQualified_flag", "Employed_flag", "Concierge_flag"]:
        if history[col].dtype != bool:
            history[col] = history[col].astype(str).str.strip().str.upper().isin(["TRUE", "1"])

    nq = history["NotQualified_flag"]

    # This week's history numbers + prior-week baseline
    qualified = int(history["Qualified_flag"].sum()) + BASELINE["Qualified"]
    not_qualified = int(nq.sum()) + BASELINE["Not Qualified"]
    employed = int((nq & history["Employed_flag"]).sum()) + BASELINE["Employed"]
    concierge = int((nq & history["Concierge_flag"]).sum()) + BASELINE["Concierge"]
    other = int((nq & ~history["Employed_flag"] & ~history["Concierge_flag"]).sum()) + BASELINE["Other"]

    total = qualified + not_qualified

    def pct(n):
        return f"{(n / total * 100):.1f}%" if total else "0.0%"

    rows = [
        ("Employed", employed, pct(employed)),
        ("Concierge", concierge, pct(concierge)),
        ("Other", other, pct(other)),
        ("Not Qualified", not_qualified, pct(not_qualified)),
        ("Qualified", qualified, pct(qualified)),
        ("TOTAL", total, "100.0%"),
    ]
    return pd.DataFrame(rows, columns=["Team Cumulative Totals", "Count", "Share"])


def week_label(df: pd.DataFrame) -> str:
    dates = df["_Date"].dropna()
    if dates.empty:
        return "Activity"
    start, end = dates.min(), dates.max()
    return f"Activity - {start.month}/{start.day}/{start:%y}-{end.month}/{end.day}/{end:%y}"


def write_combined_csv(weekly: pd.DataFrame, cumulative: pd.DataFrame, label: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(label + "\n")
        weekly.to_csv(f, index=False)
        f.write("\n")
        cumulative.to_csv(f, index=False)


def write_combined_html(weekly: pd.DataFrame, cumulative: pd.DataFrame, label: str, path: str) -> None:
    """Render the weekly + cumulative tables as a styled HTML document."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    css = """
    <style>
        body {
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #1f2a36;
            color: #e6edf3;
            padding: 24px;
        }
        h2 {
            margin: 0 0 8px 0;
            font-size: 16px;
            font-weight: 600;
        }
        table {
            border-collapse: collapse;
            margin-bottom: 32px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.4);
        }
        table.weekly { width: 900px; }
        table.cumulative { width: 420px; }
        thead.title-row th {
            background: #2a3b4d;
            color: #cfe2f3;
            font-weight: 600;
            text-align: left;
            padding: 8px 12px;
            border: 1px solid #3b4a5e;
            font-size: 13px;
        }
        thead.header-row th {
            background: #3a4c63;
            color: #ffffff;
            font-weight: 600;
            text-align: center;
            padding: 8px 12px;
            border: 1px solid #4a5d75;
            font-size: 13px;
        }
        thead.header-row th.first-col { text-align: left; }
        tbody td {
            background: #2a3b4d;
            color: #e6edf3;
            padding: 6px 12px;
            border: 1px solid #3b4a5e;
            text-align: center;
            font-size: 13px;
        }
        tbody td.first-col { text-align: left; }
        tbody td.indent { padding-left: 32px; }
        tbody tr.total-row td {
            background: #324558;
            font-weight: 700;
        }
    </style>
    """

    # Labels in the cumulative table that should be visually indented as components of "Not Qualified"
    indented_labels = {"Employed", "Concierge", "Other"}

    def render_table(df: pd.DataFrame, title: str | None, table_class: str, total_label: str) -> str:
        cols = list(df.columns)
        ncols = len(cols)
        parts = [f'<table class="{table_class}">']

        if title:
            parts.append(
                f'<thead class="title-row"><tr><th colspan="{ncols}">{title}</th></tr></thead>'
            )

        # Column header row
        header_cells = "".join(
            f'<th class="first-col">{c}</th>' if i == 0 else f'<th>{c}</th>'
            for i, c in enumerate(cols)
        )
        parts.append(f'<thead class="header-row"><tr>{header_cells}</tr></thead>')

        # Body rows
        body_rows = []
        for _, row in df.iterrows():
            first_val = str(row[cols[0]]).strip()
            row_cls = ' class="total-row"' if first_val.upper() == total_label.upper() else ""
            indent_first = table_class == "cumulative" and first_val in indented_labels
            cells = "".join(
                (
                    f'<td class="first-col indent">{row[c]}</td>'
                    if i == 0 and indent_first
                    else f'<td class="first-col">{row[c]}</td>' if i == 0
                    else f'<td>{row[c]}</td>'
                )
                for i, c in enumerate(cols)
            )
            body_rows.append(f"<tr{row_cls}>{cells}</tr>")
        parts.append(f"<tbody>{''.join(body_rows)}</tbody>")

        parts.append("</table>")
        return "".join(parts)

    weekly_html = render_table(weekly, label, "weekly", "TOTAL")
    # Cumulative table already has "Team Cumulative Totals" as its first column header — no title row needed
    cumulative_html = render_table(cumulative, None, "cumulative", "TOTAL")

    doc = (
        "<!DOCTYPE html>"
        '<html lang="en"><head><meta charset="utf-8">'
        f"<title>SDR Executive Summary - {label}</title>"
        f"{css}</head><body>"
        f"{weekly_html}{cumulative_html}"
        "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)


def main():
    input_file = find_latest_input(INPUT_DIR, INPUT_PATTERN)
    print(f"Using input file: {input_file}")

    df = load_and_flag(input_file)
    weekly = build_weekly_table(df)
    history = append_history(df, HISTORY_FILE)
    cumulative = build_cumulative_table(history)
    label = week_label(df)

    print(label)
    print(weekly.to_string(index=False))
    print()
    print(cumulative.to_string(index=False))

    write_combined_csv(weekly, cumulative, label, OUTPUT_FILE)
    write_combined_html(weekly, cumulative, label, HTML_OUTPUT_FILE)
    print(f"\\nSaved CSV to {OUTPUT_FILE}")
    print(f"Saved HTML to {HTML_OUTPUT_FILE}")
    print(f"History updated at {HISTORY_FILE}")


if __name__ == "__main__":
    main()