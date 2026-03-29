import re
import pandas as pd
from pathlib import Path


# NHS A&E provider files have junk rows before the real headers.
# We look for a row where one cell is exactly "Code" or "Org Code" — that's the true header row.
HEADER_CELL_VALUES = {"code", "org code", "period", "org"}

COLUMN_RENAMES = {
    # Monthly file column names (vary by year)
    "code": "org_code",
    "system": "org_name",
    "name": "org_name",    # 2018-19 format
    # Legacy column names (older file formats)
    "period": "period",
    "org code": "org_code",
    "org name": "org_name",
    "type 1 departments - major a&e": "type1_attendances",
    "type 2 departments - single specialty": "type2_attendances",
    "type 3 departments - other a&e/minor injury units": "type3_attendances",
    "total attendances": "total_attendances",
    "total emergency admissions via a&e": "total_admissions",
    "number of patients spending >4 hours from decision to admit to admission": "over_4hr_dtoa",
    "percentage in 4 hours or less (all)": "pct_within_4hr",
}


def _find_header_row(df_raw: pd.DataFrame) -> int:
    """Return index of the row whose first non-null cell exactly matches a known header value."""
    for i, row in df_raw.iterrows():
        cells = [str(v).strip().lower() for v in row.values if pd.notna(v)]
        if cells and cells[0] in HEADER_CELL_VALUES:
            return i
    raise ValueError("Could not locate header row in file")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + strip column names, then apply known renames."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    rename_map = {k: v for k, v in COLUMN_RENAMES.items() if k in df.columns}
    return df.rename(columns=rename_map)


def _parse_period(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the period column to a proper datetime."""
    if "period" not in df.columns:
        return df
    df["period"] = pd.to_datetime(df["period"], errors="coerce")
    return df


def _coerce_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """Force numeric columns to numbers, replace non-parseable with NaN."""
    numeric_cols = [
        c for c in df.columns
        if c not in ("period", "org_code", "org_name")
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def parse_ae_file(file_path: str | Path) -> pd.DataFrame:
    """
    Parse a single NHS A&E provider XLS/XLSX file into a clean DataFrame.

    Returns a DataFrame with columns:
        period, org_code, org_name, total_attendances,
        total_admissions, over_4hr_dtoa, pct_within_4hr, ...
    """
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    # Some NHS files are xlsx saved with a .xls extension — try xlrd first,
    # fall back to openpyxl if it fails.
    if suffix == ".xls":
        try:
            df_raw = pd.read_excel(file_path, engine="xlrd", header=None)
            engine = "xlrd"
        except Exception:
            df_raw = pd.read_excel(file_path, engine="openpyxl", header=None)
            engine = "openpyxl"
    else:
        engine = "openpyxl"
        df_raw = pd.read_excel(file_path, engine=engine, header=None)

    header_row = _find_header_row(df_raw)

    # Re-read skipping junk rows, using the detected header as row 0
    df = pd.read_excel(
        file_path,
        engine=engine,
        header=0,
        skiprows=range(0, header_row),
    )

    df = _normalise_columns(df)

    # Monthly files don't have a period column per row — extract from metadata
    if "period" not in df.columns:
        period_val = None
        for _, row in df_raw.iterrows():
            for v in row.values:
                s = str(v).strip()
                if s.lower().startswith("period:") or re.match(r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}", s, re.IGNORECASE):
                    period_val = re.sub(r"^period:\s*", "", s, flags=re.IGNORECASE).strip()
                    break
            if period_val:
                break
        df["period"] = pd.to_datetime(period_val, errors="coerce") if period_val else pd.NaT

    # Drop rows where org_code is missing — these are usually blank spacers
    id_col = "org_code" if "org_code" in df.columns else df.columns[0]
    df = df.dropna(subset=[id_col])

    # Drop columns that are entirely NaN (common in these files)
    df = df.dropna(axis=1, how="all")

    df = _parse_period(df)
    df = _coerce_numerics(df)

    # Add source filename so we can trace each row back to its file
    df["source_file"] = file_path.name

    return df.reset_index(drop=True)


def parse_all_files(raw_dir: str | Path) -> pd.DataFrame:
    """
    Parse every XLS/XLSX file in raw_dir and concatenate into one DataFrame.
    Files that fail to parse are skipped with a warning.
    """
    raw_dir = Path(raw_dir)
    files = list(raw_dir.glob("*.xls")) + list(raw_dir.glob("*.xlsx"))

    if not files:
        raise FileNotFoundError(f"No XLS/XLSX files found in {raw_dir}")

    frames = []
    for f in sorted(files):
        try:
            df = parse_ae_file(f)
            frames.append(df)
            print(f"Parsed {f.name} — {len(df)} rows")
        except Exception as e:
            print(f"WARNING: skipping {f.name} — {e}")

    if not frames:
        raise RuntimeError("All files failed to parse — check warnings above")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows after combining: {len(combined)}")
    return combined


if __name__ == "__main__":
    import yaml

    with open("configs/ingestion.yaml") as f:
        config = yaml.safe_load(f)

    df = parse_all_files(config["raw_data_dir"])
    print(df.head())
    print(df.dtypes)

    out_path = Path(config.get("processed_data_dir", "data/processed")) / "ae_combined.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")
