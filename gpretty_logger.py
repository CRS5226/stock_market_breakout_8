# pretty_forecasts.py

import os
from pathlib import Path
from typing import Dict, List, Optional
import math
import pandas as pd

try:
    import gspread  # optional
except Exception:
    gspread = None

from googleapiclient.discovery import build
from google.oauth2.service_account import Credentials


# ============================= helpers =============================


def _latest_per_stock(df: pd.DataFrame) -> pd.DataFrame:
    """Return latest row per (stock_code, model) based on timestamp-like column if present."""
    ts_col = None
    for cand in ("timestamp", "last_updated", "last_updated_at"):
        if cand in df.columns:
            ts_col = cand
            break

    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df["_ts"] = df[ts_col]
    else:
        df["_ts"] = pd.NaT

    if "stock_code" in df.columns:
        df["stock_code"] = df["stock_code"].astype(str).str.upper()
    if "model" not in df.columns:
        df["model"] = "ALGO"

    df = df.sort_values(["stock_code", "model", "_ts"])
    idx = (
        df.groupby(["stock_code", "model"], dropna=False)["_ts"].transform("max")
        == df["_ts"]
    )
    out = df[idx].copy()
    out.drop(columns=["_ts"], inplace=True, errors="ignore")
    return out


TF_LABELS = {
    0: "1min",
    1: "5min",
    2: "15min",
    3: "30min",
    4: "45min",
    5: "1hour",
    6: "4hour",
    7: "1day",
    8: "1month",
}


def _get_num(row: Optional[Dict], *keys):
    if not row:
        return None
    for k in keys:
        if k in row and row[k] not in (None, ""):
            try:
                v = row[k]
                if isinstance(v, str) and v.strip().lower() in (
                    "nan",
                    "na",
                    "null",
                    "none",
                    "inf",
                    "+inf",
                    "-inf",
                ):
                    continue
                fv = float(v)
                if math.isnan(fv) or math.isinf(fv):
                    continue
                return fv
            except Exception:
                pass
    return None


def _get_str(row: Optional[Dict], *keys, default=""):
    if not row:
        return default
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return default


def _sr_pair(row: Optional[Dict], idx: int):
    if idx == 0:
        s = _get_num(row, "support", "S")
        r = _get_num(row, "resistance", "R")
    else:
        s = _get_num(row, f"support{idx}", f"S{idx}")
        r = _get_num(row, f"resistance{idx}", f"R{idx}")
    return s, r


def _triplet(row: Optional[Dict], idx: int):
    if idx == 0:
        e = _get_num(row, "entry")
        t = _get_num(row, "target")
        sl = _get_num(row, "stoploss")
    else:
        e = _get_num(row, f"entry{idx}")
        t = _get_num(row, f"target{idx}")
        sl = _get_num(row, f"stoploss{idx}")
    return e, t, sl


def _sr_pct(row: Optional[Dict], idx: int):
    if idx == 0:
        v = _get_num(row, "sr_range_pct", "SR_pct")
    else:
        v = _get_num(row, f"SR{idx}_pct")
    if v is not None:
        return v
    s, r = _sr_pair(row, idx)
    if s is None or r is None or s == 0:
        return None
    try:
        return ((r - s) / s) * 100.0
    except Exception:
        return None


def _respect_pair(row: Optional[Dict], idx: int):
    if idx == 0:
        rs = _get_num(row, "respected_S")
        rr = _get_num(row, "respected_R")
    else:
        rs = _get_num(row, f"respected_S{idx}")
        rr = _get_num(row, f"respected_R{idx}")

    def _i(v, default=0):
        try:
            if v is None:
                return default
            vv = float(v)
            if math.isnan(vv) or math.isinf(vv):
                return default
            return int(vv)
        except Exception:
            return default

    return _i(rs), _i(rr)


def _ts_of(row: Optional[Dict]) -> str:
    v = _get_str(row, "timestamp", "last_updated", "last_updated_at")
    if not v:
        return ""
    try:
        return pd.to_datetime(v).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        return str(v)


# ============================ Filter (Sheets v4) ============================


def add_filter_to_sheet(
    spreadsheet_id: str, sheet_id: int, service_account_json: str = "cred.json"
):
    """Apply a basic filter to the whole used range (header at row 1)."""
    creds = Credentials.from_service_account_file(
        service_account_json, scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    service = build("sheets", "v4", credentials=creds)

    body = {
        "requests": [
            {
                "setBasicFilter": {
                    "filter": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 0,  # include header row
                            "startColumnIndex": 0,  # from column A
                            # no end indices => until the end of used grid
                        }
                    }
                }
            }
        ]
    }
    service.spreadsheets().batchUpdate(
        spreadsheetId=spreadsheet_id, body=body
    ).execute()


# ============================= table builder =============================

# ============================= table builder =============================

# ============================= table builder =============================

TABLE_HEADER = [
    "stock_code",
    "model",
    "last_updated",
    "close_price",
    "consolidation_score",
    "respected_S",
    "respected_R",
    "support_diff_pct",  # 🆕 added new column here
    # "resistance_diff_pct",  # existing one stays
    "signal",
    "tf",
    "S",
    "R",
    "SR_pct",
    "entry",
    "target",
    "stoploss",
    "entry_target_pct",
]


def _row_for_model(
    stock: str, model_tag: str, row: Optional[Dict], idx: int
) -> List[str]:
    ts = _ts_of(row)
    close = _get_num(row, "close") or _get_num(
        row.get("ohlcv", {}) if row else {}, "close"
    )

    # --- Fix: signal per TF
    sig_key = "signal" if idx == 0 else f"signal{idx}"
    sig = _get_str(row, sig_key, default="")

    tf = TF_LABELS.get(idx, f"tf{idx}")
    s, r = _sr_pair(row, idx)
    srp = _sr_pct(row, idx)
    e, t, sl = _triplet(row, idx)
    rs, rr = _respect_pair(row, idx)

    # 🧮 Compute support_diff_pct
    support_diff_pct = None
    if close is not None and s is not None and close != 0:
        try:
            support_diff_pct = ((close - s) / close) * 100.0
        except Exception:
            support_diff_pct = None

    # 🧮 Compute resistance_diff_pct
    resistance_diff_pct = None
    if close is not None and r is not None and close != 0:
        try:
            resistance_diff_pct = ((r - close) / close) * 100.0
        except Exception:
            resistance_diff_pct = None

    et_key = "entry_target_pct" if idx == 0 else f"entry_target_pct{idx}"
    cons_key = "consolidation_score" if idx == 0 else f"consolidation_score{idx}"
    entry_target_pct = _get_num(row, et_key)
    consolidation_score = _get_num(row, cons_key)

    def fmt(x):
        return (
            "" if x is None else (f"{x:.6g}" if isinstance(x, (int, float)) else str(x))
        )

    return [
        stock,
        model_tag,
        ts,
        (f"{close:.6g}" if close is not None else ""),
        fmt(consolidation_score),
        str(rs),
        str(rr),
        fmt(support_diff_pct),  # 🆕 new column value
        # fmt(resistance_diff_pct),  # existing
        sig,
        tf,
        fmt(s),
        fmt(r),
        fmt(srp),
        fmt(e),
        fmt(t),
        fmt(sl),
        fmt(entry_target_pct),
    ]


def _col_letter(n: int) -> str:
    """
    Convert a 1-based column index to an Excel-style letter (A, B, ... AA, AB, ...).
    """
    result = ""
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(65 + remainder) + result
    return result


def _build_all_rows(combined_df: pd.DataFrame) -> List[List[str]]:
    """
    Build ONE big, contiguous table:
      - header once
      - for each stock: 9 ALGO rows
      - adds one blank separator row between stocks
    """
    df = _latest_per_stock(combined_df)

    rows: List[List[str]] = []
    rows.append(TABLE_HEADER)

    # Real stocks
    for stock_idx, (stock, g) in enumerate(df.groupby("stock_code", dropna=False)):
        algo_row = g[g["model"] == "ALGO"].sort_values("timestamp").tail(1)
        algo_row = algo_row.iloc[0].to_dict() if not algo_row.empty else None

        # --- Add ALGO rows only ---
        for idx in range(9):
            rows.append(_row_for_model(stock, "ALGO", algo_row, idx))

        # 🧱 Add one blank separator row between stocks (but not after the last one)
        if stock_idx < len(df["stock_code"].unique()) - 1:
            rows.append([""] * len(TABLE_HEADER))

    return rows


# ============== PUBLIC ENTRYPOINT (WRITE DIRECTLY TO SHEET TAB) ==============


def write_pretty_to_sheet_from_sheets(
    spreadsheet_name: str,
    gpt_tab: str,
    algo_tab: str,
    pretty_tab: str = "stocks_30_pretty",
    service_account_json: Optional[str] = None,
):
    if gspread is None:
        raise RuntimeError("gspread is not installed. pip install gspread oauth2client")

    # Authenticate
    gc = (
        gspread.service_account(filename=service_account_json)
        if service_account_json
        else gspread.service_account()
    )
    sh = gc.open(spreadsheet_name)

    # Read GPT + ALGO tabs
    ws_gpt = sh.worksheet(gpt_tab)
    ws_algo = sh.worksheet(algo_tab)

    df_gpt = pd.DataFrame(ws_gpt.get_all_records())
    df_algo = pd.DataFrame(ws_algo.get_all_records())
    df_gpt["model"] = "GPT"
    df_algo["model"] = "ALGO"

    combined = pd.concat([df_algo], ignore_index=True)

    # Build table rows (header + data)
    flat_rows = _build_all_rows(combined)

    # Ensure target tab
    try:
        ws_pretty = sh.worksheet(pretty_tab)
    except Exception:
        ws_pretty = sh.add_worksheet(title=pretty_tab, rows=2000, cols=50)

    spreadsheet_id = sh.id
    try:
        sheet_id = ws_pretty.id  # gspread >=5
    except Exception:
        sheet_id = ws_pretty._properties.get("sheetId")

    # === Write logic ===
    existing = ws_pretty.get_all_values()
    new_header = flat_rows[0]
    new_body = flat_rows[1:]

    if not existing:
        # First run → write full header + body, then apply filter
        ws_pretty.update("A1", flat_rows, value_input_option="RAW")
        if spreadsheet_id and sheet_id is not None:
            add_filter_to_sheet(
                spreadsheet_id, sheet_id, service_account_json or "cred.json"
            )
    else:
        # Compare header row
        cur_header = existing[0] if existing else []
        if cur_header != new_header:
            # overwrite header if changed
            ws_pretty.update("A1", [new_header], value_input_option="RAW")

        # Pad body rows to same width
        tgt_cols = max(len(new_header), len(new_body[0]) if new_body else 0)
        padded = [row + [""] * (tgt_cols - len(row)) for row in new_body]

        # Only update body rows (A2 onwards)
        end_col = _col_letter(tgt_cols or 1)
        end_row = len(padded) + 1  # +1 because header row is row 1
        a1 = f"A2:{end_col}{end_row}"
        ws_pretty.update(a1, padded, value_input_option="RAW")
