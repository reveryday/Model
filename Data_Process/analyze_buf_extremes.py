import os
import numpy as np
import pandas as pd


def analyze_extremes(csv_path: str, buf_cols: list[str], out_path: str) -> str:
    df = pd.read_csv(csv_path)

    # Ensure BUF columns are numeric
    for c in buf_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in CSV. Available: {list(df.columns)}")
        df[c] = pd.to_numeric(df[c], errors="coerce")

    lines: list[str] = []
    lines.append(f"Source: {csv_path}")
    lines.append(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    lines.append(f"BUF columns: {', '.join(buf_cols)}")
    lines.append("Definitions of 'extreme' used:")
    lines.append(" - Tukey upper fence: Q3 + 1.5*IQR")
    lines.append(" - Percentile-based: >= 99th and >= 99.9th percentile")
    lines.append("")

    for col in buf_cols:
        s = df[col].dropna().astype(float)
        total = s.size
        lines.append(f"=== {col} ===")

        if total == 0:
            lines.append("No valid numeric data.")
            lines.append("")
            continue

        # Basic stats
        Q1 = s.quantile(0.25)
        Q3 = s.quantile(0.75)
        IQR = Q3 - Q1
        tukey_high = Q3 + 1.5 * IQR
        p99 = s.quantile(0.99)
        p999 = s.quantile(0.999)

        extremes_tukey = s[s > tukey_high]
        extremes_p99 = s[s >= p99]
        extremes_p999 = s[s >= p999]

        def fmt(v: float) -> str:
            try:
                return f"{float(v):.6g}"
            except Exception:
                return str(v)

        def range_or_na(x: pd.Series) -> tuple[float, float] | tuple[str, str]:
            if len(x):
                return (float(x.min()), float(x.max()))
            return (np.nan, np.nan)

        r_tukey = range_or_na(extremes_tukey)
        r_p99 = range_or_na(extremes_p99)
        r_p999 = range_or_na(extremes_p999)

        lines.extend([
            f"count={total}",
            f"min={fmt(s.min())} max={fmt(s.max())} mean={fmt(s.mean())} std={fmt(s.std())}",
            f"Q1={fmt(Q1)} Q3={fmt(Q3)} IQR={fmt(IQR)} TukeyHigh={fmt(tukey_high)}",
            f"p99={fmt(p99)} p99.9={fmt(p999)}",
            f"Tukey extremes: {len(extremes_tukey)} ({len(extremes_tukey)/total:.2%}), range=({fmt(r_tukey[0])}, {fmt(r_tukey[1])})",
            f">=99th percentile: {len(extremes_p99)} ({len(extremes_p99)/total:.2%}), range=({fmt(r_p99[0])}, {fmt(r_p99[1])})",
            f">=99.9th percentile: {len(extremes_p999)} ({len(extremes_p999)/total:.2%}), range=({fmt(r_p999[0])}, {fmt(r_p999[1])})",
            f"Top5 largest: {', '.join(fmt(v) for v in s.nlargest(5).tolist())}",
            "",
        ])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    return out_path


if __name__ == "__main__":
    csv_path = r"d:\DL_单层BUF\Model\Data Analysis\BUF_DATA_with_MAC_no_material.csv"
    out_path = r"d:\DL_单层BUF\Model\Data_Process\buf_extremes_report.txt"
    buf_cols = [
        "Inf_Flu_BUF",
        "Fin_Flu_BUF",
        "Inf_Exp_BUF",
        "Fin_Exp_BUF",
        "Inf_Eff_BUF",
        "Fin_Eff_BUF",
    ]

    analyze_extremes(csv_path, buf_cols, out_path)