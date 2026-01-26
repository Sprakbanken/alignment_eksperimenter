import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


def is_year_dir(name: str) -> bool:
    return re.fullmatch(r"maalfrid_(\d{4})", name) is not None


def extract_year(dir_name: str) -> int:
    m = re.fullmatch(r"maalfrid_(\d{4})", dir_name)
    return int(m.group(1)) if m else -1


def parse_filename(fname: str) -> Tuple[str, str, str]:
    """
    Forventer filnavn som <domain>_<lang>_<type>.jsonl, f.eks. '113.no_nob_pdf.jsonl'.
    Returnerer (domain, lang, filetype).
    """
    if not fname.endswith(".jsonl"):
        raise ValueError(f"Ikke en .jsonl-fil: {fname}")
    stem = fname[:-6]  # fjern .jsonl
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Ugyldig filnavn-format: {fname}")
    ftype = parts[-1]
    lang = parts[-2]
    domain = "_".join(parts[:-2])
    return domain, lang, ftype


def find_grouped_files(data_dir: str) -> Dict[str, List[Tuple[int, str]]]:
    """
    Finn alle jsonl-filer i maalfrid_YYYY-mapper og grupper dem etter <domain>_<lang>.
    Returnerer: { group_key: [(year, filepath), ...] } sortert etter year asc, filepath asc.
    """
    groups: Dict[str, List[Tuple[int, str]]] = {}
    for name in os.listdir(data_dir):
        if not is_year_dir(name):
            continue
        year = extract_year(name)
        year_path = os.path.join(data_dir, name)
        if not os.path.isdir(year_path):
            continue
        for entry in os.listdir(year_path):
            fp = os.path.join(year_path, entry)
            if not os.path.isfile(fp):
                continue
            if not entry.endswith(".jsonl"):
                continue
            try:
                domain, lang, _ = parse_filename(entry)
            except ValueError:
                continue
            key = f"{domain}_{lang}"
            groups.setdefault(key, []).append((year, fp))
    # Sorter slik at tidligere år kommer først (så "keep last" beholder nyeste)
    for key in groups:
        groups[key].sort(key=lambda t: (t[0], t[1]))
    return groups


def read_jsonl(fp: str) -> List[dict]:
    """Les JSONL-fil og returner liste av dicts."""
    out = []
    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if isinstance(rec, dict):
                    out.append(rec)
            except json.JSONDecodeError:
                tqdm.write(f"[ADVARSEL] Klarte ikke parse linje i {fp}: {line[:50]}...")
                continue
    return out


def build_group_df(items: List[Tuple[int, str]], group_key: str = "") -> pd.DataFrame:
    """
    items: [(year, filepath), ...]
    Returnerer en DataFrame med alle rader.
    """
    frames = []
    for year, fp in tqdm(
        items,
        leave=False,
        position=1,
        desc=(group_key if group_key else "Filer"),
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            rows = read_jsonl(fp)
        except Exception as e:
            tqdm.write(f"[ADVARSEL] Klarte ikke lese fil: {fp} ({e})")
            continue
        if not rows:
            continue

        df = pd.DataFrame(rows)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def dedupe_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dedupliser basert på URL eller doc-hash.
    Rader med samme URL eller samme hash regnes som duplikater.
    Den nyeste (basert på 'date'-kolonnen) beholdes.
    """
    if df.empty:
        return df

    url_col = "url" if "url" in df.columns else None
    hash_col = "doc_hash" if "doc_hash" in df.columns else None

    # Konverter date til streng for sortering (noen verdier kan være int av en eller annen grunn)
    df["_date_str"] = df["date"].astype(str)

    # Sorter etter dato (eldst først) så "keep last" velger nyeste
    # ISO 8601-datoer sorteres som strenger (https://stackoverflow.com/questions/9576860/sort-iso-8601-dates-forward-or-backwards)
    df = df.sort_values("_date_str", kind="stable", na_position="first")

    # Dedupliser på url
    if url_col is not None:
        has_url = df[url_col].notna() & (df[url_col] != "")

        df_with_url = df[has_url].copy()
        df_without_url = df[~has_url].copy()

        df_with_url = df_with_url.drop_duplicates(subset=[url_col], keep="last")

        df = pd.concat([df_with_url, df_without_url], ignore_index=True)
        df = df.sort_values("_date_str", kind="stable", na_position="first")

    # Dedupliser på hash
    if hash_col is not None:
        has_hash = df[hash_col].notna() & (df[hash_col] != "")

        df_with_hash = df[has_hash].copy()
        df_without_hash = df[~has_hash].copy()

        df_with_hash = df_with_hash.drop_duplicates(subset=[hash_col], keep="last")

        df = pd.concat([df_with_hash, df_without_hash], ignore_index=True)

    # Fjern hjelpekolonnen
    df = df.drop(columns=["_date_str"], errors="ignore")

    return df


def is_na_value(v) -> bool:
    """Sjekk om en verdi er NA/NaN, håndterer også lister og andre typer."""
    if v is None:
        return True
    if isinstance(v, (list, dict)):
        return False  # Lister og dicts er alltid gyldige verdier
    try:
        return pd.isna(v)
    except (ValueError, TypeError):
        return False  # Hvis pd.isna feiler, anta at verdien er gyldig


def write_jsonl(path: str, df: pd.DataFrame, chunksize: int = 50_000):
    """Skriv DataFrame til JSONL-fil uten å endre innholdet."""
    tqdm.write(f"  Skriver {len(df)} rader til {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start : start + chunksize]
            for record in chunk.to_dict(orient="records"):
                # Fjern NaN-verdier som pandas legger til, men behold lister og dicts
                clean_record = {k: v for k, v in record.items() if not is_na_value(v)}
                f.write(json.dumps(clean_record, ensure_ascii=False) + "\n")

    tqdm.write(f"  Ferdig: {path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lag et deduplisert superset av Målfrid-datasett på tvers av år (per domene+språk)."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
        help="Rotmappe som inneholder maalfrid_YYYY-mapper (default: ./data)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data", "maalfrid_superset"
        ),
        help="Utdatamappe for superset (default: ./data/maalfrid_superset)",
    )
    parser.add_argument(
        "--exclude_domains",
        type=str,
        nargs="*",
        default=[],
        help="Liste over domener som skal utelates (f.eks. 'regjeringen.no').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    groups = find_grouped_files(data_dir)

    # Finn hvilke grupper som allerede er ferdig
    completed = {
        os.path.splitext(fn)[0]
        for fn in os.listdir(output_dir)
        if fn.endswith(".jsonl") and os.path.isfile(os.path.join(output_dir, fn))
    }

    exclude = set(args.exclude_domains or [])

    def domain_from_key(k: str) -> str:
        return k.rsplit("_", 1)[0]

    keys_sorted = sorted(k for k in groups.keys() if domain_from_key(k) not in exclude)
    keys_to_process = [k for k in keys_sorted if k not in completed]

    print(f"Fant {len(groups)} grupper (domene+språk).")
    print(f"Allerede ferdige: {len(completed)}. Gjenstår: {len(keys_to_process)}.")

    for key in tqdm(
        keys_to_process,
        desc="Lager supersett",
        unit="group",
        dynamic_ncols=True,
    ):
        items = groups[key]
        tqdm.write(f"\nProsesserer: {key} ({len(items)} filer)")

        df = build_group_df(items, group_key=key)

        if df.empty:
            out_path = os.path.join(output_dir, f"{key}.jsonl")
            with open(out_path, "w", encoding="utf-8"):
                pass
            tqdm.write("  Tom gruppe, skrev tom fil.")
            continue

        rows_before = len(df)
        df = dedupe_keep_last(df)
        rows_after = len(df)

        tqdm.write(
            f"  Rader før dedupe: {rows_before}, etter: {rows_after} (fjernet {rows_before - rows_after})"
        )

        out_path = os.path.join(output_dir, f"{key}.jsonl")
        write_jsonl(out_path, df)

    print(f"\nFerdig! Supersett skrevet til: {output_dir}")


if __name__ == "__main__":
    main()
