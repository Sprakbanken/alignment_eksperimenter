import argparse
import json
import os
import re
from typing import Dict, List, Tuple

import pandas as pd
from tqdm import tqdm


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


def is_year_dir(name: str) -> bool:
    # godta kun maalfrid_YYYY, ikke *_hashed
    return re.fullmatch(r"maalfrid_(\d{4})", name) is not None


def extract_year(dir_name: str) -> int:
    m = re.fullmatch(r"maalfrid_(\d{4})", dir_name)
    return int(m.group(1)) if m else -1


def parse_filename(fname: str) -> Tuple[str, str, str]:
    """
    Forventer filnavn som <domain>_<lang>_<type>.jsonl, f.eks. '113.no_nob_pdf.jsonl'.
    """
    if not fname.endswith(".jsonl"):
        raise ValueError("not jsonl")
    stem = fname[:-6]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Ugyldig filnavn: {fname}")
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
    for key in groups:
        groups[key].sort(key=lambda t: (t[0], t[1]))
    return groups


def robust_read_jsonl(fp: str) -> List[dict]:
    """Les JSONL robust, ignorer linjer som ikke er gyldig JSON-objekt."""
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
            except Exception:
                # hopp over ødelagte linjer
                print(f"[ADVARSEL] Klarte ikke parse linje i {fp}: {line[:50]}...")
                continue
    return out


def ensure_cols(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA


def build_group_df(items: List[Tuple[int, str]], group_key: str = "") -> pd.DataFrame:
    """
    items: [(year, filepath), ...] i sortert rekkefølge (eldst -> nyest)
    Returnerer en DataFrame som inneholder alle rader + hjelpekolonner for dedupe.
    """
    frames = []
    order_id = 0
    for year, fp in tqdm(
        items,
        leave=False,
        position=1,
        desc=(group_key if group_key else "Filer"),
        unit="file",
        dynamic_ncols=True,
    ):
        print(f"  Leser: {fp} ({os.path.getsize(fp) / 1e6:.1f} MB)")  # <-- Legg til

        try:
            rows = robust_read_jsonl(fp)
        except Exception as e:
            tqdm.write(f"[ADVARSEL] Klarte ikke lese fil: {fp} ({e})")
            continue
        if not rows:
            continue
        df = pd.DataFrame(rows)

        # legg til hjelpekolonner
        df["_year"] = year
        df["_src"] = os.path.basename(fp)
        # bevar stabil totalrekkefølge: eldre år < nyere år, og filnavn a..z
        # Vi øker order_id per rad i lese-rekkefølge slik at den siste blir beholdt
        n = len(df)
        df["_order"] = range(order_id, order_id + n)
        order_id += n

        # normaliser URL/hash-kolonner
        ensure_cols(
            df, ["url", "source_url", "doc_hash", "doc-hash", "hash", "content_hash"]
        )
        df["_url_norm"] = df[["url", "source_url"]].bfill(axis=1).iloc[:, 0]
        df["_hash_norm"] = (
            df[["doc_hash", "doc-hash", "hash", "content_hash"]]
            .bfill(axis=1)
            .iloc[:, 0]
        )

        frames.append(df)

    if not frames:
        return pd.DataFrame()

    # concat bevarer rekkefølgen vi bygget
    print(f"  Concat {len(frames)} frames...")  # <-- Legg til
    return pd.concat(frames, ignore_index=True)


def dedupe_keep_last(df: pd.DataFrame) -> pd.DataFrame:
    """
    De-dupliser på OR-logikk: duplikat hvis samme URL ELLER samme doc-hash.
    'Sist vinner' ifølge _order (nyeste år kommer sist i byggingen).
    """
    if df.empty:
        return df

    # Sorter etter '_order' slik at keep='last' velger nyeste
    df = df.sort_values("_order", kind="stable")

    # Pass 1: de-dupe på URL
    if "_url_norm" in df.columns:
        df = df.drop_duplicates(subset=["_url_norm"], keep="last")

    # Pass 2: de-dupe på hash
    if "_hash_norm" in df.columns:
        df = df.sort_values("_order", kind="stable")
        df = df.drop_duplicates(subset=["_hash_norm"], keep="last")

    # Fjern hjelpekolonner før vi skriver til fil
    drop_cols = ["_year", "_src", "_order", "_url_norm", "_hash_norm"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    return df.drop(columns=drop_cols, errors="ignore")


def write_jsonl(path: str, df: pd.DataFrame, chunksize: int = 50_000):
    # Noen av filene er for store til å bruke pd-to_json på, så vi chunker det og bruker json.dumps istedet
    print(f"  Skriver {len(df)} rader til {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for start in range(0, len(df), chunksize):
            chunk = df.iloc[start : start + chunksize]
            for record in chunk.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"    Skrev rader {start}-{min(start + chunksize, len(df))}")

    print(f"  Ferdig: {path}")


def main():
    args = parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    groups = find_grouped_files(data_dir)

    # Finn hvilke grupper som allerede er skrevet (basert på eksisterende filer i output_dir)
    completed = {
        os.path.splitext(fn)[0]
        for fn in os.listdir(output_dir)
        if fn.endswith(".jsonl") and os.path.isfile(os.path.join(output_dir, fn))
    }
    print(completed)

    # Filtrer bort grupper basert på ekskluderte domener
    exclude = set(args.exclude_domains or [])

    def domain_from_key(k: str) -> str:
        # key-format: <domain>_<lang>
        return k.rsplit("_", 1)[0]

    keys_sorted = sorted(k for k in groups.keys() if domain_from_key(k) not in exclude)
    keys_to_process = [k for k in keys_sorted if k not in completed]

    print(
        f"Fant {len(groups)} domener. Ferdige: {len(completed)}. Gjenstår: {len(keys_to_process)}."
    )

    for key in tqdm(
        keys_to_process,
        desc="Lager supersett",
        unit="group",
        dynamic_ncols=True,
    ):
        items = groups[key]
        tqdm.write(f"prosesserer domene: {key} ({len(items)} filer)")
        df = build_group_df(items, group_key=key)
        if df.empty:
            # skriv likevel en tom fil slik at vi ikke forsøker på nytt
            out_path = os.path.join(output_dir, f"{key}.jsonl")
            with open(out_path, "w", encoding="utf-8"):
                pass
            continue
        df = dedupe_keep_last(df)
        print(df.head())

        out_path = os.path.join(output_dir, f"{key}.jsonl")
        write_jsonl(out_path, df)

    print(f"Ferdig. Superset skrevet til: {output_dir}")


if __name__ == "__main__":
    main()
