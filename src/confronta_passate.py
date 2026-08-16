#!/usr/bin/env python3
"""Tabella comparativa delle passate registrate in output/sessioni.jsonl,
per capire a colpo d'occhio quale combinazione di parametri (dilatazione,
alpha, ombre...) cambia il risultato su un capo — invece di scorrere il
JSON a mano o fidarsi della memoria tra un test e l'altro.

Uso: python3 src/confronta_passate.py MGL10708-M-BE.png
     python3 src/confronta_passate.py                    (tutti i capi)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
LOG = ROOT / "output" / "sessioni.jsonl"

COLS = ["quando", "engine", "dilate_px", "use_alpha", "catch_shadows",
        "shadow_strength", "conf", "mask_px", "collateral_px", "secondi", "nota"]


def main():
    filtro = sys.argv[1] if len(sys.argv) > 1 else None
    if not LOG.exists():
        print(f"Nessun log trovato in {LOG}")
        return

    righe = []
    for line in LOG.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "engine" not in rec:  # scarta i record di warp, non di rimozione
            continue
        if filtro and rec.get("capo") != filtro:
            continue
        righe.append(rec)

    if not righe:
        print(f"Nessuna passata di rimozione trovata{' per ' + filtro if filtro else ''}.")
        return

    capi = sorted(set(r["capo"] for r in righe))
    for capo in capi:
        print(f"\n=== {capo} ===")
        sub = [r for r in righe if r["capo"] == capo]
        widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in sub)) for c in COLS}
        header = " | ".join(c.ljust(widths[c]) for c in COLS)
        print(header)
        print("-" * len(header))
        for r in sub:
            row = " | ".join(str(r.get(c, "")).ljust(widths[c]) for c in COLS)
            print(row)


if __name__ == "__main__":
    main()
