"""Data for the same-state panels (ed03a, ed03c): same-state examples and same-patient pairs.
Reuses the builders' data functions; nothing is retyped."""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "explore"))
from build_fig1_same_dot_draft import collect, find_matched_cell, progression_times  # noqa: E402
import build_fig1_same_dot_v2 as V  # noqa: E402
from build_phase_plane_grammar_draft import fit_combined_axis  # noqa: E402
from composite_tumor_host_phase_space import enrich_snapshots, load_tables  # noqa: E402
import build_population_vs_patient_field_exercise as M  # noqa: E402

TUMOUR = "LMS"
NAMES = {"gemini31_lite": "Gemini 3.1 L", "glm52_fp8": "GLM-5.2", "gemma4": "Gemma 4",
         "claude_sonnet5": "Sonnet", "qwen36_flash": "Qwen 3.6", "kimi_k3_off": "Kimi K3"}


@lru_cache(maxsize=1)
def same_state():
    tables = load_tables()
    s, _p, _sc, _sg = fit_combined_axis(enrich_snapshots(tables))
    prog = progression_times(tables.events)
    bymodel = V.per_model_contrast(s, prog)
    sub = s[s["tumor_oncotree_code"].astype(str).eq(TUMOUR)].copy()
    box = find_matched_cell(sub)
    entries = collect(sub, box, prog)
    # display only: frozen 20-coordinate state-map positions of the same snapshots (selection stays in the older combined axis)
    import pandas as pd
    fz = pd.read_csv(Path(__file__).resolve().parents[3] / "outputs" / "paper_state_map" / "state_projection.csv.gz",
                     usecols=["_gid", "snapshot_index", "pc1", "pc2"]).rename(columns={"pc1": "fz1", "pc2": "fz2"})
    for e in entries:
        for k in ("trail", "future"):
            e[k] = e[k].merge(fz, on=["_gid", "snapshot_index"], how="left")  # death/terminal states have no map position
    return entries, box, bymodel


@lru_cache(maxsize=1)
def same_patient():
    atlas = M.add_phase_labels(M.fit_display_map(M.enrich_snapshots(M.load_tables()))[1])
    display = M.fit_display_map(M.enrich_snapshots(M.load_tables()))[0]
    pairs6 = M.horizon_pairs(atlas, horizon_months=6, tolerance_months=2.5)
    examples = M.select_divergent_within_trajectory_pairs(pairs6)
    branch = M.transform_to_map(M.enrich_snapshots(M.load_derivation(M.BRANCH_ROOT)), display)
    limits = M.map_limits(atlas, branch)
    return atlas, examples, limits
