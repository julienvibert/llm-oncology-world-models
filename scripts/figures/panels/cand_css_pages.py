"""Clinical State Space pages built with _css_template: the same figure for each use case.

cand_final_prime_wt, cand_final_prime_mut, cand_final_egfr, cand_final_atlas, cand_final_genie_nsclc, cand_final_genie_crc.
Run: uv run python scripts/figures/panels/cand_css_pages.py [prime egfr atlas genie]
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _css_axes as A  # noqa: E402
import _f4 as F4  # noqa: E402
import _css_grammar as Gm  # noqa: E402
import _css_template as T  # noqa: E402
import _genie_state as GS  # noqa: E402
from _f3util import axes_mm  # noqa: E402
from common import BLUE, CHEMO, FAMILY_COLOURS, PAL, ROOT, STATE_COLOURS  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "explore"))
sys.path.insert(0, str(ROOT / "scripts" / "explore" / "figpanels"))
M = 30.4375
WHICH = sys.argv[1:] or ["prime", "egfr", "egfr_wt", "egfr_four", "atlas", "genie"]
AX_DH = "Axes: first component of the 10 disease and of the 10 host variables, each fitted on all atlas positions (30.6% and 43.6% of block variance); these are not the two components of the frozen state map."
AX_PRIME = "Axes: first two components of 14 trial features, fitted on PRIME; a two-dimensional display, comparisons use the recorded variables."
AX_GENIE = "Axes: first two components of the tumour block at time t, fitted on observed positions with equal patient weights; generated positions are projected onto them."


def rmst(t: np.ndarray, e: np.ndarray, tau: float = 24.0) -> float:
    g = np.linspace(0, tau, 241)
    return float(np.trapezoid(T.km_curve(t, e, np.ones(len(t)), g), g))


def split(b: pd.DataFrame, tau: float, w: np.ndarray | None = None) -> np.ndarray:
    """Restricted mean months progression-free, after progression and dead, of tau."""
    g = np.linspace(0, tau, 241)
    ww = np.ones(len(b)) if w is None else w
    pfm = float(np.trapezoid(T.km_curve(b.t_pf.to_numpy(float), b.e_pf.to_numpy(int), ww, g), g))
    alv = float(np.trapezoid(T.km_curve(b.t_os.to_numpy(float), b.e_os.to_numpy(int), ww, g), g))
    return np.array([pfm, alv - pfm, tau - alv])


def side_inputs(fig, x: float, y: float, w: float, h: float) -> None:
    """Compact input effects from the decision-aligned perturbation matrix (full matrix in Extended Data)."""
    d = json.loads((ROOT / "outputs/perturbation_matrix_decision_aligned/readout.json").read_text())
    m = np.array(d["effect_null_corrected"], float)[:, [d["readouts"].index("pfs1"), d["readouts"].index("os")]]
    b = float(np.nanmax(np.abs(np.array(d["effect_null_corrected"], float))))
    ax = axes_mm(fig, x + 20, y, w - 20, h - 9)
    fig.text((x + 20) / 180, (y + h - 0.2) / (fig.get_size_inches()[1] * 25.4), "Factorial experiment", fontsize=6.5, va="top")
    ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-b, vmax=b)
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, f"{v:.2f}".replace("-", "−"), ha="center", va="center", fontsize=6, color="white" if abs(v) > 0.6 * b else "#000000")
    ax.set_xticks([0, 1], ["First-line PFS", "OS"], fontsize=6)
    ax.set_yticks(range(4), ["EGFR status", "Condition", "History", "Treatment"], fontsize=6)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, pad=1)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text((x + 20) / 180, (y - 1.2) / (fig.get_size_inches()[1] * 25.4), "Effect size, null-corrected", fontsize=6, va="top")


def side_treatment(fig, x: float, y: float, w: float, h: float) -> None:
    """Documented treatment burden, validated estimate of the four-arm experiment (Fig. 4 tables)."""
    t = F4.final_table("treatment_scalars_scores_20261012.csv")
    t = t[(t.horizon_months == 12) & (t.quantity == "months_any")].set_index("row")
    arms = [("patient_only", "Record alone", PAL["arm_patient_only"]), ("setting", "+ setting", PAL["arm_setting"]), ("generic_cohort", "+ generic", PAL["arm_generic"]), ("patients_like_me", "+ matched", PAL["arm_matched"])]
    ax = axes_mm(fig, x + 15, y + 7, w - 15, h - 16)
    obs = float(t.loc["forecaster:patient_only"].mean_observed)
    ax.plot([obs, obs], [-3.5, 0.5], color="#000000", lw=0.8)
    for i, (k, lab, c) in enumerate(arms):
        r = t.loc[f"forecaster:{k}"]
        ax.plot(r.mean_forecast, -i, "o", ms=4, color=c, mec="#000000", mew=0.3)
    ax.set_yticks([-i for i in range(4)], [a[1] for a in arms], fontsize=6)
    ax.set_ylim(-3.5, 0.5)
    ax.set_xlim(0, 12)
    ax.set_xticks([0, 4, 8, 12])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Months on systemic therapy,\nfirst 12 months", labelpad=1, fontsize=6)
    fig.text((x + 15) / 180, (y + h - 0.2) / (fig.get_size_inches()[1] * 25.4), "All cancers, n = 317", fontsize=6.5, va="top")
    ax.text(obs, 0.62, f"Documented {obs:.1f}", ha="center", va="bottom", fontsize=6)


def prime() -> None:
    s = pd.read_csv(ROOT / "outputs/exploratory_pds_prime_trial_geometry/state_snapshots.csv.gz", low_memory=False)
    v = pd.read_csv(ROOT / "outputs/exploratory_prime_trial_geometry/prime_state_visits.csv.gz", usecols=["SUBJID", "day"])
    s = s[s.kras_exon2.isin(["Wild-type", "Mutant"]) & s.SUBJID.isin(v.SUBJID[v.day.le(14)])].copy()
    s["t"] = s.VISITDY / M
    s = s.sort_values(["SUBJID", "t"]).rename(columns={"SUBJID": "id", "trial_pc1": "x", "trial_pc2": "y"})
    ev_ok = s.RSRESP.notna() & s.RSRESP.ne("Unable to evaluate")  # unevaluable stays missing
    s["crpr"] = s.RSRESP.isin(["Complete response", "Partial response"]).where(ev_ok).astype(float)
    s["sd"] = s.RSRESP.eq("Stable disease").where(ev_ok).astype(float)
    s["pd"] = s.RSRESP.eq("Progressive disease").where(ev_ok).astype(float)
    s["sev"] = (pd.to_numeric(s.active_ae_max_grade, errors="coerce") >= 3).astype(float)
    s["alb"] = (pd.to_numeric(s.Albumin, errors="coerce") < 35).where(s.Albumin.notna()).astype(float)
    s["hb"] = (pd.to_numeric(s.Hemoglobin, errors="coerce") < 100).where(s.Hemoglobin.notna()).astype(float)
    xl = (float(s.x.min()) - 0.2, float(s.x.max()) + 0.2)
    yl = (float(s.y.min()) - 0.2, float(s.y.max()) + 0.2)
    n8 = pd.read_csv(ROOT / "outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv")
    DIFF = "difference (panitumumab + FOLFOX minus FOLFOX)"
    rows = [("Alive", "ALIVE", Gm.TEAL), ("Progression-free", "PF", Gm.TEAL), ("CR or PR", "crpr", STATE_COLOURS["response"]), ("Stable", "sd", STATE_COLOURS["stable"]), ("Progressive", "pd", STATE_COLOURS["progression"]),
            ("Severe AE", "sev", Gm.PURPLE), ("Albumin <35 g/L", "alb", Gm.OCHRE), ("Hb <100 g/L", "hb", Gm.OCHRE)]
    PR = {}
    for kras, tag in (("Wild-type", "wt"), ("Mutant", "mut")):
        gs = []
        for arm, lab, col in (("Panitumumab + FOLFOX", "Panitumumab + FOLFOX", BLUE), ("FOLFOX", "FOLFOX", CHEMO)):
            q = s[s.kras_exon2.eq(kras) & s.arm.eq(arm)]
            b0 = q[q.t <= 14 / M].groupby("id").tail(1).set_index("id")
            p = q.groupby("id").first()
            base = pd.DataFrame({"x": b0.x, "y": b0.y}).reindex(p.index)
            base["t_os"], base["e_os"], base["t_pf"], base["e_pf"] = p.os_months, p.os_event, p.pfs_months, p.pfs_event
            ev = []
            for k, tt, ee in (("progression", "t_pf", "e_pf"), ("death", "t_os", "e_os")):
                hit = T.endpoint_progression(base) if k == "progression" else base[base[ee] > 0]
                for i, r in hit.iterrows():
                    c = q[(q.id == i) & (q.t <= r[tt])]
                    if len(c):
                        ev.append((i, r[tt], k, c.x.iloc[-1], c.y.iloc[-1]))
            gs.append(dict(label=lab, rowlab=lab.replace(" + ", "\n+ "), col=col, base=base, track=q, events=pd.DataFrame(ev, columns=["id", "t", "kind", "x", "y"]), small_cells=True))  # trial patients: counts 1-4 masked
        d = n8[n8.kras.eq(kras) & n8.arm.eq(DIFF)].set_index("part")
        rng = np.random.default_rng(20261106)

        def parts(b: pd.DataFrame) -> np.ndarray:
            pfm = rmst(b.t_pf.to_numpy(float), b.e_pf.to_numpy(int))
            alv = rmst(b.t_os.to_numpy(float), b.e_os.to_numpy(int))
            return np.array([pfm, alv - pfm, 24 - alv])

        A_, B_ = gs[0]["base"], gs[1]["base"]
        est = parts(A_) - parts(B_)
        bs = np.array([parts(A_.iloc[rng.integers(0, len(A_), len(A_))]) - parts(B_.iloc[rng.integers(0, len(B_), len(B_))]) for _ in range(500)])
        forest = [(n_, d.loc[k].months, d.loc[k].lo, d.loc[k].hi, est[i], *np.percentile(bs[:, i], [2.5, 97.5]))
                  for i, (k, n_) in enumerate((("alive_progression_free", "Progression-free"), ("alive_after_progression", "After progression"), ("dead", "Dead")))]
        PR[tag] = (gs, forest)
        T.page(f"cand_final_prime_{tag}", gs, f"randomization. KRAS {kras.lower()},\npatients assessed by day 14", "first two components of\n14 trial features", [0, 3, 6, 9, 12], 12, 6, 3.0, xl, yl, rows, [3, 6, 12, 18], 24, forest,
               "P+F minus F (months of 24)\nopen: map patients",
               f"Same starting patients, three views. PRIME, KRAS {kras.lower()}, patients with a baseline lesion assessment by day 14, by randomized arm. a, Kaplan-Meier curves of these patients, and how the first 24 months divide, panitumumab + FOLFOX minus FOLFOX: all patients with known KRAS status (filled; 514 wild-type and 352 mutant, 866 of 935 trial records) and the 439 map patients (open; 260 and 179; 95% bootstrap intervals, 500 resamples within arm). b, Starting-state outcome maps: survival at 12 months by position at the start (kernel-weighted Kaplan-Meier; blank where the kernel-weighted mass of patients with known status at the horizon is under 12, or under 80% of the local mass), and mean displacement from the start to month 6 among survivors located then. c, Follow-up frames: positions at each month among patients assessed in the preceding three months, with endpoint-derived progressions (a PFS exit on the date of a death is shown only as that death) and deaths drawn at the last assessment, bright when new since the previous frame, pale when earlier. d, Clinical ribbon: survival (Kaplan-Meier) and, among patients assessed in the interval, RECIST, severe adverse events (code 3 or 4) and labs. Comparisons between arms are from randomization; later frames describe the patients and are not randomized comparisons. Unevaluable responses stay missing. Counts of 1 to 4 patients are shown as <5 (and one other count as ≥5 where needed to prevent recovery by subtraction) and the matching ribbon cells are blank (with one more RECIST cell when needed to prevent recovery by subtraction). Observed trial data, no language model; exploratory.",
               ["outputs/exploratory_pds_prime_trial_geometry/state_snapshots.csv.gz", "outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv"], [],
               "Same starting patients: classical summary, starting-state map, follow-up frames.", gap=6.6, tlabel="Months from randomization", xlab="Trial PC1 →", ylab="Trial PC2 →", axes_note=AX_PRIME)
    gs, fw = PR["wt"]  # wild-type map in the main figure; mutant pages in Extended Data
    fm = PR["mut"][1]
    fo = [(f"{a[0]}, {k}", *a[1:]) for pair in zip(fw, fm) for a, k in zip(pair, ("WT", "mutant"))]
    rc = [q for q in rows if q[1] in ("ALIVE", "PF", "crpr", "pd", "sev", "alb")]
    T.page("prop_fig6_prime", gs, "randomization. Map: KRAS wild-type,\npatients assessed by day 14", "first two components of\n14 trial features", [0, 3, 6, 9, 12], 12, 6, 3.0, xl, yl, rc, [3, 6, 12, 18], 24, fo,
           "P+F minus F (months of 24)\nopen: map patients",
           "PRIME, from randomization. a, Kaplan-Meier curves of KRAS wild-type patients with a baseline assessment by day 14 (shaded, 95% interval) and how the first 24 months divide, panitumumab + FOLFOX minus FOLFOX, in KRAS wild-type (WT) and mutant disease: all patients with known KRAS status (filled; 514 wild-type and 352 mutant, 866 of 935 trial records) and the 439 map patients (open; 260 and 179). Map, frames and ribbon show KRAS wild-type map patients; mutant maps in Extended Data; frames after randomization are not arm comparisons. b, Survival at 12 months by starting position, and mean move to month 6 among survivors located then. c, Follow-up frames. d, Clinical ribbon. Counts of 1 to 4 patients are shown as <5 (and one other count as ≥5 where needed to prevent recovery by subtraction) and the matching ribbon cells are blank. Observed trial data, no language model.",
           ["outputs/exploratory_pds_prime_trial_geometry/state_snapshots.csv.gz", "outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv"], [], "PRIME in three views.", gap=6.6, tlabel="Months from randomization",
           xlab="Trial PC1 →", ylab="Trial PC2 →", axes_note=AX_PRIME, height=95.0, ha=28.0, hb=22.0, rib_h=24.0, key_pad=7.0, parts="cd", letters="xxbc")


def egfr(context: str = "EGFR_1L", tag: str = "egfr", draw: bool = True):
    import build_paper_precision_ladder_paired_views as paired
    st, en, _ = paired.load_phase1()
    st = st[st.context.eq(context) & st.arm.isin(["osimertinib", "chemo"])].copy()
    en = en[en._gid.isin(st._gid.unique())].set_index("_gid")
    st["x"], st["y"] = A.project(st)
    st["t"] = (st.days_from_diagnosis - st.cutoff_days) / M
    st = st.sort_values(["_gid", "t", "snapshot_index"]).rename(columns={"_gid": "id"})
    st["treated"] = st.active_treatment_regimens_joined.notna().astype(float)
    st["osi"] = st.active_treatment_regimens_joined.fillna("").str.lower().str.contains("osimertinib").astype(float)
    st["ecog2"] = (st.ecog_ps >= 2).astype(float)
    st["g3"] = (st.host_active_ae_max_grade >= 0.6 - 1e-9).astype(float)
    for f, c in (("liver", "tumour_hepatic"), ("thorax", "tumour_thoracic"), ("bone", "tumour_osseous"), ("cns", "tumour_cns")):
        st[f] = (st[c] > 0).astype(float)
    cut = st.groupby("id").cutoff_days.first()
    xl = (float(st.x.min()) - 0.1, float(st.x.max()) + 0.1)
    yl = (float(st.y.min()) - 0.1, float(st.y.max()) + 0.1)
    rows = [("Alive", "ALIVE", Gm.TEAL), ("Progression-free", "PF", Gm.TEAL), ("Treated", "treated", Gm.BLUE), ("On osimertinib", "osi", Gm.BLUE), ("Grade ≥3 AE", "g3", Gm.PURPLE), ("ECOG ≥2", "ecog2", Gm.OCHRE), ("Liver", "liver", Gm.ORANGE), ("Thoracic", "thorax", Gm.ORANGE), ("Bone", "bone", Gm.ORANGE), ("CNS", "cns", Gm.ORANGE)]
    gs = []
    for arm, lab, col in (("osimertinib", "Osimertinib", PAL["osimertinib"]), ("chemo", "Chemotherapy", PAL["chemotherapy"])):
        q = st[st.arm.eq(arm)]
        ids = q.id.unique()
        b0 = q[q.t <= 0].groupby("id").tail(1).set_index("id")
        base = pd.DataFrame({"x": b0.x.round(6), "y": b0.y.round(6)}).reindex(ids)
        base["t_os"] = (en.os_days.reindex(ids) - cut.reindex(ids)) / M
        base["e_os"] = 1
        base["t_pf"] = en.pfs_line_1_days.reindex(ids) / M
        base["e_pf"] = 1
        base["cluster"] = base.index.str.extract(r"__(S\d+__H\d+)__")[0].to_numpy()
        ev = []
        for i, r in base.iterrows():
            c = q[q.id == i]
            for k, tt in (("progression", r.t_pf), ("death", r.t_os)):
                z = c[c.t <= tt]
                ev.append((i, tt, k, z.x.iloc[-1], z.y.iloc[-1]))
        gs.append(dict(label=lab, col=col, base=base, track=q, events=pd.DataFrame(ev, columns=["id", "t", "kind", "x", "y"])))
    rng = np.random.default_rng(20261106)
    tau = 24.0
    keys = [g["base"].index.str.extract(r"__(S\d+__H\d+)__")[0].to_numpy() for g in gs]
    u = np.unique(keys[0])
    assert len(u) == 15 and set(u) == set(keys[1])
    bs = []
    for _ in range(1000):
        c = dict(zip(*np.unique(rng.choice(u, len(u)), return_counts=True)))
        ws = [np.array([c.get(k, 0) for k in kk], float) for kk in keys]
        bs.append(split(gs[0]["base"], tau, ws[0]) - split(gs[1]["base"], tau, ws[1]))
    est = split(gs[0]["base"], tau) - split(gs[1]["base"], tau)
    lo, hi = np.percentile(np.array(bs), [2.5, 97.5], axis=0)
    forest = [(n_, est[i], lo[i], hi[i]) for i, n_ in enumerate(("Progression-free", "After progression", "Dead"))]
    if not draw:
        return gs, forest, xl, yl, [q for q in rows if q[1] in ("ALIVE", "PF", "osi", "g3", "ecog2", "cns")]
    T.page(f"cand_final_{tag}", gs, "treatment decision. EGFR-mutant\nlung, five starting states,\n9 futures each per arm", "disease and host\n(first component of each block)", [0, 6, 12, 24, 36, 48], 36, 12, 1e9, xl, yl, rows, [0, 6, 12, 24, 36, 48], 48, forest, "Osimertinib minus chemo\n(months of 24)",
           "Same starting records, three views. Claude Sonnet 5, EGFR-mutant lung adenocarcinoma, assigned osimertinib or carboplatin-pemetrexed (45 futures per arm: five starting states, three histories, three futures). a, Kaplan-Meier curves (solid, overall survival; dashed, first-line progression-free; shaded, 95% interval, 400 resamples of the 15 starting records; none censored), and how the first 24 months divide, osimertinib minus chemotherapy (95% bootstrap intervals, 1,000 resamples of the 15 starting records). b, Starting-state outcome maps: survival at 36 months for each starting state, and mean displacement from it to month 12 among survivors. c, Follow-up frames: positions at each month, first-line progression and death at the last state, bright when new, pale when earlier. d, Clinical ribbon. The chemotherapy arm includes early switches to osimertinib. Model-generated futures.",
           ["outputs/paper_state_map (via load_phase1)"], [], "Same starting records: classical summary, starting-state map, follow-up frames.", discrete=True, tlabel="Months from treatment decision", xlab="Disease →", ylab="Host impairment →", klabel="Futures (%)", axes_note=AX_DH)
    rc = [q for q in rows if q[1] in ("ALIVE", "PF", "treated", "g3", "ecog2", "cns")]
    if tag != "egfr":
        return
    T.page("prop_fig3_egfr", gs, "treatment decision. EGFR-mutant\nlung, five starting states,\n9 futures each per arm", "disease and host\n(first component of each block)", [0, 12, 24, 48], 36, 12, 1e9, xl, yl, rc, [0, 12, 24, 36, 48], 48, forest,
           "Osimertinib minus chemo\n(months of 24)",
           "Claude Sonnet 5, EGFR-mutant lung adenocarcinoma, assigned osimertinib or carboplatin-pemetrexed (45 futures per arm: five starting states, three histories, three futures). a, Kaplan-Meier curves (solid, overall survival; dashed, first-line progression-free; shaded, 95% interval) and how the first 24 months divide (95% intervals; all intervals resample the 15 starting records). b, Survival at 36 months by starting state, and mean move to month 12 among survivors. c, Follow-up frames; right, a separate experiment: Claude Sonnet 5 factorial, 360 futures (270 active-treatment futures for first-line PFS). Each value is the mean within-stratum range of input-level means, minus its permutation-null mean, divided by the median within-cell s.d.; negative, below the shuffled spread; descriptive (full matrix in Extended Data). d, Clinical ribbon. PFS runs from first-line treatment initiation. The chemotherapy arm includes early switches to osimertinib. Model-generated futures.",
           ["outputs/paper_state_map (via load_phase1)"], [], "EGFR in three views.", discrete=True, tlabel="Months from treatment decision", xlab="Disease →", ylab="Host impairment →", klabel="Futures (%)", axes_note=AX_DH, height=139.5, ha=26.0, hb=22.0, rib_h=24.0, key_pad=7.0, frame_w=112.0, side=side_inputs)


def egfr_four() -> None:
    """EGFR-mutant and wild-type, both treatments, identical axes, horizons and scales."""
    gm, fm, xm, ym, rc = egfr(draw=False)
    gw, fw_, xw, yw, _ = egfr("EGFR_WT_1L", "egfr_wt", draw=False)
    xl, yl = (min(xm[0], xw[0]), max(xm[1], xw[1])), (min(ym[0], yw[0]), max(ym[1], yw[1]))
    gs = []
    for block, grp in (("EGFR-mutant", gm), ("EGFR wild-type", gw)):
        for g in grp:
            gs.append({**g, "arm": g["label"], "block": block, "label": f"{block}, {g['label'].lower()}"})
    forest = [(a[0], a[1], a[2], a[3], b[1], b[2], b[3]) for a, b in zip(fm, fw_)]
    T.page("cand_fig3_four", gs, "", "", [6, 24, 48], 36, 6, 1e9, xl, yl, rc, [0, 6, 12, 24, 36, 48], 48, forest, "Osimertinib minus chemo (months of 24)\nfilled, mutant; open, wild-type",
           "EGFR-mutant and wild-type on one page.", ["outputs/paper_state_map (via load_phase1)"], [], "EGFR-mutant and wild-type on one page.", discrete=True,
           tlabel="Months from decision", xlab="Disease →", ylab="Host impairment →", klabel="Futures (%)", axes_note=AX_DH, height=139.5, ha=23.0, hb=18.0, rib_h=24.0, key_pad=7.0, frame_w=165.0,
           paired=True, key_title="Claude Sonnet 5 futures")


def atlas() -> None:
    from build_atlas_model_dissection import MODEL_LABEL  # noqa: F401
    from build_field_trellis import CODE2FAM
    import fig1_data as F1
    sn = F1.enrich_snapshots(F1.load_tables())  # true endpoints: the projection leaves out terminal states
    end = pd.DataFrame({"t": sn.groupby("_gid").days_from_diagnosis.max() / M, "e": sn.groupby("_gid").outcome_type.first().eq("death").astype(int)})
    a = pd.read_csv(ROOT / "outputs/paper_state_map/state_projection.csv.gz", low_memory=False)
    a["fam"] = a.tumor_oncotree_code.astype(str).map(CODE2FAM)
    a = a[a.fam.isin(["Lung", "Gastrointestinal", "Sarcoma", "Breast"])].copy()
    a["x"], a["y"] = A.project(a)
    a["t"] = a.days_from_diagnosis / M
    a = a.sort_values(["_gid", "t", "snapshot_index"]).rename(columns={"_gid": "id"})
    a["ecog2"] = (a.host_ecog >= 0.5 - 1e-9).astype(float)
    a["g3"] = (a.host_active_ae_max_grade >= 0.6 - 1e-9).astype(float)
    for f, c in (("liver", "tumour_hepatic"), ("thorax", "tumour_thoracic"), ("bone", "tumour_osseous"), ("cns", "tumour_cns")):
        a[f] = (a[c] > 0).astype(float)
    xl = (float(a.x.min()) - 0.1, float(a.x.max()) + 0.1)
    yl = (float(a.y.min()) - 0.1, float(a.y.max()) + 0.1)
    rows = [("Alive", "ALIVE", Gm.TEAL), ("ECOG ≥2", "ecog2", Gm.OCHRE), ("Grade ≥3 AE", "g3", Gm.PURPLE), ("Liver", "liver", Gm.ORANGE), ("Thoracic", "thorax", Gm.ORANGE), ("Bone", "bone", Gm.ORANGE), ("CNS", "cns", Gm.ORANGE)]
    gs = []
    for f, col in (("Lung", FAMILY_COLOURS["lung"]), ("Gastrointestinal", FAMILY_COLOURS["gi"]), ("Sarcoma", FAMILY_COLOURS["sarcoma"]), ("Breast", FAMILY_COLOURS["breast"])):
        q = a[a.fam.eq(f)]
        g0 = q.groupby("id")
        base = pd.DataFrame({"x": g0.x.first(), "y": g0.y.first()})
        base["t_os"], base["e_os"] = end.t.reindex(base.index), end.e.reindex(base.index)
        assert base.t_os.notna().all() and (base.t_os >= g0.t.max() - 1e-9).all()
        last = g0.tail(1).set_index("id")
        ev = pd.DataFrame({"id": last.index, "t": base.t_os.reindex(last.index), "kind": "death", "x": last.x, "y": last.y}).reset_index(drop=True)
        gs.append(dict(label=f.replace("Gastrointestinal", "GI"), col=col, base=base, track=q, events=ev))
    rng = np.random.default_rng(20261106)
    forest = []
    for g in gs:
        b = g["base"]
        t, e = b.t_os.to_numpy(float), b.e_os.to_numpy(int)
        v = [100 * T.km_curve(t[i], e[i], np.ones(len(i)), [18.0])[0] for i in (rng.integers(0, len(b), len(b)) for _ in range(1000))]
        forest.append((g["label"], 100 * T.km_curve(t, e, np.ones(len(b)), [18.0])[0], *np.percentile(v, [2.5, 97.5])))
    T.page("cand_final_atlas", gs, "diagnosis. Generated patients of four\nfamilies, six models pooled", "disease and host\n(first component of each block)", [0, 6, 12, 18, 24, 36], 18, 12, 1e9, xl, yl, rows, [0, 12, 24, 36], 48, forest, "Alive at 18 months (%)",
           "Generated patients of four cancer families, six models pooled, from diagnosis (the Kaplan-Meier curves and follow-up frames are in Fig. 2c). a, Starting-state outcome maps: survival at 18 months by position at diagnosis (kernel-weighted Kaplan-Meier estimate; blank where the kernel-weighted mass of patients with known status at the horizon is under 12, or under 80% of the local mass), and mean displacement from diagnosis to month 12 among survivors. b, Clinical course. Each map describes these starting patients, not an intrinsic risk of a position. Model-generated patients.",
           ["outputs/paper_state_map/state_projection.csv.gz"], [], "Each family: classical summary, starting-state map, follow-up frames.", pf=False, points=False, tlabel="Months from diagnosis", xlab="Disease →", ylab="Host impairment →", klabel="Generated patients (%)", axes_note=AX_DH,
           parts="bd", letters="xaxb", height=96.0)


def genie() -> None:
    D = ROOT / "outputs/exploratory_cand_css_four_arm"
    series = pd.read_csv(D / "series_cal.csv.gz", keep_default_na=False, na_values=[""])
    state = pd.read_csv(D / "state_cal.csv.gz")
    fol = pd.read_csv(D / "sites.csv.gz", usecols=["source", "query", "repeat", "death", "follow_end"])
    STATE = {"no_active_disease": 0.0, "responding": 1.0, "stable": 2.0, "progressing": 3.0}
    SRC = [("observed", "Observed", "#222222"), ("patient_only", "Record alone", PAL["arm_patient_only"]), ("patients_like_me", "Matched examples", PAL["arm_matched"])]
    for cancer, tag in (("NSCLC", "nsclc"), ("CRC", "crc")):
        keep = list(GS.SITES)
        d = series[series.cancer.eq(cancer)].merge(state[state.cancer.eq(cancer)], on=["source", "cancer", "query", "repeat", "t"], how="left")
        st0 = state[state.cancer.eq(cancer) & state.t.eq(0) & state.source.eq("observed")]
        d = d[d["query"].isin(st0["query"][GS.located(st0)])].copy()
        d["extent"] = np.minimum(d[keep].sum(axis=1), 5) / 5
        d["loc"] = GS.located(d)
        F = keep + ["extent"]
        o = d[d.source.eq("observed") & d.status.eq("alive") & d["loc"]]
        mu, sd, v1, v2, var = GS.weighted_pca(o, F, o["query"])
        z = ((d[F].fillna(mu) - mu) / sd).to_numpy()
        d["x"] = np.where(d["loc"], z @ v1, np.nan)
        d["y"] = np.where(d["loc"], z @ v2, np.nan)
        d["id"] = d["query"] + "#" + d.source + "#" + d["repeat"].astype(str)
        for k, n_ in (("responding", "resp"), ("stable", "stab"), ("progressing", "prog")):
            d[n_] = d.state.eq(k).where(d.state.isin(list(STATE))).astype(float)
        q_ = d[d.source.isin([s_[0] for s_ in SRC])].dropna(subset=["x"])
        xl = (float(q_.x.min()) - 0.3, float(q_.x.max()) + 0.3)
        yl = (float(q_.y.min()) - 0.3, float(q_.y.max()) + 0.3)
        rows = [("Alive", "ALIVE", Gm.TEAL), ("Responding", "resp", STATE_COLOURS["response"]), ("Stable", "stab", STATE_COLOURS["stable"]), ("Progressing", "prog", STATE_COLOURS["progression"]), ("Treated", "on_tx", Gm.BLUE)] + [(GS.LAB[f], f, Gm.ORANGE) for f in keep[:5]]
        gs = []
        for src, lab, col in SRC:
            q = d[d.source.eq(src)]
            b0 = q[q.t.eq(0)].set_index("id")
            f = fol[fol.source.eq(src)].assign(id=lambda z: z["query"] + "#" + src + "#" + z["repeat"].astype(str)).set_index("id")
            base = b0[["x", "y"]].copy()
            dd = f.death.reindex(base.index).astype(float)
            end = f.follow_end.reindex(base.index).astype(float).fillna(np.inf)
            end = np.minimum(end, 24.5)
            base["t_os"] = np.minimum(dd, end)
            base["e_os"] = (dd <= end).astype(int)
            base["w"] = 1 / base.groupby(q[q.t.eq(0)].set_index("id")["query"]).x.transform("size")
            base["cluster"] = q[q.t.eq(0)].set_index("id")["query"].reindex(base.index)
            gs.append(dict(label=lab, short=lab.split()[0], col=col, base=base, track=q, events=None, small_cells=src == "observed"))
        rng = np.random.default_rng(20261106)
        qs = [pd.Series(g["base"].index).str.split("#").str[0].to_numpy() for g in gs]
        u = np.unique(qs[0])

        def alive(g: dict, w: np.ndarray) -> np.ndarray:
            b = g["base"]
            return 100 * T.km_curve(b.t_os.to_numpy(float), b.e_os.to_numpy(int), w, [12.0, 24.0])

        est = [alive(g, g["base"].w.to_numpy(float)) for g in gs]
        bs = []
        for _ in range(500):
            c = pd.Series(rng.choice(u, len(u))).value_counts()
            v = [alive(g, g["base"].w.to_numpy(float) * pd.Series(k).map(c).fillna(0).to_numpy(float)) for g, k in zip(gs, qs)]
            bs.append([v[1] - v[0], v[2] - v[0]])
        bs = np.array(bs)
        forest = [(f"{gs[i + 1]['short']}, {m} mo", (est[i + 1] - est[0])[j], *np.percentile(bs[:, i, j], [2.5, 97.5])) for j, m in enumerate((12, 24)) for i in range(2)]
        T.page(f"cand_final_genie_{tag}", gs, f"the landmark. {cancer}, located real\npatients, read on their real calendar", "first two components of the\ntumour block at time t", [0, 3, 6, 12, 18, 24], 12, 6, 0.5, xl, yl, rows, [0, 6, 12, 24], 24, forest, "Alive, generated minus\nobserved (points)",
               f"The same real patients, observed and generated. GENIE-BPC {cancer}; DeepSeek V4 Flash, record alone and matched examples; every course read on the patient's real assessment calendar. a, Survival from the landmark (Kaplan-Meier; generated curves pool each patient's futures with equal patient weight; shaded, 95% interval, 400 resamples of patients; truncated at 24 months), and survival at 12 and 24 months, generated minus observed in percentage points (95% bootstrap intervals, 500 resamples of patients). b, Starting-state outcome maps: survival at 12 months by position at the landmark (blank where the kernel-weighted mass of patients with known status at the horizon is under 10, or under 80% of the local mass), and mean displacement from the landmark to month 6 among survivors located then. The same patients start at the same positions in all rows. c, Follow-up frames: positions at each month among patients located then; generated counts are patient equivalents. d, Clinical ribbon: survival, then disease state, treatment and sites among those known. Aggregates only: observed counts of 1 to 4 are shown as <5 (and one other count as ≥5 where needed to prevent recovery by subtraction) and the matching ribbon cells are blank.",
               ["outputs/exploratory_cand_css_four_arm/series_cal.csv.gz", "outputs/exploratory_cand_css_four_arm/state_cal.csv.gz"], [],
               "The same real patients: observed and generated, from the same starts.", points=False, sigma=2.8, pf=False, min_eff=10.0, bw=0.18, gap=6.6, tlabel="Months from the landmark", xlab="Tumour PC1 →", ylab="Tumour PC2 →", klabel="Alive (%)", axes_note=AX_GENIE)
        if tag == "nsclc":
            rc = [q for q in rows if q[0] in ("Alive", "Responding", "Stable", "Progressing", "Treated", "Thoracic")]
            T.page("prop_fig5_genie", gs, "the landmark. NSCLC, located real\npatients, read on their real calendar", "first two components of the\ntumour block at time t", [0, 6, 12, 24], 12, 6, 0.5, xl, yl, rc, [0, 6, 12, 24], 24, forest, "Alive, generated minus\nobserved (points)",
                   "The same real patients, observed and generated. GENIE-BPC NSCLC; DeepSeek V4 Flash, record alone and matched examples; every course read on the patient's real assessment calendar. a, Survival from the landmark (shaded, 95% interval; generated curves pool each patient's futures with equal patient weight), and survival at 12 and 24 months, generated minus observed (95% bootstrap intervals, 500 resamples of patients). b, Survival at 12 months by position at the landmark, and mean move to month 6 among survivors located then. c, Follow-up frames, a selected comparison of observed and matched examples (all three groups in Extended Data; counts of generated courses are patient equivalents); right, an exception: documented systemic treatment against mean forecast months, in 317 evaluable patients of all three cancers. d, Clinical ribbon. e, Whole-course score, all cancers, with n per horizon. Panels a to d show 109 NSCLC patients, except the inset at c; panel e covers all cancers. Aggregates only: observed counts of 1 to 4 are shown as <5 (and one other count as ≥5 where needed to prevent recovery by subtraction) and the matching ribbon cells are blank.",
                   ["outputs/exploratory_cand_css_four_arm/series_cal.csv.gz", "outputs/exploratory_cand_css_four_arm/state_cal.csv.gz"], [], "Observed and generated in three views.",
                   points=False, sigma=2.8, pf=False, min_eff=10.0, bw=0.18, gap=6.6, tlabel="Months from the landmark", xlab="Tumour PC1 →", ylab="Tumour PC2 →", klabel="Alive (%)", axes_note=AX_GENIE, height=96.0, ha=26.0, hb=22.0, rib_h=24.0, key_pad=7.0, frame_groups=["Observed", "Matched examples"], frame_w=112.0, side=side_treatment, parts="cd", letters="xxbc")


for name in WHICH:
    {"prime": prime, "egfr": egfr, "egfr_wt": lambda: egfr("EGFR_WT_1L", "egfr_wt"), "egfr_four": egfr_four, "atlas": atlas, "genie": genie}[name]()
