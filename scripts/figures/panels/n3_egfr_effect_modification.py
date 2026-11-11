"""Panel n3_egfr_effect_modification: osimertinib minus chemotherapy, EGFR-mutant versus wild-type, per model.

Message: a model can rank osimertinib first with the mutation and still keep its advantage without the target.
Osimertinib minus chemotherapy in median first-line PFS, EGFR-mutant versus EGFR wild-type, per model:
Sonnet reverses the difference (+7.3 to -7.1 months), Kimi K3 keeps a positive one (+6.3 to +3.8) and GLM-5.2 a
large one (+30.7 to +22.3).
Data: per-future endpoints (phase1_endpoint_long.csv of the three phase-1 runs, endpoint_days / 30.4375), the same
futures as f3d_wrong_target. Medians are checked against _egfr_days.context_medians().
Interval: 95% bootstrap percentile interval resampling the starting records (condition_id, 15 per status) with both
arms of a record kept together, 4,000 resamples, seed 20261028. Futures are not independent patients; the interval
describes the generated set.
Run: uv run python scripts/figures/panels/n3_egfr_effect_modification.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, INK, GREY, MODEL_COLOURS, save_panel  # noqa: E402
from _f3util import axes_mm, minus, new_fig  # noqa: E402
from _egfr_days import RUNS, CONTEXT, MONTH_DAYS, context_medians, value  # noqa: E402

B, SEED = 4000, 20261028
MODELS = [("claude_sonnet5", "Sonnet"), ("kimi_k3_off", "Kimi K3"), ("glm52_fp8", "GLM-5.2")]
STATUS = [("mut", "EGFR-mutant"), ("wt", "EGFR wild-type")]

futures = pd.concat([pd.read_csv(ROOT / d / "phase1_endpoint_long.csv") for d in RUNS.values()], ignore_index=True)
futures["months"] = futures["endpoint_days"] / MONTH_DAYS
MED = context_medians()


def diff(t: pd.DataFrame) -> float:
    return float(t.loc[t.arm_key.eq("osimertinib"), "months"].median() - t.loc[t.arm_key.eq("chemo"), "months"].median())


rng = np.random.default_rng(SEED)
rows = []
for m, name in MODELS:
    for st, lab in STATUS:
        t = futures[futures.model.eq(m) & futures.context.eq(CONTEXT[st]) & futures.arm_key.isin(["osimertinib", "chemo"])]
        for arm in ("osimertinib", "chemo"):  # same medians as Fig. 3f
            assert abs(t.loc[t.arm_key.eq(arm), "months"].median() - value(MED, m, st, arm)) < 1e-9, (m, st, arm)
        groups = {c: g for c, g in t.groupby("condition_id")}
        keys = list(groups)
        boot = [diff(pd.concat([groups[k] for k in rng.choice(keys, len(keys))])) for _ in range(B)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(model=m, name=name, status=st, est=diff(t), lo=lo, hi=hi, n_records=len(keys),
                         n_osi=int(t.arm_key.eq("osimertinib").sum()), n_chemo=int(t.arm_key.eq("chemo").sum())))
R = pd.DataFrame(rows)
print(R.round(2).to_string(index=False))

W, H = 58.0, 42.0
fig = new_fig(W, H)
ax = axes_mm(fig, 10.5, 4.6, 29.0, 36.0)
ax.axhline(0, color=GREY, lw=0.5, ls="--", zorder=0)
for k, (m, name) in enumerate(MODELS):
    r = R[R.model.eq(m)].set_index("status")
    col = MODEL_COLOURS[name]
    xs = np.array([0, 1]) + (k - 1) * 0.06
    ax.plot(xs, [r.loc["mut", "est"], r.loc["wt", "est"]], color=col, lw=1.0, zorder=2)
    for x, st in zip(xs, ("mut", "wt")):
        ax.plot([x, x], [r.loc[st, "lo"], r.loc[st, "hi"]], color=col, lw=0.8, zorder=2)
        ax.plot(x, r.loc[st, "est"], "o", color=col, ms=4, mec="white", mew=0.5, zorder=3)
    ax.text(1.12, r.loc["wt", "est"], f"{name} {minus(r.loc['wt', 'est'])}", fontsize=6, color=INK, va="center", ha="left",
            clip_on=False)
ax.set_xlim(-0.25, 1.15)
ax.set_xticks([0, 1])
ax.set_xticklabels([lab for _, lab in STATUS])
ax.set_ylabel("Osimertinib minus chemotherapy,\nmedian first-line PFS (months)")
save_panel(
    fig, "n3_egfr_effect_modification",
    legend=("Difference in median duration of first-line control between osimertinib and carboplatin-pemetrexed "
            "chemotherapy generated from the same starting records, in EGFR-mutant and EGFR wild-type lung cancer, for three "
            "models (colours). Osimertinib has no target in wild-type disease, so its advantage over chemotherapy is expected "
            "to disappear. Points, difference of medians; lines, 95% bootstrap intervals resampling the 15 starting records "
            "per status with both arms kept together (4,000 resamples). Dashed line, no difference. 45 futures per arm and "
            "status (15 records, three repeats), except GLM-5.2 chemotherapy, 35 (mutant) and 42 (wild-type). Futures are "
            "not independent patients, so intervals describe the generated set. Months are days / 30.4375."),
    sources=[f"{d}/phase1_endpoint_long.csv" for d in RUNS.values()] + ["scripts/figures/panels/_egfr_days.py"],
    confirm=["Kimi K3 passed all five predefined ordering checks, yet keeps osimertinib ahead of chemotherapy without the "
             "target: the check 'osimertinib is not the best arm' passes because another arm is ranked first. Text that "
             "says Kimi 'ranked the treatments correctly' should be qualified.",
             "Bootstrap unit = starting record (condition_id), paired across arms."],
    message="Only Sonnet removes the osimertinib advantage when the target is absent; Kimi and GLM keep it.")
