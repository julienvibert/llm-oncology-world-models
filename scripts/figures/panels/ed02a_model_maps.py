"""Panel ed02a_model_maps: where each model's generated states sit on the frozen state map."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import MODEL_COLOURS, MODEL_KEYS  # noqa: E402
from _ed12 import projection  # noqa: E402
from common import MODEL_ORDER, panel, save_panel  # noqa: E402

p = projection()
bg = p.sample(8000, random_state=1)
fig, axs = panel(180, 64, 2, 3, sharex=True, sharey=True)
rev = {v: k for k, v in MODEL_KEYS.items()}
for ax, name in zip(axs.ravel(), MODEL_ORDER):
    g = p[p.model_short == rev[name]]
    ax.scatter(bg.pc1, bg.pc2, s=1.0, c="#E3E7EA", linewidths=0, rasterized=True)
    s = g.sample(min(len(g), 5000), random_state=2)
    ax.scatter(s.pc1, s.pc2, s=1.2, c=MODEL_COLOURS[name], alpha=0.5, linewidths=0, rasterized=True)
    ax.text(0.07, 1.02, f"{name} (n = {len(g):,} states)", transform=ax.transAxes, va="bottom", ha="left")
    ax.plot([0.02], [1.085], marker="s", ms=5, color=MODEL_COLOURS[name], mew=0, transform=ax.transAxes, clip_on=False, ls="none")
for ax in axs[1]:
    ax.set_xlabel("State-map PC1 (a.u.)")
for ax in axs[:, 0]:
    ax.set_ylabel("State-map PC2 (a.u.)")
save_panel(
    fig, "ed02a_model_maps",
    legend=("Where each model places its generated patients on the state map (the map defined in Extended Data Fig. 1). "
            "One panel per model; coloured dots, a random 5,000 of that model's living clinical states (n states shown in "
            "each title); pale dots, 8,000 states drawn from all six models for reference. Axes, state-map PC1 and PC2 "
            "in arbitrary units (loadings described in Extended Data Fig. 1b; neither is a validated clinical scale). Models cover the same map but with different densities, for example at high PC1 "
            "(right). Source: generated atlas, 4,434 trajectories (105,839 living states), six models."),
    sources=["outputs/paper_state_map/state_projection.csv.gz"],
    confirm=["Legend sentence on density differences is a reading of the figure.",
             "Subsample sizes and seeds (5,000 per model, 8,000 reference; seeds 2 and 1) are display-only choices."],
    message="Six models fill the same state map with different densities.")
