"""Panel ed01a_state_map: the frozen state map (PC1 x PC2) coloured by mean encoded burden."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _ed12 import AX1, AX2, CMAP, D, coord_cols, projection  # noqa: E402
from common import panel, save_panel  # noqa: E402

p = projection()
sample = p.sample(24000, random_state=20260730)
sev = sample[coord_cols(p)].mean(axis=1)
ev = json.loads((D / "manifest.json").read_text())["pca_explained_variance_ratio"]

fig, ax = panel(88, 74)
sc = ax.scatter(sample["pc1"], sample["pc2"], c=sev, cmap=CMAP, s=1.4, alpha=0.45, linewidths=0, rasterized=True)
ax.set_xlabel(AX1)
ax.set_ylabel(AX2)
cb = fig.colorbar(sc, ax=ax, location="right", fraction=0.045, pad=0.02)
cb.set_label("Mean of the 20 encoded coordinates")
cb.outline.set_linewidth(0.5)
save_panel(
    fig, "ed01a_state_map",
    legend=("The state map. Each dot is one generated, living clinical state (24,000 of 105,839 shown), placed by two "
            "composite scores that combine 10 tumour and 10 host findings. Axes are the first two components of a fixed linear map (PC1, "
            f"{100 * ev[0]:.1f}% of variance; PC2, {100 * ev[1]:.1f}%), in arbitrary units. PC1 is mainly liver and lung disease, disease "
            "extent, symptoms and performance status, so higher PC1 means more of these findings. PC2 mainly contrasts marrow disease and marrow "
            "reserve (higher) against liver, thoracic and other anatomical disease (lower), with symptoms and ECOG also positive; a high PC2 "
            "can therefore mean less lung or liver disease and is not a host-health score. Neither axis is a validated clinical scale. Colour, mean of the 20 encoded findings (pale, low; wine, high). "
            "Death is outside the map. Source: generated atlas, six models, 4,434 trajectories."),
    sources=["outputs/paper_state_map/state_projection.csv.gz", "outputs/paper_state_map/manifest.json"],
    confirm=[],
    message="The state map places generated clinical states by two composite scores of 10 tumour and 10 host findings.")
