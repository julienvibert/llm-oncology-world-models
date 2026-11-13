# Large language models as oncology world models: code

Code for "Large language models as oncology world models" (Vibert and Kather, 2026). Release v1.0.1: the code of v1.0, under the license below.
Repository: https://github.com/julienvibert/llm-oncology-world-models
Code and data: Zenodo, https://doi.org/10.5281/zenodo.24137513
License: Non-Commercial Software License – Heidelberg University Version 1.0 (NC-SA-UHDV1.0; `LICENSE`), non-commercial use only. The generated data are under CC BY 4.0.

The pipeline asks a large language model (LLM) to write the complete course of a cancer patient, from diagnosis
to death, as a structured record: a baseline description and a dated list of events (disease, treatment,
toxicity and condition). The records are validated, reconstructed into the patient's state over time, and
analyzed. This release contains the generation pipeline, the analysis code of the paper and the scripts of its
figures.

Three files describe the release:

- `REPRODUCTION_INDEX.md`: every figure panel, the script that draws it, the files it reads, and whether it can
  be rebuilt from public data (**public**) or needs registry or trial data (**restricted**).
- `PROVENANCE.md`: models, dates, code commits and checksums of the generated data.
- `FILE_MANIFEST.tsv`: every file with its purpose and SHA-256 (`SNAPSHOT_MANIFEST.json` gives the hash of
  `FILE_MANIFEST.tsv` itself).

## 1. Install

You need [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and the packages pinned in `uv.lock`.
For the figures, you also need `cairo` and the Arial or Helvetica font (the figure scripts check the font; on
Linux, install Arial, for example from a Microsoft core-fonts package).

```bash
uv sync --locked
uv run pytest -q
```

## 2. Generate patients

In the paper, the models were called through OpenRouter, which gives access to most LLMs with one API key,
except GLM-5.2, which ran on an institutional OpenAI-compatible server. Add your OpenRouter key to
`~/.config/openrouter.env`:

```bash
mkdir -p ~/.config && echo "OPENROUTER_API_KEY=sk-or-..." >> ~/.config/openrouter.env
```

A run is described by one YAML file, which sets the model and the cancer types. The configurations of the paper
are in `configs/`: for example, `configs/paper_v1_pillar1_anchored_atlas_kimi_k3_n5.yaml` generates five
patients for each of 148 cancer types (OncoTree codes). To use another model, copy the file and change its
`models:` entry to the OpenRouter identifier of that model:

```yaml
models:
- id: anthropic/claude-sonnet-5   # OpenRouter model identifier
  short: sonnet5                  # short name used in folder names
```

The pipeline is not tied to OpenRouter. `src/synthonccases/generate/providers.py` also has direct clients for the
Anthropic, OpenAI and Google APIs, chosen by the model name (`claude-...`, `gpt-...`, `gemini-...`), and for an
OpenAI-compatible server, which ran GLM-5.2 in the paper. To use them, set `use_openrouter: false` under
`provider:`, remove `response_format`, which only the OpenRouter client sends, and set the API key of that provider in its
usual environment variable (for example `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`). The direct clients limit the
output to 8,192 tokens, against 16,384 in the runs of the paper. Other providers can be added in
the same file. A model outside those of the paper may need such adjustments: the dry run checks the prompts, not
the provider.

The cancer types are listed under `cells:`. To generate one patient for each of the first two cancer types:

```bash
# check the configuration without calling the model
uv run python scripts/run_paper_v1.py --config configs/paper_v1_pillar1_anchored_atlas_kimi_k3_n5.yaml \
  --out-root runs --run-id demo_preview --cells-count 2 --accepted-futures-per-cell 1 --dry-run
# generate
uv run python scripts/run_paper_v1.py --config configs/paper_v1_pillar1_anchored_atlas_kimi_k3_n5.yaml \
  --out-root runs --run-id demo --cells-count 2 --accepted-futures-per-cell 1
# tables of baselines, events, treatments and endpoints
uv run python scripts/build_rev5_derivation.py --run-root runs/demo
```

Each run folder holds the configuration, the rendered prompt and raw response of every attempt, the accepted
records (`trajectories/`), the rejected attempts with the reason (`failed_attempt.json`), a manifest and, after
the last command, the derivation tables (`derivation/*.csv`). Generation costs API credits.

## 3. Download the generated data of the paper

The Zenodo record holds this code (`paper_code_release_*.zip`) and the data. Download the data files into one
folder that does not contain the code archive: the five `synthonccases_deposit_*.tar.gz` archives,
`ARCHIVES_SHA256SUMS`, `SHA256SUMS`, `DEPOSIT_MANIFEST.tsv`, `TRANSFORMATIONS.tsv`, `DEPOSIT_README.md` and
`LICENSE.txt`. Then, from the root of this release:

```bash
python3 scripts/release/fetch_and_verify.py <download-folder> --checkout .
```

The script checks every archive and file against their SHA-256 checksums and copies the files to their places
(`runs/<run>/...`, `outputs/<folder>/...`). The data hold the six-model cancer atlas, the EGFR experiment and
the EGFR checks of GLM-5.2 and Kimi K3, the 12-scenario benchmark, and the result tables that the figure scripts
read.

## 4. Rebuild the figures

```bash
uv run python scripts/figures/rebuild_panels.py --public   # every panel that does not need restricted data
uv run python scripts/figures/assemble.py --allow-missing  # the figures whose panels are all present
```

Of the 84 panels, 22 are computed from the generated runs, 27 are drawn from deposited result tables, and 35
need GENIE-BPC or PRIME data (`--public` skips them). With this code and the data, the 49 public panels and the
17 figures made only of them are rebuilt identical to those of the paper. One panel is drawn by
`uv run python scripts/figures/rebuild_panels.py --only <id>`, one figure by
`uv run python scripts/figures/assemble.py "Figure 1"`.

## 5. Recompute the analyses

All commands read the deposited runs.

- Derivation tables of a run: `uv run python scripts/build_rev5_derivation.py --run-root runs/<run> --out-dir <folder>`.
- EGFR experiment and EGFR checks (Fig. 3; Extended Data Fig. 4):

  ```bash
  uv run python scripts/explore/build_interventional_phase1.py --out-dir <folder>/egfr_design
  uv run python scripts/explore/score_interventional_phase1.py --run-root runs/interventional_phase1_sonnet_full_v1c \
    --manifest <folder>/egfr_design/arm_manifest.json --out-dir <folder>/sonnet
  uv run python scripts/explore/score_interventional_phase1.py --run-root runs/interventional_phase1_glm52_full_v1c \
    --manifest <folder>/egfr_design/arm_manifest.json --out-dir <folder>/glm52
  uv run python scripts/explore/build_kimi_k3_combined.py --out-dir <folder>/kimi
  ```

  The last command compares its tables with those of the deposit; tables that the deposit does not hold are
  reported as not compared.
- EGFR numbers of the text: `uv run python scripts/explore/cand_css_text_numbers.py`,
  `scripts/explore/cand_css_crossover_numbers.py` and `scripts/explore/cand_css_forest_numbers.py egfr`.
- Scenario benchmark (Extended Data Fig. 4a,b):
  `mkdir -p outputs/ed6_provenance_2026-10-14 && uv run python scripts/explore/ed6_provenance/rescore.py`.
- Fig. 1d P values: `uv run python scripts/explore/figpanels/fig1d_survival_test.py`.
- Other figure inputs: `REPRODUCTION_INDEX.md` names the script of each panel.

## 6. GENIE-BPC and PRIME data

The registry experiment (Figs 4 and 5) and the PRIME analyses (Fig. 6) use data that we cannot redistribute:

- AACR Project GENIE Biopharma Collaborative: NSCLC v2.0-public, CRC v2.0-public and BrCa v1.0-public, for
  registered users who accept the AACR Project GENIE terms of access (Synapse, syn27056172).
- PRIME trial (NCT00364013; dataset Colorec_Amgen_2006_264), from Project Data Sphere
  (https://www.projectdatasphere.org) after registration and acceptance of its data sharing agreement.

```bash
export GENIE_ROOT=/path/to/GenieBPC   # holds NSCLC_Genie/, CRC_GENIE/, BRCA_GENIE/
export PDS_ROOT=/path/to/PDS          # Project Data Sphere downloads, including PRIME
```

Some scripts read `~/GenieBPC` and `~/PDS` directly, so a copy or link there also works. The registry
experiment is scored by `scripts/score_four_arm_final_v1.py` (`--precheck-only`, then `--score`), following
`manuscript/SCORING_SPEC_2026-09-28.md`. The futures generated for registry patients are not deposited, because
each was conditioned on a registry record, so the registry comparisons cannot be reproduced exactly from the
public material. Without these data, a script that needs them stops at its first missing input with a
`FileNotFoundError`.

No patient-level GENIE-BPC or PRIME data are included in this release or in the data deposit. Aggregate counts
from both sources appear in `manuscript/supplementary_tables/supplementary_table_3_patients.md` and in the
scoring specifications.

## Acknowledgements and data terms

The authors would like to acknowledge the American Association for Cancer Research and its financial and
material support in the development of the AACR Project GENIE registry, as well as members of the consortium
for their commitment to data sharing. Interpretations are the responsibility of study authors. This
publication is based on research using information obtained from www.projectdatasphere.org, which is
maintained by Project Data Sphere. Neither Project Data Sphere nor the owner(s) of any information from the
web site have contributed to, approved or are in any way responsible for the contents of this publication.
