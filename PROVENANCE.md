# Provenance

Models, dates, code commits and checksums of the generated data. Commits refer to the development repository; this release is a snapshot of commit `80cfc8f`.

## Generated runs (deposited on Zenodo)

| run | role | models | generation commit | date | prompt template | template SHA-256 | accepted | rejected |
|---|---|---|---|---|---|---|---|---|
| `pillar1_anchored_atlas_tranche1_v4_n5_live01` | atlas | qwen/qwen3.6-flash, google/gemini-3.1-flash-lite | `5df4cdd22f29` | 2026-06-03 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 1480 | 52 |
| `pillar1_anchored_atlas_gemma4_n5_live01` | atlas | google/gemma-4-31b-it | `5383d6e47886` | 2026-06-06 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 740 | 57 |
| `glm52_full_atlas_n5` | atlas | GLM-5.2-FP8 | `cc46cb59a72b` | 2026-07-25 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 737 | 43 |
| `kimi_k3_off_full_atlas_n5_complete` | atlas | moonshotai/kimi-k3 | `d2953ff1532e` | 2026-07-27 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 737 | 39 |
| `sonnet5_mini_atlas_n100_2026_07_08` | atlas | anthropic/claude-sonnet-5 | `439950937899` | 2026-07-08 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 100 | 3 |
| `sonnet5_full_atlas_topup_remaining128_n5_2026_07_09` | atlas | anthropic/claude-sonnet-5 | `f333f7b4e917` | 2026-07-09 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 640 | 90 |
| `interventional_phase1_sonnet_full_v1c` | EGFR experiment | anthropic/claude-sonnet-5 | `1f029d00e591` | 2026-07-23 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 360 | 0 |
| `interventional_phase1_glm52_full_v1c` | EGFR check | GLM-5.2-FP8 | `6dd5709db446` | 2026-07-26 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 341 | 19 |
| `interventional_phase1_kimi_k3_off_full_v1c` | EGFR check | see `config.yaml` of the run | not recorded | | | | | |
| `interventional_phase1_kimi_k3_off_topup75_v1c_retry2` | EGFR check | moonshotai/kimi-k3 | `04a478458d93` | 2026-07-28 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 75 | 0 |
| `sentinel_usecase_braf_melanoma_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `7eac14b6330c` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 48 | 0 |
| `sentinel_usecase_ccrcc_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `7eac14b6330c` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 48 | 0 |
| `sentinel_usecase_crc_flip_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `619706b1d7c3` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 96 | 0 |
| `sentinel_usecase_egfr_flip_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `619706b1d7c3` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 72 | 0 |
| `sentinel_usecase_egfr_post_osi_explicit_resistance_v5_v2_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `7eac14b6330c` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 36 | 0 |
| `sentinel_usecase_egfr_post_osi_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `619706b1d7c3` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 36 | 0 |
| `sentinel_usecase_her2_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `7eac14b6330c` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 96 | 0 |
| `sentinel_usecase_hgsoc_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `7eac14b6330c` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 96 | 0 |
| `sentinel_usecase_pancreas_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `7eac14b6330c` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 96 | 0 |
| `sentinel_usecase_pdl1_v5_2026_07_13` | scenario benchmark | google/gemini-3.1-flash-lite, qwen/qwen3.6-flash, google/gemma-4-31b-it, anthropic/claude-sonnet-5 | `7eac14b6330c` | 2026-07-13 | `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f9` | 48 | 0 |

Available run manifests contain the full hashes. Each run folder also holds its `config.yaml`, the rendered prompt of every attempt and the rejected attempts.

## Registry experiment (GENIE-BPC data; not deposited)

| run | model | generation commit |
|---|---|---|
| `four_arm_v1_definitive_p40b20_20260826` | deepseek/deepseek-v4-flash-0731 | `28c12e43c4e1` |

Code: `scripts/explore/build_query_universe_v1.py`, `emit_four_arm_prompts_v1.py`, `run_four_arm_v1.py` and `scripts/score_four_arm_final_v1.py`.

## Prompts and specifications

| file | SHA-256 |
|---|---|
| `docs/interventional_run_spec_v1.md` | `6aeb37659c89a438abb3bc5961602cc6180f57aa9a5ed41c32e27d5fa3e28221` |
| `manuscript/specifications/2026-10-28_N6_N8_N11_specification_excerpts.md` | `cd4437b7c17d252b02e9f4285704d85dabdb8f182784190ee5a45c21a9dfaf41` |
| `prompts/anchored_generation_v1/lean_ledger_v2_rich_neutralized_anchored.md` | `f0750f24ccb4cbf17558425e201310e95e12bfe7c6eafdbad2477ba108a19d94` |
| `prompts/anchored_generation_v1/lean_ledger_v2_rich_neutralized_continuation.md` | `e150ea142d4219b3e9cb5d5e414521081c35882e030d62180805a339800a32c5` |
| `prompts/oncollm_v1/experimental_lean_ledger_v2_rich_neutralized.md` | `3109486f7e2f96f94c61685bd000a15fdef52e6759b13b11517af15407d0bac6` |
| `scripts/explore/sentinel_battery_manifest_reconstructed_2026-10-14.json` | `c53d968ccd5f3be5cf9f269cf10f7b35df2232c0e31978a194a00e64f923a114` |
| `scripts/explore/sentinel_benchmark_spec_v1.yaml` | `849b50cf1bfaaab850a67692a7cb9f101574c56a17831c8e882126079aa434a8` |

## Code hashes recorded in the outputs

Outputs record the SHA-256 of the code that wrote them. Where the released file differs, it differs in comments only.

| file | recorded SHA-256 | released SHA-256 |
|---|---|---|
| `scripts/build_paper_state_embedding_bakeoff.py` | `283844f42ec86ab4` | `283844f42ec86ab4` |
| `scripts/build_paper_state_feature_ledger.py` | `cdd4340c27e765cc` | `cdd4340c27e765cc` |
| `scripts/build_paper_state_map.py` | `5b0fd465e5894ed9` | `5b0fd465e5894ed9` |
| `scripts/build_perturbation_matrix_decision_aligned.py` | `ff7d969013edf78a` | `d826b30434e85d3f` |
| `scripts/explore/audit_native_atlas_geometry_v1.py` | `20d8c8e69d056b22` | `20d8c8e69d056b22` |
| `scripts/explore/n11_event_conditioned_fields_v1.py` | `12dd9b7952afe602` | `eb7835deb1130c1d` |
| `scripts/explore/n11_phase_support_v1.py` | `954fce5912d58ea0` | `914335c31d79d618` |
| `scripts/explore/trajectory_representation_v1.py` | `e1b1a30a552a5c7b` | `e1b1a30a552a5c7b` |
| `src/synthonccases/oncollm/replay.py` | `75fbd26cd33a535b` | `75fbd26cd33a535b` |
