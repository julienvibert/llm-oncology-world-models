# Source Data index (generated; do not edit)

From `scripts/release/build_source_data.py` at commit 33131c5. Existing tables only, no recomputation. Classes: included (synthetic aggregate, in the package); included (registry/trial aggregate): plotted values or cohort-level tables of GENIE-BPC or PRIME patients, cells showing or revealing counts of 1 to 4 masked (scripts/release/mask_small_cells.py); deposit (synthetic runs or patient-level synthetic tables, in the data deposit); plotted values in the panel table (the panel's own plotted-value table is included instead of its GENIE-BPC or PRIME source table); not needed (metadata or readout); not used (listed in panel metadata, not read by the panel script); not included (patient-level registry or trial data; an aggregate table must be derived for release); no data (schematic, text or code).

Distinct sources by class: deposit 3, included 19, included (registry/trial aggregate) 24, no data 9, not included 8, not needed 4, not used 2, plotted values in the panel table 11

| Figure | Panel | Panel id | Source | Class | Note |
|---|---|---|---|---|---|
| Extended Data Figure 5 | a | ed07b_gap_by_horizon | `outputs/figures/panels/ed07b_gap_by_horizon/ed07b_gap_by_horizon_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 5 | b | f4g_calibration | `outputs/figures/panels/f4g_calibration/f4g_calibration_aggregates.csv` | included (registry/trial aggregate) | plotted values; 2 cells masked |
| Extended Data Figure 5 | c | f4e_fan_coverage | `outputs/figures/panels/f4e_fan_coverage/f4e_fan_coverage_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 5 | d | ed07e_donor_agreement | `outputs/figures/panels/ed07e_donor_agreement/ed07e_donor_agreement_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 5 | e | f4f_c_index | `outputs/figures/panels/f4f_c_index/f4f_c_index_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 5 | f | f4d_bias_deaths | `outputs/figures/panels/f4d_bias_deaths/f4d_bias_deaths_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 6 | all | cand_final_genie_crc | `outputs/figures/panels/cand_final_genie_crc/cand_final_genie_crc_ribbon_shown.csv` | included (registry/trial aggregate) | plotted values; 8 cells masked |
| Extended Data Figure 7 | a | f5c_course_disagreement | `outputs/figures/panels/f5c_course_disagreement/f5c_course_disagreement_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 7 | b | f5d_course_contrasts | `outputs/figures/panels/f5d_course_contrasts/f5d_course_contrasts_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 7 | c | f4i_treatment | `outputs/figures/panels/f4i_treatment/f4i_treatment_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 7 | d | ed08c_sensitivity | `outputs/figures/panels/ed08c_sensitivity/ed08c_sensitivity_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 7 | e | ed08d_regimen_timing | `outputs/figures/panels/ed08d_regimen_timing/ed08d_regimen_timing_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Extended Data Figure 9 | all | cand_final_prime_mut | `outputs/figures/panels/cand_final_prime_mut/cand_final_prime_mut_ribbon_shown.csv` | included (registry/trial aggregate) | plotted values; 15 cells masked |
| Figure 4 | c | f4b_early_deaths | `outputs/figures/panels/f4b_early_deaths/f4b_early_deaths_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Figure 4 | d | f4c_survival_12m | `outputs/figures/panels/f4c_survival_12m/f4c_survival_12m_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Figure 5 | all | prop_fig5_genie | `outputs/figures/panels/prop_fig5_genie/prop_fig5_genie_ribbon_shown.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Figure 5 | d | f4h_joint_score | `outputs/figures/panels/f4h_joint_score/f4h_joint_score_aggregates.csv` | included (registry/trial aggregate) | plotted values; no small cell |
| Figure 6 | all | prop_fig6_prime | `outputs/figures/panels/prop_fig6_prime/prop_fig6_prime_ribbon_shown.csv` | included (registry/trial aggregate) | plotted values; 3 cells masked |
| Figure 1 | a | f1b_prism | `schematic` | no data |  |
| Figure 1 | b | f2a_modal_share | `outputs/exploratory_figure_panels/fig1/data/modal_share.csv` | included |  |
| Figure 1 | c | f1a_patient_timeline | `outputs/exploratory_figure_panels/fig1/data/exemplar.json` | included |  |
| Figure 1 | c | f1a_patient_timeline | `outputs/paper_state_map/state_projection.csv.gz` | deposit |  |
| Figure 1 | d | f2c_survival_by_family | `outputs/exploratory_figure_panels/fig1/data/survival_by_family.csv` | included |  |
| Figure 1 | d | f2c_survival_by_family | `outputs/exploratory_figure_panels/fig1/data/survival_by_family_test.csv` | included |  |
| Figure 2 | a | f1b_patient_state_map | `outputs/paper_state_map/state_projection.csv.gz` | deposit |  |
| Figure 2 | a | f1b_patient_state_map | `outputs/exploratory_figure_panels/fig1/data/exemplar.json` | included |  |
| Figure 2 | b | f1d_whole_course | `schematic` | no data |  |
| Figure 2 | c | fig2_families_km | `outputs/paper_state_map/state_projection.csv.gz` | deposit |  |
| Figure 2 | c | fig2_families_km | `atlas` | no data |  |
| Figure 3 | all | cand_fig3_four | `outputs/paper_state_map` | deposit |  |
| Figure 3 | e | f3d_wrong_target | `outputs/exploratory_interventional_phase1/phase1_context_arm_medians.csv` | included |  |
| Figure 3 | e | f3d_wrong_target | `outputs/exploratory_interventional_phase1_kimi_k3_combined/phase1_context_arm_medians.csv` | included |  |
| Figure 3 | e | f3d_wrong_target | `outputs/exploratory_interventional_phase1_glm52_full/phase1_context_arm_medians.csv` | included |  |
| Figure 3 | e | f3d_wrong_target | `outputs/paper_interventional_trust_map/egfr_gate_scores_used.csv` | included |  |
| Figure 3 | e | f3d_wrong_target | `outputs/paper_interventional_trust_map/readout.json` | included |  |
| Figure 3 | e | f3d_wrong_target | `scripts/explore/score_interventional_phase1.py` | no data |  |
| Figure 3 | f | n3_egfr_effect_modification | `outputs/exploratory_interventional_phase1/phase1_endpoint_long.csv` | included |  |
| Figure 3 | f | n3_egfr_effect_modification | `outputs/exploratory_interventional_phase1_kimi_k3_combined/phase1_endpoint_long.csv` | included |  |
| Figure 3 | f | n3_egfr_effect_modification | `outputs/exploratory_interventional_phase1_glm52_full/phase1_endpoint_long.csv` | included |  |
| Figure 3 | f | n3_egfr_effect_modification | `scripts/figures/panels/_egfr_days.py` | no data |  |
| Figure 4 | a | f4a_design | `manuscript/DRAFT.md` | no data |  |
| Figure 4 | a | f4a_design | `outputs/four_arm_preprint_20260928/population_composition_scores_20261012.csv` | included (registry/trial aggregate) | 4 cells masked |
| Figure 4 | b | ed07a_km_at_risk | `outputs/exploratory_four_arm_deep/generated_side.csv` | not included |  |
| Figure 4 | b | ed07a_km_at_risk | `outputs/exploratory_four_arm_deep/real_side.csv` | not included |  |
| Figure 4 | b | ed07a_km_at_risk | `outputs/exploratory_query_universe/queries.jsonl` | not included |  |
| Figure 4 | b | ed07a_km_at_risk | `GENIE-BPC` | not included |  |
| Figure 4 | d | f4c_survival_12m | `outputs/four_arm_preprint_20260928/survival_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Figure 4 | c | f4b_early_deaths | `outputs/exploratory_four_arm_deep/real_side.csv` | not included |  |
| Figure 4 | c | f4b_early_deaths | `outputs/exploratory_four_arm_deep/generated_side.csv` | not included |  |
| Figure 5 | a | n2_state_occupancy_main | `outputs/exploratory_n2_state_occupancy/n2_state_occupancy_v1.csv` | included (registry/trial aggregate) | 72 cells masked |
| Figure 5 | a | n2_state_occupancy_main | `outputs/exploratory_n2_state_occupancy/manifest.json` | not needed | metadata or readout; not needed to plot |
| Figure 5 | all | prop_fig5_genie | `outputs/exploratory_cand_css_four_arm/series_cal.csv.gz` | not included |  |
| Figure 5 | all | prop_fig5_genie | `outputs/exploratory_cand_css_four_arm/state_cal.csv.gz` | not included |  |
| Figure 5 | d | f4h_joint_score | `outputs/four_arm_preprint_20260928/joint_scores_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Figure 5 | d | f4h_joint_score | `outputs/four_arm_preprint_20260928/population_composition_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Figure 6 | a | n8_prime_time_in_state | `outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv` | included (registry/trial aggregate) | no small cell |
| Figure 6 | a | n8_prime_time_in_state | `outputs/exploratory_n8_prime_time_in_state/manifest.json` | not needed | metadata or readout; not needed to plot |
| Figure 6 | all | prop_fig6_prime | `outputs/exploratory_pds_prime_trial_geometry/state_snapshots.csv.gz` | not included |  |
| Figure 6 | all | prop_fig6_prime | `outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Figure 6 | d | n6_support_ladder | `outputs/exploratory_n6_support_ladder/n6_support_ladder_v1.csv` | included (registry/trial aggregate) | 5 cells masked |
| Figure 6 | d | n6_support_ladder | `outputs/exploratory_n6_support_ladder/manifest.json` | not needed | metadata or readout; not needed to plot |
| Extended Data Figure 1 | a | edc_atlas_phenotype | `outputs/exploratory_atlas_phenotype_six_model/phenotype_by_tissue_model.csv` | included |  |
| Extended Data Figure 1 | b | ed02d_specificity | `outputs/exploratory_native_atlas_geometry/readout.json` | included |  |
| Extended Data Figure 1 | c | ed02e_blocks | `outputs/exploratory_native_atlas_geometry/readout.json` | included |  |
| Extended Data Figure 2 | a | f1a_history | `schematic` | no data |  |
| Extended Data Figure 2 | b | ed03c_prior_progression | `outputs/exploratory_fig1_same_dot/readout_v2.json` | included | synthetic part only (per_model), as same_dot_per_model.csv |
| Extended Data Figure 2 | b | ed03c_prior_progression | `scripts/explore/build_fig1_same_dot_v2.py` | no data |  |
| Extended Data Figure 2 | c | f1b_world_model | `schematic` | no data |  |
| Extended Data Figure 3 | all | cand_final_atlas | `outputs/paper_state_map/state_projection.csv.gz` | deposit |  |
| Extended Data Figure 4 | a | ed04a_scenario_matrix | `outputs/exploratory_sentinel_benchmark/sentinel_trust_map_table.csv` | included |  |
| Extended Data Figure 4 | a | ed04a_scenario_matrix | `scripts/build_paper_interventional_trust_map.py` | no data |  |
| Extended Data Figure 4 | b | ed04b_scenario_counts | `outputs/exploratory_sentinel_benchmark/sentinel_trust_map_table.csv` | included |  |
| Extended Data Figure 4 | c | si04a_input_matrix | `outputs/perturbation_matrix_decision_aligned/readout.json` | included |  |
| Extended Data Figure 4 | c | si04a_input_matrix | `scripts/build_perturbation_matrix_decision_aligned.py` | no data |  |
| Extended Data Figure 4 | d | f3c_forest | `runs/interventional_phase1_sonnet_full_v1c` | deposit |  |
| Extended Data Figure 4 | d | f3c_forest | `outputs/paper_state_map/state_map_contract.json` | included |  |
| Extended Data Figure 4 | d | f3c_forest | `outputs/paper_survival_in_state_space/readout.json` | included |  |
| Extended Data Figure 4 | d | f3c_forest | `scripts/build_paper_survival_in_state_space.py` | no data |  |
| Extended Data Figure 4 | e | f3d_occupancy_map | `runs/interventional_phase1_sonnet_full_v1c` | deposit |  |
| Extended Data Figure 4 | e | f3d_occupancy_map | `outputs/paper_state_map/state_map_contract.json` | included |  |
| Extended Data Figure 4 | e | f3d_occupancy_map | `outputs/paper_survival_in_state_space/readout.json` | included |  |
| Extended Data Figure 4 | e | f3d_occupancy_map | `scripts/build_paper_survival_in_state_space.py` | no data |  |
| Extended Data Figure 5 | a | ed07b_gap_by_horizon | `outputs/four_arm_preprint_20260928/survival_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 5 | b | f4g_calibration | `outputs/four_arm_preprint_20260928/survival_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 5 | c | f4e_fan_coverage | `outputs/exploratory_four_arm_deep/deep_results.json` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 5 | c | f4e_fan_coverage | `outputs/exploratory_four_arm_deep/FINDINGS.md` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 5 | d | ed07e_donor_agreement | `outputs/four_arm_preprint_20260928/donor_diagnostics_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 5 | e | f4f_c_index | `outputs/four_arm_preprint_20260928/survival_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 5 | f | f4d_bias_deaths | `outputs/exploratory_four_arm_deep/deep_results.json` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 6 | all | cand_final_genie_crc | `outputs/exploratory_cand_css_four_arm/series_cal.csv.gz` | not included |  |
| Extended Data Figure 6 | all | cand_final_genie_crc | `outputs/exploratory_cand_css_four_arm/state_cal.csv.gz` | not included |  |
| Extended Data Figure 7 | a | f5c_course_disagreement | `outputs/four_arm_preprint_20260928/course_diagnostics_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 7 | a | f5c_course_disagreement | `outputs/four_arm_preprint_20260928/population_composition_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 7 | b | f5d_course_contrasts | `outputs/four_arm_preprint_20260928/course_diagnostics_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 7 | c | f4i_treatment | `outputs/four_arm_preprint_20260928/treatment_scalars_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 7 | d | ed08c_sensitivity | `outputs/four_arm_preprint_20260928/sensitivity_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 7 | d | ed08c_sensitivity | `outputs/four_arm_preprint_20260928/population_composition_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 7 | e | ed08d_regimen_timing | `outputs/four_arm_preprint_20260928/timing_scores_20261012.csv` | plotted values in the panel table | the panel's plotted-value table is included |
| Extended Data Figure 7 | f | n2_state_occupancy_all_arms | `outputs/exploratory_n2_state_occupancy/n2_state_occupancy_v1.csv` | included (registry/trial aggregate) | 72 cells masked |
| Extended Data Figure 7 | f | n2_state_occupancy_all_arms | `outputs/exploratory_n2_state_occupancy/manifest.json` | not needed | metadata or readout; not needed to plot |
| Extended Data Figure 8 | a | f6a_prime_rmst | `outputs/exploratory_pds_prime_trial_geometry/subgroup_rmst24.csv` | included (registry/trial aggregate) | no small cell |
| Extended Data Figure 8 | b | ed09a_forest_hr | `outputs/exploratory_prime_corrections_2026_10_12/prime_forest_hr_itt935.csv` | included (registry/trial aggregate) | 2 cells masked |
| Extended Data Figure 8 | b | ed09a_forest_hr | `outputs/exploratory_prime_corrections_2026_10_12/manifest.json` | not needed | metadata or readout; not needed to plot |
| Extended Data Figure 8 | c | n12a_prime_km | `PRIME` | not included |  |
| Extended Data Figure 8 | c | n12a_prime_km | `outputs/exploratory_pds_prime_trial_geometry/subgroup_rmst24.csv` | included (registry/trial aggregate) | see above |
| Extended Data Figure 8 | c | n12a_prime_km | `outputs/exploratory_n12_prime_decomposition/n12_components_v1.csv` | not used | listed in the panel metadata, not read by the panel script |
| Extended Data Figure 8 | c | n12a_prime_km | `outputs/exploratory_n12_prime_decomposition/n12_fine_cells_v1.csv` | not used | listed in the panel metadata, not read by the panel script |
| Extended Data Figure 9 | all | cand_final_prime_mut | `outputs/exploratory_pds_prime_trial_geometry/state_snapshots.csv.gz` | not included |  |
| Extended Data Figure 9 | all | cand_final_prime_mut | `outputs/exploratory_n8_prime_time_in_state/n8_time_in_state_v1.csv` | plotted values in the panel table | the panel's plotted-value table is included |
