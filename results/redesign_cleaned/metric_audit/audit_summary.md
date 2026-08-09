# Metric audit machine summary

## conditioner
- **eps_f**: `0.19841410553977548`
- **eps_r**: `0.2024037892718708`
- **eps_f_equals_eps_r**: `False`
- **Sf_symmetric**: `True`
- **Sr_symmetric**: `True`
- **Sf_min_eig**: `0.19841410553976818`
- **Sr_min_eig**: `0.2024037892718703`
- **Sf_min_eig_ge_eps_f**: `True`
- **Sr_min_eig_ge_eps_r**: `True`
- **forecast_independent_of_target**: `True`
- **both_spd**: `True`

## stein_kl
- **stein_S_S**: `0.0`
- **stein_near_zero**: `True`
- **kl_S_S**: `-1.7763568394002505e-15`
- **kl_near_zero**: `True`
- **kl_direction**: `KL(N(0,S_true) || N(0,S_hat)) = 0.5*(tr(S_hat^{-1}S_true)-n+logdet(S_hat)-logdet(S_true))`
- **stein_formula**: `tr(S_hat^{-1}S_true) - logdet(S_hat^{-1}S_true) - n`
- **kl_inflated_hat**: `1.4486038537666568`
- **stein_inflated_hat**: `2.897207707533326`
- **evaluate_on**: `C_eta(hat), C_eta(real) with independent thresholds`

## frobenius_units
- **returns_scale**: `decimal daily returns (median |r|≈0.01)`
- **redesign_raw_frob_definition**: `sum_{ij} (A_ij-B_ij)^2 = ||A-B||_F^2 (SQUARED Frobenius)`
- **old_paper_frob_definition**: `np.linalg.norm(A-B, ord='fro') = ||A-B||_F (unsquared)`
- **example_squared**: `5.377430750234001e-05`
- **example_unsquared**: `0.007333096719827169`
- **example_sqrt_squared**: `0.007333096719827171`
- **match_sqrt**: `True`
- **typical_diag_entry**: `0.00022322645567972633`
- **typical_||S||_F**: `0.007944552608100773`
- **oos_panel_mean_raw_frob_is_squared**: `True`
- **scale_note**: `Old manuscript ~0.02–0.03 is unsquared Frobenius on similarly scaled covs; redesign panel ~10^2–10^3 is squared Frobenius. Rankings under monotone x->x^2 are identical for nonnegative values, but absolute levels are not comparable.`

## regret
- **example_R_raw**: `-0.00020153411941696737`
- **example_R_cond**: `2.6842412918642866e-06`
- **R_cond_nonneg_example**: `True`
- **R_raw_can_be_negative_in_MC**: `False`
- **neg_count_of_50**: `0`
- **naming**: `{'frozen_metric': 'raw_realized_excess_vs_conditioned_oracle (R_raw)', 'robustness_metric': 'conditioned_gmvp_variance_regret (R_cond >= 0)'}`
