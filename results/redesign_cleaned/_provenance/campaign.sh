#!/bin/bash
set -e
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
PY=$HOME/m279iso/bin/python
cd $HOME/rerun_cleaned
echo "=== CAMPAIGN START $(date) ==="
echo "--- stage A: pca8 ---";        $PY scripts/redesign/run_stage_a.py --embedder pca_only --pca-k 8 --tag stage_a_pca8
echo "--- stage A: market_state ---"; $PY scripts/redesign/run_stage_a.py --embedder market_state --tag stage_a_market_state
echo "--- stage A: spectral ---";     $PY scripts/redesign/run_stage_a.py --embedder spectral --tag stage_a_spectral
echo "--- stage A: hybrid ---";       $PY scripts/redesign/run_stage_a.py --embedder hybrid --pca-k 5 --tag stage_a_hybrid
echo "--- collapse decomposition ---"; $PY scripts/redesign/run_collapse_decomp.py
echo "--- stage C regime diag ---";    $PY scripts/redesign/run_stage_c_regime_diag.py
echo "--- stage D ---";                $PY scripts/redesign/run_stage_d.py
echo "--- incremental bootstrap ---";  $PY scripts/redesign/run_incremental_bootstrap.py
echo "--- FREEZE DECISION (pre-test) ---"; $PY scripts/redesign/freeze_decide.py
echo "--- FROZEN OOS TEST ---";        $PY scripts/redesign/run_frozen_oos.py
echo "--- metric audit ---";           $PY scripts/redesign/run_metric_audit.py
echo "--- oos robustness ---";         $PY scripts/redesign/run_oos_robustness.py
echo "=== CAMPAIGN DONE $(date) ==="
