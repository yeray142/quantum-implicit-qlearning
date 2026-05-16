#!/usr/bin/env python3
"""
Statistical verification script for Task 11 report.
Run: python scripts/analysis/verify_statistics.py

This script computes all statistical tests reported in the Task 11 report
and prints the results for verification.
"""

import json
import math
from scipy import stats
import numpy as np

# Data from W&B (classical n=8, quantum-multi-qubit n=8)

CLASSICAL_RETURNS = [3486.6, 3504.6, 3526.3, 3545.1, 3485.3, 3486.9, 3540.5, 3422.9]
CLASSICAL_P_NEG = [0.702, 0.709, 0.876, 0.477, 0.760, 0.417, 0.837, 0.677]

# From task11_report.tex - existing Fix-A family data
FIXED_C_RETURNS = [2468.3, 3520.7, 3474.8, 3519.8, 3591.5, 2996.1, 3611.3, 2589.4]
FIXED_C_P_NEG = [0.773, 0.878, 0.670, 0.693, 0.733, 0.679, 0.438, 0.687]

QUANTUM_MQR_RETURNS = [3554.9, 3582.2, 3412.6, 2213.4, 3566.5, 2375.1, 3626.9, 3318.6]
QUANTUM_MQR_P_NEG = [0.654, 0.588, 0.537, 0.776, 0.686, 0.601, 0.639, 0.695]

# From report - other configs for reference
QUANTUM_FIXED_RETURNS = [3565.1, 3517.0, 3495.0, 3516.7, 3298.9, 2630.0, 3038.7, 1610.7]
CONSTANT_V_RETURNS = [3013.5]  # Single value from report

def welch_ttest(x, y):
    """Welch's two-sample t-test"""
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    se = math.sqrt(vx/nx + vy/ny)
    t = (mx - my) / se
    df = ((vx/nx + vy/ny)**2) / ((vx/nx)**2/(nx-1) + (vy/ny)**2/(ny-1))
    p = 2 * stats.t.sf(abs(t), df)
    # Cohen's d
    pooled_std = math.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    d = (mx - my) / pooled_std
    return t, p, d

def fisher_exact(n1, n2):
    """Fisher's exact test for proportion comparison"""
    # 2x2 contingency table
    table = [[n1[0], n1[1]], [n2[0], n2[1]]]
    OR, p = stats.fisher_exact(table)
    return OR, p

def wilson_ci(k, n, z=1.96):
    """Wilson score 95% confidence interval for proportion"""
    p = k / n
    denom = 1 + z**2/n
    center = (p + z**2/(2*n)) / denom
    margin = z * math.sqrt((p*(1-p) + z**2/(4*n))/n) / denom
    return (center - margin, center + margin)

def cohen_d(x, y):
    nx, ny = len(x), len(y)
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = math.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    return (mx - my) / pooled_std

print("="*70)
print("STATISTICAL VERIFICATION FOR TASK 11 REPORT")
print("="*70)

# Classical n=8 stats
print("\n1. CLASSICAL (n=8) STATISTICS")
print("-"*40)
c_mean = np.mean(CLASSICAL_RETURNS)
c_std = np.std(CLASSICAL_RETURNS, ddof=1)
c_cv = c_std / c_mean * 100
print(f"  Returns: {CLASSICAL_RETURNS}")
print(f"  Mean: {c_mean:.1f} ± {c_std:.1f}")
print(f"  CV: {c_cv:.1f}%")
c_dir_ok = sum(1 for p in CLASSICAL_P_NEG if p > 0.5)
c_on_target = sum(1 for p in CLASSICAL_P_NEG if 0.6 <= p <= 0.8)
print(f"  Direction-correct (P>0.5): {c_dir_ok}/8")
print(f"  On-target (0.6-0.8): {c_on_target}/8")
wilson_c = wilson_ci(c_dir_ok, 8)
print(f"  Wilson 95% CI for direction-correct: [{wilson_c[0]:.3f}, {wilson_c[1]:.3f}]")

# Quantum Multi-Qubit Readout stats
print("\n2. QUANTUM MULTI-QUBIT READOUT (n=8) STATISTICS")
print("-"*40)
mqr_mean = np.mean(QUANTUM_MQR_RETURNS)
mqr_std = np.std(QUANTUM_MQR_RETURNS, ddof=1)
mqr_cv = mqr_std / mqr_mean * 100
print(f"  Returns: {QUANTUM_MQR_RETURNS}")
print(f"  Mean: {mqr_mean:.1f} ± {mqr_std:.1f}")
print(f"  CV: {mqr_cv:.1f}%")
mqr_dir_ok = sum(1 for p in QUANTUM_MQR_P_NEG if p > 0.5)
mqr_on_target = sum(1 for p in QUANTUM_MQR_P_NEG if 0.6 <= p <= 0.8)
print(f"  Direction-correct (P>0.5): {mqr_dir_ok}/8")
print(f"  On-target (0.6-0.8): {mqr_on_target}/8")
wilson_mqr = wilson_ci(mqr_dir_ok, 8)
print(f"  Wilson 95% CI for direction-correct: [{wilson_mqr[0]:.3f}, {wilson_mqr[1]:.3f}]")
wilson_mqr_ot = wilson_ci(mqr_on_target, 8)
print(f"  Wilson 95% CI for on-target: [{wilson_mqr_ot[0]:.3f}, {wilson_mqr_ot[1]:.3f}]")

# Comparison: Quantum MQR vs Classical
print("\n3. PAIRWISE COMPARISONS: Quantum MQR vs Classical")
print("-"*40)
t, p, d = welch_ttest(QUANTUM_MQR_RETURNS, CLASSICAL_RETURNS)
print(f"  Returns: t={t:.2f}, p={p:.3f}, d={d:.2f}")
if p < 0.05:
    print(f"  → SIGNIFICANT at p<0.05")
else:
    print(f"  → NOT significant at p<0.05")

# Calibration comparison
print(f"\n  Calibration comparison (Fisher exact):")
n1 = [mqr_dir_ok, 8 - mqr_dir_ok]
n2 = [c_dir_ok, 8 - c_dir_ok]
OR, p_fisher = fisher_exact(n1, n2)
print(f"    Quantum MQR: {mqr_dir_ok}/8, Classical: {c_dir_ok}/8")
print(f"    Odds ratio: {OR:.2f}, p={p_fisher:.3f}")

# On-target comparison
print(f"\n  On-target calibration comparison (Fisher exact):")
n1_ot = [mqr_on_target, 8 - mqr_on_target]
n2_ot = [c_on_target, 8 - c_on_target]
OR_ot, p_fisher_ot = fisher_exact(n1_ot, n2_ot)
print(f"    Quantum MQR: {mqr_on_target}/8, Classical: {c_on_target}/8")
print(f"    Odds ratio: {OR_ot:.2f}, p={p_fisher_ot:.3f}")

# New: Quantum MQR vs Fix-A family
print("\n4. NEW COMPARISON: Quantum MQR vs Fix-A Family")
print("-"*40)
print(f"  Fix-C returns: {FIXED_C_RETURNS}")
fc_mean = np.mean(FIXED_C_RETURNS)
fc_std = np.std(FIXED_C_RETURNS, ddof=1)
print(f"  Fix-C mean: {fc_mean:.1f} ± {fc_std:.1f}")
t, p, d = welch_ttest(QUANTUM_MQR_RETURNS, FIXED_C_RETURNS)
print(f"  Quantum MQR vs Fix-C: t={t:.2f}, p={p:.3f}, d={d:.2f}")
if p < 0.05:
    print(f"  → SIGNIFICANT at p<0.05")
else:
    print(f"  → NOT significant at p<0.05")

# Fix-A pool (Fixed + Fixed-Warmup + Fixed-C) for reference
FIXED_RETURNS = [3565.1, 3517.0, 3495.0, 3516.7, 3298.9, 2630.0, 3038.7, 1610.7]
FIXED_P_NEG = [0.121, 0.610, 0.431, 0.735, 0.588, 0.682, 0.689, 0.610]

FIXED_WARMUP_RETURNS = [3466.1, 3503.4, 3566.8, 3563.5, 2226.4, 1921.0, 3539.4, 3427.7]
FIXED_WARMUP_P_NEG = [0.829, 0.753, 0.364, 0.683, 0.646, 0.857, 0.621, 0.671]

FIX_A_POOL_RETURNS = FIXED_C_RETURNS + FIXED_RETURNS + FIXED_WARMUP_RETURNS
print(f"\n  Fix-A pool (n=24) mean: {np.mean(FIX_A_POOL_RETURNS):.1f}")

t, p, d = welch_ttest(QUANTUM_MQR_RETURNS, FIX_A_POOL_RETURNS)
print(f"  Quantum MQR vs Fix-A pool: t={t:.2f}, p={p:.3f}, d={d:.2f}")

print("\n4b. NEW COMPARISON: Classical (n=8) vs Fix-A Pool (n=24)")
print("-"*40)
t, p, d = welch_ttest(CLASSICAL_RETURNS, FIX_A_POOL_RETURNS)
print(f"  Classical vs Fix-A pool: t={t:.2f}, p={p:.3f}, d={d:.2f}")
if p < 0.05:
    print(f"  → SIGNIFICANT at p<0.05 (classical is better)")
else:
    print(f"  → NOT significant at p<0.05")

print("\n5. CALIBRATION COMPARISONS (Fisher Exact)")
print("-"*40)

# Quantum MQR vs No-Warmup
mqr_dir = 8
no_warmup_dir = 2
n1 = [mqr_dir, 8 - mqr_dir]
n2 = [no_warmup_dir, 8 - no_warmup_dir]
OR, p_f = fisher_exact(n1, n2)
print(f"  Quantum MQR (8/8) vs No-Warmup (2/8): OR={OR:.2f}, p={p_f:.3f}")
if p_f < 0.05:
    print(f"    → SIGNIFICANT at p<0.05")

# Classical vs Fix-A pool
c_dir = 6
fixa_dir = 20
n1 = [c_dir, 8 - c_dir]
n2 = [fixa_dir, 24 - fixa_dir]
OR, p_f = fisher_exact(n1, n2)
print(f"  Classical (6/8) vs Fix-A pool (20/24): OR={OR:.2f}, p={p_f:.3f}")

print("\n5. SUMMARY TABLE")
print("-"*40)
print(f"  {'Config':<30} {'Mean':>10} {'Std':>10} {'CV':>8} {'DirOK':>8} {'OnTgt':>8}")
print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*8} {'-'*8} {'-'*8}")
print(f"  {'Classical (n=8)':<30} {c_mean:>10.1f} {c_std:>10.1f} {c_cv:>7.1f}% {c_dir_ok:>8}/8 {c_on_target:>8}/8")
print(f"  {'Quantum MQR (n=8)':<30} {mqr_mean:>10.1f} {mqr_std:>10.1f} {mqr_cv:>7.1f}% {mqr_dir_ok:>8}/8 {mqr_on_target:>8}/8")
print(f"  {'Fix-C (n=8)':<30} {fc_mean:>10.1f} {fc_std:>10.1f} {fc_std/fc_mean*100:>7.1f}% {sum(1 for p in FIXED_C_P_NEG if p>0.5):>8}/8 {sum(1 for p in FIXED_C_P_NEG if 0.6<=p<=0.8):>8}/8")
print(f"  {'Fix-A pool (n=24)':<30} {np.mean(FIX_A_POOL_RETURNS):>10.1f} {np.std(FIX_A_POOL_RETURNS, ddof=1):>10.1f} {np.std(FIX_A_POOL_RETURNS,ddof=1)/np.mean(FIX_A_POOL_RETURNS)*100:>7.1f}% {sum(1 for p in FIXED_C_P_NEG+FIXED_P_NEG+FIXED_WARMUP_P_NEG if p>0.5):>8}/24 {sum(1 for p in FIXED_C_P_NEG+FIXED_P_NEG+FIXED_WARMUP_P_NEG if 0.6<=p<=0.8):>8}/24")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)
print("\nTo regenerate this output, run:")
print("  python scripts/analysis/verify_statistics.py")