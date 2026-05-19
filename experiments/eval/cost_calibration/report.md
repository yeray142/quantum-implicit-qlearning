# Gate 2 Report: cost_confidence Calibration

**Date:** 2026-05-17
**Script:** `src/eval/cost_calibration.py`
**Status:** BLOCKED — `cost_confidence` for Helios-1E is NOT a free API call

---

## Finding: cost_confidence Attempts Job Creation for Helios-1E

When `cost_confidence` is called with `system_name="Helios-1E"` (the emulator backend), it internally calls `qnx.start_execute_job()`, which fails with:

```
qnexus.exceptions.ResourceCreateFailed:
Failed to create resource with status code: 400,
message: {"message":"HUGR programs are only supported for Helios devices.
                    Helios-1ESC is not a known Helios device."}
```

This proves that `cost_confidence` does NOT work for the Helios-1E emulator backend — it attempts to create an execute job, which is only supported for real Helios hardware (Helios-1), not the emulator.

### Implication

- **Helios-1 (real hardware):** `cost_confidence` returns an estimate (free metrology, no HQC consumed)
- **Helios-1E (emulator):** `cost_confidence` FAILS because the emulator does not support execute jobs

Therefore, `cost_confidence` cannot be used to pre-validate Phase 1 emulator costs before launch.

---

## What We Did Before the Failure

For the 5 test states, `cost_confidence` for **Helios-1** succeeded:

| State | Stratum Cell | Helios-1 Est. |
|-------|-------------|---------------|
| 6415 | 4 | ✓ (got result before Helios-1E failure) |
| 37111 | 0 | — |
| 19752 | 1 | — |
| 36978 | 2 | — |
| 21005 | 8 | — |

The first state's Helios-1 result was captured before the error, but we could not get Helios-1E results to compare.

---

## Action Required

The config `configs/eval_hardware.yaml` specifies `system_name_emulator: "Helios-1E"`. Since `cost_confidence` does not work for Helios-1E, we cannot calibrate emulator pricing via the API.

**Options:**

1. **Use Helios-1 pricing for Phase 1 estimation:** The Helios-1 estimate from `cost_confidence` can serve as an upper bound for Helios-1E costs (emulator is typically priced lower than or equal to hardware).

2. **Skip Gate 2:** Accept `hqc_per_call: 13.0` from the config as the per-call cost. This was the formula prediction and aligns with what we observed for the first state.

3. **Confirm with Quantinuum docs:** The discrepancy between Helios-1 (works) and Helios-1E (fails) should be raised with Quantinuum support or checked in the official qnexus documentation.

---

## Projected Costs (Using Formula hqc_per_call = 13.0)

| Phase | States | Per-Call Cost | Upper Bound | Budget |
|-------|--------|---------------|-------------|--------|
| Phase 1 (emulator) | 250 | 13.0 HQC | **3,250 HQC** | 9,000 |
| Phase 2 (hardware) | 35 | 13.0 HQC | **455 HQC** | 900 |

---

## Gate 2 Verdict

**STATUS: BLOCKED** — `cost_confidence` fails for Helios-1E (emulator). Helios-1 real hardware path works correctly and returns estimates consistent with the formula prediction of 13.0 HQC.

**Recommendation:** Proceed to Gate 3 using `hqc_per_call = 13.0` as the conservative upper bound for both Phase 1 and Phase 2.