"""Background GPU utilization monitor using nvidia-smi.

Polls SM utilization and memory usage every N seconds on a daemon thread.
Produces a structured JSON summary and an optional two-panel chart.

The key decision metric is sm_util_mean_active_pct: mean SM utilization
filtered to samples where the GPU was actually doing work (above
active_util_threshold_pct). This filters out CPU-bound idle periods between
epochs so the number reflects true GPU efficiency rather than pipeline
structure.

Upgrade decision logic:
  sm_util_mean_active_pct >= upgrade_sm_threshold_pct  → upgrade_recommended: true
  mem_peak_pct >= 85                                    → upgrade_recommended: true
  (either condition is sufficient)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import threading
import time
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class GpuMonitor:
    """1-second GPU sampler for training diagnostics.

    All public methods are safe to call regardless of whether the monitor
    is enabled — they silently no-op when disabled.

    Usage:
        monitor = GpuMonitor(enabled=True, ...)
        monitor.start()
        for fold_num, ... in enumerate(folds, 1):
            monitor.mark_fold(fold_num)
            ... train ...
        monitor.stop()
        monitor.save_summary("/path/to/gpu_utilization.json")
        monitor.save_chart("/path/to/gpu_utilization.png")
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        poll_interval: float = 1.0,
        active_util_threshold: int = 10,
        upgrade_sm_threshold: int = 70,
        save_chart_enabled: bool = True,
    ) -> None:
        """
        Args:
            enabled:               Master switch. If False, all methods
                                   are no-ops regardless of CUDA availability.
            poll_interval:         Seconds between nvidia-smi polls.
            active_util_threshold: SM utilization (%) above which a sample
                                   is counted as "active" for
                                   sm_util_mean_active_pct.
            upgrade_sm_threshold:  Active-mean SM utilization (%) at or above
                                   which upgrade_recommended is set True.
            save_chart_enabled:    If False, save_chart() skips PNG generation
                                   but still logs the verdict.
        """
        self._config_enabled      = enabled
        self._interval            = poll_interval
        self._active_threshold    = active_util_threshold
        self._upgrade_sm_thresh   = upgrade_sm_threshold
        self._save_chart_enabled  = save_chart_enabled

        # Each sample: (elapsed_s, sm_util_pct, mem_used_mib, mem_total_mib)
        self._samples: List[Tuple[float, int, int, int]] = []
        # Each mark:   (elapsed_s, fold_num)
        self._fold_marks: List[Tuple[float, int]] = []

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._t0: Optional[float] = None
        self._enabled = False          # set True only after successful start()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start background polling.

        No-op with a log message if disabled via config, CUDA is absent,
        or nvidia-smi is not found.
        """
        if not self._config_enabled:
            logger.info("GPU monitor disabled via config (GPU_MONITOR.enabled: false).")
            return
        if not self._check_available():
            return
        self._t0      = time.perf_counter()
        self._enabled = True
        self._thread  = threading.Thread(
            target=self._poll_loop, daemon=True, name="GpuMonitor"
        )
        self._thread.start()
        logger.info(
            "GPU monitor started — poll interval: %.1fs, "
            "active threshold: %d%%, upgrade threshold: %d%%.",
            self._interval,
            self._active_threshold,
            self._upgrade_sm_thresh,
        )

    def mark_fold(self, fold_num: int) -> None:
        """Record a fold boundary at the current wall-clock time."""
        if not self._enabled or self._t0 is None:
            return
        self._fold_marks.append((time.perf_counter() - self._t0, fold_num))

    def stop(self) -> None:
        """Signal the polling thread to stop and wait for it to exit."""
        if not self._enabled or self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        n = len(self._samples)
        dur = self._samples[-1][0] if self._samples else 0.0
        logger.info(
            "GPU monitor stopped — %d samples over %.0fs.", n, dur
        )

    def summary(self) -> Dict:
        """Return a structured dict of utilization metrics.

        Returns an empty dict if the monitor was never enabled or collected
        no samples. All percentage values are rounded to one decimal place.

        Keys:
            total_elapsed_s         Wall-clock seconds from start() to stop().
            n_samples               Total number of 1-second samples.
            n_active_samples        Samples with SM util >= active_util_threshold_pct.
            sm_util_mean_pct        Mean SM utilization across ALL samples.
            sm_util_mean_active_pct Mean SM utilization across active samples only.
                                    This is the primary decision metric.
            sm_util_peak_pct        Maximum observed SM utilization.
            mem_used_peak_mib       Maximum observed GPU memory used (MiB).
            mem_total_mib           Total GPU memory capacity (MiB).
            mem_peak_pct            Peak memory as % of total capacity.
            active_util_threshold_pct  The threshold used to define "active".
            upgrade_sm_threshold_pct   The threshold used for upgrade verdict.
            verdict                 "low" | "moderate" | "high"
            upgrade_recommended     True if GPU is the bottleneck.
            upgrade_reason          Human-readable explanation of the verdict.
        """
        if not self._enabled or not self._samples:
            return {}

        elapsed     = self._samples[-1][0]
        sm_all      = [s[1] for s in self._samples]
        sm_active   = [v for v in sm_all if v >= self._active_threshold]
        mem_used    = [s[2] for s in self._samples]
        mem_total   = self._samples[0][3]

        mean_all    = sum(sm_all)    / len(sm_all)
        mean_active = sum(sm_active) / len(sm_active) if sm_active else 0.0
        peak_sm     = max(sm_all)
        peak_mem    = max(mem_used)
        mem_pct     = peak_mem / max(mem_total, 1) * 100

        # Upgrade decision
        sm_bound  = mean_active >= self._upgrade_sm_thresh
        mem_bound = mem_pct     >= 85.0

        upgrade_recommended = sm_bound or mem_bound

        if mean_active >= self._upgrade_sm_thresh:
            verdict = "high"
        elif mean_active >= 50:
            verdict = "moderate"
        else:
            verdict = "low"

        if sm_bound and mem_bound:
            reason = (
                f"GPU is both compute-bound (active SM mean: {mean_active:.0f}% "
                f">= {self._upgrade_sm_thresh}%) and near VRAM capacity "
                f"(peak: {mem_pct:.0f}%). Upgrading GPU tier is strongly recommended."
            )
        elif sm_bound:
            reason = (
                f"GPU is compute-bound (active SM mean: {mean_active:.0f}% "
                f">= {self._upgrade_sm_thresh}%). A higher-tier GPU should "
                f"meaningfully reduce training time."
            )
        elif mem_bound:
            reason = (
                f"GPU memory is near capacity (peak: {mem_pct:.0f}% of "
                f"{mem_total} MiB). A GPU with more VRAM is needed before "
                f"a faster compute tier."
            )
        elif verdict == "moderate":
            reason = (
                f"GPU utilization is moderate (active SM mean: {mean_active:.0f}%). "
                f"A higher-tier GPU may provide some benefit but the bottleneck "
                f"is partially CPU-side (threshold tuning, calibration, metrics)."
            )
        else:
            reason = (
                f"GPU utilization is low (active SM mean: {mean_active:.0f}%). "
                f"The bottleneck is CPU-side work between epochs. Upgrading GPU "
                f"tier is unlikely to reduce training time."
            )

        return {
            "total_elapsed_s":           round(elapsed, 1),
            "n_samples":                 len(self._samples),
            "n_active_samples":          len(sm_active),
            "sm_util_mean_pct":          round(mean_all,    1),
            "sm_util_mean_active_pct":   round(mean_active, 1),
            "sm_util_peak_pct":          peak_sm,
            "mem_used_peak_mib":         peak_mem,
            "mem_total_mib":             mem_total,
            "mem_peak_pct":              round(mem_pct, 1),
            "active_util_threshold_pct": self._active_threshold,
            "upgrade_sm_threshold_pct":  self._upgrade_sm_thresh,
            "verdict":                   verdict,
            "upgrade_recommended":       upgrade_recommended,
            "upgrade_reason":            reason,
        }

    def save_summary(self, output_path: str) -> None:
        """Write the summary dict to a JSON file.

        Always written when monitoring is enabled and samples were collected,
        regardless of save_chart_enabled.
        """
        if not self._enabled:
            return
        s = self.summary()
        if not s:
            logger.warning("GPU monitor: no samples collected, skipping summary.")
            return
        try:
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(s, f, indent=2)
            logger.info(
                "GPU monitor summary saved to %s. "
                "upgrade_recommended=%s — %s",
                output_path,
                s["upgrade_recommended"],
                s["upgrade_reason"],
            )
        except Exception as exc:
            logger.warning("GPU monitor: failed to save summary: %s", exc)

    def save_chart(self, output_path: str) -> None:
        """Generate and save the two-panel utilization chart.

        Skipped if save_chart_enabled is False in config, but the verdict
        from summary() is still logged.

        Top panel:    SM utilization (%) with active-threshold and
                      upgrade-threshold reference lines.
        Bottom panel: GPU memory used (MiB) with total capacity and
                      85% warning lines.
        Vertical dashed lines mark fold boundaries.
        """
        if not self._enabled:
            return
        s = self.summary()
        if not s:
            logger.warning("GPU monitor: no samples to chart.")
            return

        # Always log the verdict even if chart is disabled
        logger.info(
            "GPU monitor verdict: upgrade_recommended=%s — %s",
            s["upgrade_recommended"],
            s["upgrade_reason"],
        )

        if not self._save_chart_enabled:
            logger.info(
                "GPU monitor chart disabled via config "
                "(GPU_MONITOR.save_chart: false)."
            )
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            times    = [sp[0] for sp in self._samples]
            sm_utils = [sp[1] for sp in self._samples]
            mem_used = [sp[2] for sp in self._samples]
            mem_total = s["mem_total_mib"]

            fig, (ax1, ax2) = plt.subplots(
                2, 1, figsize=(14, 7), sharex=True
            )
            fig.suptitle(
                f"GPU Utilization — "
                f"active SM mean: {s['sm_util_mean_active_pct']}%  "
                f"peak SM: {s['sm_util_peak_pct']}%  "
                f"peak mem: {s['mem_used_peak_mib']} / {mem_total} MiB  "
                f"upgrade_recommended: {s['upgrade_recommended']}",
                fontsize=11,
            )

            # Top panel: SM utilization
            ax1.plot(times, sm_utils, color="#2196F3", linewidth=0.8,
                     label="SM Utilization")
            ax1.axhline(
                self._active_threshold,
                color="gray", linestyle=":", linewidth=0.8, alpha=0.6,
                label=f"Active threshold ({self._active_threshold}%)",
            )
            ax1.axhline(
                self._upgrade_sm_thresh,
                color="red", linestyle="--", linewidth=0.9, alpha=0.8,
                label=f"Upgrade threshold ({self._upgrade_sm_thresh}%) "
                      f"— sustained above → GPU-bound",
            )
            ax1.set_ylabel("SM Utilization (%)")
            ax1.set_ylim(0, 108)
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(axis="y", alpha=0.2)

            # Bottom panel: memory
            ax2.plot(times, mem_used, color="#4CAF50", linewidth=0.8,
                     label="Memory Used (MiB)")
            ax2.axhline(
                mem_total, color="black", linestyle="-",
                linewidth=0.6, alpha=0.35,
                label=f"Total capacity ({mem_total} MiB)",
            )
            ax2.axhline(
                mem_total * 0.85, color="red", linestyle="--",
                linewidth=0.9, alpha=0.8,
                label=f"85% capacity ({int(mem_total * 0.85)} MiB)",
            )
            ax2.set_ylabel("GPU Memory (MiB)")
            ax2.set_xlabel("Elapsed Time (s)")
            ax2.set_ylim(0, mem_total * 1.08)
            ax2.legend(loc="upper right", fontsize=8)
            ax2.grid(axis="y", alpha=0.2)

            # Fold boundary lines
            for elapsed_s, fold_num in self._fold_marks:
                for ax in (ax1, ax2):
                    ax.axvline(elapsed_s, color="gray",
                               linestyle=":", linewidth=0.7, alpha=0.5)
                ax1.text(
                    elapsed_s + max(times[-1] * 0.003, 0.5), 101,
                    f"F{fold_num}", fontsize=7, color="gray", va="top",
                )

            plt.tight_layout()
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            plt.savefig(output_path, dpi=120, bbox_inches="tight")
            plt.close(fig)
            logger.info("GPU monitor chart saved to %s.", output_path)

        except Exception as exc:
            logger.warning("GPU monitor: chart generation failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_available(self) -> bool:
        """Return True if CUDA is present and nvidia-smi is callable."""
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("GPU monitor disabled: no CUDA device detected.")
                return False
        except ImportError:
            logger.info("GPU monitor disabled: torch not importable.")
            return False
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                logger.warning(
                    "GPU monitor disabled: nvidia-smi returned non-zero exit."
                )
                return False
            return True
        except FileNotFoundError:
            logger.warning("GPU monitor disabled: nvidia-smi not found on PATH.")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("GPU monitor disabled: nvidia-smi timed out.")
            return False
        except Exception as exc:
            logger.warning("GPU monitor disabled: %s", exc)
            return False

    def _poll_loop(self) -> None:
        """Daemon thread: poll nvidia-smi every _interval seconds."""
        while not self._stop_event.wait(timeout=self._interval):
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 3:
                        elapsed = time.perf_counter() - self._t0  # type: ignore[operator]
                        self._samples.append((
                            round(elapsed, 2),
                            int(parts[0].strip()),
                            int(parts[1].strip()),
                            int(parts[2].strip()),
                        ))
            except Exception:
                pass    # silently drop failed samples; never crash training
