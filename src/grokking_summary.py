from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

from grokking_experiment import CurveRow, GradRow


@dataclass(frozen=True)
class SummaryRow:
    policy: str
    eval_policy: str
    base_policy: str
    sketch_dim: int | None
    fixed_sketch: bool
    resample_sketch: bool
    correction_period: int | None
    architecture: str
    n_seeds: int
    p: int
    train_fraction: float
    lr: float
    weight_decay: float
    final_train_acc_mean: float
    final_test_acc_mean: float
    final_train_acc_std: float
    final_test_acc_std: float
    fit_rate: float
    grok_rate: float
    t_mem_median: float
    t_grok_median: float
    grok_delay_median: float
    censored_fraction: float
    grad_cos_final_mean: float
    grad_relerr_final_mean: float
    update_snr_final_mean: float
    innovation_prob_kl_final_mean: float
    uniform_attn_drop_mean: float
    fourier_energy_final_mean: float
    outcome_class: str


def _classify_outcome(grok_rate: float, fit_rate: float) -> str:
    if fit_rate < 0.5:
        return "failed_to_fit"
    if grok_rate >= 0.9:
        return "preserved"
    if grok_rate > 0.0:
        return "partial"
    return "memorized_only"


def build_summary(
    curves: Sequence[CurveRow],
    grads: Sequence[GradRow],
) -> list[SummaryRow]:
    final_curves: dict[tuple[str, int], CurveRow] = {}
    for c in curves:
        key = (c.train_policy, c.seed)
        if key not in final_curves or c.step > final_curves[key].step:
            final_curves[key] = c

    final_grads: dict[tuple[str, int], GradRow] = {}
    for g in grads:
        key = (g.approx_policy, g.seed)
        if key not in final_grads or g.step > final_grads[key].step:
            final_grads[key] = g

    policy_seeds: dict[str, set[int]] = defaultdict(set)
    for (pol, seed), c in final_curves.items():
        policy_seeds[pol].add(seed)

    rows: list[SummaryRow] = []
    for pol in sorted(policy_seeds):
        seeds = sorted(policy_seeds[pol])
        n = len(seeds)

        tr_accs = []
        te_accs = []
        mem_steps = []
        grok_steps = []
        delays = []
        grok_count = 0
        fit_count = 0
        censored = 0
        c0 = final_curves[(pol, seeds[0])]

        for s in seeds:
            c = final_curves[(pol, s)]
            tr_accs.append(c.train_acc)
            te_accs.append(c.test_acc)
            if c.train_acc >= 0.99:
                fit_count += 1
            if c.memorization_step > 0:
                mem_steps.append(c.memorization_step)
            if c.grokking_step > 0:
                grok_steps.append(c.grokking_step)
                grok_count += 1
            if c.grokking_reached == 0:
                censored += 1
            if c.grokking_reached and c.memorization_reached:
                delays.append(c.grokking_delay)

        tr_mean = sum(tr_accs) / n
        te_mean = sum(te_accs) / n
        tr_std = math.sqrt(sum((x - tr_mean) ** 2 for x in tr_accs) / n) if n > 1 else 0.0
        te_std = math.sqrt(sum((x - te_mean) ** 2 for x in te_accs) / n) if n > 1 else 0.0

        mem_median = sorted(mem_steps)[len(mem_steps) // 2] if mem_steps else float("nan")
        grok_median = sorted(grok_steps)[len(grok_steps) // 2] if grok_steps else float("nan")
        delay_median = sorted(delays)[len(delays) // 2] if delays else float("nan")

        fit_rate = fit_count / n
        grok_rate = grok_count / n
        censored_frac = censored / n

        grad_cos_vals = []
        grad_rel_vals = []
        snr_vals = []
        inno_kl_vals = []
        for s in seeds:
            g = final_grads.get((pol, s))
            if g is not None:
                if not math.isnan(g.grad_cos):
                    grad_cos_vals.append(g.grad_cos)
                if not math.isnan(g.grad_relerr):
                    grad_rel_vals.append(g.grad_relerr)
                if not math.isnan(g.update_snr):
                    snr_vals.append(g.update_snr)
                if not math.isnan(g.innovation_prob_kl):
                    inno_kl_vals.append(g.innovation_prob_kl)

        uniform_drops = []
        fourier_vals = []
        for s in seeds:
            c = final_curves[(pol, s)]
            if not math.isnan(c.uniform_attn_eval_drop):
                uniform_drops.append(c.uniform_attn_eval_drop)
            if not math.isnan(c.fourier_energy_ratio):
                fourier_vals.append(c.fourier_energy_ratio)

        outcome = _classify_outcome(grok_rate, fit_rate)

        rows.append(SummaryRow(
            policy=pol,
            eval_policy=c0.eval_policy,
            base_policy=c0.base_policy,
            sketch_dim=c0.sketch_dim,
            fixed_sketch=c0.fixed_sketch,
            resample_sketch=c0.resample_sketch,
            correction_period=c0.correction_period,
            architecture=c0.architecture,
            n_seeds=n,
            p=c0.p,
            train_fraction=c0.train_fraction,
            lr=c0.lr,
            weight_decay=c0.weight_decay,
            final_train_acc_mean=tr_mean,
            final_test_acc_mean=te_mean,
            final_train_acc_std=tr_std,
            final_test_acc_std=te_std,
            fit_rate=fit_rate,
            grok_rate=grok_rate,
            t_mem_median=mem_median,
            t_grok_median=grok_median,
            grok_delay_median=delay_median,
            censored_fraction=censored_frac,
            grad_cos_final_mean=sum(grad_cos_vals) / len(grad_cos_vals) if grad_cos_vals else float("nan"),
            grad_relerr_final_mean=sum(grad_rel_vals) / len(grad_rel_vals) if grad_rel_vals else float("nan"),
            update_snr_final_mean=sum(snr_vals) / len(snr_vals) if snr_vals else float("nan"),
            innovation_prob_kl_final_mean=sum(inno_kl_vals) / len(inno_kl_vals) if inno_kl_vals else float("nan"),
            uniform_attn_drop_mean=sum(uniform_drops) / len(uniform_drops) if uniform_drops else float("nan"),
            fourier_energy_final_mean=sum(fourier_vals) / len(fourier_vals) if fourier_vals else float("nan"),
            outcome_class=outcome,
        ))

    return rows
