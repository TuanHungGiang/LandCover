"""Fast invariants for the conservation-aware INFO policy search (no dataset needed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools.search_token_policy_info import (DEFAULT, EXACT_RATIO, dominates, encode,
                                            info_generation, pareto_front, policy_overrides,
                                            repair, simulate_result)


def main():
    z = encode(DEFAULT)
    decoded = repair(z)
    assert decoded == DEFAULT, (decoded, DEFAULT)
    overrides = policy_overrides(decoded)
    assert overrides.count(f"model_config.seghead.scan_cfg.ratio={EXACT_RATIO}") == 1
    assert all("ratio" not in item or item.endswith("ratio=0.25") for item in overrides)

    fast = dict(miou=.51, hard_iou=.40, latency_ms=7.)
    accurate = dict(miou=.54, hard_iou=.43, latency_ms=9.)
    dominated = dict(miou=.50, hard_iou=.39, latency_ms=10.)
    assert dominates(fast, dominated) and dominates(accurate, dominated)
    assert len(pareto_front([fast, accurate, dominated])) == 2

    rng = np.random.default_rng(7)
    pop = [rng.random(len(z)) for _ in range(6)]
    rows = [{**simulate_result(repair(x)), "policy": repair(x)} for x in pop]
    children = info_generation(pop, rows, rng)
    assert len(children) == len(pop)
    assert all(x.shape == z.shape and np.all((0 <= x) & (x <= 1)) for x in children)
    print("INFO policy search invariants: PASS (ratio fixed at 0.25)")


if __name__ == "__main__":
    main()
