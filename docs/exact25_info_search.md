# Exact-25% policy search (C3)

## What is optimized

C3 keeps the C2 forward path and its exact `N/4` context sequence. The value `ratio=0.25` is a
constant outside the chromosome. The offline search changes only how the budget is spent:

- merge/split fraction;
- boundary, entropy, and rare-class evidence weights;
- spectral graph temperature, self-loop, and iteration count;
- SSM state size and scan chunk.

The deployment model does **not** run a metaheuristic. It uses the selected constants and therefore
adds no population-search cost at inference.

## Conservation-aware multi-fidelity Pareto INFO

The base optimizer is INFO, whose update is built from a weighted mean of population vectors, vector
combination, and local search. The land-cover version adds:

1. **Conservation projection:** mixed continuous/discrete candidates are repaired to legal values;
   the 25% ratio is injected only after repair and cannot mutate.
2. **Three-objective archive:** maximize validation mIoU and mean IoU of barren/forest, while minimizing
   measured adaptive-block latency. Final selection comes from the non-dominated set.
3. **Leakage-safe cross-validated evaluation:** LoveDA Train is deterministically partitioned within
   each Urban/Rural domain into three disjoint folds. Every policy is trained three times, each time
   holding out a different fold. Selection uses `fold_mean - 0.5 * fold_std`, so an unstable one-fold
   winner is penalized. Twelve-epoch screening precedes 24-epoch confirmation. Official LoveDA Val is
   never loaded by the search. Only the selected policy is retrained on full Train for 96 epochs and
   then evaluated on official Val.
4. **C2 warm start:** the known policy is one population member, preventing a small GPU budget from
   spending every trial rediscovering the baseline.

This is a search method, not evidence of improvement by itself. A paper should compare equal search
budgets against random search and at least one standard method such as TPE or NSGA-II, then retrain the
selected policies with the same seeds.

## Run

```bash
python tools/check_info_token_search.py
python tools/check_loveda_search_split.py
python tools/search_token_policy_info.py --gpus 0 1 --population 4 --generations 1 --confirm_top 2
```

The search is restartable through `work_dirs/info_token_search/cache.json`. Copy the command written
to `best_policy.json` for the fresh 96-epoch run. Never report a 12/24-epoch screening score as the
final model result. The script fails closed if its search validation loader points to official Val,
and writes every selected image id plus a SHA-256 digest to `split_manifest.json`. The digest is part
of the cache key, so trials from official Val or a different internal split cannot be reused.
