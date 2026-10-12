"""Verify the internal policy-search split without loading image pixels."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rsseg.datasets.loveda_dataset import LoveDA
from tools.search_token_policy_info import validate_internal_holdout


def main():
    ids = [f'image_{i:04d}' for i in range(100)]
    fit = LoveDA.partition_ids(ids, 'Urban', 'fit', .20, 2026)
    holdout = LoveDA.partition_ids(ids, 'Urban', 'holdout', .20, 2026)
    assert len(fit) == 80 and len(holdout) == 20
    assert not set(fit) & set(holdout)
    assert set(fit) | set(holdout) == set(ids)
    assert holdout == LoveDA.partition_ids(ids[::-1], 'Urban', 'holdout', .20, 2026)
    meta = validate_internal_holdout('configs/loveda/gee_repvit_landcover_adaptive_spectral_search.py')
    assert meta == {'source_split': 'Train', 'holdout_fraction': .20, 'holdout_seed': 2026}
    try:
        validate_internal_holdout('configs/loveda/gee_repvit_landcover_adaptive_spectral.py')
    except ValueError:
        pass
    else:
        raise AssertionError('leakage guard accepted the official-Val final config')
    print('LoveDA internal search split: PASS (80 fit / 20 holdout, official Val rejected)')


if __name__ == '__main__':
    main()
