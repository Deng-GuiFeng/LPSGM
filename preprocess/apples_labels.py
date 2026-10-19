# -*- coding: utf-8 -*-
"""
apples_labels.py

Derive the APPLES screening labels directly from the NSRR APPLES dataset table
(``apples-dataset-0.1.0.csv``, distributed by the NSRR together with the PSG
files). No per-subject label file is shipped with this repository; labels are
recomputed from the raw NSRR fields every time a task loads its subjects.

The NSRR table has one row per participant visit. Clinical variables used
here are recorded at the baseline visit, while the PSG ``fileid`` is recorded
at the diagnostic visit; the two are linked through ``appleid``.

Depression (surrogate label; see Supplementary Methods):
    - Depressed (1): ``depressionmedhxhp`` = 2 (ongoing depression history) and
      not both HAMD < 5 and BDI < 5.
    - Non-depressed (0): ``depressionmedhxhp`` = 0 (no depression history),
      HAMD <= 4 and BDI <= 4.
    - Subjects with a missing HAMD or BDI score are excluded (negative values
      are NSRR missing-value codes); all other subjects are excluded.

OSA severity:
    - Severe (1): ``osaseveritypostqc`` = 3 (severe OSA, AHI > 30).
    - Non-severe (0): ``osaseveritypostqc`` = 1 or 2 (mild or moderate OSA).
    - Subjects without a severity grade (0 = non-randomized, or missing) are
      excluded.

Usage:
    python preprocess/apples_labels.py --nsrr_csv /path/to/apples-dataset-0.1.0.csv
"""

import argparse
import os
from typing import Dict

import pandas as pd


DEFAULT_NSRR_CSV = r"/nvme1/denggf/PSG_datasets/public_datasets/APPLES/files/datasets/apples-dataset-0.1.0.csv"

DEPRESSION_FIELDS = ['depressionmedhxhp', 'hamdtotalscore', 'bditotalscore']
OSA_FIELDS = ['osaseveritypostqc']


def _per_subject_table(nsrr_csv: str, fields) -> pd.DataFrame:
    """Return one row per PSG ``fileid`` with the requested baseline fields."""
    df = pd.read_csv(nsrr_csv, low_memory=False)
    psg = df.loc[df['fileid'].notna(), ['appleid', 'fileid']].drop_duplicates('appleid')
    baseline = df.loc[df['visitn'] == 1, ['appleid'] + list(fields)].drop_duplicates('appleid')
    table = psg.merge(baseline, on='appleid', how='left')
    for f in fields:
        # Categorical fields are stored as "<code>) <text>" (e.g. "3) severe");
        # keep the leading numeric code. Numeric fields pass through unchanged.
        values = table[f].astype(str).str.extract(r'^\s*(-?\d+(?:\.\d+)?)', expand=False)
        table[f] = pd.to_numeric(values, errors='coerce')
    return table


def depression_labels(nsrr_csv: str = DEFAULT_NSRR_CSV) -> Dict[str, int]:
    """Map PSG ``fileid`` to the surrogate depression label (0 / 1)."""
    t = _per_subject_table(nsrr_csv, DEPRESSION_FIELDS)
    hist, hamd, bdi = t['depressionmedhxhp'], t['hamdtotalscore'], t['bditotalscore']
    scales_valid = (hamd >= 0) & (bdi >= 0)
    depressed = scales_valid & (hist == 2) & ~((hamd < 5) & (bdi < 5))
    non_depressed = scales_valid & (hist == 0) & (hamd <= 4) & (bdi <= 4)
    labels = {fid: 1 for fid in t.loc[depressed, 'fileid']}
    labels.update({fid: 0 for fid in t.loc[non_depressed, 'fileid']})
    return labels


def osa_labels(nsrr_csv: str = DEFAULT_NSRR_CSV) -> Dict[str, int]:
    """Map PSG ``fileid`` to the OSA severity label (0 = Non-severe, 1 = Severe)."""
    t = _per_subject_table(nsrr_csv, OSA_FIELDS)
    grade = t['osaseveritypostqc']
    labels = {fid: 1 for fid in t.loc[grade == 3, 'fileid']}
    labels.update({fid: 0 for fid in t.loc[grade.isin([1, 2]), 'fileid']})
    return labels


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Derive APPLES screening labels from the NSRR dataset table')
    p.add_argument('--nsrr_csv', type=str, default=DEFAULT_NSRR_CSV)
    args = p.parse_args()
    if not os.path.exists(args.nsrr_csv):
        raise FileNotFoundError(args.nsrr_csv)
    for name, fn in [('depression', depression_labels), ('osa', osa_labels)]:
        labels = fn(args.nsrr_csv)
        n1 = sum(labels.values())
        print(f'{name}: {len(labels)} subjects ({len(labels) - n1} negative, {n1} positive)')
