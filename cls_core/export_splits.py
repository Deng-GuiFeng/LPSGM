# -*- coding: utf-8 -*-
"""
export_splits.py

Export the per-fold train / validation / test subject splits written by
``cls_core.trainer.PooledTrainer`` (``{run_root}/fold{N}/splits/subjects.json``)
into a single CSV file per task:

    subject_id,fold,split
    apples-130001,0,train
    ...

Labels are not written: they are recomputed from the original dataset files by
each task's ``load_subjects`` (for APPLES, from the NSRR dataset table).

Every subject appears exactly once per fold. The same file is produced by
re-running the splitting code on the same subject list with the task's split
seed (``StratifiedKFold(5, shuffle=True, random_state=seed)`` and a 25 %
validation hold-out of each training fold, drawn with the same seed).

Usage:
    python -m cls_core.export_splits --run_root ./run_osa --out splits/apples_osa_5fold.csv
"""

import argparse
import csv
import glob
import json
import os


def export(run_root: str, out_path: str, with_labels: bool = False) -> int:
    """Write the per-fold splits under ``run_root`` to ``out_path``; return the number of rows written."""
    rows = []
    for path in sorted(glob.glob(os.path.join(run_root, 'fold*', 'splits', 'subjects.json'))):
        with open(path, encoding='utf-8') as f:
            split = json.load(f)
        for name in ('train', 'val', 'test'):
            for item in split[name]:
                rows.append((item['id'], int(split['fold']), name, int(item['label'])))
    rows.sort(key=lambda r: (r[1], ('train', 'val', 'test').index(r[2]), r[0]))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if with_labels:
            w.writerow(['subject_id', 'fold', 'split', 'label'])
            w.writerows(rows)
        else:
            w.writerow(['subject_id', 'fold', 'split'])
            w.writerows(r[:3] for r in rows)
    return len(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Export per-fold subject splits to CSV')
    p.add_argument('--run_root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--with_labels', action='store_true',
                   help='Also write the label column.')
    args = p.parse_args()
    n = export(args.run_root, args.out, with_labels=args.with_labels)
    print(f'Saved: {args.out} ({n} rows)')
