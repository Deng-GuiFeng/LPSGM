# -*- coding: utf-8 -*-
"""
simple_eval.py

CLI wrapper that invokes ``cls_core.simple_eval`` for the MNC tasks.

Usage:
    python -m nar_cls.simple_eval --fold 0 --run-root ./run_nar
    python -m nar_cls.simple_eval --task binary --fold 0 --run-root ./run_nar_binary
"""

import argparse
import functools
import sys

from cls_core.simple_eval import cli_main

from .dataset import TASKS, load_subjects


if __name__ == '__main__':
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--task', type=str, default='3class', choices=sorted(TASKS))
    known, rest = pre.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    num_classes, _ = TASKS[known.task]
    cli_main(
        functools.partial(load_subjects, task=known.task),
        default_run_root='./run_nar' if known.task == '3class' else './run_nar_binary',
        default_num_classes=num_classes,
    )
