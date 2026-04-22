# -*- coding: utf-8 -*-
"""
simple_eval.py

CLI wrapper that invokes ``cls_core.simple_eval`` with APPLES depression defaults.

Usage:
    python -m dep_cls.simple_eval --fold 0 --run-root ./run_dep
"""

from cls_core.simple_eval import cli_main

from .dataset import load_subjects


if __name__ == '__main__':
    cli_main(load_subjects, default_run_root='./run_dep', default_num_classes=2)
