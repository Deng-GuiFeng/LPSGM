# -*- coding: utf-8 -*-
"""
channel_maps.py

EDF channel mapping, canonical channel order, and sleep stage labels consumed
by the Grad-CAM pipeline.

The default ``MASS_CHANNEL_MAP`` targets the MASS-SS1 / MASS-SS3 datasets
(https://borealisdata.ca/dataverse/MASS) as a reproducible public-data example
and uses the same channel options as ``preprocess/MASS-SS1-SS3.py``. Users
working with other recordings should provide their own mapping following the
same schema: each target channel lists one or more fallback options, where a
tuple of two names denotes a differential pair and a plain string denotes a
pre-referenced single channel. The first matching option (in listed order)
wins at load time.
"""

# 9-channel mapping for MASS-SS1 / MASS-SS3 EDF recordings.
# Each value is a tuple of fallback options, resolved in order by ``load_sig``.
# MASS bundles dual electrode references (CLE and LER), both of which point at
# the same physical derivation.
MASS_CHANNEL_MAP = {
    'F3':   ('EEG F3-CLE', 'EEG F3-LER'),
    'F4':   ('EEG F4-CLE', 'EEG F4-LER'),
    'C3':   ('EEG C3-CLE', 'EEG C3-LER'),
    'C4':   ('EEG C4-CLE', 'EEG C4-LER'),
    'O1':   ('EEG O1-CLE', 'EEG O1-LER'),
    'O2':   ('EEG O2-CLE', 'EEG O2-LER'),
    'E1':   ('EOG Left Horiz',),
    'E2':   ('EOG Right Horiz',),
    'Chin': (('EMG Chin1', 'EMG Chin2'),),
}

# Ordered list of canonical channels LPSGM was pre-trained with.
CANONICAL_CHANNELS = ['F3', 'F4', 'C3', 'C4', 'O1', 'O2', 'E1', 'E2', 'Chin']

# Sleep stage labels (LPSGM 5-class convention: 0=W, 1=N1, 2=N2, 3=N3, 4=R).
SLEEP_STAGES = ['W', 'N1', 'N2', 'N3', 'R']
