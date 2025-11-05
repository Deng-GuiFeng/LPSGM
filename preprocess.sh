# !/bin/bash

# Prepare Sleep Staging Datasets

python ./preprocess/HANG7.py   # private dataset
python ./preprocess/SYSU.py    # private dataset
python ./preprocess/DOD-H.py
python ./preprocess/DOD-O.py
python ./preprocess/APPLES.py
python ./preprocess/HOMEPAP.py
python ./preprocess/STAGES_1.py
python ./preprocess/STAGES_2.py
python ./preprocess/NCHSDB.py
python ./preprocess/MROS.py
python ./preprocess/SVUH.py
python ./preprocess/SHHS.py
python ./preprocess/CFS.py
python ./preprocess/DCSM.py
python ./preprocess/HMC.py
python ./preprocess/CCSHS.py
python ./preprocess/ABC.py
python ./preprocess/CHAT.py
python ./preprocess/ISRUC.py
python ./preprocess/P2018.py
python ./preprocess/MASS-SS1-SS3.py
python ./preprocess/MESA.py


# Prepare Sleep Disorder Datasets

python ./preprocess/MNC/CNC.py
python ./preprocess/MNC/FHC.py
python ./preprocess/MNC/IHC.py
python ./preprocess/MNC/KHC.py
python ./preprocess/MNC/SSC.py
python ./preprocess/MNC/DHC.py

