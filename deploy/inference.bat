@echo off

python main.py ^
    --input data/input/2025021001.edf ^
    --output data/output/2025021001.txt ^
    --scoredata_xml data/input/2025021001.scoredata.xml ^
    --log_file data/output/2025021001.log ^
    --batch_size 64
