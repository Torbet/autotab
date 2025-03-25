#!/bin/bash

python train.py
python train.py --ft
python train.py --ft --freeze
python train.py --ss
python train.py --ft --ss
python train.py --ft --freeze --ss
