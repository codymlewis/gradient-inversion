#!/bin/bash

for model in ResNetRS50; do
    python create_model.py --model "$model"
    python create_model.py --model "$model" --robust
    python create_model.py --model "$model" --dp
    python create_model.py --model "$model" --robust --dp
    python create_model.py --model "$model" --dp 0.5 0.3
    python create_model.py --model "$model" --robust --dp 0.5 0.3
    python create_model.py --model "$model" --dp 1 0.1
    python create_model.py --model "$model" --robust --dp 1 0.1
done
