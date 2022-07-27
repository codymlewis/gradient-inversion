#!/bin/bash

for model in Softmax LeNet LeNet_300_100 CNN; do
    python create_model.py --model "$model"
    python create_model.py --model "$model" --robust
done
