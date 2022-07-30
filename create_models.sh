#!/bin/bash

for model in Softmax LeNet LeNet_300_100 CNN; do
    python create_model.py --model "$model"
    python create_model.py --model "$model" --robust
    python create_model.py --model "$model" --dp
    python create_model.py --model "$model" --robust --dp
done
