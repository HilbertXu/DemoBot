#!/bin/bash
cd FoundationPose

# use gdown to download FoundationPose weights
# the rest setup will be done in FoundationPose docker
conda activate 100doh

mkdir -p weights/2023-10-28-18-33-37
mkdir -p weights/2024-01-11-20-02-45

cd weights/2023-10-28-18-33-37
gdown https://drive.google.com/uc\?id\=1iChCUvu91eNVb2z2Z6_K6s0ChiT72t9_ # config.yml
gdown https://drive.google.com/uc\?id\=1u5Ul82iOVwAeX-PH9HNoP8pHlFPZYxx6 # model_best.pth

cd ../2024-01-11-20-02-45
gdown https://drive.google.com/uc\?id\=1HQQsCQSFiljBl3BAaISQKf3yUASUzCwo # config.yml
gdown https://drive.google.com/uc\?id\=1dUHEF1iw5JKM3jkxpxgkMfW56ig-Yif6 # model_best.pth
