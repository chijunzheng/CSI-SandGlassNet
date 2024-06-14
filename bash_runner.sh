#!/usr/bin/env bash

#WORKDIR="/content/drive/My Drive/2.Learning/Masters_Thesis/Final_Report/CSI-SandGlassNet"
#CONDAENV=""
#CONDAROOT=""

#source $CONDAROOT/etc/profile.d/conda.sh
#conda activate ${CONDAENV}

#cd $WORKDIR

export CUDA_VISIBLE_DEVICES=0

CR=4
DS=indoor
nohup python main.py \
    --label SandglassNet-$DS-p2-s1.64-s2.32-4heads-ffnR4-cr$CR-NoPosEnc \
    --wandb_proj test \
    --data_scenario $DS \
    --cr $CR \
    --lr 0.0001 \
    --nepoch 1000 \
    --rundir /content/drive/MyDrive/2.Learning/Masters_Thesis/Final_Report/ &
