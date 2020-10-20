#!/bin/bash
LAYERS=1
DATASET="origin"
WIDTH=50
NUM_EPOCHS=100000
SIGMA_W=4.0
SIGMA_B=1.0
MINIBATCH_SIZE=100
DROPOUT_RATE=0.05
LEARNING_RATE=0.001
NOISE_STD=0.1
SAVEDIR="./1HL_results"
for INFERENCE in "DropoutBNN" "FFGBNN" "GPBNN"
  do
    python train_2d.py --dataset $DATASET --inference $INFERENCE --layers $LAYERS \
      --width $WIDTH --num_epochs $NUM_EPOCHS --sigma_w $SIGMA_W --sigma_b $SIGMA_B \
      --minibatch_size $MINIBATCH_SIZE --dropout_rate $DROPOUT_RATE \
      --learning_rate $LEARNING_RATE --noise_std $NOISE_STD --savedir $SAVEDIR --save \
      --plot
  done
