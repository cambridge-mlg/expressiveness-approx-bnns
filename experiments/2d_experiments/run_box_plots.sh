#!/bin/bash
DATASET="origin"
WIDTH=50
NUM_EPOCHS=100000
SIGMA_B=1.0
MINIBATCH_SIZE=100
DROPOUT_RATE=0.05
LEARNING_RATE=0.001
NOISE_STD=0.1

SAVEDIR="./box_plot_results"
for INFERENCE in "DropoutBNN" "FFGBNN" "GPBNN"
  do
    LAYERS=1
    for SIGMA_W in 4.0 3.0 2.25 2.0 2.0 1.9 1.75 1.75 1.7 1.65
      do
        python train_2d.py --dataset $DATASET --inference $INFERENCE --layers $LAYERS \
          --width $WIDTH --num_epochs $NUM_EPOCHS --sigma_w $SIGMA_W --sigma_b $SIGMA_B \
          --minibatch_size $MINIBATCH_SIZE --dropout_rate $DROPOUT_RATE \
          --learning_rate $LEARNING_RATE --noise_std $NOISE_STD --savedir $SAVEDIR --save
        LAYERS=$((LAYERS+1)) # increment depth to match sigma_w
      done
  done

SAVEDIR="./root2_box_plot_results"
SIGMA_W=1.414213562 # use sigma_w=root 2 for all depths
for INFERENCE in "DropoutBNN" "FFGBNN" "GPBNN"
  do
    for LAYERS in {1..10}
      do
        python train_2d.py --dataset $DATASET --inference $INFERENCE --layers $LAYERS \
          --width $WIDTH --num_epochs $NUM_EPOCHS --sigma_w $SIGMA_W --sigma_b $SIGMA_B \
          --minibatch_size $MINIBATCH_SIZE --dropout_rate $DROPOUT_RATE \
          --learning_rate $LEARNING_RATE --noise_std $NOISE_STD --savedir $SAVEDIR --save
      done
  done
