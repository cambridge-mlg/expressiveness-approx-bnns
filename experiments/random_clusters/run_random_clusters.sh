#!/bin/bash
DIRNAME="data"
NUM_EPOCHS=100000
NOISE_STD=0.1
DIM=5
NUM_DATASETS=10
WIDTH=50
MINIBATCH_SIZE=200
SAMPLES=32
DROPOUT_RATE=0.05
SIGMA_W=1.41421356
SIGMA_B=1.0
LEARNING_RATE=0.001

I=0
for INFERENCE in "GPBNN" "FFGBNN" "DropoutBNN"
  do
    for LAYERS in 1 3
      do
        if (( $I == 0 )) # We must generate the data in the first instance
          then
             python random_clusters_regression.py --gen --dirname $DIRNAME \
               -n $NUM_DATASETS -d $DIM -i $INFERENCE --layers $LAYERS --width $WIDTH \
               -e $NUM_EPOCHS --noise_std $NOISE_STD --minibatch_size $MINIBATCH_SIZE \
               --samples $SAMPLES --dropout_rate $DROPOUT_RATE --sigma_w $SIGMA_W \
               --sigma_b $SIGMA_B --learning_rate $LEARNING_RATE
              I=1
            else
              python random_clusters_regression.py --dirname $DIRNAME \
                -n $NUM_DATASETS -d $DIM -i $INFERENCE --layers $LAYERS --width $WIDTH \
                -e $NUM_EPOCHS --noise_std $NOISE_STD --minibatch_size $MINIBATCH_SIZE \
                --samples $SAMPLES --dropout_rate $DROPOUT_RATE --sigma_w $SIGMA_W \
                --sigma_b $SIGMA_B --learning_rate $LEARNING_RATE
        fi
      done
  done
