#!/bin/bash
DATASET="Naval"
ACTIVE_BATCH_SIZE=1
NUM_SPLITS=20
INITIAL_POINTS=5
NUM_STEPS=50
WIDTH=50
NUM_EPOCHS=20000
SAMPLES=32
DROPOUT_RATE=0.05
SIGMA_W=1.41421356
SIGMA_B=1.0
NOISE_STD=0.01
LEARNING_RATE=0.001
MINIBATCHSIZE=100

ACQUISITION_FUNCTION="max_variance"
for INFERENCE in "DropoutBNN" "FFGBNN" "GPBNN"
  do
    for DEPTH in 1 2 3 4
      do
        LOGDIR="results/"$DATASET"/"$INFERENCE"/"$DEPTH"HL/"$ACQUISITION_FUNCTION
        echo $LOGDIR
        python active_main.py -d $DATASET -i $INFERENCE -l $DEPTH --width $WIDTH \
         --acquisition_function $ACQUISITION_FUNCTION --active_batch_size $ACTIVE_BATCH_SIZE \
         --num_splits $NUM_SPLITS --num_epochs $NUM_EPOCHS --num_initial_points $INITIAL_POINTS \
         --samples $SAMPLES --dropout_rate $DROPOUT_RATE --sigma_w $SIGMA_W \
         --sigma_b $SIGMA_B --noise_std $NOISE_STD --learning_rate $LEARNING_RATE \
         -mb $MINIBATCHSIZE --logdir $LOGDIR --num_steps $NUM_STEPS
      done
  done

ACQUISITION_FUNCTION="random"
INITIAL_POINTS=5
ACTIVE_BATCH_SIZE=50 # when making random acquisitions, they can all be taken at once
NUM_STEPS=1
for INFERENCE in "DropoutBNN" "FFGBNN" "GPBNN"
  do
    for DEPTH in 1 2 3 4
      do
        LOGDIR="results/"$DATASET"/"$INFERENCE"/"$DEPTH"HL/"$ACQUISITION_FUNCTION
        echo $LOGDIR
        python active_main.py -d $DATASET -i $INFERENCE -l $DEPTH --width $WIDTH \
         --acquisition_function $ACQUISITION_FUNCTION --active_batch_size $ACTIVE_BATCH_SIZE \
         --num_splits $NUM_SPLITS --num_epochs $NUM_EPOCHS --num_initial_points $INITIAL_POINTS \
         --samples $SAMPLES --dropout_rate $DROPOUT_RATE --sigma_w $SIGMA_W \
         --sigma_b $SIGMA_B --noise_std $NOISE_STD --learning_rate $LEARNING_RATE \
         -mb $MINIBATCHSIZE --logdir $LOGDIR --num_steps $NUM_STEPS
      done
  done
