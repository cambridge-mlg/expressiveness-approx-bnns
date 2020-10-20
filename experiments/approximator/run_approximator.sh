#!/bin/bash
WIDTH=50
NUM_EPOCHS=50000
MINIBATCH_SIZE=100
SAMPLES=128
DROPOUT_RATE=0.05
LEARNING_RATE=0.001

for INFERENCE in "ApproximatorFFGBNN" "ApproximatorDropoutBNN"
  do
    for LAYERS in 1 2
      do
        python train_approximator.py -i $INFERENCE --layers $LAYERS --width $WIDTH \
          -e $NUM_EPOCHS --minibatch_size $MINIBATCH_SIZE --samples $SAMPLES \
          --dropout_rate $DROPOUT_RATE --learning_rate $LEARNING_RATE
      done
  done

python plot_approximator.py -d 2
python plot_approximator.py -d 1 -b --inference_types ApproximatorFFGBNN
python plot_approximator.py -d 1 --inference_types ApproximatorDropoutBNN
