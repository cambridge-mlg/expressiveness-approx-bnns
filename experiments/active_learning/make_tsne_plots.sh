#!/bin/bash
DATASET="Naval"
ITER=50
SPLIT="split0"
TSNE_DIRPATH="tsne"
# to be run after running active.sh

# create the tsne file
python active_plotting.py --dirpath results/Naval/DropoutBNN/1HL/max_variance/$SPLIT \
  --iter $ITER --run-tsne --tsne_dirpath tsne

ACQUISITION_FUNCTION="max_variance"
for INFERENCE in "DropoutBNN" "FFGBNN" "GPBNN"
  do
    for DEPTH in 1 3
      do
        DIRPATH="results/"$DATASET"/"$INFERENCE"/"$DEPTH"HL/"$ACQUISITION_FUNCTION"/"$SPLIT
        python active_plotting.py --dirpath $DIRPATH --iter $ITER --tsne_dirpath $TSNE_DIRPATH --savedir "tsne_plots" --depth $DEPTH --inference $INFERENCE --acquisition $ACQUISITION_FUNCTION
        python active_plotting.py --dirpath $DIRPATH --iter 0 --tsne_dirpath $TSNE_DIRPATH --savedir "tsne_plots" --depth $DEPTH --inference $INFERENCE --acquisition $ACQUISITION_FUNCTION

      done
  done

ACQUISITION_FUNCTION="random"
ITER=1
for INFERENCE in "DropoutBNN" "FFGBNN" "GPBNN"
  do
    for DEPTH in 1 3
      do
        DIRPATH="results/"$DATASET"/"$INFERENCE"/"$DEPTH"HL/"$ACQUISITION_FUNCTION"/"$SPLIT
        python active_plotting.py --dirpath $DIRPATH --iter $ITER --tsne_dirpath $TSNE_DIRPATH --savedir "tsne_plots" --depth $DEPTH --inference $INFERENCE --acquisition $ACQUISITION_FUNCTION

        python active_plotting.py --dirpath $DIRPATH --iter 0 --tsne_dirpath $TSNE_DIRPATH --savedir "tsne_plots" --depth $DEPTH --inference $INFERENCE --acquisition $ACQUISITION_FUNCTION

      done
  done
