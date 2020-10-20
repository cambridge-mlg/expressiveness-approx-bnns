# On the Expressiveness of Approximate Inference in Bayesian Neural Networks

This repository is the official implementation of [On the Expressiveness of Approximate Inference in Bayesian Neural Networks](https://arxiv.org/abs/1909.00719). The code in the repository was jointly written by Andrew Y. K. Foong and David R. Burt. 

## Requirements

To install requirements:

```setup
python setup.py install
```

Additionally, in order to run the active learning experiments, you will need to install [Bayesian benchmarks](https://github.com/hughsalimbeni/bayesian_benchmarks) by cloning the git repository and running
```
python setup.py install
```
in the root of the `bayesian_benchmarks` repository.

## Recreating Experiments

To recreate the experiments from the paper:

To minimise squared loss in `experiments/approximator` run:
```
bash run_approximator.sh
```
in order to recreate figures 2 and 4 in the main text. 

To run the 2d experiments with 1HL networks (Figure 3) in `experiments/2d_experiments` run:

```
bash run_2d.sh
```

To run the 2d experiments with deeper networks and generate the boxplots (Figure 5 in the main text, Figures 9 and 10 in the supplement) in `experiments/2d_experiments` run:

```
bash run_box_plots.sh
```

After which boxplots can be generated using `python box_plots.py` (run ```python box_plots.py --help``` to view syntax for arguments to be passed to the plotting script).
Samples from HMC, generated using [pyro](https://pyro.ai/) are provided in `experiments/2d_experiments/HMC`.


To run the active learning experiments (Table 1 and Figure 6 in the main body, Figures 12-14 in the supplement) in `experiments/active_learning` run (this will take a while!):

```
bash active.sh
```

After this is complete, the t-sne plots (Figs 6, Figs 12-14 supplement) can be made by running:

```
bash make_tsne_plots.sh
```

and the results in table 1 can be collected by running,

```
python collect_results.py
```

Use `python collect_results.py --help` to view syntax for arguments to be passed to this script.

To run experiments with inputs generated in pairs of clusters on a 5-dimensional sphere (Figures 6 and 7 in the supplement), in `experiments/random_clusters` run:

```
bash run_random_clusters.sh
```
