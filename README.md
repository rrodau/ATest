# FairRanking

This is a Python implementation of the models and tests done in the paper Fair pairwise learning to rank. 

All the models are located in FairRanking/models while the data is located in the data folder. For running all 
experiments one also need to download the Wiki Talk Page Comments dataset in place it in the data folder 
(link in the folder README).

## Dependencies

The libraries required for the python scripts are shown in the requirements file


## Installation

First install the FairRanking package localy on your machine with:

```bash
...
pip install -e .
...
``` 

## Run experiments

For running the full gridsearch the script run_gridsearch needs to be executed. The used grid.json is passed with the -j 
option (for debugging the grid_debug.json can be taken). The -n argument can set the number of CPUs employed by the
gridsearch. You can also use the -o argument to choose the folder the experiment results will be written to. The folder(s)
will be created automatically. Please choose a two-leveled structure for your grid results, as this is needed to parse the
results correctly by the plot_utils script.

```bash
...
python run_gridsearch.py -j grid.json -n 4 -o results/name
...
```

For ploting the results the the script plot_tools/plot_utils needs to be executed. In this script the variable debug_data
can be set to True or False if grid.json or grid_debug.json was used. The -p value points to the results folders and can
be also multiple results.
```bash
...
python plot_tools/plot_utils.py -p path
...
```

## Remarks

For running the Wiki dataset a GPU and RAM > 32GB is at least required since the used embedding model is kind of big.  
