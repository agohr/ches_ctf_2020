# Supplementary Code and Data to the Paper _Breaking Masked Implementations of the Clyde-Cipher by Means of Side-Channel Analysis_

## Installing the Requirements

To install the requirements, do the following:

- Clone this repository to a suitable local directory.
- Run `python -m venv <name_of_virtualenv>` to create a virtual environment under python 3.8.8 or later. We will in the sequel assume that `name_of_virtualenv` is `venv`.
- Activate the virtual environment by running `source ./venv/bin/activate`
- Change into the directory where this repository has been cloned to.
- Run `pip install -r requirements.txt`.

Now all requirements should be automatically set up.

## Overview of the Repository

The repository contains the following files and directories:

- The main directory contains this readme file, the requirements file, the license file, gitignore file, and the python code implementing the training and evaluation of our attack.
- The `results` subdirectory contains `.npz` files that contain the evaluation results for our attack that are reported in the paper.
- The `npy_files` subdirectory contains our scattershot masks.
- The `models` subdirectory contains the trained models that are used for the evaluation of our attack.	
- The `interface` subdirectory contains code provided by the organisers of the competition that is used by our attack. This code was not written by us and is licensed under the MIT license. The original copyright notice is included.
- The `additional_tests` subdirectory contains the implementation of our synthetic data experiments on the scattershot encoding. It also contains the data on those experiments that are described in the paper.

## Running the experiments

The main script in this repository is `test_main.py`. It can be used to run the attack described in our paper, to train the main neural network described in our paper, or to perform our synthetic data experiment on the scattershot encoding. We will describe below how the script is to be used in each case:

### Running the Attack

#### Generic Instructions

To run the attack, first visit https://ctf.spook.dev/ and download (some of) the _fixed key_ datasets for the challenge you wish to attack (sw3, sw4, sw6 or sw8) from the UCLouvain open data repository. Put the downloaded datasets in a folder on your local machine. Also download the secret key file for the fixed-key dataset in question from the open data repository. Then run the script as follows:

`python test_main.py --target <target> --data-folder <data_folder> --mode attack --key-file <key_file> --num_samples <number of traces to use>`

If you want to supply your own model file, you can do so by running

`python test_main.py --target <target> --data-folder <data_folder> --mode attack --key-file <key_file> --num_samples <number of traces to use> --model-file <model_file>`

where `model_file` is the path to the model file you want to use. The model file must be a keras model and has to use the same inputs and outputs as the pretrained network we provide for the same target in this repository.

#### Examples

To illustrate these instructions, we will provide a fully specified example call. Suppose you want to run the sw3 challenge on fixed key datasets stored in `./data/sw3/fkey/` after downloading the relevant traces files from the open data repository; for concreteness, we may assume that `./data/sw3/fkey/` contains just one file, say `fkey_sw3_K0_1000_0.npz`. You have further downloaded the corresponding secret key to `./data/sw3/secret_sw3_K0` and you want to test the attack using 30 samples in each attack run. Then you would run the following command:

`python test_main.py --target sw3 --data-folder ./data/sw3/fkey/ --mode attack --key-file ./data/sw3/secret_sw3_K0 --num_samples 30`

To obtain a key rank below 2<sup>32</sup>, you should use the following number of samples:

| Target | Number of Samples |
|--------|-------------------|
| sw3    | 25                |
| sw4    | 105               |
| sw6    | 3000              |
| sw8    | 35000             |

In order to obtain different datasets in each attack run, the number of traces in the attack data folder should be higher than the number of samples specified in the command line. For example, if you want to run the attack on 30 samples, you should have at least 100 traces in the attack data folder.

### Training the network

To train the network, download a subset of the _random_ key files from the UCLouvain repository. Then train using

`python test_main.py --target <target> --data-folder <data_folder> --mode train --model-file <model_file_name> --traces-file <trace_file_name>`

The `model_file_name` and the `trace_file_name` tell the script where to store the model and where to store a collection of all the traces to be used for training.

We recommend using _all_ of the random-key data available in the open data repository for each challenge for training. However, for sw3, sw4 and sw6, reasonable results should be achievable already when training on 10000 traces. We expect that there is significant room for optimisation of the training procedure for the setting where the number of training traces is small.

Assuming that you have downloaded the random key data for sw3 to `./data/sw3/rkey/` (i.e. this directory will contain all of the relevant `.npz` files obtained from the UCLouvain repository), you might train the network as follows:

`python test_main.py --target sw3 --data-folder ./data/sw3/rkey/ --mode train --model-file ./models/sw3_model.h5 --traces-file ./data/sw3_traces.npz`

### Running the Synthetic Scattershot Data Experiment

To replicate the experiment on using the scattershot encoding on a simple synthetic problem, run the main script as follows:

`python test_main.py --mode scatter

Optionally, saving the plots generated by the scattershot experiment can be specified by adding the `--save-plots` flag, i.e. by running

`python test_main.py --mode scatter --save-plots <path_to_save_plots>`

where `path_to_save_plots` is the path to the file where the plots should be saved. The script supports saving the plots in `png` or `tex` format (the latter will output a tikzpicture).

### Troubleshooting

On Mac, installing the `tensorflow` dependency has been reported to be sometimes problematic. In this case, manually installing `tensorflow-macos` should fix the problem.
