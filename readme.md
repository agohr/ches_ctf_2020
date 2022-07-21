# Supplementary Code and Data to the Paper _Breaking Masked Implementations of the Clyde-Cipher by Means of Side-Channel Analysis_

## Installing the Requirements

To install the requirements, do the following:

- Clone this repository to a suitable local directory.
- Run `python -m venv <name_of_virtualenv>` to create a virtual environment under python 3.8.8 or later. We will in the sequel assume that `name_of_virtualenv` is `venv`.
- Activate the virtual environment by running `source ./venv/bin/activate'
- Change into the directory where this repository has been cloned to.
- Run `pip install -r requirements.txt`.

Now all requirements should be automatically set up.

## Running the experiments

The main script in this repository is `test_main.py`. It can be used to run the attack described in our paper, to train the main neural network described in our paper, or to perform our synthetic data experiment on the scattershot encoding. We will describe below how the script is to be used in each case:

### Running the Attack

To run the attack, first visit https://ctf.spook.dev/ and download (some of) the _fixed key_ datasets for the challenge you wish to attack (sw3, sw4, sw6 or sw8) from the UCLouvain open data repository. Put the downloaded datasets in a folder on your local machine. Also download the secret key file for the fixed-key dataset in question from the open data repository. Then run the script as follows:

`python test_main.py --target <target> --data-folder <data_folder> --mode attack --key-file <key_file> --num_samples <number of traces to use>`

### Training the network

To train the network, download a subset of the _random_ key files from the UCLouvain repository. Then train using

`python test_main.py --target <target> --data-folder <data_folder> --mode train --model-file <model_file_name> --traces-file <trace_file_name>`

The `model_file_name` and the `trace_file_name` tell the script where to store the model and where to store a collection of all the traces to be used for training.

### Running the Synthetic Scattershot Data Experiment

To replicate the experiment on using the scattershot encoding on a simple synthetic problem, run the main script as follows:

`python test_main.py --mode scatter`
