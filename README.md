# Assignment *number, 3*

- name: Matthew Pereira Wilson
- student ID: 20644035

## Dependencies

- json
- numpy
- matplotlib
- argparse
- torch
- collections

## Running `main.py`

To run `main.py`, use

```sh
python3 main.py data/in.txt data/params.json -v 2
```

## Hyper-parameter `json` file

A sample `params.json` file is provided in `data/`. It contains,

- n_epoch    : number of epochs to train the model
- eta        : learning rate of the Boltzmann Machine
- n_epoch_v  : interval at which diagnosis output is printed
- batch_size : number of samples generated from MCMC

## Outputs

This script outputs a plot, `loss_function.png`, of the KL divergence loss and a text file, `output.txt`, with the coupling constants of the Hamiltonian. An output directory can be given to the script with `--output_path dir/`.
