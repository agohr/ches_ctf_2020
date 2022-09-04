import numpy as np
import attack
import train
import click

from additional_tests import scatter
from matplotlib import pyplot as plt

import tikzplotlib


net_params = {}
net_params['sw3'] = {'depth': 10, 'q1': 100, 'q2': 625, 'outputs': 100, 'D': 3, 'netfile': './models/model_hw_sbox_masks3.h5'}
net_params['sw4'] = {'depth': 10, 'q1': 167, 'q2': 499, 'outputs': 100, 'D': 4, 'netfile': './models/model_hw_sbox_masks4.h5'}
net_params['sw6'] = {'depth': 10, 'q1': 250, 'q2': 625, 'outputs': 100, 'D': 6, 'netfile': './models/model_hw_sbox_masks6.h5'}
net_params['sw8'] = {'depth': 10, 'q1': 250, 'q2': 875, 'outputs': 100, 'D': 8, 'netfile': './models/model_hw_sbox_masks8.h5'}


def get_attack_stats(data_folder, target, key_file, num_samples):
    if target not in list(net_params.keys()):
        raise ValueError('Target must be one of: {}'.format(list(net_params.keys())))
    net_parameters = net_params[target]
    depth, q1, q2, outputs, D, netfile = net_parameters['depth'], net_parameters['q1'], net_parameters['q2'], net_parameters['outputs'], net_parameters['D'], net_parameters['netfile']
    attack.D = D
    predictions = attack.run_neural_network(data_folder, N=outputs, q1=q1, q2=q2, netfile=netfile)
    tweaks = attack.get_tweaks(data_folder)
    key = attack.read_key(key_file)
    masks = np.load('./npy_files/testmasks.npy')
    stats = attack.stats_log_key_rank(predictions, masks, tweaks, key, num_samples)
    return stats


@click.command()
@click.option('--target', default='sw3', help='Target implementation')
@click.option('--data-folder', default=None, help='Folder containing data')
@click.option('--mode', default='attack', help='Mode: attack or train or scatter')	
@click.option('--key-file', default=None, help='File containing key')
@click.option('--num-samples', default=20, help='Number of samples to use in attack')
@click.option('--model-file', default='./models/model.h5', help='Model checkpoint file')
@click.option('--traces-file', default='./all_traces.npy', help='File to collect all traces in')
@click.option('--save-plots', default=None, help='Save plots')
def main(target, data_folder, mode, key_file, num_samples, model_file, traces_file, save_plots):
    if mode == 'attack':
        print('Running attack')	
        stats = get_attack_stats(data_folder, target, key_file, num_samples)
        print("Key ranks in attack trials:")
        print(stats)
        print("Median key rank (log2): {}".format(np.median(stats)))
    elif mode == 'train':
        if target not in list(net_params.keys()):
            raise ValueError('Target must be one of: {}'.format(list(net_params.keys())))
        net_parameters = net_params[target]
        depth, q1, q2, outputs, D, _ = net_parameters['depth'], net_parameters['q1'], net_parameters['q2'], net_parameters['outputs'], net_parameters['D'], net_parameters['netfile']
        train.train(data_folder, traces_file, model_file, N=outputs, D=D, q1=q1, q2=q2)
    elif mode == 'scatter':
        h1, h2, model1, model2, Xt, Yt, mse2, enc = scatter.test()
        # draw a figure with two subplots, one for h1 and one for h2
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(h1)
        ax[0].set_title('Validation loss Model 1')
        ax[1].plot(h2)
        ax[1].set_title('Validation loss Model 2')
        if save_plots is not None:
            if save_plots.endswith('.png'):
                plt.savefig(save_plots)
            else:
                if save_plots.endswith('.tex'):
                    tikzplotlib.save(save_plots)
        plt.show()

    else:
        raise ValueError('Mode must be one of: attack or train')

if __name__ == '__main__':
    main()
    print('Done')
    exit(0)



