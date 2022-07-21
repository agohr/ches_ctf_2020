# Try to learn the state after the first S-box layer.

import numpy as np
import net_ches_2020 as nc
import utils

from keras.layers import Input, Concatenate
from keras.models import Model, load_model

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler


from keras import backend as K

from os import urandom
from pathlib import Path

from interface import spook_masked

def train(data_folder, traces_file, model_filename, N=100, D=3, q1=100, q2=625):
    S = 4 * D
    batch_size = 128

    # find all npz files in the data_folder
    data_path = Path(data_folder)
    npz_files = [f for f in data_path.iterdir() if f.suffix == '.npz']

    if (N == 100):
        masks = np.load('./npy_files/testmasks.npy')
    else:
        masks = np.frombuffer(urandom(4 * N), dtype=np.uint32)
        np.save('masks.npy', masks)

    masks = np.tile(masks, S)

    nets = [nc.make_net(depth=10, q1=q1, q2=q2, outputs=N) for i in range(S)]

    inp = Input(shape=(q1*q2,))

    for i in range(len(nets)):
        nets[i] = nets[i](inp)

    out = Concatenate()(nets)
    net = Model(inputs=inp, outputs=out)
    net.compile(optimizer='adam', loss='mse', metrics=['acc'])

    X, keys, ukeys, tweaks, seeds = utils.uncompress_data(
        npz_files, traces_outfile=traces_file)

    Y = spook_masked.clyde128_encrypt_masked(
        np.zeros(4, dtype=np.uint32), tweaks.T, keys.T, seeds.T, D, Nr=1, step=0)

    Y = np.repeat(Y.flatten(), N).reshape(-1, S*N)
    Y = Y & masks

    n = len(Y)
    Y = utils.hw(Y)

    n = len(X)
    perm = np.random.permutation(np.arange(n, dtype=np.uint32))
    train_set = perm[0:n-n//10]
    val_set = perm[n-n//10:]

    check = ModelCheckpoint(model_filename, save_best_only=True,
                            monitor='val_loss', save_weights_only=True)
    rl = ReduceLROnPlateau(monitor='val_loss', patience=5,
                        factor=0.5, min_lr=0.00002)
    lrs = LearningRateScheduler(utils.cyclic_lr(10, 0.001, 0.00002))

    Xl = [X]

    net.fit(utils.gen_samples(Xl, Y, train_set, batch_size=batch_size), max_queue_size=5, epochs=800, steps_per_epoch=500,
            validation_data=utils.gen_samples(Xl, Y, val_set, batch_size=batch_size), validation_steps=50, callbacks=[check, lrs], verbose=2, workers=1)
