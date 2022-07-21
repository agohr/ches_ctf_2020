import numpy as np
import utils

import tikzplotlib

import net_ches_2020 as nc

from matplotlib import pyplot as plt

from scipy.stats import norm
from math import log2, sqrt

from tqdm import tqdm

from pathlib import Path

from interface.spook import clyde_encrypt

D = 3

p2 = 2 ** np.arange(32, dtype=np.uint32)


def convert_to_binary(v, n=32):
    res = np.zeros((len(v), n), dtype=np.uint8)
    for i in range(n):
        res[:, i] = (v >> i) & 1
    return(res)


def extract_share(v, mbin, best_guess_only=True):
    tmp = np.linalg.lstsq(mbin, v)[0]
    tbin = tmp > 0.5
    res = np.sum(p2 * tbin)
    if (not best_guess_only):
        res = (res, tmp)
    return(res)


def guess_shares(Z, masks, k=4*D, weights=None):
    n = len(masks)
    if (weights is None):
        weights = np.ones(k * n)
    mbin = convert_to_binary(masks)
    res = np.zeros((len(Z), k), dtype=np.uint32)
    conf = np.zeros((len(Z), k, 32))
    for i in range(k):
        a, b = n * i, n * (i+1)
        for j in range(len(Z)):
            res[j][i], conf[j][i] = extract_share(
                Z[j][a:b] / weights[a:b], (mbin.T / weights[a:b]).T, best_guess_only=False)
    return(res, conf)


def guess_bit_probabilities(conf, eps=0):
    n = len(conf)
    c = np.maximum(conf, eps)
    c = np.minimum(c, 1-eps)
    c2 = [c[:, (i, D+i, 2*D+i, 3*D+i), :].reshape(n, -1) for i in range(D)]
    tmp1 = np.arange(2**D, dtype=np.uint32)
    tmp2 = utils.hw(tmp1) & 1
    p0 = np.zeros(128)
    for i in tmp1[tmp2 == 1]:
        ptmp = np.ones(128)
        b = np.array([(i >> j) & 1 for j in range(D)])
        for j in range(D):
            #ptmp0 = c2[j]; ptmp1 = 1 - c2[j];
            ptmp0 = 1 - c2[j]
            ptmp1 = c2[j]
            ptmp = ptmp * (b[j] * ptmp1 + (1 - b[j]) * ptmp0)
        p0 = p0 + ptmp
    return(p0)


def evaluate_sbox_guess(bp1, m, tweaks, i, x, eps=1e-100, weights=None):
    if (weights is None):
        weights = np.ones(128)
    n = len(bp1)
    #cost_in = np.log2(np.maximum(np.array([bp, 1 - bp]), eps));
    #cost_out = np.log2(np.maximum(np.array([1 - bp1, bp1]),eps));
    k0 = x & 1
    k1 = (x >> 1) & 1
    k2 = (x >> 2) & 1
    k3 = (x >> 3) & 1
    msk = 0xffffffff ^ (1 << i)
    #kg = key_guess & msk;
    kg = np.zeros(4, dtype=np.uint32)
    kg = kg ^ (k0 << i, k1 << i, k2 << i, k3 << i)
    #kgbin = convert_to_binary(kg.flatten()).flatten();
    state = clyde_encrypt(m, tweaks.T, kg, num_steps=1, nr=0)
    state = np.array(state, dtype=np.uint32).T
    sbin = convert_to_binary(state.flatten()).reshape(n, -1)
    #cost = np.sum(cost_in[0][kgbin == 0]) + np.sum(cost_in[1][kgbin == 1]) + np.sum(cost_out[0].flatten()[sbin.flatten() == 0]) + np.sum(cost_out[1].flatten()[sbin.flatten() == 1]);
    #cost = np.linalg.norm(kgbin - (1 - bp)) + np.linalg.norm(sbin - bp1);
    l = [i, i+32, i+64, i+96]
    cost = np.linalg.norm((sbin[:, l] - bp1[:, l]) * weights[l])
    #cost = -np.corrcoef(sbin.flatten(), bp1.flatten())[0][1];
    #cost = -np.sum(cost_out[0].flatten()[sbin.flatten() == 0]) - np.sum(cost_out[1].flatten()[sbin.flatten() == 1]);
    return(cost)


def recover_key(X, masks, tweaks, model1=None, batch_size=100, weights=None):
    n = len(X)
    #Z = model.predict(X,batch_size=batch_size);
    if (model1 is None):
        Z1 = np.copy(X)
    else:
        Z1 = model1.predict(X, batch_size=batch_size)
    #shares, conf = guess_shares(Z, masks);
    shares1, conf1 = guess_shares(Z1, masks, k=4*D)
    #bp = guess_bit_probabilities(conf);
    #bp = combine_bit_probabilities(conf);
    bp1 = guess_bit_probabilities(conf1)
    #weights = np.repeat(np.array([(i+100) for i in range(32)]),4);
    #res = bp.reshape(-1,32) < 0.5; key_guess = np.sum(p2 * res, axis=1).astype(np.uint32);
    #print([hex(x) for x in key_guess]);
    #key_guess = np.zeros(4,dtype=np.uint32);
    vals = np.zeros((32, 16))
    for i in range(32):
        for j in range(16):
            vals[i][j] = evaluate_sbox_guess(bp1, np.zeros(
                4, dtype=np.uint32), tweaks, i, j, weights=weights)
    return(vals)


def reassemble_key(nibbles):
    key = np.zeros(4, dtype=np.uint32)
    bits = np.array([nibbles & 1, (nibbles >> 1) & 1, (nibbles >> 2)
                    & 1, (nibbles >> 3) & 1], dtype=np.uint32).T
    for i in range(32):
        key = key ^ (bits[i] << i)
    return(key)


def deassemble_key(key):
    nibbles = np.zeros(32, dtype=np.uint8)
    for i in range(32):
        nibbles[i] = np.dot((key >> i) & 1, np.array([1, 2, 4, 8]))
    return(nibbles)


def combine_bit_probabilities(conf):
    n = len(conf)
    bp = guess_bit_probabilities(conf)
    m = np.mean(bp, axis=0)
    h0 = norm.pdf(m, loc=p0mean, scale=p0s/sqrt(n))
    h1 = norm.pdf(m, loc=1-p0mean, scale=p0s/sqrt(n))
    p0 = h0/(h0+h1)
    return(p0)

# Guess the shares using each trace; derive guesses for the keys; look for bit biases in the key guesses in order to combine information across traces.


def run_attack(Z, masks, return_confidence=False):
    conf = np.zeros((4, 32))
    n = len(Z)
    k = np.zeros((n, 4), dtype=np.uint32)
    guess = np.zeros(4, dtype=np.uint32)
    s, c = guess_shares(Z, masks)
    for i in range(4):
        k[:, i] = s[:, 3*i] ^ s[:, 3*i+1] ^ s[:, 3*i+2]
    for i in range(4):
        tmp = convert_to_binary(k[:, i])
        conf[i] = np.sum(tmp, axis=0)
        res = conf[i] > (n//2)
        guess[i] = np.sum(p2 * res)
    if (return_confidence):
        guess = (guess, conf)
    return(guess)


def log_key_rank(Z, masks, tweaks, key, num_bins=100):
    v = recover_key(Z, masks, tweaks)
    val_best_guess = np.min(v, axis=1)
    for i in range(32):
        v[i] = v[i] - val_best_guess[i]
    nibbles = deassemble_key(key)
    val_true_key = 0
    for i in range(32):
        val_true_key += v[i][nibbles[i]]
    bins = np.zeros(num_bins + 1, dtype=np.float64)
    v = num_bins * v / max(val_true_key, 1e-10)
    bins[0] = 1
    for i in range(32):
        tmp = np.zeros(num_bins + 1)
        for j in range(num_bins):
            for k in range(16):
                s = v[i, k] + j
                if (s <= num_bins):
                    tmp[int(s)] += bins[j]/16
        bins = tmp
    rank = np.log2(np.sum(bins)) + 128
    return(rank)


def stats_log_key_rank(Z, masks, tweaks, key, samples, n=20, num_bins=100):
    results = np.zeros(n)
    for i in tqdm(range(n)):
        choice = np.random.choice(
            np.arange(len(Z), dtype=np.uint32), samples, replace=False)
        results[i] = log_key_rank(
            Z[choice], masks, tweaks[choice], key, num_bins=num_bins)
    return(results)


def calc_log_key_ranks(Z, masks, tweaks, key, num_samples, num_bins=100):
    ranks = np.zeros(len(num_samples))
    for i in tqdm(range(len(num_samples))):
        ranks[i] = log_key_rank(Z[:num_samples[i]], masks,
                                tweaks[:num_samples[i]], key, num_bins=num_bins)
    return(ranks)


def plot_log_key_ranks(Z, masks, tweaks, key, num_samples, num_bins=100, save_path=None, tex=None):
    ranks = calc_log_key_ranks(
        Z, masks, tweaks, key, num_samples, num_bins=num_bins)
    plt.plot(num_samples, ranks)
    plt.xlabel('Number of samples')
    plt.ylabel('Log key rank')
    if (save_path is not None):
        plt.savefig(save_path)
    if (tex is not None):
        tikzplotlib.save(tex)
    plt.show()


def output_probabilities(Z, masks, tweaks, show_guess=False):
    p = np.zeros((16, 256))
    n = len(Z)
    v = recover_key(Z, masks, tweaks)
    v = v * v
    guess = reassemble_key(np.argmin(v, axis=1))
    if (show_guess):
        print('#', [hex(x) for x in guess])
    def rank2nibble0(r): return (r & 0x1) | (((r >> 0x2) & 0x1) << 1) | (((r >> 0x4)
                                                                          & 0x1) << 2) | (((r >> 0x6) & 0x1) << 3)

    def rank2nibble1(r): return rank2nibble0(r >> 1)
    for i in range(0, 32, 2):
        #p[i//2] = -np.array([x+y for x in v[i] for y in v[i+1]]);
        p[i//2] = -np.array([v[i][rank2nibble0(j)] + v[i+1]
                            [rank2nibble1(j)] for j in range(256)])
    return(p)


def print_probabilities(Z, tweaks, masks):
    p = output_probabilities(Z, masks, tweaks, show_guess=True)
    msk = [(0x3 << (2*i)) | (0x3 << (2*i + 32)) | (0x3 << (2*i + 64))
           | (0x3 << (2*i + 96)) for i in range(16)]
    for i in range(len(msk)):
        print('{:x}:{}'.format(msk[i], ','.join(f'{x:.04f}' for x in p[i])))


def ctf_attack(X, tweaks, N=100, q1=100, q2=625, netfile='./models/model_hw_sbox_masks3.h5'):
    masks = np.load('./npy_files/testmasks.npy')
    model = nc.make_masked_model(N=N, q1=q1, q2=q2, D=D)
    model.load_weights(netfile)
    print('#Attacke beginnt')
    Z = model.predict(X, batch_size=10)
    print_probabilities(Z, tweaks, masks)
    print('#Attacke beendet.')


def get_predictions(X, N=100, q1=100, q2=625, netfile='./models/model_hw_sbox_masks3.h5'):
    model = nc.make_masked_model(N=N, q1=q1, q2=q2, D=D)
    model.load_weights(netfile)
    Z = model.predict(X, batch_size=10)
    return(Z)


def run_neural_network(data_dir, N=100, q1=100, q2=625, netfile='./models/model_hw_sbox_masks3.h5'):
    # process all the npz files in data_dir and return the predictions
    data_path = Path(data_dir)
    data_files = [x for x in data_path.iterdir() if x.suffix == '.npz']
    results = []
    for file in data_files:
        X = np.load(file)['traces']
        Z = get_predictions(X, N=N, q1=q1, q2=q2, netfile=netfile)
        results.append(Z)
    results = np.concatenate(results, axis=0)
    return(results)

def get_tweaks(data_dir):
    data_path = Path(data_dir)
    data_files = [x for x in data_path.iterdir() if x.suffix == '.npz']
    tweaks = []
    for file in data_files:
        X = np.load(file)['nonces']
        tweaks.append(X)
    tweaks = np.concatenate(tweaks, axis=0)
    return(tweaks)


def read_key(filename):
    with open(filename, 'rb') as f:
        key = f.read()
    # key contains a hexadecimal value of length 16 bytes
    # remove newlines from the string
    key = key.replace(b'\n', b'')
    # chop it into 4 values of size 4 bytes
    key = [key[i:i+8] for i in range(0, len(key), 8)]
    # turn this into an array of np.uint32
    key = np.array([int(x, 16) for x in key]).astype(np.uint32)
    key = key[::-1]
    return(key)


def scatter_plot(samples, stats, tex=None):
    plt.scatter(samples, np.median(stats, axis=1))
    plt.scatter(samples, np.log2(np.mean(2 ** stats, axis=1)))
    plt.scatter(samples, np.min(stats, axis=1))
    plt.scatter(samples, np.max(stats, axis=1))
    plt.legend(['Median rank', 'Mean rank', 'Min rank', 'Max rank'])
    plt.xlabel('Number of samples')
    plt.ylabel('Key rank (log2)')
    if (tex is not None):
        tikzplotlib.save(tex)
    plt.show()
