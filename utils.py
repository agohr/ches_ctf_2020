import numpy as np
from math import log2


def to_nibbles(arr, width=32, axis=None):
    s = arr.shape
    if (axis is None):
        axis = len(s) - 1
    X = np.frombuffer(np.ndarray.tobytes(arr), dtype=np.uint8)
    X0 = X & 0xf
    X1 = (X >> 4) & 0xf
    res = np.concatenate([X0, X1])
    s_neu = list(s)
    s_neu[axis] = s[axis] * width // 4
    res = res.reshape(s_neu)
    return(res)


def to_bitorder(arr, width=32):
    n = arr.shape[1]
    res = np.zeros((arr.shape[0], width), dtype=np.uint32)
    v = np.arange(n, dtype=np.uint32)
    v = 2**v
    for i in range(width):
        tmp = (arr >> i) & 1
        tmp = tmp * v
        tmp = np.sum(tmp, axis=1)
        res[:, i] = tmp
    return(res)


def hw(x, width=32):
    s = x.shape
    res = np.zeros(s, dtype=np.uint8)
    for i in range(width):
        res = res + ((x >> i) & 1)
    return(res)


def shuffle_together(l):
    state = np.random.get_state()
    for x in l:
        np.random.set_state(state)
        np.random.shuffle(x)


def load_data(s, x):
    flist = [s + str(i) + '.npz' for i in x]
    data = [np.load(x) for x in flist]
    X = np.concatenate([x['traces'] for x in data])
    keys = np.concatenate([x['msk_keys'] for x in data])
    ukeys = np.concatenate([x['umsk_keys'] for x in data])
    tweaks = np.concatenate([x['nonces'] for x in data])
    seeds = np.concatenate([x['seeds'] for x in data])
    return(X, keys, ukeys, tweaks, seeds)


def uncompress_data(flist, traces_outfile):
    data = [np.load(x) for x in flist]
    #X = np.concatenate([x['traces'] for x in data]);
    keys = np.concatenate([x['msk_keys'] for x in data])
    ukeys = np.concatenate([x['umsk_keys'] for x in data])
    tweaks = np.concatenate([x['nonces'] for x in data])
    seeds = np.concatenate([x['seeds'] for x in data])
    X0 = data[0]['traces']
    n = len(X0)
    m = X0.shape[1]
    f = open(traces_outfile, 'ab')
    for i in range(len(data)):
        data[i]['traces'].tofile(f)
    f.close()
    X = np.memmap(traces_outfile, dtype=np.int16,
                  shape=(len(flist) * n, m), mode='r')
    return(X, keys, ukeys, tweaks, seeds)


def get_samples(X, Y, subset, batch_size=64):
    c = np.random.choice(subset, batch_size, replace=False)
    return(X[c], Y[c])


def gen_samples(Xl, Y, subset, batch_size=64, ram_bound=10**9):
    X = Xl[0]
    n = len(X[0])
    counter = 0
    while(True):
        if (counter == 0):
            fn, shape = X.filename, X.shape
            Xl[0] = np.memmap(fn, dtype=np.int16, shape=shape, mode='r')
            #del X;
        if (counter * batch_size * n > ram_bound):
            counter = 0
        else:
            counter = counter + 1
        yield(get_samples(Xl[0], Y, subset, batch_size=batch_size))


def cyclic_lr(num_epochs, high_lr, low_lr):
    hlr = log2(high_lr)
    llr = log2(low_lr)

    def res(i): return 2 ** (llr + ((num_epochs-1) - i %
                                    num_epochs)/(num_epochs-1) * (hlr - llr))
    return(res)
