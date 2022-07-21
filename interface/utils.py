# MIT License
#
# Copyright (c) 2020 UCLouvain
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from interface.spook import shadow, bytes2state, state2bytes
import numpy as np
from interface.parameters import *

bor = np.bitwise_or
bxor = np.bitwise_xor
band = np.bitwise_and


def mask(k, D, PRGON=1):
    """
        is used to mask a given key

        k: is a byte array
        D: is the order
        PRGON: is PRNG active
    """
    uk = np.frombuffer(k, dtype=np.uint32)
    muk = np.zeros(len(uk)*D).astype(np.uint32)
    for i, k in enumerate(uk):
        acc = k
        for d in range(D-1):
            r = np.random.randint(0, (((2**32)-1)*PRGON)+1)
            muk[i*D + d] = r
            acc = np.bitwise_xor(r, acc)
        muk[i*D + (D-1)] = acc
    return np.ndarray.tobytes(muk)


def simple_refresh(out, inp, seed, D):
    """
        perform similar refresh as the one done on the MCU

        out: refreshed key
        inp: input key
        seed: state of the PRNG
    """
    for i in range(4):
        r = 0
        for d in range(0, (D-1)):
            s, seed = get_random_tape(seed, 1)
            out[(i*D)+d] = s ^ inp[(i*D)+d]
            r ^= s[0]
        out[(i*D)+(D-1)] = r ^ inp[(i*D)+(D-1)]


def umask(k, D):
    """"
        unmask N sharing
        N x (4*D) unmask N data in parallel
    """
    out = np.zeros((len(k[:, 0]), 4), dtype=np.uint32)
    for i in range(4*D):
        out[:, i//D] ^= k[:, i]
    return out


MAX = 16


def get_random_tape(seed, l):
    """
        return the randomness from the on-board PRNG
        which is a shadow in sponge mode.

        seed: the actual PRNG state. If a tuple (i,tab,prng_state) , then the PRNG
            is already initialized. If not, the PRNG is initialized.

        l: number of random values to return
    """
    if not type(seed) is tuple:
        # The PRNG is not initialized yet,
        # it is initialized with randomness from the t-function
        seed = np.uint32(seed)
        if isinstance(seed, np.ndarray) and seed.ndim == 2:
            rng = np.zeros((l, len(seed[0, :])), dtype=np.uint32)
            prng_state_core = [np.zeros((4, 4), dtype=np.uint32)
                               for i in range(len(seed[0, :]))]
            for i, state in enumerate(prng_state_core):
                state[0, :] = seed[:, i]
            prng_tab = np.zeros((MAX, len(seed[0, :])), dtype=np.uint32)
        else:
            rng = np.zeros(l, dtype=np.uint32)
            prng_tab = np.zeros((MAX, 1), dtype=np.uint32)
            prng_state_core = [np.zeros((4, 4), dtype=np.uint32)
                               for i in range(1)]
            for i, state in enumerate(prng_state_core):
                state[0, :] = seed

        prng_index = MAX
    else:
        # The PRNG is initalized,
        # getting back the actual index and LFSR state
        prng_index, prng_tab, prng_state_core = seed
        rng = np.zeros((l, len(prng_tab[0, :])), dtype=np.uint32)

    # dump l random words
    for i in range(l):
        if prng_index >= MAX:
            fill_table(prng_state_core, prng_tab)
            prng_index = 0
        rng[i] = prng_tab[prng_index]
        prng_index += 1

    if PRGON == 0:
        rng[:] = 0
    return rng, (prng_index, prng_tab, prng_state_core)


def fill_table(state_all, tab_all):
    for i in range(0, MAX, 8):
        for j, state in enumerate(state_all):
            state_all[j] = shadow(state)
            for n in range(8):
                tab_all[i+n, j] = state_all[j][n//4][n % 4]
