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

"""
This files allows to derive all the intermediate states of a masked 
Clyde execution. It takes as inputs key shares, PRNG state and a plaintext.

It reimplements to on board functions in embedded_src/spook_masked/ which
are actually used by the MCU.
"""

import numpy as np
from interface.utils import get_random_tape, umask
bor = np.bitwise_or
bxor = np.bitwise_xor
band = np.bitwise_and

seed_tmp = None


def clyde128_encrypt_masked(state, t, key, seed, D, Nr=6, step=2):
    """
        This function simulates the behavior of 
        multiple executions of Nc masked clyde128

        inputs:
            - state (4,Nc) matrix with each column being a plaintext of an execution
            - t (4,Nc) matrix with each column being a tweak of an execution
            - key ((4*D),Nc)  matrix with each column being a masked key of an execution
            - seed (4,Ns) with each column being the state of the PRNG seed at the beginning of the encryption

            - Nr: number of rounds to simulate
            - step: on what step to stop

        output:
            - (Nc,4*D) where each column is the masked state of the corresponding inputs.

    """
    global seed_tmp
    seed_tmp = seed
    tk = np.array([[t[0], t[1], t[2], t[3]],
                   [t[0] ^ t[2], t[1] ^ t[3], t[0], t[1]],
                   [t[2], t[3], t[0] ^ t[2], t[1] ^ t[3]]], dtype=np.uint32)
    masked_state = key.copy()
    XORLS_MASK(masked_state, tk[0], D=D)
    XORLS_MASK(masked_state, state, D=D)
    off = 0x924
    lfsr = 0x8
    for s in range(0, Nr):
        sbox_layer_masked(masked_state[0:D],
                          masked_state[D:D*2],
                          masked_state[D*2:D*3],
                          masked_state[D*3:D*4], refresh_flag=1, D=D)
        if s == (Nr-1) and step == 0:
            return masked_state.T
        lbox_masked(masked_state, D=D)
        XORCST_MASK(masked_state, lfsr, D=D)
        b = lfsr & 0x1
        lfsr = (lfsr ^ (b << 3) | b << 4) >> 1  # update LFSR
        sbox_layer_masked(masked_state[0:D],
                          masked_state[D:D*2],
                          masked_state[D*2:D*3],
                          masked_state[D*3:D*4], D=D)
        if s == (Nr-1) and step == 1:
            return masked_state.T
        lbox_masked(masked_state, D=D)
        XORCST_MASK(masked_state, lfsr, D=D)
        b = lfsr & 0x1
        lfsr = (lfsr ^ (b << 3) | b << 4) >> 1  # update LFSR
        off = off >> 2

        masked_state ^= key
        XORLS_MASK(masked_state, tk[off & 0x3], D=D)
    return masked_state.T

###############
# Various operation used by Clyde (see C code for more detailed)
###############


def XORLS_MASK(DEST, OP, D):
    """
        Performs addition of the unmaksed value OP
        with the shared DEST
    """
    DEST[0, :] ^= OP[0]
    DEST[1*D, :] ^= OP[1]
    DEST[2*D, :] ^= OP[2]
    DEST[3*D, :] ^= OP[3]


def XORCST_MASK(DEST, LFSR, D):
    """ 
        Performs round constant addition
        within the masked state
    """
    DEST[0] ^= (LFSR >> 3) & 0x1
    DEST[D] ^= (LFSR >> 2) & 0x1
    DEST[2*D] ^= (LFSR >> 1) & 0x1
    DEST[3*D] ^= (LFSR >> 0) & 0x1


def add_shares(out, a, b, D):
    """ 
        Performs addition between two sharing a and b
        and store the result in out.
    """
    for i in range(0, D):
        out[i] = a[i] ^ b[i]


def add_clyde128_masked_state(out, a, b, D):
    """ 
        Performs addition of two Clyde states and store 
        the result in out.
    """
    for d in range(D):
        for i in range(4):
            j = (i*D)+d
            out[j] = a[j] ^ b[j]


def refresh(shares, D):
    """
        Performs SNI refresh on the sharing it is implemented 
        up to 8 shares
    """
    if D < 4:
        refresh_block_j(shares, 1, D=D)
    elif D < 9:
        refresh_block_j(shares, 1, D=D)
        refresh_block_j(shares, 3, D=D)
    else:
        raise Exception(
            "SNI refresh is not implemented for D = %d, max D=8" % (D))


def refresh_block_j(shares, j, D):
    global seed_tmp
    for i in range(D):
        r, seed_tmp = get_random_tape(seed_tmp, 1)
        shares[i] ^= r[0]
        shares[(i+j) % D] ^= r[0]


def mult_shares(out, a, b, D):
    """
        performs ISW multiplication on two sharings a and b. 
        Stores the result in out.
        Takes randomness from the prng state seed_tmp.
    """
    global seed_tmp

    for i in range(D):
        out[i, :] = a[i, :] & b[i, :]

    for i in range(D):
        for j in range(i+1, D):
            rng, seed_tmp = get_random_tape(seed_tmp, 1)
            s = rng[0, :]
            tmp = (a[i, :] & b[j, :]) ^ s
            sp = tmp ^ (a[j, :] & b[i, :])
            out[i, :] ^= s
            out[j, :] ^= sp


def sbox_layer_masked(a, b, c, d, D, refresh_flag=0):
    """
        Applies inplace sbox to the inputs sharings a,b,c,d
        if refresh_flag, a refresh is inserted after the 
        first XOR of the Sbox according to Tornado tool. 
    """
    y0 = np.zeros(a.shape, dtype=np.uint32)
    y1 = np.zeros(a.shape, dtype=np.uint32)
    y3 = np.zeros(a.shape, dtype=np.uint32)
    tmp = np.zeros(a.shape, dtype=np.uint32)

    mult_shares(tmp, a, b, D=D)
    y1[:] = tmp ^ c
    if refresh_flag:
        refresh(y1, D=D)
    mult_shares(tmp, d, a, D=D)
    y0[:] = tmp ^ b
    mult_shares(tmp, y1, d, D=D)
    y3[:] = tmp ^ a
    mult_shares(tmp, y0, y1, D=D)
    c[:] = tmp ^ d

    a[:] = y0
    b[:] = y1
    d[:] = y3


def lbox_masked(masked_state, D):
    """
    Applies lbox to a masked clyde state. Because it is linear, it is a share-wise
    operation.
    """
    for i in range(D):
        masked_state[(0*D) + i], masked_state[(1*D) +
                                              i] = lbox(masked_state[(0*D) + i], masked_state[(1*D)+i])
        masked_state[(2*D) + i], masked_state[(3*D) +
                                              i] = lbox(masked_state[(2*D) + i], masked_state[(3*D)+i])


def lbox(x, y):
    """
        In place Clyde lbox
    """
    a = x ^ rotr(x, 12)
    b = y ^ rotr(y, 12)
    a = a ^ rotr(a, 3)
    b = b ^ rotr(b, 3)
    a = a ^ rotr(x, 17)
    b = b ^ rotr(y, 17)
    c = a ^ rotr(a, 31)
    d = b ^ rotr(b, 31)
    a = a ^ rotr(d, 26)
    b = b ^ rotr(c, 25)
    a = a ^ rotr(c, 15)
    b = b ^ rotr(d, 15)
    return (a, b)


def rotr(x, c):
    return (x >> c) | ((x << (32-c)) & 0xFFFFFFFF)


if __name__ == "__main__":
    Nc = 100
    D = 2
    # random seeds
    seeds = np.random.randint(0, 2**32, (4, Nc), dtype=np.uint32)
    # random masked keys
    msk_key = np.random.randint(0, 2**32, (4*D, Nc), dtype=np.uint32)
    umsk_key = umask(msk_key.T)
    # random tweak
    tweak = np.random.randint(0, 2**32, (4, Nc), dtype=np.uint32)
    # random plaintext
    plaintext = np.random.randint(0, 2**32, (4, Nc), dtype=np.uint32)

    # output masked state
    msk_state = clyde128_encrypt_masked(plaintext, tweak, msk_key, seeds, D=D)
    umsk_state = umask(msk_state, D=D)
