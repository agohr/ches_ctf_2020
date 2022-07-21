import numpy as np
from keras.layers import Input, Dense, BatchNormalization, Add
from keras.models import Model


# learn f^-1 given samples of applying f with some noise
def make_train_data(n, d, sigma, f, reverse=False):
    X = np.random.randint(0, 2, (n, d))
    Y = f(X)
    Y = Y + np.random.normal(0, sigma, Y.shape)
    if (reverse):
        return(Y, X)
    else:
        return(X, Y)


def make_resnet(num_in, num_out, width, depth, activation='relu', output_activation='sigmoid'):
    inp = Input(shape=(num_in,))
    bn = BatchNormalization()(inp)
    dense = Dense(width, activation=activation)(bn)
    for i in range(depth):
        shortcut = dense
        bn = BatchNormalization()(dense)
        dense = Add()([shortcut, Dense(width, activation=activation)(bn)])
    out = Dense(num_out, activation=output_activation)(
        BatchNormalization()(dense))
    model = Model(inputs=inp, outputs=out)
    return(model)

# train a model to invert f, first without scattershot encoding
# default is a linear model


def experiment_no_scatter(X, Y, Xt, Yt, epochs=10, model=None):
    if (model is None):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='mse')
    h = model.fit(X, Y, epochs=epochs, batch_size=5000,
                  validation_data=(Xt, Yt))
    return(h.history['val_loss'])


def experiment_scatter(X, Y, Xt, Yt, d, scatter_size, epochs=10, model=None):
    if (model is None):
        model = Model(inputs=inp, outputs=out)
        model.compile(optimizer='adam', loss='mse')
    enc = np.random.randint(0, 2, (scatter_size, d))
    Y2 = np.dot(enc, Y.T).T
    h = model.fit(X, Y2, epochs=epochs, batch_size=5000, validation_split=0.1)
    Z = model.predict(Xt, batch_size=5000)
    Z2 = np.zeros(Yt.shape)
    for i in range(len(Z)):
        Z2[i] = np.linalg.lstsq(enc, Z[i])[0]
    diff = Z2 - Yt
    mse = np.mean(diff * diff, axis=0)
    return(h.history['val_loss'], mse, enc)


def f1(X, sigma=0.1, n=100, reg_size=16):
    d = len(X[0])
    Y = np.zeros((len(X), n))
    # make the noise in our experiment into a random walk
    r = np.random.normal(0, sigma, Y.shape)
    for i in range(1, n):
        r[:, i] = r[:, i-1] + r[:, i]
    S = np.zeros((len(X), n))
    for i in range(1, n):
        index = (i*d)//n
        last_index = ((i-1)*d)//n
        k = index % reg_size
        S[:, i] = S[:, i-1] + X[:, index] - X[:, last_index]
    Y = r + S
    return(Y)

# try multiplication with a matrix over F2


def f2(X, A):
    Y = np.dot(X, A) & 1
    return(Y)


def make_matrix(s, t=0.1):
    A = np.random.rand(*s)
    A = A < t
    return(A)


def test():
    # create a 32x32 matrix A with all ones at and above the diagonal
    A = np.zeros((32, 32), dtype=np.uint32)
    for i in range(32):
        A[i, i:] = 1
    X, Y = make_train_data(10**6, 32, 0, lambda x: f2(x, A))
    Xt, Yt = make_train_data(10**4, 32, 0, lambda x: f2(x, A))
    model1 = make_resnet(32, 32, 100, 10, activation='relu',
                         output_activation='sigmoid')
    model2 = make_resnet(32, 100, 100, 10, activation='relu',
                         output_activation='relu')
    model1.compile(optimizer='adam', loss='mse')
    model2.compile(optimizer='adam', loss='mse')
    h1 = experiment_no_scatter(X, Y, Xt, Yt, epochs=100, model=model1)
    h2, mse2, enc = experiment_scatter(
        X, Y, Xt, Yt, 32, 100, epochs=100, model=model2)
    np.savez_compressed('data_scatter.npz', history_model1=h1,
                        history_model2=h2, mse2=mse2)
    return h1, h2, model1, model2, Xt, Yt, mse2, enc


def scatter_predict(X, enc, model):
    # predict f(X) using the trained model and the scattershot matrix enc
    Z = model.predict(X, batch_size=5000)
    Z2 = np.zeros(X.shape)
    for i in range(len(Z)):
        Z2[i] = np.linalg.lstsq(enc, Z[i])[0]
    return(Z2)
