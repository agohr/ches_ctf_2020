from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Add, Conv1D, Reshape, AveragePooling1D, Flatten, Concatenate
from keras.regularizers import l2


def make_net(outputs=176, width=None, q1=100, q2=650, depth=2, reg_term=10**-5, output_activation=None):
    if (width is None):
        width = outputs
    p = q1 * q2
    inp = Input(shape=(p,))
    res1 = Reshape((q1, -1))(inp)
    bn = BatchNormalization()(res1)
    conv = Conv1D(width, 1, activation='relu', padding='same',
                  kernel_regularizer=l2(reg_term))(bn)
    shortcut = conv
    for i in range(depth):
        bn = BatchNormalization()(conv)
        conv = Conv1D(width, 3, activation='relu', padding='same',
                      kernel_regularizer=l2(reg_term))(bn)
        conv = Add()([conv, shortcut])
        shortcut = conv
    if (output_activation is None):
        out = AveragePooling1D(pool_size=q1)(conv)
        out = Flatten()(out)
    else:
        out = Dense(outputs, activation=output_activation)(
            BatchNormalization(Flatten()(conv)))
    model = Model(inputs=inp, outputs=out)
    return(model)


def make_masked_model(N=100, width=None, q1=100, q2=625, depth=10, D=3):
    S = 4*D
    nets = [make_net(outputs=N, width=width, q1=q1, q2=q2, depth=depth)
            for i in range(S)]
    inp = Input(shape=(q1*q2,))
    nets = [net(inp) for net in nets]
    out = Concatenate()(nets)
    model = Model(inputs=inp, outputs=out)
    return(model)
