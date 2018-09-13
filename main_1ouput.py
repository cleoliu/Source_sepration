import numpy as np
import librosa
import os.path
import sys
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import SGD, Adam, RMSprop, Adadelta, Adagrad
from keras.objectives import mean_absolute_error, kullback_leibler_divergence, mse_weiner
from keras.regularizers import l2, activity_l2
from scipy.io import wavfile
from mir_eval import separation as sp


def get_bss_result(s1, s2, es1, es2):
    """
    s1, s2: the targeted sources
    es1, es2: the esperated sources
    Return the (SDR, SIR, SAR)
    """
    size = min(len(s1), len(s2), len(es1), len(es2))

    sdr, sir, sar, perm = sp.bss_eval_sources(np.array([s1[1:size], s2[1:size]]),
                                              np.array([es1[1:size], es2[1:size]]))

    return sdr, sir, sar


def get_data(file):
    """Read in audio file and computes STFT."""
    y, sr_ = librosa.load(file)
    S = librosa.core.stft(y=y, n_fft=512)

    return np.transpose(S)


def reconstruct(file, spec1):
    y, sr_ = librosa.load(file)
    D = librosa.core.stft(y, n_fft=512)
    mag, phase = librosa.magphase(D)

    spec1 = np.transpose(spec1)

    mask = np.abs(spec1) / np.abs(D)

    rec_a = D * mask
    rec_b = D * (1 - mask)

    o_a = librosa.core.istft(rec_a)
    o_b = librosa.core.istft(rec_b)

    return o_a, o_b




def custom_obj(out_true, out_pred):

    loss = mse_weiner(out_true, out_pred)

    return loss


# def my_init(shape, name=None):
#     value = np.random.random(shape)
#     return K.variable(value, name=name)

if __name__ == '__main__':
    mixture = get_data('valid_train_piano_mixture.wav')
    instrument_1 = get_data('valid_train_piano.wav')

    mixture_test = get_data('piano_jaychou.wav')


    Norm = np.amax(np.abs(mixture))
    mixture_spec = np.abs(mixture) / Norm

    Norm = np.amax(np.abs(instrument_1))
    instrument_spec_1 = np.abs(instrument_1) / Norm

    Norm = np.amax(np.abs(mixture_test))
    mixture_test_spec = np.abs(mixture_test) / Norm


    # fit
    batch_size = 256
    nb_epoch = 250

    model = Sequential()
    model.add(Dense(500, input_shape=(257,)))
    # model.add(Dense(50, input_shape=(257,)))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(500))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(1000))
    model.add(Activation('relu'))

    model.add(Dense(257))
    model.add(Activation('linear'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    # checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

    model.fit(mixture_spec, instrument_spec_1,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1)

    model.save_weights("bestweights.hdf5", overwrite=True)
    # model.load_weights('weights.hdf5')
    out_spec = model.predict(mixture_test_spec)

    out_1, out_2 = reconstruct('piano_jaychou.wav', out_spec)

    wavfile.write('predicted_target.wav', 22050, out_1)
    wavfile.write('remain.wav', 22050, out_2)

