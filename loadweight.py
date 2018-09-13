import numpy as np
import librosa
import os.path
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import SGD, Adam, RMSprop,Adadelta,Adagrad
from scipy.io import wavfile
import mir_eval



def get_data(file):
    """Read in audio file and computes STFT."""
    y, sr_ = librosa.load(file)
    S = librosa.core.stft(y=y, n_fft=512)

    return np.transpose(S)



def reconstruct(file,spec1, spec2):
    y, sr_ = librosa.load(file)
    D = librosa.core.stft(y, n_fft=512)
    mag, phase = librosa.magphase(D)

    spec1 = np.transpose(spec1)
    spec2 = np.transpose(spec2)
    mask = np.abs(spec1)/(np.abs(spec1)+np.abs(spec2))


    rec_a = D*mask
    rec_b = D*(1 - mask)

    out_a = librosa.core.istft(rec_a)
    out_b = librosa.core.istft(rec_b)

    return out_a, out_b

if __name__ == "__main__":

    test = get_data('piano_jaychou.wav')
    image_test = test.imag
    test_spec = np.abs(test)

    model = Sequential()
    model.add(Dense(200, input_shape=(257,)))
    model.add(Activation('relu'))

    model.add(Dense(200))
    model.add(Activation('relu'))


    model.add(Dense(200))
    model.add(Activation('relu'))


    model.add(Dense(500))
    model.add(Activation('relu'))


    model.add(Dense(500))
    model.add(Activation('relu'))

    model.add(Dense(257))
    model.add(Activation('linear'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=Adagrad(),
                  metrics=['accuracy'])

    model.load_weights('weights_200_200_200_500_500.hdf5')

    out_spec = model.predict(test_spec)

    out_spec_1 = out_spec[:,0:257]
    out_spec_2 = out_spec[:, 257:514]

    out_1, out_2 = reconstruct('piano_jaychou.wav', out_spec_1, out_spec_2)

    wavfile.write('piano_jaychou_voice.wav', 22050, out_1)
    wavfile.write('piano_jaychou_music.wav', 22050, out_2)
