import numpy as np
import librosa
import os.path
import sys
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM, SimpleRNN, GRU
from keras.optimizers import SGD, Adam, RMSprop,Adadelta,Adagrad
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
    size = min( len(s1), len(s2), len(es1), len(es2) )

    sdr, sir, sar, perm = sp.bss_eval_sources(np.array( [s1[1:size],  s2[1:size]] ),
                        np.array( [es1[1:size], es2[1:size]] ))

    return sdr, sir, sar

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




def custom_obj(out_true, out_pred):

    loss = mse_weiner(out_true, out_pred)

    return loss

# def my_init(shape, name=None):
#     value = np.random.random(shape)
#     return K.variable(value, name=name)

if __name__ == '__main__':

    mixture = get_data('./timit_shift/10000/mixture_train.wav')
    instrument_1 = get_data('./timit_shift/10000/female_train.wav')
    instrument_2 = get_data('./timit_shift/10000/male_train.wav')
    mixture_dev = get_data('./timit_shift/10000/mixture_dev.wav')
    instrument_dev_1 = get_data('./timit_shift/10000/female_dev.wav')
    instrument_dev_2 = get_data('./timit_shift/10000/male_dev.wav')
    mixture_test = get_data('./timit_shift/10000/mixture_test.wav')
    instrument_test_1 = get_data('./timit_shift/10000/female_test.wav')
    instrument_test_2 = get_data('./timit_shift/10000/male_test.wav')




    Norm = np.amax(np.abs(mixture))
    mixture_spec = np.abs(mixture)/Norm

    Norm = np.amax(np.abs(instrument_1))
    instrument_spec_1 = np.abs(instrument_1)/Norm

    Norm = np.amax(np.abs(instrument_2))
    instrument_spec_2 = np.abs(instrument_2)/Norm

    Norm = np.amax(np.abs(mixture_dev))
    mixture_dev_spec = np.abs(mixture_dev)/Norm
    
    Norm = np.amax(np.abs(instrument_dev_1))
    instrument_dev_spec_1 = np.abs(instrument_dev_1)/Norm

    Norm = np.amax(np.abs(instrument_dev_2))
    instrument_dev_spec_2 = np.abs(instrument_dev_2)/Norm

    Norm = np.amax(np.abs(mixture_test))
    mixture_test_spec = np.abs(mixture_test)/Norm
    
    instrument_test_spec_1 = np.abs(instrument_test_1.real)

    instrument_test_spec_2 = np.abs(instrument_test_2)

    instrument_spec = np.concatenate((instrument_spec_1,instrument_spec_2), axis=1)

    instrument_dev_spec = np.concatenate((instrument_dev_spec_1, instrument_dev_spec_2), axis=1)

    instrument_test_spec = np.concatenate((instrument_test_spec_1, instrument_test_spec_2), axis=1)

    #fit
    batch_size = 256
    nb_epoch = 200

    Callback()

    model = Sequential()
    model.add(Dense(100, input_shape=(257,)))
    #model.add(Dense(50, input_shape=(257,)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.2))

    model.add(Dense(100))
    model.add(Activation('relu'))
    # model.add(Dropout(0.2))

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(514))
    model.add(Activation('linear'))

    model.summary()

    model.compile(loss='mse',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    #checkpointer = ModelCheckpoint(filepath="weights.hdf5", monitor='val_acc', verbose=1, save_best_only=True)

    model.fit(mixture_spec, instrument_spec,
                        batch_size=batch_size, nb_epoch=nb_epoch,
                        verbose=1,validation_data=(mixture_dev_spec,instrument_dev_spec))

    #model.save_weights("bestweights.hdf5", overwrite=True)
    #model.load_weights('weights.hdf5')
    out_spec = model.predict(mixture_test_spec)

    out_1, out_2 = reconstruct('./timit_shift/10000/mixture_test.wav',out_spec_1,out_spec_2)




    wavfile.write('./timit_shift/10000/predict_output_female.wav',22050, out_1)
    wavfile.write('./timit_shift/10000/predict_output_male.wav', 22050, out_2)

    out_instruemnt_1, sr_ = librosa.load('./timit_shift/10000/female_test.wav')
    out_instruemnt_2, sr_ = librosa.load('./timit_shift/10000/male_test.wav')
    minlength = np.minimum(out_1.shape[0], out_instruemnt_1.shape[0])
    out_instruemnt_1 = out_instruemnt_1[0:minlength]
    out_instruemnt_2 = out_instruemnt_2[0:minlength]
    #sdr, sir, sar = get_bss_result(librosa.core.istft(np.transpose(instrument_test_1)),librosa.core.istft(np.transpose(instrument_test_2)),out_1,out_2)
    #print "SDR: {}, SIR: {}, SAR: {}".format(sdr, sir, sar)
    (sdr, sir, sar, _) = sp.bss_eval_sources(out_instruemnt_1, out_1)
    print "SDR: {}, SIR: {}, SAR: {}".format(sdr, sir, sar)
    (sdr, sir, sar, _) = sp.bss_eval_sources(out_instruemnt_2, out_2)
    print "SDR: {}, SIR: {}, SAR: {}".format(sdr, sir, sar)