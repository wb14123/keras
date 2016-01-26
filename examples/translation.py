# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers.core import Activation, TimeDistributedDense, RepeatVector
from keras.layers.embeddings import Embedding
from keras.layers import recurrent
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


RNN = recurrent.LSTM
BATCH_SIZE = 128
LAYERS = 1

print('Generating data...')
# path = get_file('training-parallel-europarl-v7.tgz', origin="http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz")
en_text = ["abcdefg", "abcfg"]
fr_text = ["efg", "dfgg"]

tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(en_text + fr_text)
vocab_size = len(tokenizer.word_counts)
seq1 = pad_sequences(tokenizer.texts_to_sequences(en_text))
seq2 = pad_sequences(tokenizer.texts_to_sequences(fr_text))


print('Build model...')
HIDDEN_SIZE = seq2.shape[1]
model = Sequential()
# "Encode" the input sequence using an RNN, producing an output of HIDDEN_SIZE
# note: in a situation where your input sequences have a variable length,
# use input_shape=(None, nb_feature).
model.add(Embedding(vocab_size, HIDDEN_SIZE, mask_zero=True))
model.add(RNN(HIDDEN_SIZE))
# For the decoder's input, we repeat the encoded input for each time step
model.add(RepeatVector(HIDDEN_SIZE))
# The decoder RNN could be multiple layers stacked or a single layer
for _ in range(LAYERS):
    model.add(RNN(HIDDEN_SIZE, return_sequences=True))

# For each of step of the output sequence, decide which character should be chosen
model.add(TimeDistributedDense(vocab_size))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(seq1, seq2, batch_size=BATCH_SIZE)
