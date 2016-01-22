# -*- coding: utf-8 -*-

from keras.models import Sequential, Graph
from keras.layers.core import Activation, TimeDistributedDense
from keras.layers.embeddings import Embedding
from keras.layers import recurrent
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


EMBED_SIZE = 10
HIDDEN_SIZE = 20
LAYERS = 4
RNN = recurrent.LSTM
SEQ_FILE1 = "/Users/wangbin/training/europarl-v7.fr-en.en"
SEQ_FILE2 = "/Users/wangbin/training/europarl-v7.fr-en.fr"

print("Parse data...")
# en_text = tuple(open(SEQ_FILE1, 'r'))
# fr_text = tuple(open(SEQ_FILE2, 'r'))
en_text = ["abcdefg", "abcfg"]
fr_text = ["efg", "dfgg"]
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(en_text + fr_text)
vocab_size = len(tokenizer.word_counts)
seq1 = pad_sequences(tokenizer.texts_to_sequences(en_text))
seq2 = pad_sequences(tokenizer.texts_to_sequences(fr_text))
print(seq1)
print(seq2)
print(seq1.shape)
print(seq2.shape)

print('Build model...')
encoder = Sequential()
encoder.add(Embedding(vocab_size, EMBED_SIZE, mask_zero=True))
encoder.add(RNN(HIDDEN_SIZE, return_sequences=False))

# decoder = Sequential()
# decoder.add(RNN(HIDDEN_SIZE, return_sequences=True))
# decoder.add(TimeDistributedDense(vocab_size))
# decoder.add(Activation("softmax"))

model = Graph()
model.add_input("seq1", input_shape = (seq1.shape[1], ))
model.add_node(encoder, "encoder", input="seq1")

model.add_input("seq2", input_shape = (seq2.shape[1], ))
model.add_node(Embedding(vocab_size, EMBED_SIZE, mask_zero=True),"seq2_embed", input="seq2")

model.add_node(RNN(HIDDEN_SIZE, return_sequences=True), "seq2_rnn", inputs=["seq2_embed", "encoder"])
model.add_node(TimeDistributedDense(vocab_size), "seq2_dense", input="seq2_rnn")
model.add_node(Activation("softmax"), "seq2_active", input="seq2_dense")
model.add_output("output", input="seq2_active")

model.compile(loss='categorical_crossentropy', optimizer='adam')

print('Training...')
model.fit({"seq1": seq1, "seq2": seq2, "output": seq2}, show_accuracy=True)
