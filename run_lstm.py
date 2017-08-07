from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import re
from keras.models import Sequential
from keras.layers import Dense, Embedding, Dropout
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from math import ceil

print ('load text documents in shape of lists of comments and lists of labels') 

def cleanup_text(doc_text):
    #doc_text = re.sub("[^a-zA-Z]"," ", doc_text)    
    doc_text = re.sub(" http[.s]*\:.* ", " url ", doc_text)
    doc_text = re.sub("[\r\n]", " ", doc_text)
    return doc_text

# load data
train_comments = []
train_labels = []
with open('train_comments.csv', 'rb') as f:
    for line in f:
        sep_loc = line.rfind(bytes(',','utf-8'))
        train_comments.append(cleanup_text(line[:sep_loc].decode('utf-8')).lower())
        #print(train_comments[-1])
        train_labels.append(int(line[sep_loc+1:]))
print ('there are a total of %d train comments in dataset ' % len(train_comments))
print(len(train_labels))

test_comments = []
test_labels = []
with open('test_comments.csv', 'rb') as f:
    for line in f:
        sep_loc = line.rfind(bytes(',','utf-8'))
        test_comments.append(cleanup_text(line[:sep_loc].decode('utf-8')).lower())
        test_labels.append(int(line[sep_loc+1:]))
print ('there are a total of %d test comments in dataset ' % len(test_comments))
print(len(test_labels))

MAX_NB_WORDS = 50000
MAX_SEQUENCE_LENGTH = 200
batch_size = 32
valid_iter = 500

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train_comments)
x_train = tokenizer.texts_to_sequences(train_comments)
x_test = tokenizer.texts_to_sequences(test_comments)

y_train = to_categorical(np.asarray(train_labels, dtype='int8'), num_classes=9)
y_test = to_categorical(np.asarray(test_labels, dtype='int8'), num_classes=9)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

print('Pad sequences')
# use 10% for validation
np.random.seed(123)
idx = np.random.permutation(len(x_train))
train_idx = idx[:int(len(x_train)*0.9)]
val_idx = idx[int(len(x_train)*0.9):]
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_val = x_train[val_idx,:]
x_train = x_train[train_idx,:]
y_val = y_train[val_idx,:]
y_train = y_train[train_idx,:]
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

print('Shape of trainset X and Y:', x_train.shape, y_train.shape)
print('Shape of textset X and Y:', x_test.shape, y_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, 256, mask_zero=True))
model.add(Dropout(0.1))
#model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2,return_sequences=False))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(9, activation='softmax'))

# try using different optimizers and different optimizer configs
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
best_val_score = float('inf')
print('Train...')

epochs = 4 
n_batches = ceil(x_train.shape[0]/batch_size)
best_val_acc = 0
for e in range(epochs):
    for i in range(n_batches):
        batch_start = i*batch_size
        batch_end = (i+1)*batch_size
        if batch_end > x_train.shape[0]:
            batch_end = x_train.shape[0]
            batch_start = batch_end -batch_size
        ret = model.train_on_batch(x_train[batch_start:batch_end], 
                                   y_train[batch_start:batch_end])
        print('{}: {}\t\t{}:{}'.format(model.metrics_names[0], ret[0], model.metrics_names[1], ret[1]),
              end='\r')
    	
        if (i+1) % valid_iter==0:
             score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
             if acc > best_val_acc:
                 print('\n****Better validation found, val loss: {} \t val acc: {}'.format(score, acc))
                 best_val_acc = acc

                 # test
                 score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
                 print('\n****Corresponding Test acc: {}'.format(acc))
   
    score, acc = model.evaluate(x_val, y_val,
                            batch_size=batch_size)
    if acc > best_val_acc:
        print('\n****Better validation found, val loss: {} \t val acc: {}'.format(score, acc))
        best_val_acc = acc

        # test
        score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
        print('\n****Corresponding Test acc: {}'.format(acc))
