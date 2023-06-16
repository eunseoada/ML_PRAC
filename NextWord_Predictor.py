# ---------------------------
# reference : https://thecleverprogrammer.com/2020/07/20/next-word-prediction-model/
# dataset : https://drive.google.com/file/d/1GeUzNVqiixXHnTl8oNiQ2W3CynX_lsu2/view
# local_dataaset : /Users/eunseo/Desktop/eunseo/ML_PRAC/NextWordPrediction
# algorithm : RNN in scikit-learn
# date : 2023.06.13 
# writter: eunseo.choi
# ----------------------------

import numpy as np
from nltk.tokenize import RegexpTokenizer
from keras.models import Sequential,load_model
from keras.layers import LSTM
from keras.layers.core import Dense,Activation
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import heapq

#------------
# Load data
#------------

path = '/Users/eunseo/Desktop/eunseo/ML_PRAC/DATA/NextWordPrediction/1661-0.txt' # local data path
text = open(path).read().lower()
# print('corpus length: ',len(text)) # num of words

#------------------------------------------------------------------------
# Split the words in text (without special characters. e.g. ?,!,#....)
#------------------------------------------------------------------------

'''
RegexpTokenizer's Parameter means : 

1. \w+ : split for words 
2. \s+ : split for space 
3. [.!?]\s+ : split for end of sentences

c.f) here are website to check regular expression

 >> https://regex101.com/r/fLntOd/1 

'''

tokenizer = RegexpTokenizer(r'\w+') 
words = tokenizer.tokenize(text)

# print(words)

#------------------------------------------------
# Remove duplicated words and sort in order 
#------------------------------------------------


unique_words = np.unique(words) 
unique_words_index = dict((c,i) for i,c in enumerate(unique_words))
print(unique_words_index)

#------------------------
# Feature engineering
#------------------------

'''

Convert number to words info to build feature matrix

Step 1. Define length of previous word to decide next word
Step 2. Create 2 array (x,y) : for storing features(x) and label(y)

'''
#---------
# Step 1
#---------

WORDS_LENGTH = 5
prev_words=[]
next_words=[]

for i in range(len(words)-WORDS_LENGTH):
    prev_words.append(words[i:i+WORDS_LENGTH])
    next_words.append(words[i+WORDS_LENGTH])

#---------
# Step 2
#---------


x = np.zeros((len(prev_words),WORDS_LENGTH,len(unique_words)),dtype=bool)
y = np.zeros((len(next_words),len(unique_words)),dtype=bool)

for i,each_words in enumerate(prev_words):
    for j,each_word in enumerate(each_words):
        
        x[i,j,unique_words_index[each_word]] =1
    y[i,unique_words_index[next_words[i]]] =1

# print(x.shape)
# print(prev_words)

#------------------------
# Build Model (RNN)
#------------------------

model = Sequential()
model.add(LSTM(128,input_shape=(WORDS_LENGTH,len(unique_words))))
model.add(Dense(len(unique_words)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
history = model.fit(x,y,validation_split=0.05,batch_size=128,epochs=2,shuffle=True).history

model.save('keras_next_word_model.h5')
pickle.dump(history,open(history.p,"wb"))
model = load_model('keras_next_word_model.h5')
history = pickle.load(open("history.p", "rb"))






