import random
import pandas as pd
from keras.layers import Dense, Input, Embedding, Conv1D, Dropout, MaxPooling1D, Bidirectional, LSTM
from keras.models import Model, load_model
import re
import unicodedata
from nltk.corpus import stopwords
import string
import emoji
from keras.layers import Dense, Dropout, Bidirectional, LSTM, dot, Flatten

# Shuffle
def shuffle(x, y):
    ind = [i for i in range(len(x))]
    random.shuffle(ind)
    x_tmp, y_tmp = [], []
    for i in ind:
        x_tmp.append(x[i])
        y_tmp.append(y[i])
    return x_tmp, y_tmp

def clean_text(text):
    text = re.sub(r'<.*?>', '', text) # 移除HTML標籤
    text = re.sub(r'[^\w\s]', '', text) # 移除數字和標點符號
    text = re.sub(r'\s+', ' ', text) # 移除多個空格
    text = emoji.demojize(text)
    text = text.lower() # 小寫轉換
    output = ""
    for char in text:
        if unicodedata.category(char) != "Lo":  # Lo表示中文字符
            output += char
    return output

def remove_stopwords(text_list):
    stop_words = set(stopwords.words('english')) #載入停用詞
    punctuation = set(string.punctuation)
    cleaned_texts = []
    for text in text_list:
        tokens = text.split()
        tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
        tokens = [token for token in tokens if token not in punctuation]
        cleaned_text = ' '.join(tokens)
        cleaned_texts.append(cleaned_text)
    return cleaned_texts

# Define model
def define_model(emb_dim, voc_size, embedding_matrix, NUM_LABELS):

    seq_input = Input(shape=(None,), name='Input_layer')
    emb = Embedding(voc_size, emb_dim, weights=[embedding_matrix], input_length=128, trainable=False)(seq_input) # Embedding
    conv1 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(emb)
    conv2 = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling1D(pool_size=2)(conv2)
    drop1 = Dropout(0.2)(pool1)
    
    # BiLSTM 雙向LSTM
    bilstm = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(drop1)
    
    # Attention
    attention_probs = Dense(128, activation='softmax', name='attention_vec')(bilstm)
    attention_mul = dot([bilstm, attention_probs], axes=[1, 1])
    
    flatten = Flatten()(attention_mul)
    drop2 = Dropout(0.2)(flatten)
    
    dense1 = Dense(128, activation='relu')(drop2)
    drop3 = Dropout(0.2)(dense1)
    output = Dense(NUM_LABELS, activation='softmax')(drop3)

    model = Model(inputs=seq_input, outputs=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model