import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
import tool

test_data = pd.read_csv("test.csv")
train_data = pd.read_csv("train.csv")

# Configs
label_map = {}
index_map = {}
BATCH_SIZE = 64
EPOCHS = 5
MAXLEN = 128
emb_dim = 300
unit = 128

# Counting labels
labels = train_data['category'].unique()
for l in range(len(labels)):
    label_map[labels[l]] = l
    index_map[l] = labels[l]
NUM_LABELS = len(label_map)

if __name__ == '__main__':

    train_x, train_y = [], []
    test_x = []

    # clean
    train_data['headline'] = train_data['headline'].apply(tool.clean_text)
    test_data['headline'] = test_data['headline'].apply(tool.clean_text)
    train_data['short_description'] = train_data['short_description'].apply(tool.clean_text)
    test_data['short_description'] = test_data['short_description'].apply(tool.clean_text)

    for i in range(len(train_data)):
        train_x.append(str(train_data['headline'][i] + train_data['short_description'][i]))
        train_y.append(train_data['category'][i])

    for i in range(len(test_data)):
        test_x.append(test_data['headline'][i] + test_data['short_description'][i])
    
    # remove stop word
    # train_x = remove_stopwords(train_x)
    # test_x = remove_stopwords(test_x)

    print(f'Train x: {len(train_x)}, Train y: {len(train_y)}')

    # Tokenize
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_x)
    tokenizer.fit_on_texts(test_x)

    print(f'Token amount: {len(tokenizer.word_index)}')

    train_x, train_y = tool.shuffle(train_x, train_y)

    # Input encoding -> change word into number sequence
    train_seq = tokenizer.texts_to_sequences(train_x)
    test_seq = tokenizer.texts_to_sequences(test_x)

    # Padding -> pad every sentence into same length
    train_seq = pad_sequences(train_seq, maxlen=MAXLEN, truncating='post', padding='post')
    test_seq = pad_sequences(test_seq, maxlen=MAXLEN, truncating='post', padding='post')
    train_seq = np.array(train_seq)
    test_seq = np.array(test_seq)

    # Output encoding -> change label into number
    train_ans = []
    for i in range(len(train_y)):
        train_ans.append(label_map[train_y[i]])
    train_ans = np.array(train_ans)

    print(f'Train x: {len(train_seq)}, Train y: {len(train_ans)}')

    # 加載glove
    embeddings_index = {}
    f = open('./tmp/glove.6B.300d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    
    # 創建embedding_matrix
    embedding_matrix = np.zeros((len(tokenizer.word_index)+1, emb_dim))
    for word, i in tokenizer.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = tool.define_model(emb_dim, len(tokenizer.word_index)+1, embedding_matrix, NUM_LABELS)
    model.fit(train_seq, train_ans, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, validation_split=0.02)
    # model.save('model')

    # Inference stage --> Codes below can be separate into another python script
    # model = load_model('model')

    pred_ans = {'id' : [], 'category' : []}

    for i in tqdm(range(len(test_seq))):
        pred = model.predict(np.array([test_seq[i]]))
        ans = np.argmax(pred)
        # print(ans)
        pred_ans['id'].append(i)
        pred_ans['category'].append(index_map[ans])

    df = pd.DataFrame(pred_ans)
    df.to_csv('submission.csv',index=False)