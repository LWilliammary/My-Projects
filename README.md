**Project:** Chatbot (Covichatbot) 

**Repository Details:** Repository contains Covichatbot’s Data File, Notbook, Source File. The Covichatbot is trained to respond users input about coronavirus (COVID-19).

**Objectives **

-> User Interactions - trained to respond customers queries 
-> Helps in responding 24/7 

**Data** 

JSON file containing replies for each intent

**Instructions**

->Install Packages pip install tensorflow pip install keras -> Install all the required packages ->Import libraries

**Source File** 

source.py

**Notebook** 

Notebook.ipynb

**Python Code Snippet**

#load the json file and extract the required data

with open('Data.json') as file: data = json.load(file)

training_sentences = [] training_labels = [] labels = [] responses = []

for intent in data['intents']: for pattern in intent['patterns']: training_sentences.append(pattern) training_labels.append(intent['tag']) responses.append(intent['responses'])

if intent['tag'] not in labels:
    labels.append(intent['tag'])
num_classes = len(labels)

vocab_size = 1000 embedding_dim = 16 max_len = 20 oov_token = ""

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token) tokenizer.fit_on_texts(training_sentences) word_index = tokenizer.word_index sequences = tokenizer.texts_to_sequences(training_sentences) padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

model = Sequential() model.add(Embedding(vocab_size, embedding_dim, input_length=max_len)) model.add(GlobalAveragePooling1D()) model.add(Dense(16, activation='relu')) model.add(Dense(16, activation='relu')) model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

**Script**

epochs = 500 history = model.fit(padded_sequences, np.array(training_labels), epochs=epochs) model.save("covichat_model") import pickle with open('tokenizer.pickle', 'wb') as handle: pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL) with open('label_encoder.pickle', 'wb') as ecn_file: pickle.dump(lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)
