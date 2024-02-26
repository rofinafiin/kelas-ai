# Tugas 1
Rofi Nafiis Zain - 1214017
## Install Tensorflow
1. Pastikan device anda memiliki python untuk menginstall library tensorflow
2. kemudian buka terminal dan jalankan perintah berikut
	```
	$ pip install tensorflow
	```
3. Maka tampilan terminal akan berubah seperti berikut ini:
	``` image
	image
	```
4. Jika sudah terinstall maka tampilan pada terminal akan menjadi seperti pada gambar berikut ini:
5. Jika sudah selesai download pada terminal maka Tensorflow sudah bisa digunakan.

## Tugas

1. Repository fork [rofinafiin/nlp-ai](https://github.com/rofinafiin/nlp-ai)
2. Buka project dengan IDE yang dimiliki (disarankan menggunakan [Visual Studio Code](https://code.visualstudio.com/download)
3. Install Requirements.txt dengan perintah:
	```
	$ pip install -r requirements.txt
	```
4. Jalankan preprocessing.py dengan perintah
	```python
	$ python preprocessing.py
	```
5. Maka hasil dari preprocessing.py akan membuat file [clean_qa.txt]()
6. Tampilan pada terminal seperti gambar berikut:
7. Jalankan training.py, kemudian tunggu sampai proses selesai

## Penjelasan Baris Kode

### Pre-processing
1. Pada line 1 - 18 merupakan import dari library yang akan digunakan dalam script
	```python
	from  Sastrawi.Stemmer.StemmerFactory  import  StemmerFactory
	import  io
	import  os
	import  re
	import  requests
	import  csv
	import  datetime
	import  numpy  as  np
	import  pandas  as  pd
	import  random
	import  pickle
	```
2. Line selanjutnya akan membuast stemmerFactory() yang diassign kedalam variable bernama factory. Kemudian gunakan create_stemmer.
	```python
	factory  =  StemmerFactory()
	stemmer  =  factory.create_stemmer()
	```
3.  Pada variable punct_re_escape berisi fungsi untuk mengidentifikasi setiap tanda baca yang ada. Pada baris ini akan mengidentifikasi semua karakter yang berada diantara kutip tunggal. dan variable Uknowns berisi kata kata yang akan dipakai jika question yang disampaikan tidak memiliki answer.
	```python
	punct_re_escape  =  re.compile('[%s]'  %  re.escape('!"#$%&()*+,./:;<=>?@[\\]^_`{|}~'))
	unknowns  = ["gak paham","kurang ngerti","I don't know"]
	```
4. Pada baris selanjutnya akan membuka dataset slang indonesia yang dalam bentuk csv.
	```python
	list_indonesia_slang  =  pd.read_csv('./dataset/daftar-slang-bahasa-indonesia.csv', header=None).to_numpy()
	```
5. Kemudian definisikan sebuah list kosong dalam variable bernama data_slang. Kemudian lakukan looping dengan memasukan setiap data slang yang ada pada csv ke dalam list
	```python
	data_slang  = {}
	for  key, value  in  list_indonesia_slang:
	data_slang[key] =  value
	```
6. Definisikan sebuah fungsi bernama dynamic_switcher 
	```python
	def  dynamic_switcher(dict_data, key):
	return  dict_data.get(key, None)
	```
7. 
    ```python
    def check_normal_word(word_input):
    slang_result = dynamic_switcher(data_slang, word_input)
    if slang_result:
        return slang_result
    return word_input
    ```
8. 
    ```python
    def normalize_sentence(sentence):
  sentence = punct_re_escape.sub('', sentence.lower())
  sentence = sentence.replace('iteung', '').replace('\n', '').replace(' wah','').replace('wow','').replace(' dong','').replace(' sih','').replace(' deh','')
  sentence = sentence.replace('teung', '')
  sentence = re.sub(r'((wk)+(w?)+(k?)+)+', '', sentence)
  sentence = re.sub(r'((xi)+(x?)+(i?)+)+', '', sentence)
  sentence = re.sub(r'((h(a|i|e)h)((a|i|e)?)+(h?)+((a|i|e)?)+)+', '', sentence)
  sentence = ' '.join(sentence.split())
  if sentence:
    sentence = sentence.strip().split(" ")
    normal_sentence = " "
    for word in sentence:
      normalize_word = check_normal_word(word)
      root_sentence = stemmer.stem(normalize_word)
      normal_sentence += root_sentence+" "
    return punct_re_escape.sub('',normal_sentence)
  return sentence
    ```
9. 
    ```python
    df = pd.read_csv('./dataset/qa.csv', sep='|',usecols= ['question','answer'])
    df.head()
    ```
10. 
    ```python
    question_length = {}
    answer_length = {}
    ```
11. 
    ```python
    for index, row in df.iterrows():
    question = normalize_sentence(row['question'])
    question = normalize_sentence(question)
    question = stemmer.stem(question)

    if question_length.get(len(question.split())):
        question_length[len(question.split())] += 1
    else:
        question_length[len(question.split())] = 1

    if answer_length.get(len(str(row['answer']).split())):
        answer_length[len(str(row['answer']).split())] += 1
    else:
        answer_length[len(str(row['answer']).split())] = 1

    question_length

    answer_length
    ```
12. 
    ```python
    val_question_length = list(question_length.values())
    key_question_length = list(question_length.keys())
    key_val_question_length = list(zip(key_question_length, val_question_length))
    df_question_length = pd.DataFrame(key_val_question_length, columns=['length_data', 'total_sentences'])
    df_question_length.sort_values(by=['length_data'], inplace=True)
    df_question_length.describe()
    ```
13. 
    ```python
    val_answer_length = list(answer_length.values())
    key_answer_length = list(answer_length.keys())
    key_val_answer_length = list(zip(key_answer_length, val_answer_length))
    df_answer_length = pd.DataFrame(key_val_answer_length, columns=['length_data', 'total_sentences'])
    df_answer_length.sort_values(by=['length_data'], inplace=True)
    df_answer_length.describe()

    data_length = 0
    ```
14. 
    ```python
    filename= './dataset/clean_qa.txt'
    with open(filename, 'w', encoding='utf-8') as f:
    for index, row in df.iterrows():
        question = normalize_sentence(str(row['question']))
        question = normalize_sentence(question)
        question = stemmer.stem(question)

        answer = str(row['answer']).lower().replace('iteung', 'aku').replace('\n', ' ')

        if len(question.split()) > 0 and len(question.split()) < 13 and len(answer.split()) < 29:
        body="{"+question+"}|<START> {"+answer+"} <END>"
        print(body, file=f)
    ```

### Training
1. 
    ```python
    import json
    import os
    import pickle

    import pandas as pd
    import tensorflow as tf
    from keras import Input, Model
    from keras.activations import softmax
    from keras.callbacks import ModelCheckpoint, TensorBoard
    from keras.layers import Embedding, LSTM, Dense, Bidirectional, Concatenate
    from keras.optimizers import RMSprop
    from keras.utils import to_categorical
    from keras_preprocessing.sequence import pad_sequences
    from keras_preprocessing.text import Tokenizer
    ```
2. 
    ```python
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))
    ```
3. 
    ```python
    path = "output_dir/"
    try:
        os.makedirs(path)
    except:
        pass
    ```
4. 
    ```python
    dataset = pd.read_csv('./dataset/clean_qa.txt', delimiter="|", header=None,lineterminator='\n')
    ```
5. 
    ```python
    dataset_val = dataset.iloc[1794:].to_csv('output_dir/val.csv')

    dataset_train = dataset.iloc[:1794]

    questions_train = dataset_train.iloc[:, 0].values.tolist()
    answers_train = dataset_train.iloc[:, 1].values.tolist()

    questions_test = dataset_train.iloc[:, 0].values.tolist()
    answers_test = dataset_train.iloc[:, 1].values.tolist()
    ```
6. 
    ```python
    def save_tokenizer(tokenizer):
    with open('output_dir/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    ```
7. 
    ```python
    def save_config(key, value):
    data = {}
    if os.path.exists(path + 'config.json'):
        with open(path + 'config.json') as json_file:
            data = json.load(json_file)

    data[key] = value
    with open(path + 'config.json', 'w') as outfile:
        json.dump(data, outfile)
    ```
8. 
    ```python
    target_regex = '!"#$%&()*+,-./:;<=>?@[\]^_`{|}~\t\n\'0123456789'
    tokenizer = Tokenizer(filters=target_regex, lower=True)
    tokenizer.fit_on_texts(questions_train + answers_train + questions_test + answers_test)
    save_tokenizer(tokenizer)
    ```
9. 
    ```python
    VOCAB_SIZE = len(tokenizer.word_index) + 1
    save_config('VOCAB_SIZE', VOCAB_SIZE)
    print('Vocabulary size : {}'.format(VOCAB_SIZE))
    ```
10. 
    ```python
    tokenized_questions_train = tokenizer.texts_to_sequences(questions_train)
    maxlen_questions_train = max([len(x) for x in tokenized_questions_train])
    save_config('maxlen_questions', maxlen_questions_train)
    encoder_input_data_train = pad_sequences(tokenized_questions_train, maxlen=maxlen_questions_train, padding='post')
    ```
11. 
    ```python
    tokenized_questions_test = tokenizer.texts_to_sequences(questions_test)
    maxlen_questions_test = max([len(x) for x in tokenized_questions_test])
    save_config('maxlen_questions', maxlen_questions_test)
    encoder_input_data_test = pad_sequences(tokenized_questions_test, maxlen=maxlen_questions_test, padding='post')
    ```
12. 
    ```python
    tokenized_answers_train = tokenizer.texts_to_sequences(answers_train)
    maxlen_answers_train = max([len(x) for x in tokenized_answers_train])
    save_config('maxlen_answers', maxlen_answers_train)
    decoder_input_data_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')
    ```
13. 
    ```python
    tokenized_answers_test = tokenizer.texts_to_sequences(answers_test)
    maxlen_answers_test = max([len(x) for x in tokenized_answers_test])
    save_config('maxlen_answers', maxlen_answers_test)
    decoder_input_data_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')
    ```
14. 
    ```python
    for i in range(len(tokenized_answers_train)):
    tokenized_answers_train[i] = tokenized_answers_train[i][1:]
    padded_answers_train = pad_sequences(tokenized_answers_train, maxlen=maxlen_answers_train, padding='post')
    decoder_output_data_train = to_categorical(padded_answers_train, num_classes=VOCAB_SIZE)
    ```
15. 
    ```python
    for i in range(len(tokenized_answers_test)):
    tokenized_answers_test[i] = tokenized_answers_test[i][1:]
    padded_answers_test = pad_sequences(tokenized_answers_test, maxlen=maxlen_answers_test, padding='post')
    decoder_output_data_test = to_categorical(padded_answers_test, num_classes=VOCAB_SIZE)
    ```
16. 
    ```python
    enc_inp = Input(shape=(None,))
    enc_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(enc_inp)
    enc_outputs, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(256, return_state=True, dropout=0.5, recurrent_dropout=0.5))(enc_embedding)
    ```
17. 
    ```python
    state_h = Concatenate()([forward_h, backward_h])
    state_c = Concatenate()([forward_c, backward_c])
    enc_states = [state_h, state_c]
    ```
18. 
    ```python
    dec_inp = Input(shape=(None,))
    dec_embedding = Embedding(VOCAB_SIZE, 256, mask_zero=True)(dec_inp)
    dec_lstm = LSTM(256 * 2, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)
    dec_outputs, _, _ = dec_lstm(dec_embedding, initial_state=enc_states)
    ```
19. 
    ```python
    dec_dense = Dense(VOCAB_SIZE, activation=softmax)
    output = dec_dense(dec_outputs)
    ```
20. 
    ```python
    logdir = os.path.join(path, "logs")
    tensorboard_callback = TensorBoard(logdir, histogram_freq=1)
    ```
21. 
    ```python
    checkpoint = ModelCheckpoint(os.path.join(path, 'model-{epoch:02d}-{loss:.2f}.hdf5'),
                             monitor='loss',
                             verbose=1,
                             save_best_only=True, mode='auto', period=10)
    ```
22. 
    ```python
    model = Model([enc_inp, dec_inp], output)
    model.compile(optimizer=RMSprop(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    ```
23. 
    ```python
    batch_size = 64
    epochs = 40
    model.fit([encoder_input_data_train, decoder_input_data_train],
            decoder_output_data_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=([encoder_input_data_test, decoder_input_data_test], decoder_output_data_test),
            callbacks=[tensorboard_callback, checkpoint])
    model.save(os.path.join(path, 'model-' + path.replace("/", "") + '.h5'))
    ```