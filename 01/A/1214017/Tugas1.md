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
