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