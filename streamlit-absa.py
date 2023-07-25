from nlp_id.lemmatizer import Lemmatizer
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
import nltk
from nltk.tokenize import word_tokenize
import streamlit as st
import re
import numpy as np
import pandas as pd
import pickle

# preprocessing


def casefolding(content):
    content = content.casefold()
    return content


# Remove Puncutuation dan karakter yg tdk dibutuhkan
clean_spcl = re.compile('[/(){}\[\]\|@,;]')
clean_symbol = re.compile('[^0-9a-zA-Z]')


def clean_punct(content):
    content = clean_spcl.sub(' ', content)
    content = clean_symbol.sub(' ', content)
    content = re.sub(
        '((www\.[^\s]+) | (https?://[^\s]+))', ' ', content)  # remove url
    content = re.sub('@[^\s]+', ' ', content)  # remove username
    content = re.sub(':v', ' ', content)  # menghilangkan :v
    content = re.sub(';v', ' ', content)  # menghilangkan ;v
    # mengganti dgn sampai dengan
    content = re.sub('s/d', ' sampai dengan', content)
    return content


def replaceThreeOrMore(content):
    # Pattern to look for three or more repetitions of any character, including newlines (contoh goool -> gool).
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", content)


# Kamus alay
alay_dict = pd.read_csv('new_kamusalay1.csv', names=[
                        'original', 'replacement'], encoding='latin-1')
alay_dict_map = dict(zip(alay_dict['original'], alay_dict['replacement']))
# Mengganti kata-kata yang dianggap alay


def normalize_alay(text):
    return ' '.join([alay_dict_map[word] if word in alay_dict_map else word for word in text.split(' ')])


def removeangkachar(content):
    content = re.sub('\d+', '', content)  # Remove angka
    return content.strip(" ")
# Menghapus Double atau Lebih Whitespace


def normalize_whitespace(content):
    content = str(content)
    content = re.sub(r"//t", r"\t", content)
    content = re.sub(r"( )\1+", r"\1", content)
    content = re.sub(r"(\n)\1+", r"\1", content)
    content = re.sub(r"(\r)\1+", r"\1", content)
    content = re.sub(r"(\t)\1+", r"\1", content)
    return content.strip(" ")


nltk.download('punkt')


def tokenisasi(content):
    content = nltk.tokenize.word_tokenize(content)
    return content


factory = StopWordRemoverFactory()
stopwords = factory.get_stop_words()
exclude_stopword = ['tidak', 'belum', 'selain',
                    'tidak bisa', 'bisa', 'belum bisa', 'ada']
remove_words = ([word for word in stopwords if word not in exclude_stopword])
final_stop_words = ArrayDictionary(remove_words)


def remove_stopwords(content):
    factory = StopWordRemover(final_stop_words)
    content = [factory.remove(x) for x in content]
    return content


def lemma_preproc(content):
    lemmatizer = Lemmatizer()
    content = [lemmatizer.lemmatize(x) for x in content]
    return content


def token_kosong(content):
    content = [x for x in content if len(x) > 0]
    return content


def jointeks(content):
    for i in content:
        return " ".join(content)


def preprocess(content):
    content = casefolding(content)
    content = clean_punct(content)
    content = replaceThreeOrMore(content)
    content = normalize_alay(content)
    content = removeangkachar(content)
    content = normalize_whitespace(content)
    content = tokenisasi(content)
    content = remove_stopwords(content)
    content = lemma_preproc(content)
    content = token_kosong(content)
    content = jointeks(content)
    return content


# load save model
model_aspek = pickle.load(open('model_aspek.sav', 'rb'))
model_sentimen = pickle.load(open('model_sentimen.sav', 'rb'))
tf_idf_data = pickle.load(open('tf_idf_data.sav', 'rb'))
data = pickle.load(open('data.sav', 'rb'))


# judul halaman
st.title("Aspect Based Sentimen Analysis Pada Ulasan")

text_input = st.text_input("Masukkan Kalimat Ulasan")
review = {'review': [text_input]}
new_data = pd.DataFrame(review)
new_data['preprocess'] = new_data['review'].apply(preprocess)
new_data = new_data.loc[:, ['preprocess']]
new_data = new_data.rename(columns={"preprocess": "review"})


def Tokenize(data):
    data['review_token'] = ""
    data['review'] = data['review'].astype('str')
    for i in range(len(data)):
        data['review_token'][i] = data['review'][i].lower().split()
    all_tokenize = sorted(
        list(set([item for sublist in data['review_token'] for item in sublist])))
    return data, all_tokenize


def tf(data, all_tokenize):
    from operator import truediv
    token_cal = Tokenize(data)
    data_tokenize = token_cal[0]
    for item in all_tokenize:
        data_tokenize[item] = 0
    for item in all_tokenize:
        for i in range(len(data_tokenize)):
            if data_tokenize['review_token'][i].count(item) > 0:
                a = data_tokenize['review_token'][i].count(item)
                b = len(data_tokenize['review_token'][i])
                c = a / b
                data_tokenize[item] = data_tokenize[item].astype('float')
                data_tokenize[item][i] = c
    return data_tokenize


def tfidf_shopee(data, new_data=new_data, tf_idf_data=tf_idf_data):
    tf_idf = tf_idf_data
    N = len(data)
    all_tokenize = tf_idf.columns.tolist()
    df = {}
    for item in all_tokenize:
        df_ = (tf_idf[item] > 0).sum()
        df[item] = df_
        idf = (np.log(N / df_)) + 1
        tf_idf[item] = tf_idf[item] * idf

    if new_data is not None:
        new_tf = tf(new_data, all_tokenize)

        for item in all_tokenize:
            if item in new_tf.columns:
                df_ = df.get(item, 0)
                idf = (np.log(N / (df_ + 1))) + 1
                new_tf[item] = new_tf[item] * idf

        new_tf.drop(columns=['review', 'review_token'], inplace=True)

        return new_tf, df
    else:
        return tf_idf, df


tfidf_result, document_frequency = tfidf_shopee(data, new_data)

# aspect


def testing_aspek(W_aspek, data_uji_aspek):
    prediksi_aspek = np.array([])
    for i in range(data_uji_aspek.shape[0]):
        y_prediksi_aspek = np.sign(
            np.dot(W_aspek, data_uji_aspek.to_numpy()[i]))
        prediksi_aspek = np.append(prediksi_aspek, y_prediksi_aspek)
    return prediksi_aspek


def testing_onevsrest_aspek(W_aspek, data_uji_aspek):
    list_kelas_aspek = W_aspek.keys()
    hasil_aspek = pd.DataFrame(columns=W_aspek.keys())
    for kelas_aspek in list_kelas_aspek:
        hasil_aspek[kelas_aspek] = testing_aspek(
            W_aspek[kelas_aspek], data_uji_aspek)
    kelas_prediksi_aspek = hasil_aspek.idxmax(1)
    return kelas_prediksi_aspek


prediksi_aspek = testing_onevsrest_aspek(model_aspek, new_data)

# sentimen


def testing_sentimen(W_sentimen, data_uji_sentimen):
    prediksi_sentimen = np.array([])
    for i in range(data_uji_sentimen.shape[0]):
        y_prediksi_sentimen = np.sign(
            np.dot(W_sentimen, data_uji_sentimen.to_numpy()[i]))
        prediksi_sentimen = np.append(prediksi_sentimen, y_prediksi_sentimen)
    return prediksi_sentimen


y_prediksi_sentimen = testing_sentimen(model_sentimen, new_data)

prediksi = st.button("Hasil Prediksi")

if prediksi:
    for aspek, sentimen in zip(prediksi_aspek, y_prediksi_sentimen):
        st.success(
            f"Aspek {aspek}, Sentimen {'Positif' if sentimen == 1 else 'Negatif'}")
