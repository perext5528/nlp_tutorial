# -*- coding:utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

docs = ["집에는 작은 쥐가 있었습니다",
        "고양이는 쥐를 보았습니다",
        "쥐가 집에서 도망쳤습니다",
        "고양이는 마침내 쥐를 먹었습니다",
        "쥐 이야기의 끝",
        ]


#TfidTransformer

cv = CountVectorizer()

#코퍼스 내의 단어에 대한 워드카운트 벡터 생성
word_count_vector = cv.fit_transform(docs)

tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
tfidf_transformer.fit(word_count_vector)

#pandas Dataframe
df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
print(df_idf.sort_values(by=['idf_weights']))

count_vector = cv.transform(docs)

tf_idf_vector = tfidf_transformer.transform(count_vector)

feature_names = cv.get_feature_names()

first_document_vector = tf_idf_vector[0]

df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=['tfidf'])
print(df.sort_values(by=['tfidf'], ascending=False))

print("##############################################")
#TfidVectorizer

tfidf_vectorizer = TfidfVectorizer(use_idf=True)

tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)

first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

print(df.sort_values(by=['tfidf'], ascending=False))



