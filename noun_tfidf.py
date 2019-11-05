# -*- coding:utf-8 -*-
import pandas as pd
from konlpy.tag import Hannanum
from sklearn.feature_extraction.text import TfidfVectorizer


def custom_tagging(docs):
    doc_noun_list = []
    for doc in docs:
        nouns = han.nouns(doc)
        doc_nouns = ''

        for noun in nouns:
            doc_nouns += noun + ' '

        doc_noun_list.append(doc_nouns)

    for i in range(0, len(docs)):
        print('doc' + str(i + 1) + ' : ' + str(doc_noun_list[i]))


docs = ["집에는 작은 쥐가 있었습니다",
        "고양이는 쥐를 보았습니다",
        "쥐가 집에서 도망쳤습니다",
        "고양이는 마침내 쥐를 먹었습니다",
        "쥐 이야기의 끝",
        ]



han = Hannanum()

# custom_tagging(docs)


doc_noun_list = []
for doc in docs:
    nouns = han.nouns(doc)
    doc_nouns = ''

    for noun in nouns:
        doc_nouns += noun + ' '

    doc_noun_list.append(doc_nouns)

for i in range(0, len(docs)):
    print('doc' + str(i + 1) + ' : ' + str(doc_noun_list[i]))

tfidf_vectorizer = TfidfVectorizer(use_idf=True)

tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(doc_noun_list)

first_vector_tfidfvectorizer = tfidf_vectorizer_vectors[0]

df = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(),
                  columns=["tfidf"])

print(df.sort_values(by=['tfidf'], ascending=False))
