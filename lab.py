# -*- coding:utf-8 -*-
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
import nltk.data
from konlpy.corpus import kolaw
from konlpy.tag import Hannanum


han = Hannanum()

str_test = '세계 최초 5G 스마트폰의 타이틀을 가진 갤럭시S10 모델 중 LTE 모델에 세일 딱지가 붙었다. 통신요금 정보 포털인 스마트초이스에 따르면 이동통신사 3사는 갤럭시S10 LTE 모델에 대한 출고가를 인하했다. SK텔레콤과 LG유플러스는 1일 갤럭시S10 128GB 모델에 대한 출고가를 105만6000원에서 89만9800원으로 내렸다. 512GB 모델에 대해서도 기존 129만8000원에서 99만8800원으로 가격을 낮췄다. 앞서 KT도 10월 중순경 갤럭시S10 LTE 모델 출고가를 인하한 바 있다. KT의 경우 현재 갤럭시S10 128GB 모델 출고가는 94만6000원, 512GB 모델은 104만5000원이다. 이와 함께 이동통신사에서 지급하는 단말기 지원금도 늘렸다. 스마트초이스에 따르면 갤럭시S10 128G 모델의 경우 월 6만원대 요금제를 기준으로 SK텔레콤은 37만원의 지원금을 제공한다. 같은 조건에서 KT는 33만3000원, LG유플러스는 35만3000원의 지원금을 준다. LG유플러스 측 관계자는 6만9000원 요금제를 기준으로 했을 경우, 갤럭시S10 LTE 모델에 대한 지원금이 기존 14만1000원에서 37만원으로 인상됐다며 22만9000원이 인상된 것이라고 말했다. 다만 단말기 지원금을 받을 경우, 선택약정 할인(매달 요금의 25%를 감면) 혜택을 받을 수 없기 때문에 이동통신사와 요금제에 따라 어느 할인 방식을 적용하는 게 유리한지 따져 보는 것이 좋다.'
str_test_corpus = ['세계 최초 5G 스마트폰의 타이틀을 가진 갤럭시S10 모델 중 LTE 모델에 세일 딱지가 붙었다.',
                   '통신요금 정보 포털인 스마트초이스에 따르면 이동통신사 3사는 갤럭시S10 LTE 모델에 대한 출고가를 인하했다.',
                   'SK텔레콤과 LG유플러스는 1일 갤럭시S10 128GB 모델에 대한 출고가를 105만6000원에서 89만9800원으로 내렸다.',
                   '512GB 모델에 대해서도 기존 129만8000원에서 99만8800원으로 가격을 낮췄다.',
                   '앞서 KT도 10월 중순경 갤럭시S10 LTE 모델 출고가를 인하한 바 있다.',
                   'KT의 경우 현재 갤럭시S10 128GB 모델 출고가는 94만6000원, 512GB 모델은 104만5000원이다.',
                   '이와 함께 이동통신사에서 지급하는 단말기 지원금도 늘렸다.',
                   '스마트초이스에 따르면 갤럭시S10 128G 모델의 경우 월 6만원대 요금제를 기준으로 SK텔레콤은 37만원의 지원금을 제공한다.',
                   '같은 조건에서 KT는 33만3000원, LG유플러스는 35만3000원의 지원금을 준다.',
                   'LG유플러스 측 관계자는 6만9000원 요금제를 기준으로 했을 경우, 갤럭시S10 LTE 모델에 대한 지원금이 기존 14만1000원에서 37만원으로 인상됐다며 22만9000원이 인상된 것이라고 말했다.',
                   '다만 단말기 지원금을 받을 경우, 선택약정 할인(매달 요금의 25%를 감면) 혜택을 받을 수 없기 때문에 이동통신사와 요금제에 따라 어느 할인 방식을 적용하는 게 유리한지 따져 보는 것이 좋다.'
                   ]
st2_test = ""
# para_kor = '안녕하세요, 여러분. 만나서 반갑습니다. 이제 텍스트 마이닝 클래스를 시작해봅시다!'

# tfidf_vec = TfidfVectorizer()
# tfidf_mat = tfidf_vec.fit_transform(sent_tokenize(str_test))
#
# print(tfidf_mat)

# print(type(sent_tokenize(str_test)))


#
# print(sent_tokenize(str_test))
# print()
#
# for i in range(len(sent_tokenize(str_test))):
#     print(sent_tokenize(str_test)[i])
#
#
#     # print(han.nouns(sent_tokenize(str_test)[i]))
#     print()

# cv = CountVectorizer()
# wcb = cv.fit_transform(sent_tokenize(str_test))
#
# print(wcb.shape)
#
# tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
# tfidf_transformer.fit(wcb)
# df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf_weights"])
# print(df_idf.sort_values(by=['idf_weights']))
#
# print("============================")
# count_vec = cv.transform(sent_tokenize(str_test))
# tf_idf_vec = tfidf_transformer.transform(count_vec)
# feature_names = cv.get_feature_names()
# first_document_vec = tf_idf_vec[0]
# df = pd.DataFrame(first_document_vec.T.todense(), index=feature_names, columns=["tfidf"])
# print(df.sort_values(by=["tfidf"], ascending=False))

#########################################################################################################

# tfidfv = TfidfVectorizer().fit(str_test_corpus)
# print(tfidfv.transform(str_test_corpus).toarray())
# print(tfidfv.vocabulary_)

#########################################################################################################

han = Hannanum()


doc_noun_list = []
for doc in str_test_corpus:
    nouns = han.nouns(doc)
    doc_nouns = ''

    for noun in nouns:
        doc_nouns+=noun+' '

    doc_noun_list.append(doc_nouns)

for i in range(0, 11):
    print('doc' + str(i+1)+ ' : ' + str(doc_noun_list[i]))

# print('#########################################################################################################')
# doc_noun_list1 = []
# kk = Kkma()
# for doc in str_test_corpus:
#     nouns = kk.nouns(doc)
#     doc_nouns = ''
#
#     for noun in nouns:
#         doc_nouns+=noun+' '
#
#     doc_noun_list1.append(doc_nouns)
#
# for i in range(0, 11):
#     print('doc' + str(i+1)+ ' : ' + str(doc_noun_list1[i]))
#
# print('#########################################################################################################')
# doc_noun_list2 = []
# okt = Okt()
# for doc in str_test_corpus:
#     nouns = okt.nouns(doc)
#     doc_nouns = ''
#
#     for noun in nouns:
#         doc_nouns+=noun+' '
#
#     doc_noun_list2.append(doc_nouns)
#
# for i in range(0, 11):
#     print('doc' + str(i+1)+ ' : ' + str(doc_noun_list2[i]))