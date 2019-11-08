# -*- coding:utf-8 -*-
from krwordrank.hangle import normalize
from krwordrank.word import KRWordRank, summarize_with_keywords
from konlpy.tag import Hannanum

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

docs2 = ["지난 9월 부산의 한 아파트에서 불이 났습니다."
         "주인이 출근한 사이에 고양이가 주방 전기레인지를 작동시켰고, 그 위에 있던 휴대용 버너 속 부탄가스가 과열되어 폭발한 것이었죠.",
         "지난 7월에도 부산의 한 원룸에서 고양이가 전기레인지 전원 버튼을 눌러 불이 나는 등, 개와 고양이 등 반려동물이 일으킨 화재 사고가 계속되고 있습니다.",
         "소방청에 따르면 최근 몇 년간 반려동물에 의한 화재가 증가세를 보이고 있는데요.",
         "특히 동물이 전기레인지 전원을 켜거나 향초 등을 넘어뜨린 사례가 많았습니다.",
         "반려동물을 키운다면, 사용하는 전기레인지의 종류를 잘 파악할 필요가 있는데요.",
         "전기레인지는 크게 '인덕션'과 '하이라이트'로 구분할 수 있습니다.",
         "인덕션은 스테인리스 등으로 이뤄진 전용냄비로만 가열할 수 있고, 하이라이트는 알루미늄 재질, 뚝배기, 유리냄비 같은 인덕션으로 가열할 수 없는 것들도 데울 수 있죠.",
         "보유한 전기레인지가 하이라이트라면 반려동물을 두고 외출을 할 때 전원 코드를 뽑는 등 더욱 주의해야 할 필요가 있습니다.",
         "종이 등 불에 탈 수 있는 물질을 레인지 근처에 두는 것도 피해야 합니다.",
         "반려동물 유발 화재가 이어지면서 반려동물 안전기능을 탑재한 전기레인지가 출시되기도 했는데요.",
         "이 같은 제품을 사용하더라도 반려동물의 특성을 고려해 화재를 예방할 필요가 있겠죠.",
         "2017년 이후 발생한 반려동물 유발 화재에서 불을 낸 동물들 대부분이 목숨을 잃었습니다.",
         "나와 이웃의 안전, 그리고 소중한 반려동물의 생명을 위해 '불조심' 잊지 마세요.",
         ]

docs = ["집에는 작은 쥐가 있었습니다",
        "고양이는 쥐를 보았습니다",
        "쥐가 집에서 도망쳤습니다",
        "고양이는 마침내 쥐를 먹었습니다",
        "쥐 이야기의 끝",
        ]

han = Hannanum()

# Text Preprocessing
docs2 = [normalize(text, english=True, number=True) for text in docs2]

doc_noun_list = []
for doc in docs2:
    nouns = han.nouns(doc)
    doc_nouns = ''

    for noun in nouns:
        doc_nouns += noun + ' '

    doc_noun_list.append(doc_nouns)

wordrank_ext = KRWordRank(min_count=7, max_length=10, verbose=True)
beta = 0.85
max_iter = 10

keywords, rank, graph = wordrank_ext.extract(doc_noun_list, beta, max_iter)

keywordss = summarize_with_keywords(doc_noun_list, min_count=3, max_length=10, beta=0.85, max_iter=10, verbose=True)

for word, r in sorted(keywordss.items(), key=lambda x: x[1], reverse=True)[:30]:
    print('%8s:\t%.4f' % (word, r))

