import os
from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types

credential_path = "temp191107-e51d2e83b360.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

client = language.LanguageServiceClient()

text = u'집에는 작은 쥐가 있었습니다!'
document = types.Document(
    content = text,
    type=enums.Document.Type.PLAIN_TEXT)

sentiment = client.analyze_sentiment(document = document).document_sentiment


print('Text: {}'.format(text))
print('Sentiment: {}, {}'.format(sentiment.score, sentiment.magnitude))

