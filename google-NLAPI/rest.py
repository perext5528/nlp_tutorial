import os
from google.cloud import language_v1

credential_path = "temp191107-e51d2e83b360.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credential_path

client = language_v1.LanguageServiceClient()

text = u'집에는 작은 쥐가 있었습니다!'

response = client.analyze_entities(text, encoding_type='UTF-8')
