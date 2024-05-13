# multilingual.py
import json
import os
import requests
from langdetect import detect
from dotenv import load_dotenv
# languages = {
#     "English": {"code": "en", "greeting": "Hello! How can I assist you today?"},
#     "中文": {"code": "zh", "greeting": "你好！今天我怎么帮助您？"},
#     "Español": {"code": "es", "greeting": "¡Hola! ¿Cómo puedo asistirte hoy?"},
#     "Français": {"code": "fr", "greeting": "Bonjour! Comment puis-je vous aider aujourd'hui?"},
#     "Deutsch": {"code": "de", "greeting": "Hallo! Wie kann ich Ihnen heute helfen?"},
#     "日本語": {"code": "jp", "greeting": "こんにちは！今日はどのようにお手伝いしましょうか？"}
# }
language_ids = {
    "Chinese": "cmn",
    "Japanese": "jpn", 
    "Russian": "rus",
    "French": "fra",
    "German": "deu",
    "Spanish": "spa"
}
load_dotenv()
HF_API_KEY = os.getenv('HUGGING_FACE_API_KEY')
def translate_to_mul(text,target_language):
    api_url = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-mul"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    target_input = f">>{ target_language }<<{text}"
    response = requests.post(api_url, headers=headers, json={"inputs": target_input})
    response_json = response.json()
    print("API Response:", response_json)  # 打印整个 JSON 响应
    print("Response Type:", type(response_json))  # 打印响应的数据类型
    if response_json and isinstance(response_json, list) and response_json[0]:
        translation = response_json[0].get('translation_text', 'Translation error')
        return translation
    else:
         return 'Translation error'

def language_detection(text):
    api_url = "https://api-inference.huggingface.co/models/ivanlau/language-detection-fine-tuned-on-xlm-roberta-base"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    response = requests.post(api_url, headers=headers, json={"inputs": text})
    response_json = response.json()
    print("Response from API:", response_json)  # Log the response to see its structure
    if response_json and isinstance(response_json, list) and response_json[0]:
        # 获取分数最高的标签
        highest_score_label = max(response_json[0], key=lambda x: x['score'])
        label = highest_score_label['label']
        print("label: ",label)
        if '_' in label:
                return label.split('_')[0]
        else:
                return label
    else:
        return 'Translation error'
def translator(language_type, text):
    print("language_type: ", language_type, ",text: ", text)
    language_id = language_ids.get(language_type)
    if language_id:
        print("translate_to_mul:", translate_to_mul(text, language_id))
        return translate_to_mul(text, language_id)
    else:
        return "Unsupported language"
# def get_language_options():
#     return list(languages.keys())

# def update_language_selection(user_input, sidebar_selection):
#     if not sidebar_selection:
#         return detect_language(user_input)
#     return languages[sidebar_selection]["code"]

# def detect_language(text):
#     try:
#         detected_lang = detect(text)
#         for lang, details in languages.items():
#             if details["code"] == detected_lang:
#                 return details["code"]
#     except:
#         pass
#     return "en"  # 默认英文

