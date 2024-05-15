
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re

translate_model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
translate_tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")
# 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")

# 创建语种检测pipeline
language_detection_pipeline = pipeline('text-classification', model=model, tokenizer=tokenizer)

def detect_language(text):
    result = language_detection_pipeline(text)
    detected_lang = result[0]['label']
    return detected_lang

def translate(text, target_lang):
    print("text: ",text," target_lang: ",target_lang)
    if(target_lang=="en"):
        return text
    translate_tokenizer.src_lang = detect_language(text)
    encoded_input = translate_tokenizer(text, return_tensors="pt")
    output_ids = translate_model.generate(**encoded_input, forced_bos_token_id=translate_tokenizer.get_lang_id(target_lang))
    output_text = translate_tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    
    print("output_text:",output_text)
    pattern = r'^([^:]*:)(.*)'
    new_text = remove_prefix(output_text,pattern)
    print("new_text:",new_text)
    return new_text

def remove_prefix(text,pattern):
    return re.sub(pattern, r'\2', text, flags=re.MULTILINE)