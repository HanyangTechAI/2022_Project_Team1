"""
## App: NLP App with Streamlit (NLPiffy)
Author: [Seongjin Lee(GirinMan)](https://github.com/GirinMan))\n
Source: [Github](https://github.com/GirinMan/HAI-DialectTranslator/)
Credits: 2022-Fall HAI Team 1 project

실행 방법: streamlit run app.py

"""
# Core Pkgs
import streamlit as st 
import os
import requests
import torch
import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_classification_model():
    with st.spinner("Loading model for classification..."):
        os.makedirs('./models', exist_ok=True)

        model_url = "https://github.com/GirinMan/HAI-DialectTranslator/raw/main/classifier_training/model.pth"
        model_response = requests.get(model_url, allow_redirects=True)

        open('./models/model.pth', 'wb').write(model_response.content)

        model_ckpt = "monologg/koelectra-small-v3-discriminator"
        model = AutoModelForSequenceClassification.from_pretrained(model_ckpt, num_labels=2).to('cuda')
        tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

        state = torch.load('./models/model.pth', map_location='cuda')
        model.load_state_dict(state, strict=False)
        return model, tokenizer

@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_translation_models():
    with st.spinner("Loading models for translation..."):

        loaded = []
        ckpts = [
            "eunjin/kobart_gyeongsang_translator", 
            "eunjin/kobart_gyeongsang_to_standard_translator",
            ]
    
        for ckpt in ckpts:
            model = AutoModelForSeq2SeqLM.from_pretrained(ckpt).to('cuda')
            tokenizer = AutoTokenizer.from_pretrained(ckpt)
            loaded.append((model, tokenizer))

        return loaded


def classification(model, tokenizer, input_txt):
    input_tensor = torch.tensor([tokenizer.encode(input_txt)]).to('cuda')

    with torch.no_grad():
        preds = model(input_tensor).logits.cpu()

    result = np.argmax(preds, axis=1).item()
    
    return result

def translation(model, tokenizer, input_txt):
    input_tensor = torch.tensor([tokenizer.encode(input_txt)]).to('cuda')

    generated_ids = model.generate(input_tensor, num_beams=4)
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    return result

def main():
    """ NLP Based App with Streamlit """
    

    # Title
    st.title("2022-2학기 HAI 1팀 프로젝트")
    st.subheader("표준어-경상도 방언 번역기")
    st.markdown("""
        #### Description
        - 이 웹 애플리케이션은 PyTorch와 Transformers 라이브러리를 활용하여 표준어와 경상도 방언을 자동으로 인식하고, 번역해주는 프로그램 입니다.
        - 모델 서빙 및 현재 보여지는 웹 페이지는 Streamlit을 활용하여 구현되었습니다.
        """)

    class_model, class_tokenizer = load_classification_model()
    trans_models = load_translation_models()

    options = ['표준어', '경상도 방언']
    input_options = st.selectbox("입력 언어", ['자동'] + options)

    message = st.text_area("발화 텍스트 입력", "여기에 입력")

    output_options = st.selectbox("목표 언어", options)

    if st.button("번역하기"):

        st.subheader("번역 결과")

        init = -1
        target = 1
        
        for i in range(len(options)):
            if input_options == options[i]:
                init = i
            if output_options == options[i]:
                target = i
        
        if init == -1:
            init = classification(class_model, class_tokenizer, message)
            st.success("입력 텍스트 자동 분류: " + options[init])

        same = False
        if init == target:
            same = True
        elif init == 0 and target == 1:
            selected = trans_models[0]
        elif init == 1 and target == 0:
            selected = trans_models[1]
        
        if same:
            translation_result = message
        else:
            translation_result = translation(selected[0], selected[1], message)
                     
        st.text_area("", translation_result, label_visibility="collapsed")

if __name__ == '__main__':
	main()
