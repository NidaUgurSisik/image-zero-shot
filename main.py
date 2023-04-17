from PIL import Image
import pandas as pd
import requests
from transformers import AutoProcessor, CLIPModel
import streamlit as st
from streamlit_tags import st_tags
import time
from functionforDownloadButtons import download_button

def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def _max_width_():
    max_width_str = f"max-width: 1800px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}

    </style>    
    """,
        unsafe_allow_html=True,
    )

st.set_page_config(page_icon="images/icon.png", page_title="Zero Shot")



c2, c3 = st.columns([6, 1])

with c2:
    c31, c32 = st.columns([12, 2])
    with c31:
        st.caption("")
        st.title("İmage Zero Shot")
    with c32:
        st.image(
            "images/logo.png",
            width=200,
        )


uploaded_file = st.file_uploader("Upload image", accept_multiple_files=True, type=["png", "jpg"])

result = ""
list_keywords = []
df = pd.DataFrame()
image_list = []
if uploaded_file is not None:
    form = st.form(key="annotation")
    with form:

            labels_from_st_tags = st_tags(
                value=[],
                maxtags=5,
                suggestions=[],
                label="",
            )

            submitted = st.form_submit_button(label="Submit")

    for i in uploaded_file:
        image = Image.open(i)
        #st.image(image, caption='')

        if uploaded_file is not None and submitted:
            
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

            inputs = processor(text=labels_from_st_tags, images=image, return_tensors="pt", padding=True)

            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
            max_idx, max_val = max(enumerate(probs[0].tolist()), key=lambda x: x[1])

            st.write(i.name,labels_from_st_tags[max_idx], max_val)

            df2 = pd.DataFrame({'Image': str(i.name), 'Label': labels_from_st_tags[max_idx], 'Probability': float(max_val)})
            df = df.append(df2, ignore_index = True)
        csv = convert_df(df)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name= 'zero_shot_image.csv',
            mime='text/csv',
        )