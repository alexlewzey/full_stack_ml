import streamlit as st
from PIL import Image

st.set_page_config(page_title="Cat vs Dog")

st.markdown("<h1 style='text-align: center;'>Cat vs Dog!</h1>", unsafe_allow_html=True)
st.markdown("Upload your image below to find out if it is a cat or a dog!")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.markdown(
        f"<h2 style='text-align: center;'>Its at {image.size}!</h2>",
        unsafe_allow_html=True,
    )
    st.image(image, use_column_width=True)
