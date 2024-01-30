import os
import sys
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

load_dotenv()

llm = OpenAI(api_token='ADD OPENAI API KEY HERE')

st.set_page_config(layout="wide")

st.title("VDart Chatbot")

df = pd.read_csv("all-contracts-filtered.csv", encoding='utf-8', engine='python')

df = df.dropna(axis=1, how="all")

col1, col2 = st.columns(2)

# uploaded_file = st.file_uploader("Upload CSV For Analysis", type="csv")

with col1:
    st.write(df)

with col2:
    prompt = st.text_area("Enter Your Prompt:")

    generate = st.button("Generate")

    if generate:
        if prompt:
            #st.write("OpenAI is generating an answer, please wait...")
            with st.spinner("Generating answer..."):
                #st.write(pandas_ai.run(df, prompt=prompt))
                smartDf = SmartDataframe(df, config={"llm": llm})
                st.write(smartDf.chat(prompt))
            
        else:
            st.warning("Please upload a csv file and enter your prompt")
