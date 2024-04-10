import os
import sys
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
from pandasai import Agent
from pandasai.llm import OpenAI
import matplotlib

load_dotenv()

matplotlib.use(backend="TkAgg")

#Input OpenAI API key here
llm = OpenAI(api_token='PUT OPENAI KEY HERE')

st.set_page_config(layout="wide")

st.title("VDart Chatbot")

df = pd.read_csv("sample-data-set-inactives.csv", encoding='utf-8', engine='python')

df = df.dropna(axis=1, how="all")

col1, col2 = st.columns(2)

with col1:
    st.write(df)

with col2:
    prompt = st.text_area("Enter Your Prompt:")

    generate = st.button("Generate")

    if generate:
        if prompt:
            with st.spinner("Generating answer..."):
                agent = Agent(df, config={"llm": llm, "verbose": True})
                response = agent.chat(prompt)
                st.write(response)
                st.write(agent.explain())
            
        else:
            st.warning("Please upload a csv file and enter your prompt")
            
