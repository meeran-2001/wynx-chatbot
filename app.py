import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
    return tokenizer, model

tokenizer, model = load_model()

st.title("Wynx 🤖 - Empathetic Chatbot")
st.write("Hello, I'm Wynx. I'm here to chat and help you feel better. 💬")

if "history" not in st.session_state:
    st.session_state.history = []

def get_response(user_input):
    # Simple grounding instruction
    history_text = " EOS ".join([f"User: {u} Bot: {b}" for u, b in st.session_state.history])
    input_text = f"Instruction: You are a friendly and empathetic assistant named Wynx. Respond supportively. Input: {history_text} EOS User: {user_input} Bot:"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model.generate(**inputs, max_length=256)
    resp = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return resp

user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    response = get_response(user_input)
    st.markdown(f"**You:** {user_input}")
    st.markdown(f"**Wynx:** {response}")
    st.session_state.history.append((user_input, response))
