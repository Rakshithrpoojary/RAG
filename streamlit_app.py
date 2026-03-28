import streamlit as st

from app import rag_simple, rag_retriever

st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="wide")

st.title("💬 RAG PDF Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Show previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input at bottom
question = st.chat_input("Ask something about your PDF...")

if question:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate bot response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = rag_simple(question, rag_retriever)
            st.markdown(answer)

    # Save bot message
    st.session_state.messages.append({"role": "assistant", "content": answer})