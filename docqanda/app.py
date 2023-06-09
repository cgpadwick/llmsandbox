import streamlit as st
import multiprocessing

from docqanda import get_embeddings, load_model

from langchain.chains import RetrievalQA


class Chat(object):
    def __init__(self):
        self.dialog = []
        self.qa_chain = None

    def initialize_pipeline(
        self, embeddingdb, modelname, chaintype, n_ctx=1024, nthreads=-1
    ):
        model = load_model(modelname, n_ctx, nthreads)
        docretriever = get_embeddings(None, embeddingdb, modelname == "openai")
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=model, chain_type=chaintype, retriever=docretriever
        )

    def add_pair(self, user, bot):
        self.dialog.append({"user": user, "bot": bot})

    def get_response(self, user_input):
        # Here you should integrate your chatbot model logic
        # For now it just echoes the user's input
        return user_input


chat = Chat()

st.title("Chatbot Interface")


with st.sidebar:
    modelname = st.sidebar.selectbox(
        "Model Selector", ["ggml-alpaca-7b-q4.bin", "ggml-gpt4all-j-v1.3-groovy.bin"]
    )
    embeddingdb = st.sidebar.selectbox("Embedding Database", ["db1", "db2"])
    chaintype = st.sidebar.selectbox("Chain Type Selector", ["stuff", "refine"])
    n_ctx = st.sidebar.slider("Context Size", min_value=64, max_value=2048, value=1024)
    n_threads = st.sidebar.slider(
        "Number of Threads",
        min_value=1,
        max_value=multiprocessing.cpu_count(),
        value=multiprocessing.cpu_count(),
    )
    if st.button('Submit'):
        st.write("Initializing model pipeline...")
        chat.initialize_pipeline(embeddingdb, modelname, chaintype, n_ctx, n_threads)
        st.write("Initialization complete!")


user_input = st.text_input("You: ")

if st.button("Send"):
    bot_response = chat.get_response(user_input)
    chat.add_pair(user_input, bot_response)

for pair in reversed(chat.dialog):
    st.write("User: ", pair["user"])
    st.write("Bot: ", pair["bot"])
