import streamlit as st
import multiprocessing

from docqanda import get_embeddings, load_model

from langchain.chains import RetrievalQA

# Edit these definitions to your liking, the key should be a meaningful huma readable name
# and the value should be the path to the precomputed embeddings.
EMBEDDINGS_DICT = {
    "Cool File 1": "vectorstoredb/1756626953_embeddings_2023-06-09_15_46_44",
    "Cool Thing 2": "vectorstoredb/481886456_embeddings_2023-06-09_17_06_41",
}


def initialize_pipeline(embeddingdb, modelname, chaintype, n_ctx=1024, nthreads=-1):
    """
    Initializes a retrieval-based question answering pipeline using a pre-trained language model.

    :param embeddingdb: A string representing the path to the database of precomputed embeddings.
    :param modelname: A string representing the name of the pre-trained language model to use.
    :param chaintype: A string representing the type of retrieval chain to use.
    :param n_ctx: An integer representing the number of tokens to consider for each input sequence.
    :param nthreads: An integer representing the number of threads to use for parallel processing. If -1, uses all available CPUs.

    :return: A RetrievalQA object representing the initialized question answering pipeline.
    """
    model = load_model(modelname, n_ctx, nthreads)
    docretriever = get_embeddings(None, embeddingdb, modelname == "openai")
    qa_chain = RetrievalQA.from_chain_type(
        llm=model, chain_type=chaintype, retriever=docretriever
    )
    return qa_chain


class Chat(object):
    def __init__(self):
        """
        Initializes an instance of the class with an empty list assigned to the 'dialog' attribute.

        Parameters:
            self: An instance of the class.

        Returns:
            None
        """
        self.dialog = []

    def add_pair(self, user, bot):
        """
        Appends a pair of user and bot input to the dialog list.

        :param user: A user input.
        :type user: Any
        :param bot: A bot input.
        :type bot: Any
        :return: None
        """
        self.dialog.append({"user": user, "bot": bot})

    def get_response(self, user_input):
        """
        Retrieves a response for the user input.

        Args:
            user_input (Any): The user input.

        Returns:
            Any: The response for the user input.
        """
        qa_chain = st.session_state["qa_chain"]
        result = qa_chain.run(user_input)
        return result


chat = Chat()
st.title("Chatbot Interface")


with st.sidebar:
    modelname = st.sidebar.selectbox(
        "Model Selector", ["ggml-alpaca-7b-q4.bin", "ggml-gpt4all-j-v1.3-groovy.bin"]
    )
    embeddingdb = st.sidebar.selectbox("Embedding Database", EMBEDDINGS_DICT.keys())
    chaintype = st.sidebar.selectbox("Chain Type Selector", ["stuff", "refine"])
    n_ctx = st.sidebar.slider("Context Size", min_value=64, max_value=2048, value=1024)
    n_threads = st.sidebar.slider(
        "Number of Threads",
        min_value=1,
        max_value=multiprocessing.cpu_count(),
        value=multiprocessing.cpu_count(),
    )
    if st.button("Submit"):
        st.write("Initializing model pipeline...")
        qa_chain = initialize_pipeline(
            EMBEDDINGS_DICT[embeddingdb], modelname, chaintype, n_ctx, n_threads
        )
        st.session_state["qa_chain"] = qa_chain
        st.write("Initialization complete!")


user_input = st.text_input("You: ")

if st.button("Send"):
    bot_response = chat.get_response(user_input)
    chat.add_pair(user_input, bot_response)

for pair in reversed(chat.dialog):
    st.write("User: ", pair["user"])
    st.write("Bot: ", pair["bot"])
