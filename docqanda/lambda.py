import multiprocessing

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler

model = LlamaCpp(
    model_path="models/ggml-alpaca-7b-q4.bin",
    callbacks=[StreamingStdOutCallbackHandler(), StdOutCallbackHandler()],
    n_ctx=1024,
    n_threads=multiprocessing.cpu_count()
)

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

persist_directory = "database/"

database = Chroma(persist_directory=persist_directory, embedding_function=embedding)

retriever = database.as_retriever()

qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=retriever)


def handler(event, context):
    query = event["query"]

    response = qa.run(query)

    return {"response": response}


if __name__ == "__main__":
    loader = TextLoader("shakespeare.txt").load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)

    documents = text_splitter.split_documents(loader)

    database = Chroma.from_documents(documents=documents, embedding=embedding, persist_directory=persist_directory)

    database.persist()
