from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders import TextLoader


textfile = "input.txt"
loader = TextLoader(textfile).load()
print(loader)
text_split = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
text = text_split.split_documents(loader)

# from langchain.embeddings import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(text, embeddings)
vectorstore.save_local("saved_embeddings")
