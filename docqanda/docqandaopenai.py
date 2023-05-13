from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

embeddings = OpenAIEmbeddings()
docsearch = FAISS.load_local("saved_embeddings", embeddings)

# This doesn't work for some reason.
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
)
query = "Is First Citizen famished"
print(qa.run(query))

prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

#{context}

#Question: {question}
#"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

query = "Is MENENIUS an evil character?  why or why not?"
print(qa.run(query))
