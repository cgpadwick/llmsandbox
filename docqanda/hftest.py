from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# from huggface_hub import hf_hub_download
import textwrap
import glob
from langchain.llms import GPT4All

textfile = "input.txt"
loader = TextLoader(textfile).load()
print(loader)
text_split = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0)
text = text_split.split_documents(loader)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(text, hf_embeddings)
print(vectorstore)

modelfile = "/home/chrispadwick/code/gpt4all-ui/models/gpt4all-lora-quantized.bin"

model = GPT4All(model=modelfile, n_ctx=512, n_threads=8)

from langchain.chains.question_answering import load_qa_chain

chain = load_qa_chain(model, chain_type="stuff")
chain.run(input_documents=docs, question=query)

# template = """ You are going to be my assistant.
# Please try to give me the most beneficial answers to my
# question with reasoning for why they are correct.

#  Question: {input} Answer: """
# prompt = PromptTemplate(template=template, input_variables=["input"])

# chain = LLMChain(prompt=prompt, llm=model)

# my_chain = load_qa_with_sources_chain(model, chain_type="refine")
# query = "How many times does the First Citizen speak"
# documents = vectorstore.similarity_search(query)

# print(documents)
# result = with_sources_chain({"input_documents": documents, "question": query})
