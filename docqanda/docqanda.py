from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All
from langchain.llms import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler

textfile = "input.txt"
loader = TextLoader(textfile).load()
print(loader)
text_split = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
text = text_split.split_documents(loader)

hf_embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# vectorstore = FAISS.from_documents(text, hf_embeddings)
# print(vectorstore)

docsearch = Chroma.from_documents(text, hf_embeddings)

# modelfile = '/home/chrispadwick/code/gpt4all-ui/models/gpt4all-lora-quantized.bin'
# modelfile = 'gptjmodels/ggml-gpt4all-j-v1.3-groovy.bin'
# modelfile = 'gptjmodels/ggml-gpt4all-j.bin'
modelfile = "models/ggml-alpaca-7b-q4.bin"

callbacks = [StreamingStdOutCallbackHandler(), StdOutCallbackHandler()]
# model = GPT4All(model=modelfile, callbacks=callbacks, n_ctx=500, n_threads=8)
# model = GPT4All(model=modelfile, backend='gptj', callbacks=callbacks, n_ctx=500, n_threads=8)
model = LlamaCpp(model_path=modelfile, callbacks=callbacks, n_ctx=1024, n_threads=12)

# This doesn't work for some reason.
qa = RetrievalQA.from_chain_type(
    llm=model, chain_type="refine", retriever=docsearch.as_retriever()
)
query = "Does Lady MacBeth appear in the text?"
print(qa.run(query))

prompt_template = """
{context}

Question: {question}

Explain your answer in detail.
"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain_type_kwargs = {"prompt": PROMPT}
qa = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=docsearch.as_retriever(),
    chain_type_kwargs=chain_type_kwargs,
)

# query = "Does Lady Macbeth appear in the text?"
# print(qa.run(query))
