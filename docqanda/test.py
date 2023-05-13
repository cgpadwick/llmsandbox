from langchain.llms import GPT4All
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.indexes import GraphIndexCreator

textfile = "input.txt"
loader = TextLoader(textfile)

# index = VectorstoreIndexCreator().from_loaders([loader])

# query = "What did MENENIUS say about bats and clubs"
# index.query(query)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cpu"}
hf = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
print(hf)


# modelfile = '/home/chrispadwick/code/gpt4all-ui/models/gpt4all-lora-quantized.bin'

# model = GPT4All(model=modelfile, n_ctx=512, n_threads=8)
# response = model("Once upon a time, ")
# print(response)
