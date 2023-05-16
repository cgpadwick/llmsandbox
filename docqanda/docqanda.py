import argparse
import multiprocessing
import os
from typing import Union

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All, LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler


def create_embeddings(textfile: str) -> VectorStoreRetriever:
    """
    Creates embeddings from the given text file and returns a VectorStoreRetriever object to perform
    vector searches on the embeddings.
    :param textfile: The path to the text file containing the documents to be embedded.
    :type textfile: str
    :return: A VectorStoreRetriever object containing the embeddings of the documents in the text file.
    :rtype: VectorStoreRetriever
    """

    if not os.path.exists(path=textfile):
        raise FileNotFoundError(f"The file {textfile} does not exist.")

    loader = TextLoader(textfile).load()
    text_split = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
    text = text_split.split_documents(loader)

    hf_embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    docsearch = Chroma.from_documents(text, hf_embeddings)
    return docsearch.as_retriever()


def load_model(model_name: str, n_ctx: int, n_threads: int) -> Union[GPT4All, LlamaCpp]:
    """
    Loads and returns a machine learning model based on the provided `model_name`.

    :param model_name: str, the name of the model to load.
    :return: Union[GPT4All, LlamaCpp], the loaded model object.
    :raises ValueError: if an invalid `model_name` is provided.
    """
    callbacks = [StreamingStdOutCallbackHandler(), StdOutCallbackHandler()]
    n_threads = multiprocessing.cpu_count() if n_threads == -1 else n_threads
    modelfile = f"models/{model_name}"
    if not os.path.exists(modelfile):
        raise ValueError(f"Invalid model path: {modelfile}")

    if model_name in ("gpt4all-lora-quantized.bin", "ggml-gpt4all-l13b-snoozy.bin"):
        model = GPT4All(
            model=modelfile, callbacks=callbacks, n_ctx=n_ctx, n_threads=n_threads
        )
    elif model_name == "ggml-gpt4all-j-v1.3-groovy.bin":
        model = GPT4All(
            model=modelfile,
            backend="gptj",
            callbacks=callbacks,
            n_ctx=n_ctx,
            n_threads=n_threads,
        )
    elif model_name == "ggml-alpaca-7b-q4.bin":
        model = LlamaCpp(
            model_path=modelfile, callbacks=callbacks, n_ctx=n_ctx, n_threads=n_threads
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        required=True,
        type=str,
        help="Input text file to index",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=False,
        choices=[
            "ggml-gpt4all-j-v1.3-groovy.bin",
            "ggml-alpaca-7b-q4.bin",
            "ggml-gpt4all-l13b-snoozy.bin",
            "gpt4all-lora-quantized.bin",
        ],
        default="ggml-gpt4all-j-v1.3-groovy.bin",
        help="LLM Model name to use.  Must be downloaded to the models directory before starting.",
    )
    parser.add_argument("--n_ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--threads", type=int, default=-1, help="Number of threads")
    parser.add_argument(
        "--query",
        type=str,
        default="Does MENENIUS mention bats and clubs? Explain your answer",
        help="Query",
    )
    args = parser.parse_args()
    
    docsearch = create_embeddings(args.inputfile)
    model = load_model(args.model, args.n_ctx, args.threads)

    qa = RetrievalQA.from_chain_type(llm=model, chain_type="stuff", retriever=docsearch)
    print(qa.run(args.query))
