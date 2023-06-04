import argparse
import multiprocessing
import os
from typing import Union

from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import GPT4All, LlamaCpp, OpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.stdout import StdOutCallbackHandler


def warn_openai(model: str):
    """
    Warns the user about the potential cost and data sharing implications of using the OpenAI model.

    Args:
        model (str): The name of the model being used.

    Returns:
        None.
    """

    if model == "openai":
        msg = """
        WARNING: Using the OpenAI model will cost money, and will share your data with
        OpenAI.  Review OpenAIs terms of service at https://openai.com/policies/terms-of-use
        before proceeding.  Type Y to continue.
        """
        ans = input(msg)
        if ans != "Y":
            exit(1)


def check_args(inputfile, dbpath):
    """
    Check if either an input file or a database file exists.

    Args:
        inputfile (str): The path to the input file.
        dbpath (str): The path to the database file.

    Raises:
        Exception: If neither inputfile nor dbpath are specified.
        FileNotFoundError: If inputfile or dbpath does not exist.

    Returns:
        None
    """

    if not inputfile and not dbpath:
        raise Exception("Error: one of --inputfile or --dbpath must be specified")

    if inputfile:
        if not os.path.exists(inputfile):
            raise FileNotFoundError(f"The file {inputfile} does not exist.")

    if dbpath:
        if not os.path.exists(dbpath):
            raise FileNotFoundError(f"The file {dbpath} does not exist.")


def get_embeddings(inputfile: str, dbpath: str, isOpenAI: bool):
    """
    generate a retriever object from a text file or a faiss database

    Args:
        inputfile (str): path to the text file
        dbpath (str): path to the faiss database
        isOpenAI (bool): flag to indicate whether to use the sentence-transformers or the openai GPT embeddings

    Returns:
        Retriever: a Retriever object that can be used to retrieve similar documents
    """
    if inputfile:
        docsearch = create_embeddings_from_textfile(inputfile, isOpenAI=False)
    elif dbpath:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        docsearch = FAISS.load_local(dbpath, embeddings)
    else:
        raise Exception("Error: one of --inputfile or --dbpath must be specified")

    return docsearch.as_retriever()


def create_embeddings_from_textfile(
    textfile: str, isOpenAI: bool
) -> VectorStoreRetriever:
    """
    Creates embeddings from the given text file and returns a VectorStoreRetriever object to perform
    vector searches on the embeddings.
    :param textfile: The path to the text file containing the documents to be embedded.
    :param isOpenAI: bool controlling whether or not to use OpenAI's embeddings
    :return: A VectorStoreRetriever object containing the embeddings of the documents in the text file.
    :rtype: VectorStoreRetriever
    """

    if not os.path.exists(path=textfile):
        raise FileNotFoundError(f"The file {textfile} does not exist.")

    root_ext = os.path.splitext(textfile)
    if root_ext[1].lower() == ".pdf":
        print("Using the pdf document loader")
        loader = UnstructuredPDFLoader(textfile, mode="elements")
        text = loader.load()
    elif root_ext[1].lower() == ".txt":
        print("Using the text document loader")
        loader = TextLoader(textfile).load()
        text_split = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)
        text = text_split.split_documents(loader)
    else:
        raise Exception("unsupported data type")

    if isOpenAI:
        embeddings = OpenAIEmbeddings()
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    docsearch = Chroma.from_documents(text, embeddings)
    return docsearch


def load_model(
    model_name: str, n_ctx: int, n_threads: int
) -> Union[GPT4All, LlamaCpp, OpenAI]:
    """
    Loads and returns a machine learning model based on the provided `model_name`.

    :param model_name: str, the name of the model to load.
    :return: Union[GPT4All, LlamaCpp, OpenAI], the loaded model object.
    :raises ValueError: if an invalid `model_name` is provided.
    """
    callbacks = [StreamingStdOutCallbackHandler(), StdOutCallbackHandler()]
    n_threads = multiprocessing.cpu_count() if n_threads == -1 else n_threads

    if not model_name == "openai":
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
    elif model_name == "openai":
        model = OpenAI()
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputfile",
        required=False,
        type=str,
        help="Text file to index",
    )
    parser.add_argument(
        "--dbpath",
        required=False,
        type=str,
        help="Path to local FAISS embeddings database",
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
            "openai",
        ],
        default="ggml-gpt4all-j-v1.3-groovy.bin",
        help="LLM Model name to use.  Must be downloaded to the models directory before starting.",
    )
    parser.add_argument("--n_ctx", type=int, default=1024, help="Context size")
    parser.add_argument("--threads", type=int, default=-1, help="Number of threads")
    args = parser.parse_args()

    check_args(args.inputfile, args.dbpath)
    warn_openai(args.model)
    docretriever = get_embeddings(args.inputfile, args.dbpath, args.model == "openai")
    model = load_model(args.model, args.n_ctx, args.threads)
    qa = RetrievalQA.from_chain_type(
        llm=model, chain_type="stuff", retriever=docretriever
    )

    while True:
        print("\n\n")
        query = input("Type a query and press return! ")
        result = qa.run(query)
        print(result)
