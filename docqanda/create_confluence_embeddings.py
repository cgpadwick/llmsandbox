import argparse
from datetime import datetime
from typing import List
import os

from atlassian import Confluence

from bs4 import BeautifulSoup

from dotenv import dotenv_values
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import ConfluenceLoader
from langchain.schema import Document


ALL_RESULTS = []
CONFLUENCE_OBJ = None
LOADER = None


def check_config(config):
    """
    Validates the provided configuration dictionary to ensure that the CONFLUENCE_USER and CONFLUENCE_API_TOKEN
    keys are present and not empty. Raises an exception if either key is missing or has an empty value.

    Args:
        config (dict): A dictionary containing the configuration values.

    Raises:
        Exception: If CONFLUENCE_USER or CONFLUENCE_API_TOKEN are missing or have an empty value.
    """
    if not "CONFLUENCE_USER" in config or config["CONFLUENCE_USER"] == "":
        raise Exception("CONFLUENCE_USER must be specified in the .env file")

    if not "CONFLUENCE_API_TOKEN" in config or config["CONFLUENCE_API_TOKEN"] == "":
        raise Exception("CONFLUENCE_API_TOKEN must be specified in the .env file")


def descend_from_root(root_id):
    """
    Recursively descends through a Confluence page hierarchy, starting from the page with the given 'root_id',
    and collects all pages under it into a list called 'ALL_RESULTS'.

    :param root_id: The ID of the root page to start descending from.
    :type root_id: int
    :return: None
    :rtype: None
    """
    print(f'Indexing {page["title"]}')
    page = CONFLUENCE_OBJ.get_page_by_id(root_id)
    text = BeautifulSoup(page["body"]["storage"]["value"], "lxml").get_text()
    doc = Document(
        page_content=text,
        metadata={
            "title": page["title"],
            "id": page["id"],
            "source": page["_links"]["webui"],
        },
    )
    ALL_RESULTS.append(doc)

    # docs = LOADER.load(page_ids=[root_id])
    # for d in docs:
    #    ALL_RESULTS.append(d)

    children = CONFLUENCE_OBJ.get_child_pages(root_id)
    for page in children:
        descend_from_root(page["id"])


def generate_embeddings(dbdirectory, chunksize, overlap):
    """
    Generates embeddings for text data using HuggingFaceEmbeddings and RecursiveCharacterTextSplitter.

    :param dbdirectory: The directory where the embeddings will be saved
    :type dbdirectory: str
    :param chunksize: The size of text chunks to be processed at a time
    :type chunksize: int
    :param overlap: The amount of overlap between adjacent text chunks
    :type overlap: int
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=chunksize, chunk_overlap=overlap
    )
    chunked_docs = text_split.split_documents(ALL_RESULTS)

    print("Generating embeddings...")
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    print("Complete.  Saving embeddings...")
    fname = datetime.now().strftime(f"{args.rootpageid}_embeddings_%H_%M_%S")
    vectorstore.save_local(os.path.join(args.dbdirectory, fname))
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--confluenceurl",
        required=True,
        type=str,
        help="The URL of the confluence space, e.g. https://mycompanyname.atlassian.net",
    )
    parser.add_argument(
        "--rootpageid",
        required=True,
        type=str,
        help="The page id of the root confluence page to be indexed",
    )
    parser.add_argument(
        "--dbdirectory",
        type=str,
        required=False,
        default="vectorstoredb",
        help="directory to write the vectorstore database to.",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        required=False,
        default=500,
        help="Chunk size to use for splitting up documents.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        required=False,
        default=25,
        help="Overlap parameter to be used for adjacent documents when they are split up.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dbdirectory):
        os.mkdir(args.dbdirectory)

    config = dotenv_values()
    check_config(config)

    CONFLUENCE_OBJ = Confluence(
        url=args.confluenceurl,
        username=config["CONFLUENCE_USER"],
        password=config["CONFLUENCE_API_TOKEN"],
    )

    LOADER = ConfluenceLoader(
        url=args.confluenceurl,
        username=config["CONFLUENCE_USER"],
        api_key=config["CONFLUENCE_API_TOKEN"],
    )

    descend_from_root(args.rootpageid)
    generate_embeddings(args.dbdirectory, args.chunksize, args.overlap)
