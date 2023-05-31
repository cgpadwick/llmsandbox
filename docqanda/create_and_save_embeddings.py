import argparse
from datetime import datetime
from typing import List
import os

from atlassian import Confluence

from dotenv import dotenv_values
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import ConfluenceLoader
from langchain.schema import Document


ALL_RESULTS = []
CONFLUENCE_OBJ = None
LOADER = None


def check_config(config):
    if not "CONFLUENCE_USER" in config or config["CONFLUENCE_USER"] == "":
        raise Exception("CONFLUENCE_USER must be specified in the .env file")

    if not "CONFLUENCE_API_TOKEN" in config or config["CONFLUENCE_API_TOKEN"] == "":
        raise Exception("CONFLUENCE_API_TOKEN must be specified in the .env file")


def descend_from_root(root_id):
    page = CONFLUENCE_OBJ.get_page_by_id(root_id)
    print(f'Indexing {page["title"]}')
    docs = LOADER.load(page_ids=[root_id])
    for d in docs:
        ALL_RESULTS.append(d)

    children = CONFLUENCE_OBJ.get_child_pages(root_id)
    for page in children:
        descend_from_root(page["id"])


def generate_embeddings(dbdirectory):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Generating embeddings...")
    vectorstore = FAISS.from_documents(ALL_RESULTS, embeddings)
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
    generate_embeddings(args.dbdirectory)
