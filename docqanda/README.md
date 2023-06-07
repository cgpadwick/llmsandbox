# Document Q&A Using Privately Hosted Models

A demo showing how to run document Q&A on user files with a privately hosted Large Language Model.  This can be useful for companies or individuals who are not comfortable sending their private data to cloud hosted services (e.g. OpenAI).

In this demo a text file called `shakespeare.txt` which contains all of Shakespeare's plays is indexed using HuggingFace embeddings into a Vectorstore.  Then one of the pre-downloaded models is loaded, and a RetrievalQA chain from langchain is used to run the provided query on the document.  The response from the model is printed out on the screen.  The demo also supports generating document Q&A on a body of confluence documents stored in a vector DB.

OpenAI models can be used with this demo by selecting `--model openai` and answering `Y` to the warning.

## Installation

Clone the repo and create a virtual env as follows:

`virtualenv -p python3.8 venv`

You might need to install python3.8 and virtualenv too (exercise left to the reader).

Source the virtual env:

`source venv/bin/activate`

Install the pip packages:

`pip install -r requirements.txt`

## Downloading The Models

This demo supports a few different models.  Download one or more of them and stick them in the `models` folder.  You don't need them all, you only need one.  I would recommend ggml-gpt4all-j-v1.3-groovy.bin but better answers will likely come from ggml-gpt4all-l13b-snoozy.bin at the cost of slower inference.

- [ggml-gpt4all-j-v1.3-groovy.bin](https://gpt4all.io/models/ggml-gpt4all-j-v1.3-groovy.bin)
- [ggml-gpt4all-l13b-snoozy.bin](https://gpt4all.io/models/ggml-gpt4all-l13b-snoozy.bin)
- [ggml-alpaca-7b-q4.bin](https://huggingface.co/Sosaka/Alpaca-native-4bit-ggml/tree/main)
- [gpt4all-lora-quantized.bin](https://huggingface.co/aryan1107/gpt4all-llora/resolve/main/gpt4all-lora-quantized.bin)

## Optional OpenAI Api Key

If you choose `--model openai` then you will need an OpenAI API key for this demo to work.  To get a key:

* Create an account on platform.openai.com
* Under "Personal" click "View API Keys"
* Create a new secret key, and copy it to your clipboard.
* You will need to save this secret key somewhere.

## Running On A Text File

- Source the virtual env `source venv/bin/activate`
- Run the demo with `python docqanda.py --inputfile shakespeare.txt`
  - Run new queries using the --query flag
  - Adjust `--threads` for number of threads and `--n_ctx` for the context length
  - Set the OpenAI environment variable (bash) with `export OPENAI_API_KEY="your key goes here"` if you choose `--model openai`

## Running Document Q&A On A Set Of Confluence Documents

You can run document Q&A on a set of confluence documents!  It is a two step process.  First, you create a vectorstore index from confluence documents and store that index on disk.  Then you run `docqanda.py` and point it at the index.  Follow the steps below.

### Create a .env file with the confluence credentials

Create a file called ".env" in the docqanda directory.  The file should contain the following contents:

```
CONFLUENCE_USER="yourconfluenceemail"
CONFLUENCE_API_TOKEN="yourapitoken"
```
The user and api token can be be retreived from the confluence dashboard.

### Identify The Page You'd Like To Index and Generate The Index

The help for the `create_confluence_embeddings.py` script is shown below.

```
python create_confluence_embeddings.py --help
usage: create_confluence_embeddings.py [-h] --confluenceurl CONFLUENCEURL --rootpageid ROOTPAGEID
                                       [--dbdirectory DBDIRECTORY] [--chunksize CHUNKSIZE] [--overlap OVERLAP]

optional arguments:
  -h, --help            show this help message and exit
  --confluenceurl CONFLUENCEURL
                        The URL of the confluence space, e.g. https://mycompanyname.atlassian.net
  --rootpageid ROOTPAGEID
                        The page id of the root confluence page to be indexed
  --dbdirectory DBDIRECTORY
                        directory to write the vectorstore database to.
  --chunksize CHUNKSIZE
                        Chunk size to use for splitting up documents.
  --overlap OVERLAP     Overlap parameter to be used for adjacent documents when they are split up
``` 

They utility is designed to accept a root page id and will index the given page and recursively index all children pages found underneath the root.  This is done to give the user more control over what gets indexed (the confluence loader in langchain by default indexes everything in a space). With this utility you can get more fine-grained control over what gets indexed.

You can find the root page id by navigating to the page you want to index in confluence and recording the integer in the URL after the "pages" element.

You also need the base confluence url, which would typically be something like `https://companyname.atlassian.net`

As an example, let's say that your company was called "acme" and the page you wanted to index was called `https://acme.atlassian.net/wiki/spaces/ACME/pages/509870119/Acme+Software`.  Then you could run the indexing utility like this:

```
python create_confluence_embeddings.py --confluenceurl https://acme.atlassian.net --rootpageid 509870119
```

This will launch a process that will index that page and recursively index everything underneath it.  The documents will be chunked up (chunking controlled by the `--chunksize` and `--overlap` parameters) and HuggingFace embeddings will be created for each document.  The embeddings will be stored in a FAISS vector database on disk in the `vectorstoredb` directory.  The index will be stored in a directory named `vectorstoredb/509870199_embeddings_YYYY_MM_DD_HH_MM_SS`


### Run Document Q&A With The Stored Index

Now you can run document Q&A on the stored embeddings db like this:

```
python docqanda.py --dbpath vectorstoredb/509870199_embeddings_YYYY_MM_DD_HH_MM_SS --model ggml-alpaca-7b-q4.bin
```

And now the queries will be returned against the indexed confluence documents!
