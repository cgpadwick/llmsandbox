# Document Q&A On A Given Text File

A demo showing how to run document Q&A on a text file with a user hosted Large Language Model.  This can be useful for companies or individuals who are not comfortable sending their private data to cloud hosted services (e.g. OpenAI).

In this demo a text file called `shakespeare.txt` which contains all of Shakespeare's plays is indexed using HuggingFace embeddings into a Vectorstore.  Then one of the pre-downloaded models is loaded, and a RetreivalQA chain from langchain is used to run the provided query on the document.  The response from the model is printed out on the screen.

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

This demo supports a few different models.  Download one or more of them and stick them in the `models` folder.

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

## Running The Code

- Source the virtual env `source venv/bin/activate`
- Run the demo with `python docqanda.py --inputfile shakespeare.txt`
  - Run new queries using the --query flag
  - Adjust --threads for number of threads and --n_ctx for the context length
  - Set the OpenAI enviroment variable (bash) with `export OPENAI_API_KEY="your key goes here"` if you choose --model openai

