# speechgpt

A speech to text, text to speech barebones demo in python, using OpenAI's text-davinci-003 model.

The demo works as follows.  The user asks a question.  The speech is converted to text using the python speech_recognition package.  Then the text is sent to the OpenAI LLM model interface in langchain.  The model returns a response and the response is converted to speech with using the pyttsx3 package.

## Installation

Clone the repo and create a virtual env as follows:

`virtualenv -p python3.8 venv`

You might need to install python3.8 and virtualenv too (exercise left to the reader).

Source the virtual env:

`source venv/bin/activate`

Install the pip packages:

`pip install -r requirements.txt`

## OpenAI Api Key

You will need an OpenAI API key for this demo to work.  To get a key:

* Create an account on platform.openai.com
* Under "Personal" click "View API Keys"
* Create a new secret key, and copy it to your clipboard.
* You will need to save this secret key somewhere.

## Running The Code

* Source the virtual env `source venv/bin/activate`
* Set the OpenAI enviroment variable (bash) with `export OPENAI_API_KEY="your key goes here"`
* Run the demo with `python speechgpt.py`
