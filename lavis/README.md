# Document Q&A On A Given Image File

A demo showing how to run image Q&A on an image with Salesforce inference models.  

In this demo an image file called `Singapore_Skyline_2019-10.jpg` is loaded into the program.  A caption for the image is generated using the Salesforce blip_caption model.  Then the program enters a loop where the user can ask questions. The image and the question are then passed to the blip_vqa model and the answer is printed out to the screen.

This demo uses the [LAVIS](https://github.com/salesforce/LAVIS) library from Salesforce.

## Installation

Clone the repo and create a virtual env as follows:

`virtualenv -p python3.8 venv`

You might need to install python3.8 and virtualenv too (exercise left to the reader).

Source the virtual env:

`source venv/bin/activate`

Install the pip packages:

`pip install -r requirements.txt`

## Running The Code

- Source the virtual env `source venv/bin/activate`
- Run the demo with `python imageqanda.py --image Singapore_Skyline_2019-10.jpg`
- Ask questions and press return!
