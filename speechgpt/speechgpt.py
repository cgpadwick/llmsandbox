# Python program to translate
# speech to text and text to speech


import speech_recognition as sr
import pyttsx3
from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory


template = """Assistant is a large language model trained by OpenAI.

Assistant is designed to be able to assist with a wide range of tasks, from answering 
simple questions to providing in-depth explanations and discussions on a wide range of topics. 
As a language model, Assistant is able to generate human-like text based on the input it receives, 
allowing it to engage in natural-sounding conversations and provide responses that are coherent and 
relevant to the topic at hand.

Assistant is constantly learning and improving, and its capabilities are constantly evolving. 
It is able to process and understand large amounts of text, and can use this knowledge to provide 
accurate and informative responses to a wide range of questions. Additionally, Assistant is able 
to generate its own text based on the input it receives, allowing it to engage in discussions and 
provide explanations and descriptions on a wide range of topics.

Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable 
insights and information on a wide range of topics. Whether you need help with a specific question or
just want to have a conversation about a particular topic, Assistant is here to assist.

Assistant is aware that human input is being transcribed from audio and as such there may be some 
errors in the transcription. It will attempt to account for some words being swapped with 
similar-sounding words or phrases. Assistant will also keep responses concise, because human 
attention spans are more limited over the audio channel since it takes time to listen to a response.

{history}
Human: {human_input}
Assistant:"""

prompt = PromptTemplate(
    input_variables=["history", "human_input"], template=template)


chatgpt_chain = LLMChain(
    llm=OpenAI(temperature=0),
    prompt=prompt,
    verbose=True,
    memory=ConversationBufferWindowMemory(k=2),
)

# Initialize the recognizer
r = sr.Recognizer()


# Function to convert text to
# speech
def SpeakText(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


# Loop infinitely for user to
# speak

while 1:
    # Exception handling to handle
    # exceptions at the runtime
    try:
        # use the microphone as source for input.
        with sr.Microphone() as source2:
            # wait for a second to let the recognizer
            # adjust the energy threshold based on
            # the surrounding noise level
            r.adjust_for_ambient_noise(source2, duration=0.2)

            # listens for the user's input
            audio2 = r.listen(source2)

            # Using google to recognize audio
            MyText = r.recognize_google(audio2)
            MyText = MyText.lower()

            print("Did you say ", MyText)
            response_text = chatgpt_chain.predict(human_input=MyText)
            print(response_text)
            SpeakText(response_text)

    except sr.RequestError as e:
        print("Could not request results; {0}".format(e))

    except sr.UnknownValueError:
        print("unknown error occurred")
