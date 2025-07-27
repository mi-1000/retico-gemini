from retico_core import network
from retico_core.audio import MicrophoneModule
from retico_core.debug import CallbackModule
from retico_googleasr import GoogleASRModule

from gemini import GeminiModule

def callback(um):
    for iu, ut in um:
        print(f"{ut}: {iu.payload}")

def set_instructions(iu, ut):
    if hasattr(iu, "text") and "France" in iu.text:
        return "You are now a French-speaking assistant. Finish all of your utterances with 'Vive la France!'"

if __name__ == "__main__":
    mic = MicrophoneModule(rate=16000)
    asr = GoogleASRModule(rate=16000)
    llm = GeminiModule("You are a knowledgeable assistant holding a conversation with the user. Talk as if you were a noble scientist from the Renaissance.")#, update_instructions_callback=set_instructions)
    debug = CallbackModule(callback)
    
    mic.subscribe(llm)
    llm.subscribe(debug)
    
    network.run(mic)
    input("Running...\n")
    network.stop(mic)
