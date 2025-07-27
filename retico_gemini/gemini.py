import numpy as np
import threading
import webrtcvad

from collections import deque
from typing import Callable, Literal, Union

from google.genai import types

from retico_core import AbstractModule, IncrementalUnit, UpdateMessage, UpdateType
from retico_core.text import TextIU
from retico_core.audio import AudioIU

from utils import Gemini

class GeminiModule(AbstractModule):
    
    @staticmethod
    def name():
        return "Gemini LLM Module"

    @staticmethod
    def description():
        return "Queries Gemini and streams response tokens."

    @staticmethod
    def input_ius():
        return [AudioIU, TextIU]

    @staticmethod
    def output_iu():
        return TextIU

    def __init__(
        self,
        system_instructions: str = "You are a helpful assistant.",
        model: str = "gemini-2.5-flash",
        temperature: float = 1.0,
        top_p: float = 1.0,
        seed: Union[int, Literal["random"]] = "random",
        max_output_tokens: int = 65535,
        thinking_budget: int = 2048,
        contents: Union[list[types.Content], None] = None,
        keep_context: bool = True,
        timeout_interval: float = 1.5,
        update_instructions_callback: Callable[[IncrementalUnit, UpdateType], Union[str, None]] = None,
        **kwargs,
    ):
        """
        Args:
            system_instructions (str, optional): The system instructions given to the model to guide its responses.
            model (str, optional): The name of the model to be used (find the list [here](https://ai.google.dev/gemini-api/docs/models)).
            temperature (float, optional): The temperature of the model. Although it varies by model (check [here](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/adjust-parameter-values)), it is usually ranging from 0.0 to 2.0, where lower values make the model more deterministic and higher values make it more creative.
            top_p (float, optional): The top_p value of the model, which influences the way the model selects the output tokens.
            seed (Union[int, Literal[&quot;random&quot;]], optional): The seed for token generation. Leave it at 'random' to let the model choose a pseudorandom seed (which is usually what we want to do for most chatbot applications), or choose an integer to make the generation deterministic. A fixed seed with the same input prompt, system instructions and model parameters will always yield the same output.
            max_output_tokens (int, optional): The maximum number of output tokens you allow the model to generate. WHen this limit is reached, the model will stop generating tokens and return the response, whether finished or not.
            thinking_budget (int, optional): The maximum number of tokens you allow the model to use for thinking, if it supports thinking.
            contents (Union[list[google.genai.types.Content], None], optional): A list of contents to initialize the conversation with. If None, the conversation will start empty and fresh.
            keep_context (bool, optional): Whether to keep the context of the conversation between turns. If True, the model will update its memory after each turn and use previous turns to generate responses. If False, it will treat each turn independently.
            timeout_interval (float, optional): The timeout (in seconds) after which to forcefully send the current input buffer to the model, if no new incremental unit has come meanwhile.
            update_instructions_callback (Callable[[retico_core.IncrementalUnit, retico_core.UpdateType], Union[str, None]], optional): A callback function that is called whenever a new incremental unit is received by this module. If None, nothing will happen. It can be used to update the system instructions dynamically. The function takes two parameters: the current `retico_core.IncrementalUnit` and its corresponding `retico_core.UpdateType` (in that order), and returns a string with the new instructions or None if no change is needed.
        """
        super().__init__(**kwargs)
        
        # Initialize Gemini client
        self.gemini = None # Will be immediately instantiated
        self.system_instructions = system_instructions
        self._update_instructions_callback = update_instructions_callback
        
        self.system_instructions = system_instructions
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.contents = contents if contents is not None else []
        
        self.keep_context = keep_context
        
        self.in_text_buffer = []  # List of yielded text chunks
        self.in_audio_buffer = deque()  # Queue of input audio chunks
        self.out_buffer = ""
        
        self._timeout_interval = timeout_interval
        self._timeout_timer = None
        self._last_iu = None
        self._speech_started = False
        
        self._set_model_instructions() # Initialize the Gemini client with the provided parameters
    
    def _set_model_instructions(self):
        system_instructions = self.system_instructions
        if self.gemini is not None:
            contents = self.gemini.contents # We transfer the previous messages to the new instance
        else:
            contents = None
        self.gemini = Gemini(system_instructions=system_instructions, model=self.model, temperature=self.temperature, top_p=self.top_p, seed=self.seed, max_output_tokens=self.max_output_tokens, thinking_budget=self.thinking_budget, contents=contents)

    def process_update(self, update_message: UpdateMessage):
        for iu, ut in update_message:
            if isinstance(iu, TextIU):
                text = iu.text
                self._last_iu = iu
                
                if ut == UpdateType.ADD:
                    self.in_text_buffer.append(text)
                    self._reset_timer() # If no IU is coming after 1.5 seconds, we process the input

                if getattr(iu, "committed", False) or ut == UpdateType.COMMIT:
                    # If the incoming IU is committed, we cancel the timer (no elif because iu.committed can be set for other update types)
                    if self._timeout_timer is not None:
                        self._timeout_timer.cancel()
                    # And immediately process the input
                    self._on_timeout()
                
                if ut == UpdateType.REVOKE:
                    self.revoke(iu)
                
            elif isinstance(iu, AudioIU):
                self._last_iu = iu
            
                if ut == UpdateType.ADD:
                    self.in_audio_buffer.append(iu.raw_audio)

                np_audio = np.frombuffer(b"".join(self.in_audio_buffer), dtype="<i2").astype(np.float32) / 32768
                rms = np.sqrt(np.mean(np_audio**2))
                
                if rms > 0.015: # We consider that this level of voice activity means the user is speaking
                    self._speech_started = True
                
                if self._speech_started and self._has_speech(b"".join(self.in_audio_buffer), iu.rate) and self.is_tail_silence_or_noise(np_audio, iu.rate):
                    audio_input = b"".join(self.in_audio_buffer)
                    self.in_audio_buffer.clear()  # Clear the buffer after processing
                    self._speech_started = False
                    
                    if self.keep_context:
                        for chunk in self.gemini.add_audio_turn(audio_input):
                            self.process_chunk(chunk)
                    else:
                        for chunk in self.gemini.single_audio_turn(audio_input):
                            self.process_chunk(chunk)

            if callable(self._update_instructions_callback):
                new_instructions = self._update_instructions_callback(iu, ut)
                if isinstance(new_instructions, str) and new_instructions != self.system_instructions:
                    self.system_instructions = new_instructions
                    self._set_model_instructions()
    
    def _on_timeout(self):
        user_input = ' '.join(self.in_text_buffer).strip()
        self.in_text_buffer.clear()
        if not user_input:
            return
        
        if self.keep_context:
            for chunk in self.gemini.add_text_turn(user_input):
                self.process_chunk(chunk)
        else:
            for chunk in self.gemini.single_text_turn(user_input):
                self.process_chunk(chunk)
    
    def _reset_timer(self):
        if self._timeout_timer is not None:
            self._timeout_timer.cancel()
        self._timeout_timer = threading.Timer(self._timeout_interval, self._on_timeout)
        self._timeout_timer.daemon = True
        self._timeout_timer.start()
    
    def process_chunk(self, chunk: str):
        um = UpdateMessage()
        self.out_buffer += chunk
        out_iu = TextIU(iuid=0, previous_iu=self._last_iu)
        out_iu.payload = out_iu.text = self.out_buffer
        self.out_buffer = ""
        um.add_iu(out_iu, UpdateType.ADD)
        self.append(um)
    
    def _frame_generator(self, frame_duration_ms: int, audio: bytes, sample_rate: int):
        """
        Split `audio` (PCM16 bytes) into frames of `frame_duration_ms` milliseconds.
        Yields tuples of (frame_bytes, timestamp_ms).
        """
        bytes_per_frame = int(sample_rate * (frame_duration_ms / 1000.0)) * 2
        offset = 0
        timestamp = 0.0
        while offset + bytes_per_frame <= len(audio):
            yield audio[offset : offset + bytes_per_frame], timestamp
            timestamp += frame_duration_ms
            offset += bytes_per_frame

    def _has_speech(
        self,
        audio: bytes,
        sample_rate: int,
        frame_duration_ms: int = 30,
        aggressiveness: int = 2,
        speech_frame_threshold: float = 0.1
    ) -> bool:
        """
        Return True if at least `speech_frame_threshold` proportion of frames
        contain speech according to WebRTC VAD.
        """
        vad = webrtcvad.Vad(aggressiveness)
        frames = list(self._frame_generator(frame_duration_ms, audio, sample_rate))
        if not frames:
            return False
        speech_frames = 0
        for frame_bytes, _ in frames:
            if vad.is_speech(frame_bytes, sample_rate):
                speech_frames += 1
        return (speech_frames / len(frames)) >= speech_frame_threshold

    @staticmethod
    def is_tail_silence_or_noise(buffer: np.ndarray, rate: int, silent_tail_size: float = 1.0, silence_max_rms_energy_threshold = 0.01) -> bool:
        """
        Returns True if the last `tail_size` seconds of `buffer`
        are essentially silence (low RMS) or noise (high spectral flatness).

        :param buffer: the audio buffer to check
        :param rate: the sample rate of the audio buffer (in Hz)
        :param silent_tail_size: the size of the tail to check whether it's deemed silent or not (in seconds)
        :param silence_max_rootmeansquare_energy_threshold: the maximum root mean square energy to consider the tail as silence (closer to 0: silence)
        """

        n_tail = int(silent_tail_size * rate)
        if buffer.size < n_tail:
            return False
        tail = buffer[-n_tail:]
        rms = np.sqrt(np.mean(tail**2))
        if rms < silence_max_rms_energy_threshold:
            return True
        return False