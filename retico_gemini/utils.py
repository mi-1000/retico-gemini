import io
import os
import wave

from dotenv import load_dotenv
from typing import Any, Generator, Literal, Union

from google import genai
from google.genai import types

load_dotenv()


class Gemini:
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
    ):
        self.system_instructions = system_instructions
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        self.max_output_tokens = max_output_tokens
        self.thinking_budget = thinking_budget
        self.contents: list[types.Content] = contents if contents is not None else []
        self.client = genai.Client(
            api_key=os.environ.get("GOOGLE_API_KEY", "missing-api-key"),
        )

    def add_text_turn(self, user_input: str) -> Generator[str, Any, None]:
        """
        Add a text user turn and stream back text chunks.
        """
        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_input)],
            )
        )

        response = ""
        for chunk in self._generate_text(self.contents):
            if chunk is not None:
                response += chunk
                yield chunk

        self.contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=response)],
            )
        )

    def add_audio_turn(self, audio_bytes: bytes) -> Generator[str, Any, None]:
        """
        Add a user turn with raw audio and stream back the model's textual response.
        """

        buffer = io.BytesIO()  # Convert PCM bytes to WAV
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)  # mono
            wf.setsampwidth(2)  # 16‑bit = 2 bytes
            wf.setframerate(16000)  # 16000 Hz by default
            wf.writeframes(audio_bytes)
        wav_bytes = buffer.getvalue()

        self.contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
            )
        )

        response = ""
        for chunk in self._generate_text(self.contents):
            if chunk is not None:
                response += chunk
                yield chunk

        self.contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=response)],
            )
        )

    def single_text_turn(self, user_input: str) -> Generator[str, Any, None]:
        """
        One-off text turn without context.
        """
        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=user_input)])
        ]
        
        for chunk in self._generate_text(contents):
            if chunk is not None:
                yield chunk

    def single_audio_turn(self, audio_bytes: bytes) -> Generator[str, Any, None]:
        """
        One-off audio turn without context.
        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(audio_bytes)
        wav_bytes = buffer.getvalue()

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")],
            )
        ]
        
        for chunk in self._generate_text(contents):
            if chunk is not None:
                yield chunk

    def clear_context(self):
        self.contents.clear()

    def _generate_text(
        self, contents: list[types.Content]
    ) -> Generator[str, Any, None]:
        cfg = types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=self.top_p,
            seed=self.seed if isinstance(self.seed, int) else None,
            max_output_tokens=self.max_output_tokens,
            system_instruction=[types.Part.from_text(text=self.system_instructions)],
            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget),
        )
        for chunk in self.client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=cfg,
        ):
            yield chunk.text
