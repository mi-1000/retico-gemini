# retico-gemini

A Retico module that supports the Google Gemini API. This module was initially written for [this network](https://github.com/mi-1000/retico-language-practice-network) and then adapted for general purpose.

## Contents

- `gemini.py`: A Retico module that defines incremental processing logic to integrate within a network. On top of the model parameters, it also allows you to set whether to keep track of the conversation in the model's context window or not, and to dynamically change the system instructions using a callback function. Both text and audio are accepted as input IUs. The output is text.
- `utils.py`: A Python module that defines the logic for dialoguing with the Gemini API and streaming the response tokens.
- `test_network.py`: A simple script for running a test network that uses this module.

> [!NOTE]
> The proposed network will only work with English input as is, due to limitations with `GoogleASRModule`. To test it out while sending audio to the API instead of text, simply subscribe the Gemini module directly to the Microphone one, and remove the ASR module from the pipeline.

## Setup

> [!WARNING]
> It is recommended to use **Python 3.9.22** with this module in order to avoid any compatibility issues. Although the module on its own is compatible with ulterior Python versions, other Retico modules that may be linked to this module might not be optimized for later Python versions.

- If you haven't yet, create a virtual environment and activate it:

  - **Linux/MacOS**:

    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
  - **Windows**:

    ```powershell
    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    ```

    If you encounter an error when activating the virtual environment, retry the above command after running the following line:

    ```powershell
    Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```
- Set the environment variable `GOOGLE_API_KEY`. The simplest way is to create an `.env` file in the root folder of this repository, and write the following contents:

  ```dotenv
   GOOGLE_API_KEY=your-api-key
  ```

  The API key will be automatically loaded by the module.

  You can get an API key for free on the [Google Cloud website](https://cloud.google.com/vertex-ai/generative-ai/docs/start/api-keys).

> [!NOTE]
> A single API key should work for both the Google Gemini and Google ASR APIs, but in case not, follow the setup tutorial for the Google ASR Retico module [here](https://github.com/retico-team/retico-googleasr).

## Installation

- To automatically install the module, run:

  ```bash
  pip install --upgrade pip setuptools wheel git+https://github.com/retico-team/retico-gemini
  ```
- If you want to install it from source, run the following:

  ```bash
  git clone https://github.com/retico-team/retico-gemini
  cd retico_gemini
  pip install -r requirements.txt
  ```
- To try out the test network:

  ```bash
  python test_network.py
  ```
