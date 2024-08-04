
## Translation Agent WebUI

A Translation-Agent webUI based on Gradio library 🤗

### Preview

![webui](image.png)

**Features:**

- **Tokenized Text:**  Displays translated text with tokenization, highlighting differences between original and translated words.
- **Document Upload:** Supports uploading various document formats (PDF, TXT, DOC, etc.) for translation.
- **OpenAI compatible APIs Supports:** Supports for customizing any OpenAI compatible APIs.
- **Different LLM for reflection**: Now you can enable second Endpoint to use another LLM for reflection.

**Getting Started**

1. **Install Dependencies:**

    **Linux**
    ```bash
        git clone https://github.com/andrewyng/translation-agent.git
        cd translation-agent
        poetry install --with app
        poetry shell
    ```
    **Windows**
    ```bash
        git clone https://github.com/andrewyng/translation-agent.git
        cd translation-agent
        poetry install --with app
        poetry shell
    ```

2. **Set API Keys:**
   - Rename `.env.sample` to `.env`, you can add your API keys for each service:

     ```
     OPENAI_API_KEY="sk-xxxxx" # Keep this field
     GROQ_API_KEY="xxxxx"
     TOGETHER_API_KEY="xxxxx"
     ```
    - Then you can also set the API_KEY in webui.

3. **Run the Web UI:**

    **Linux**
    ```bash
    python app/app.py
    ```
    **Windows**
    ```bash
    python .\app\app.py
    ```

4. **Access the Web UI:**
   Open your web browser and navigate to `http://127.0.0.1:7860/`.

**Usage:**

1. Select your desired translation API from the Endpoint dropdown menu.
2. Input the source language, target language, and country(optional).
3. Input the source text or upload your document file.
4. Submit and get translation, the UI will display the translated text with tokenization and highlight differences.
5. Enable Second Endpoint, you can add another endpoint by different LLMs for reflection.
6. Using a custom endpoint, you can enter an OpenAI compatible API base url.

**Advanced Options:**

- **Nax tokens Per chunk:** Break down text into smaller chunks. LLMs have a limited context window, appropriate setting based on model information will ensure that the model has enough context to understand each individual chunk and generate accurate reponses. Defaults to 1000.

- **Temprature:** The sampling temperature for controlling the randomness of the generated text. Defaults to 0.3.

- **Request Per Minute:** This parameter affects the request speed. Rate limits are a common practice for APIs, such as RPM(Request Per Minute), TPM(Tokens Per Minute), please refer to the information of the API service provider and set the parameter value reasonably. Defaults to 60.

**DEMO:**

[Huggingface Demo](https://huggingface.co/spaces/vilarin/Translation-Agent-WebUI)
