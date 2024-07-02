import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import re
import gradio as gr
from glob import glob
from app.webui.process import model_load, diff_texts, translator, translator_sec
from llama_index.core import SimpleDirectoryReader

def huanik(
    endpoint,
    model,
    api_key,
    choice,
    endpoint2,
    model2,
    api_key2,
    source_lang,
    target_lang,
    source_text,
    country,
    max_tokens,
    context_window,
    num_output,
):

    if not source_text or source_lang == target_lang:
        raise gr.Error("Please check that the content or options are entered correctly.")

    try:
        model_load(endpoint, model, api_key, context_window, num_output)
    except Exception as e:
        raise gr.Error(f"An unexpected error occurred: {e}")

    source_text = re.sub(r'(?m)^\s*$\n?', '', source_text)

    if choice:
        init_translation, reflect_translation, final_translation = translator_sec(
            endpoint2=endpoint2,
            model2=model2,
            api_key2=api_key2,
            context_window=context_window,
            num_output=num_output,
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            country=country,
            max_tokens=max_tokens,
        )

    else:
        init_translation, reflect_translation, final_translation = translator(
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
            country=country,
            max_tokens=max_tokens,
        )

    final_diff = gr.HighlightedText(
        diff_texts(init_translation, final_translation),
        label="Diff translation",
        combine_adjacent=True,
        show_legend=True,
        visible=True,
        color_map={"removed": "red", "added": "green"})

    return init_translation, reflect_translation, final_translation, final_diff

def update_model(endpoint):
    endpoint_model_map = {
        "Groq": "llama3-70b-8192",
        "OpenAI": "gpt-4o",
        "Cohere": "command-r",
        "TogetherAI": "Qwen/Qwen2-72B-Instruct",
        "Ollama": "llama3",
        "Huggingface": "mistralai/Mistral-7B-Instruct-v0.3"
    }
    return gr.update(value=endpoint_model_map[endpoint])

def read_doc(file):
    docs = SimpleDirectoryReader(input_files=[file]).load_data()
    texts = ""
    for doc in docs:
        texts += doc.text
    texts = re.sub(r'(?m)^\s*$\n?', '', texts)
    return texts

def enable_sec(choice):
    if choice:
        return gr.update(visible = True), gr.update(visible = True), gr.update(visible = True)
    else:
        return gr.update(visible = False), gr.update(visible = False), gr.update(visible = False)

def update_menu(visible):
    return not visible, gr.update(visible=not visible)

def export_txt(strings):
    os.makedirs("outputs", exist_ok=True)
    base_count = len(glob(os.path.join("outputs", "*.txt")))
    file_path = os.path.join("outputs", f"{base_count:06d}.txt")
    print(file_path)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(strings)
    return gr.update(value=file_path, visible=True)

TITLE = """
    <div style="display: inline-flex;">
        <div style="margin-left: 6px; font-size:32px; color: #6366f1"><b>Translation Agent</b> WebUI</div>
    </div>
"""

CSS = """
    h1 {
        text-align: center;
        display: block;
        height: 10vh;
        align-content: center;
    }
    footer {
        visibility: hidden;
    }
    .menu_btn {
        width: 48px;
        height: 48px;
        max-width: 48px;
        min-width: 48px;
        padding: 0px;
        background-color: transparent;
        border: none;
        cursor: pointer;
        position: relative;
        box-shadow: none;
    }
    .menu_btn::before,
    .menu_btn::after {
        content: '';
        position: absolute;
        width: 30px;
        height: 3px;
        background-color: #4f46e5;
        transition: transform 0.3s ease;
    }
    .menu_btn::before {
        top: 12px;
        box-shadow: 0 8px 0 #6366f1;
    }
    .menu_btn::after {
        bottom: 16px;
    }
    .menu_btn.active::before {
        transform: translateY(8px) rotate(45deg);
        box-shadow: none;
    }
    .menu_btn.active::after {
        transform: translateY(-8px) rotate(-45deg);
    }
"""

JS = """
    function () {
        const menuBtn = document.getElementById('menu');
        menuBtn.classList.toggle('active');
    }

"""

with gr.Blocks(theme="soft", css=CSS, fill_height=True) as demo:
    with gr.Row():
        visible = gr.State(value=True)
        menuBtn = gr.Button(value="", elem_classes="menu_btn", elem_id="menu", size="sm")
        gr.HTML(TITLE)
    with gr.Row():
        with gr.Column(scale=1) as menubar:
            endpoint = gr.Dropdown(
                label="Endpoint",
                choices=["Groq","OpenAI","Cohere","TogetherAI","Ollama","Huggingface"],
                value="OpenAI",
            )
            choice = gr.Checkbox(label="Second Endpoint", info="Add second endpoint for reflection")
            model = gr.Textbox(label="Model", value="gpt-4o", )
            api_key = gr.Textbox(label="API_KEY", type="password", )
            endpoint2 = gr.Dropdown(
                label="Endpoint 2",
                choices=["Groq","OpenAI","Cohere","TogetherAI","Ollama","Huggingface"],
                value="OpenAI",
                visible=False,
            )
            model2 = gr.Textbox(label="Model 2", value="gpt-4o", visible=False, )
            api_key2 = gr.Textbox(label="API_KEY 2", type="password", visible=False,)
            source_lang = gr.Textbox(
                label="Source Lang",
                value="English",
            )
            target_lang = gr.Textbox(
                label="Target Lang",
                value="Spanish",
            )
            country = gr.Textbox(label="Country", value="Argentina", max_lines=1)
            with gr.Accordion("Advanced Options", open=False):
                max_tokens = gr.Slider(
                    label="Max tokens Per Chunk",
                    minimum=512,
                    maximum=2046,
                    value=1000,
                    step=8,
                    )
                context_window = gr.Slider(
                    label="Context Window",
                    minimum=512,
                    maximum=8192,
                    value=4096,
                    step=8,
                    )
                num_output = gr.Slider(
                    label="Output Num",
                    minimum=256,
                    maximum=8192,
                    value=512,
                    step=8,
                    )
        with gr.Column(scale=4):
            source_text = gr.Textbox(
                label="Source Text",
                value="How we live is so different from how we ought to live that he who studies "+\
                "what ought to be done rather than what is done will learn the way to his downfall "+\
                "rather than to his preservation.",
                lines=12,
            )
            with gr.Tab("Final"):
                output_final = gr.Textbox(label="FInal Translation", lines=12, show_copy_button=True)
            with gr.Tab("Initial"):
                output_init = gr.Textbox(label="Init Translation", lines=12, show_copy_button=True)
            with gr.Tab("Reflection"):
                output_reflect = gr.Textbox(label="Reflection", lines=12, show_copy_button=True)
            with gr.Tab("Diff"):
                output_diff = gr.HighlightedText(visible = False)
    with gr.Row():
        submit = gr.Button(value="Translate")
        upload = gr.UploadButton(label="Upload", file_types=["text"])
        export = gr.DownloadButton(visible=False)
        clear = gr.ClearButton([source_text, output_init, output_reflect, output_final])

    menuBtn.click(fn=update_menu, inputs=visible, outputs=[visible, menubar], js=JS)
    endpoint.change(fn=update_model, inputs=[endpoint], outputs=[model])
    choice.select(fn=enable_sec, inputs=[choice], outputs=[endpoint2, model2, api_key2])
    endpoint2.change(fn=update_model, inputs=[endpoint2], outputs=[model2])
    submit.click(fn=huanik, inputs=[endpoint, model, api_key, choice, endpoint2, model2, api_key2, source_lang, target_lang, source_text, country, max_tokens, context_window, num_output], outputs=[output_init, output_reflect, output_final, output_diff])
    upload.upload(fn=read_doc, inputs = upload, outputs = source_text)
    output_final.change(fn=export_txt, inputs=output_final, outputs=[export])
if __name__ == "__main__":
    demo.queue(api_open=False).launch(show_api=False, share=False)