import os
#import gradio as gr

from brain_of_the_doctor import encode_image, analyze_image_with_query
from voice_of_the_user import transcribe_with_groq
from voice_of_the_doctor import text_to_speech_with_gtts


SYSTEM_PROMPT = """
You are a helpful AI assistant. The user will share an image and ask a question about it (or about any topic).
Answer clearly and accurately based on what you see in the image and the user's question. You can answer questions about any field: documents, objects, nature, diagrams, screenshots, photos, etc.
Use simple, clear language. Structure your answer with short paragraphs. Do not use bullet points, numbers, symbols, or emojis unless the user's question clearly calls for them.
Do not mention that you are an AI or that you are analyzing an image. Respond naturally as a knowledgeable assistant.
"""


def process_inputs(audio_filepath, image_filepath):
    if not audio_filepath or not image_filepath:
        return "Please provide voice and image", "", None

    speech_text = transcribe_with_groq(audio_filepath)

    assistant_response = analyze_image_with_query(
        query=SYSTEM_PROMPT + " " + speech_text,
        encoded_image=encode_image(image_filepath),
        model="meta-llama/llama-4-maverick-17b-128e-instruct"
    )

    voice_path = text_to_speech_with_gtts(assistant_response)

    return speech_text, assistant_response, voice_path


def clear_all():
    return None, None, "", "", None


# ---------------- UI STYLING ---------------- #

custom_css = """
body {
    background-color: #f2f6fc;
}

.section-card {
    background: white;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.08);
}

h1 {
    text-align: center;
    color: #0b5ed7;
    font-weight: 800;
}

.subtitle {
    text-align: center;
    color: #4a6fa5;
    margin-bottom: 25px;
}

label {
    font-weight: 600 !important;
    color: white !important;
}

textarea {
    font-size: 15px !important;
}

#analyze-btn {
    background: linear-gradient(90deg, #0b5ed7, #4dabf7);
    color: white;
    border-radius: 10px;
    font-size: 17px;
    font-weight: 600;
}

#analyze-btn:hover {
    transform: scale(1.03);
}

#clear-btn {
    background: linear-gradient(90deg, #dc3545, #ff6b6b);
    color: white;
    border-radius: 10px;
    font-size: 16px;
    font-weight: 600;
}

#clear-btn:hover {
    transform: scale(1.03);
}
"""


with gr.Blocks(css=custom_css) as iface:

    gr.Markdown("""
    <h1>🔍 AI Assistant – Vision & Voice</h1>
    <p class="subtitle">
    Upload any image and ask your question by voice.<br>
    Get answers in any field: documents, objects, diagrams, photos, and more.
    </p>
    """)

    with gr.Row():

        # LEFT COLUMN – INPUTS
        with gr.Column(scale=1):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown("### 📥 Your Inputs")

                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="🎙️ Speak Your Question"
                )

                image_input = gr.Image(
                    type="filepath",
                    label="🖼️ Upload Image"
                )

                with gr.Row():
                    submit = gr.Button(
                        "Analyze 🧠",
                        elem_id="analyze-btn"
                    )

                    clear_btn = gr.Button(
                        "Clear ❌",
                        elem_id="clear-btn"
                    )

        # RIGHT COLUMN – OUTPUTS
        with gr.Column(scale=1):
            with gr.Group(elem_classes="section-card"):
                gr.Markdown("### 💬 AI Response")

                stt_output = gr.Textbox(
                    label="🗣️ Speech to Text",
                    lines=2,
                    interactive=False
                )

                assistant_text = gr.Textbox(
                    label="🤖 AI Response",
                    lines=8,
                    max_lines=12,
                    interactive=False
                )

                assistant_voice = gr.Audio(
                    label="🔊 Voice Response",
                    autoplay=True
                )

    submit.click(
        fn=process_inputs,
        inputs=[audio_input, image_input],
        outputs=[stt_output, assistant_text, assistant_voice]
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[
            audio_input,
            image_input,
            stt_output,
            assistant_text,
            assistant_voice
        ]
    )

iface.launch(
    share=True,
    theme=gr.themes.Soft()
)
