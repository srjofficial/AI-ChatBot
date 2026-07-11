import os
import gradio as gr
from groq import Groq

# ----------------------------
# Groq Client
# ----------------------------
client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

SYSTEM_PROMPT = """
You are a helpful AI assistant.

Always answer politely and accurately.

If you don't know something, say you don't know.
"""

# ----------------------------
# Chat Function
# ----------------------------
def chatbot(message, history):

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    # Convert Gradio history to Groq format
    if history:
        for msg in history:

            # Gradio 6 sends history as dictionaries
            if isinstance(msg, dict):
                messages.append(
                    {
                        "role": msg["role"],
                        "content": msg["content"]
                    }
                )

    # Current user message
    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )

    return response.choices[0].message.content


# ----------------------------
# UI
# ----------------------------
demo = gr.ChatInterface(
    fn=chatbot,
    type="messages",
    title="🤖 AI Assistant",
    description="Powered by Groq • Llama 3.3 70B",
    examples=[
        "Hello",
        "Explain Artificial Intelligence",
        "Write Python Bubble Sort",
        "Summarize Machine Learning"
    ]
)

# ----------------------------
# Launch
# ----------------------------
port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)