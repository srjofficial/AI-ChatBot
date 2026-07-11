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

    # Print history to Render logs (for debugging)
    print("History:", history)

    if history:
        for item in history:

            # Old Gradio format
            if isinstance(item, (list, tuple)) and len(item) == 2:
                messages.append({"role": "user", "content": item[0]})
                messages.append({"role": "assistant", "content": item[1]})

            # New Gradio format
            elif isinstance(item, dict):
                messages.append(item)

    messages.append(
        {
            "role": "user",
            "content": message
        }
    )

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
    )

    return response.choices[0].message.content


# ----------------------------
# Interface
# ----------------------------
demo = gr.ChatInterface(
    fn=chatbot,
    title="🤖 AI Assistant",
    description="Powered by Groq",
    examples=[
        "Hello",
        "Explain AI",
        "Write Python code",
        "What is Machine Learning?"
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
