"""webui.py — a real chat UI connected to the LIVE model, served from the Kaggle notebook.

    # in a Kaggle CODE cell:
    from serving.webui import launch
    launch()          # loads Moonlight once, opens a chat UI, prints a public link to click

Why this and not the Lumen artifact: a published artifact runs in a sandbox that cannot call an
external server, and a Kaggle notebook has no public port of its own. Gradio solves both — it runs
the UI in the notebook and opens its own share tunnel, so you get a public https link that talks
to the model actually loaded in this kernel. Base Moonlight + capability_first policy + RAG, in a
browser chat.

The model loads once when the UI starts and stays resident, so replies are just generation.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Lumen palette, applied to Gradio so the connected UI matches the design language.
CSS = """
.gradio-container{background:#070c16 !important; color:#d9e3f2 !important;
  font-family:system-ui,-apple-system,'Segoe UI',sans-serif !important; max-width:900px !important}
#lumen-head{display:flex;align-items:center;gap:12px;padding:14px 4px}
#lumen-head .orb{width:22px;height:22px;border-radius:7px;
  background:linear-gradient(150deg,#6fe6d4,#2f9e93);box-shadow:0 0 16px -2px #2f9e93}
#lumen-head h1{font-size:18px;letter-spacing:.3em;text-transform:uppercase;margin:0;font-weight:600}
#lumen-head .status{margin-left:auto;font-family:ui-monospace,Consolas,monospace;font-size:11px;
  color:#7fdca0;letter-spacing:.12em}
.message-wrap, .message{background:#101a2e !important;border:1px solid #1d2b47 !important;
  color:#d9e3f2 !important}
footer{display:none !important}
"""

HEADER = ('<div id="lumen-head"><div class="orb"></div><h1>Lumen</h1>'
          '<span class="status">● MOONLIGHT ONLINE</span></div>')


def launch(share: bool = True, use_rag: bool = True, port: int = 7860):
    try:
        import gradio as gr
    except ImportError:
        raise SystemExit("Gradio isn't installed. In a cell run:  !pip install -q gradio")

    from serving.chat import Assistant
    bot = Assistant()                       # loads the model once; stays resident

    def respond(message, history):
        try:
            return bot.ask(message, use_rag=use_rag)
        except Exception as e:              # noqa: BLE001 — surface errors in the chat, don't crash
            return f"[error: {type(e).__name__}: {e}]"

    with gr.Blocks(css=CSS, title="Lumen", theme=gr.themes.Base()) as demo:
        gr.HTML(HEADER)
        gr.ChatInterface(
            respond,
            chatbot=gr.Chatbot(height=460, show_label=False),
            textbox=gr.Textbox(placeholder="Ask Lumen — grounded in your documents…",
                               show_label=False),
            examples=["Explain SSRF and how to prevent it.",
                      "What attention mechanism does Moonlight use?",
                      "Review this for a bug: cmd = 'ping ' + host; os.system(cmd)"],
        )
    print("starting the UI — a public https link will print below; open it to chat.")
    demo.launch(share=share, server_port=port, quiet=False)
    return demo
