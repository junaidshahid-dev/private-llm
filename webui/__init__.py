"""webui — the local FastAPI + browser workstation for the agent.

This marker makes `webui` a real package so `from webui import model` resolves to this package even
when another directory on sys.path contains a module named `webui` (e.g. the legacy serving/webui.py,
which shadows a namespace package when serving/ is the script directory — as when gpu_server.py runs).
"""
