"""Exercise the real patched guard against real ChatMessage objects (no GPU)."""
import sys, textwrap
sys.path.insert(0, "/opt/vllm-dsv4/vllm-src")
from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage

src = open("/opt/vllm-dsv4/vllm-src/vllm/entrypoints/openai/chat_completion/serving.py").read()
start = src.index('            for choice in choices:\n                output_text = ""')
end = src.index("                if output_text:\n                    # Get the corresponding output token IDs", start)
block = src[start:end]
body = textwrap.dedent("\n".join(block.splitlines()[1:]))  # drop the `for` line, dedent the rest

class Choice:
    def __init__(self, message): self.message, self.index = message, 0

def evaluate(message):
    ns = {"choice": Choice(message)}
    exec(body, {"getattr": getattr}, ns)
    return ns["output_text"]

cases = [
    ("content only",             ChatMessage(role="assistant", content="OK"),                         "OK"),
    ("reasoning only (the bug)", ChatMessage(role="assistant", content=None, reasoning="thinking..."),"[reasoning: thinking...]"),
    ("reasoning + content",      ChatMessage(role="assistant", content="OK", reasoning="thinking..."),"[reasoning: thinking...] OK"),
    ("empty-string content",     ChatMessage(role="assistant", content="", reasoning="t"),            "[reasoning: t]"),
    ("reasoning_content alias",  ChatMessage(role="assistant", content=None, reasoning_content="alt"),"[reasoning: alt]"),
    ("nothing at all",           ChatMessage(role="assistant", content=None),                         ""),
]
ok = True
for name, msg, expect in cases:
    got = evaluate(msg); good = got == expect; ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:26} -> {got!r}")
print("RESULT:", "all guard cases behave as intended" if ok else "MISMATCH")
