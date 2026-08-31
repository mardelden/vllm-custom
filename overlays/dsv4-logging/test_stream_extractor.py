"""Exercise the patched streaming extractor against real chunk payloads."""
import sys
sys.path.insert(0, "/opt/vllm-dsv4/vllm-src")
from vllm.entrypoints.serve.utils.server_utils import _extract_content_from_chunk as extract

def chunk(**delta):
    return {"object": "chat.completion.chunk", "choices": [{"delta": delta}]}

cases = [
    ("content only",            chunk(content="OK"),                 "OK"),
    ("reasoning only (the bug)",chunk(reasoning="thinking..."),      "thinking..."),
    ("reasoning + content",     chunk(content="OK", reasoning="t"),  "OK"),
    ("empty delta",             chunk(),                             ""),
    ("null content, reasoning", chunk(content=None, reasoning="t"),  "t"),
    ("text_completion",         {"object":"text_completion","choices":[{"text":"hi"}]}, "hi"),
]
ok = True
for name, c, expect in cases:
    got = extract(c); good = got == expect; ok &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:26} -> {got!r}")
print("RESULT:", "streaming extractor captures reasoning" if ok else "MISMATCH")
