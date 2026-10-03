"""Measure the local configured server and recall three markers in a token-counted long prompt."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from strata_tokenizer import Tokenizer
from serve.frontend import ChatTemplate, openai_to_messages
from needle_bench import haystack

parser = argparse.ArgumentParser()
parser.add_argument("--tokens", type=int, default=199800)
parser.add_argument("--short", action="store_true")
parser.add_argument("--out", default="measurement.json")
parser.add_argument("--config", type=Path, default=ROOT / "strata-unsloth-ud-q4_k_xl.json")
parser.add_argument("--url", default="http://127.0.0.1:8080")
args = parser.parse_args()
config = json.loads(args.config.read_text())
directory = Path(config["tokenizer"])
if not directory.is_absolute():
    directory = Path(config.get("cwd", ROOT)) / directory
vocab = json.loads((directory / "vocab.json").read_text())
tokens = [None] * len(vocab)
for token, number in vocab.items():
    tokens[number] = token
tokenizer = Tokenizer(tokens, (directory / "merges.txt").read_text().splitlines(),
                      json.loads((directory / "token_type.json").read_text()))
template = ChatTemplate(directory / "chat_template.jinja")
url = args.url.rstrip("/")

def get(route):
    with urllib.request.urlopen(url + route, timeout=30) as response:
        return json.load(response)

def request(content):
    return {"model": "strata", "messages": [{"role": "user", "content": content}],
            "temperature": 0, "reasoning_effort": "none", "max_tokens": 192,
            "stream": True, "stream_options": {"include_usage": True}}

def count(req):
    messages, tools, kwargs = openai_to_messages(req)
    return len(tokenizer.encode(template.render(messages, tools, **kwargs), parse_special=True))

markers = ["falcon-quartz-729", "amber-willow-381", "glacier-copper-946"]
nonce = str(time.time_ns())
if args.short:
    req = request("Benchmark nonce " + nonce + ". Write a detailed Python implementation of an LRU cache. "
                  "Include get, put, and examples. Explain the complexity.")
else:
    text = haystack(args.tokens * 6)
    ending = ("\n\nFind the three UNIQUE_CODE records in the document. First quote each code in record order. "
              "Then explain what a bounded least recently used cache does and how to implement one in Python.")
    def make(chars):
        filler = text[:chars]
        parts, previous = [], 0
        for depth, marker in zip([0.1, 0.5, 0.9], markers):
            cut = int(len(filler) * depth)
            parts.extend([filler[previous:cut], "\nUNIQUE_CODE: " + marker + ".\n"])
            previous = cut
        parts.append(filler[previous:])
        return request("Benchmark nonce " + nonce + ". Read this reference document:\n" + "".join(parts) + ending)
    lo, hi = 0, len(text)
    while lo < hi:
        middle = (lo + hi + 1) // 2
        if count(make(middle)) <= args.tokens:
            lo = middle
        else:
            hi = middle - 1
    req = make(lo)

expected = count(req)
if not args.short and not args.tokens - 20 <= expected <= args.tokens:
    raise RuntimeError(f"prompt construction failed: {expected}")
health = get("/health")
if expected + req["max_tokens"] + 8 > health["max_context"]:
    raise RuntimeError("constructed prompt would exceed the configured context")
body = json.dumps(req).encode()
started = time.perf_counter()
first = None
text = []
usage = {}
finish = None
print("Sending", expected, "prompt tokens; context", health["max_context"], flush=True)
wire = urllib.request.Request(url + "/v1/chat/completions", data=body,
                              headers={"Content-Type": "application/json"})
with urllib.request.urlopen(wire, timeout=3600) as response:
    for line in response:
        if not line.startswith(b"data: "):
            continue
        payload = line[6:].strip()
        if payload == b"[DONE]":
            break
        chunk = json.loads(payload)
        if "error" in chunk:
            raise RuntimeError(chunk["error"])
        if "usage" in chunk:
            usage = chunk["usage"]
        for choice in chunk.get("choices", []):
            delta = choice.get("delta", {})
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                if first is None:
                    first = time.perf_counter() - started
                    print("First token after", round(first, 2), "seconds", flush=True)
                text.append(piece)
            finish = choice.get("finish_reason") or finish
elapsed = time.perf_counter() - started
engine = get("/metrics")["requests"][0]
answer = "".join(text)
result = {"health": health, "expected_prompt_tokens": expected, "engine": engine, "usage": usage,
          "request_sha256": hashlib.sha256(body).hexdigest(), "ttft_seconds": first,
          "elapsed_seconds": elapsed, "finish_reason": finish, "answer": answer,
          "markers": [] if args.short else markers,
          "markers_found": [] if args.short else [marker in answer for marker in markers]}
Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2), flush=True)
if engine["prompt_tokens"] != expected:
    raise RuntimeError("engine token count differs from rendered prompt")
if not answer or (not args.short and not all(result["markers_found"])):
    raise RuntimeError("response was empty or a marker was missed")
