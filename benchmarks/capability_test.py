#!/usr/bin/env python3
"""Task-style capability check: compare 320E against 384E on deterministic prompts.

Perplexity is a likelihood measure; task accuracy is what users actually feel. This runs
a fixed suite with temperature 0 and scores exact-answer / structural correctness.
Because both models are greedy, outputs are reproducible, so any difference is real.
"""
import json, os, re, sys, time
import requests

URL = os.environ.get("DSV41_URL", "http://localhost:9004") + "/v1/chat/completions"
MODEL = "DeepSeek-V4.1-Flash"

# (prompt, checker) — checkers look for a definite correct answer
CASES = [
    ("What is 17 * 23? Show your work, then give the final answer as a number.",
     lambda t: "391" in t),
    ("What is 847 + 1569? Give the final answer as a number.",
     lambda t: "2416" in t),
    ("Compute 144 / 12 * 7. Give the final answer as a number.",
     lambda t: "84" in t),
    ("A train travels 240 km in 3 hours. What is its average speed in km/h? Answer with the number.",
     lambda t: "80" in t),
    ("If all Bloops are Razzies and all Razzies are Lazzies, are all Bloops Lazzies? Answer yes or no first.",
     lambda t: re.search(r"\byes\b", t, re.I) is not None),
    ("List the numbers 1 to 15 in order, comma separated, and nothing else.",
     lambda t: all(str(i) in t for i in range(1, 16))),
    ("What is the 10th prime number? Answer with the number.",
     lambda t: "29" in t),
    ("Reverse the string 'abcdefg'. Give just the reversed string.",
     lambda t: "gfedcba" in t.lower()),
    ("How many letter 'r' characters are in the word 'strawberry'? Answer with the number.",
     lambda t: "3" in t or "three" in t.lower()),
    ("Write a Python one-liner using a list comprehension that squares the numbers 1..5. Give only the code.",
     lambda t: re.search(r"\[.*\*.*\*.*\bfor\b|\[.*\*\*.*\bfor\b", t) is not None or "[x*x" in t.replace(" ", "")),
    ("Convert 0b10110101 to decimal. Answer with the number.",
     lambda t: "181" in t),
    ("What is the capital of Australia? One word.",
     lambda t: "canberra" in t.lower()),
]


def ask(prompt, max_tokens=700):
    r = requests.post(URL, json={
        "model": MODEL, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.0,
    }, timeout=1200)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else "model"
    passed = 0
    out = []
    t0 = time.time()
    for i, (p, chk) in enumerate(CASES):
        try:
            txt = ask(p)
        except Exception as e:
            txt = f"<error {e}>"
        ok = chk(txt)
        passed += bool(ok)
        out.append({"i": i, "prompt": p[:70], "ok": bool(ok), "answer": txt.strip()[:150]})
        print(f"  [{i+1:2d}/{len(CASES)}] {'PASS' if ok else 'FAIL'}  {p[:58]}")
    print(f"\n[{label}] {passed}/{len(CASES)} = {100*passed/len(CASES):.0f}%  ({time.time()-t0:.0f}s)")
    with open(f"/home/dkp/capability-{label}.json", "w") as f:
        json.dump({"label": label, "passed": passed, "total": len(CASES), "cases": out}, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
