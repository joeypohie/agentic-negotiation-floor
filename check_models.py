"""Model selection rule, step 1: the real configuration (temperature 0.5, seed, thinking off), 3 calls per model."""
import time
from pathlib import Path
from openai import OpenAI

env = dict(l.split("=", 1) for l in Path(".env").read_text().splitlines()
           if "=" in l and not l.startswith("#"))
c = OpenAI(api_key=env["SOCLAAS_API_KEY"].strip(), base_url=env["SOCLAAS_BASE_URL"].strip())

P = 'A bond costs 99.5 and I will pay up to 100.2. Reply only with JSON: {"midpoint": <number>}'

for m in ["qwen3.6:35b", "qwen3.6:27b", "gemma4:26b", "ornith1.5:35b"]:
    for i in range(3):
        t = time.time()
        r = c.chat.completions.create(model=m, temperature=0.5, seed=1, reasoning_effort="none",
                                      max_tokens=400, messages=[{"role": "user", "content": P}])
        msg = r.choices[0].message
        content = msg.content or ""
        th = bool(getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
                  or "<think>" in content)
        print(f"{m:15} run{i + 1}  {time.time() - t:4.1f}s  served={r.model}  "
              f"thinking={th}  reply={content.strip()[-40:]!r}", flush=True)
