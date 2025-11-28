import json
import subprocess
from typing import List
try:
    import ollama
except Exception:
    ollama = None

def semantic_delta(text1, text2):
    # Simple "change" metric: 1 - word overlap ratio (higher = more delta)
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    overlap = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
    return 1 - overlap  # 1 = big change, 0 = same

def mrrc_prune_semantic(context, max_m=3):
    if len(context) <= max_m:
        return context
    deltas = []
    for i in range(1, len(context)):
        delta = semantic_delta(context[i-1], context[i])
        deltas.append(delta)
    # Keep highest deltas (most change)
    keep_indices = sorted(range(len(deltas)), key=lambda i: deltas[i], reverse=True)[:max_m // 2]
    pruned = [context[0]] + [context[i+1] for i in sorted(keep_indices)] + [context[-1]]  # First + top + last
    return pruned[:max_m]

def _ollama_generate(prompt: str, model: str = "llama3.1", timeout: float = 60.0) -> str:
    # Prefer Python API if available
    if ollama is not None:
        try:
            if hasattr(ollama, "Client"):
                client = ollama.Client()
                resp = client.generate(model=model, prompt=prompt)
                return resp.get("response") or resp.get("message", {}).get("content", "") or ""
            if hasattr(ollama, "generate"):
                resp = ollama.generate(model=model, prompt=prompt)
                return resp.get("response") or resp.get("message", {}).get("content", "") or ""
            if hasattr(ollama, "chat"):
                resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
                if isinstance(resp, dict):
                    msg = resp.get("message") or {}
                    return msg.get("content", "") or ""
                return str(resp)
        except Exception:
            pass  # Fall back to CLI

    # CLI fallback
    try:
        proc = subprocess.run([
            "ollama", "run", model
        ], input=prompt.encode("utf-8"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError("Ollama CLI not found. Install Ollama or add to PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama CLI timed out.")

    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore"))

    output = proc.stdout.decode("utf-8", errors="ignore").strip()
    if not output:
        return ""
    parts: List[str] = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            parts.append(obj.get("response") or obj.get("message", {}).get("content", "") or "")
        except json.JSONDecodeError:
            parts.append(line)
    text = "".join(p for p in parts if p)
    return text or output


def main():
    prompts = [
        "Summarize MRRC V5 α variation in two sentences.",
        "List falsifiable predictions of MRRC for α variation.",
    ]
    for p in prompts:
        try:
            resp = _ollama_generate(p)
        except Exception as e:
            print(f"[ERROR] {e}")
            resp = ""
        pruned = (resp or "").strip()[:256]
        print("--- PRUNED ---\n" + pruned + "\n---------------")


if __name__ == "__main__":
    main()