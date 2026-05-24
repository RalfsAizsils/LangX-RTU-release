import os, re, traceback, time, base64, subprocess, tempfile
from datetime import datetime
import ollama

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
MMPROJ = os.path.join(MODEL_DIR, "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf")
IMAGE_PATH = os.path.join(SCRIPT_DIR, "robot2.jpg")

LOCAL_MODEL_NAME = "test-qwen-vlm"
HUB_MODEL_NAME = "qwen2.5vl:7b"

INTRO_FACT = "My name is Jimmy"
INTRO_PROMPT = (
    f"Hello! I want to have a conversation with you. "
    f"Here is an important fact to remember: {INTRO_FACT}. "
    f"Now, what color is the sky on a clear day?"
)

PROMPTS = [
    {
        "name": "introduction",
        "prompt": INTRO_PROMPT,
        "image": None,
        "pattern": r"blue",
        "desc": "Basic question about sky color",
    },
    {
        "name": "image_describe",
        "prompt": "Describe what you see in this image in detail.",
        "image": IMAGE_PATH,
        "pattern": r"robot|mech|machine|armor|metal|figure|character",
        "desc": "Describe the robot2.jpg image",
    },
    {
        "name": "fact_recall",
        "prompt": "Earlier I told you an important fact. What was it? Repeat it.",
        "image": None,
        "pattern": r"Jimmy",
        "desc": "Recall fact from the introduction",
    },
]

def check(response: str, pattern: str) -> bool:
    return bool(re.search(pattern, response, re.IGNORECASE))

def load_image_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def model_exists(name: str) -> bool:
    models = ollama.list().models
    return any(name in (m.model or "") for m in models)

def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(SCRIPT_DIR, f"result_ollama_{timestamp}.txt")
    results = []
    model_name = LOCAL_MODEL_NAME

    log_lines = [f"Ollama Test – {datetime.now().isoformat()}\n"]
    log_lines.append(f"Local GGUF: {MODEL_PATH}\n")
    log_lines.append(f"Image: {IMAGE_PATH}\n")
    log_lines.append(
        "Note: Ollama cannot directly load GGUF files by path. Models must be\n"
        "imported via 'ollama create' or pulled from the Ollama hub.\n"
        "VLM models with a separate mmproj projector file cannot be imported\n"
        "locally — they must be pulled from the hub as a pre-packaged model.\n"
        "Locally imported GGUFs may also lose their embedded chat template,\n"
        "causing degraded response quality compared to other solutions.\n"
    )
    log_lines.append("=" * 60 + "\n")

    t_total = time.perf_counter()
    try:
        # --- Check if hub model already exists in Ollama ---
        t_load = time.perf_counter()
        if model_exists(HUB_MODEL_NAME):
            model_name = HUB_MODEL_NAME
            print(f"Model '{HUB_MODEL_NAME}' already available in Ollama.")
            log_lines.append(f"Model source: already pulled ('{HUB_MODEL_NAME}')\n")
        elif model_exists(LOCAL_MODEL_NAME):
            print(f"Model '{LOCAL_MODEL_NAME}' already available in Ollama.")
            log_lines.append(f"Model source: local (already imported as '{LOCAL_MODEL_NAME}')\n")
        else:
            # --- Try importing local GGUF into Ollama ---
            print(f"Attempting local GGUF import as '{LOCAL_MODEL_NAME}'...")
            try:
                modelfile = f"FROM {MODEL_PATH.replace(chr(92), '/')}\n"
                tf = tempfile.NamedTemporaryFile(mode="w", suffix=".Modelfile", delete=False)
                tf.write(modelfile)
                tf.close()
                subprocess.run(["ollama", "create", LOCAL_MODEL_NAME, "-f", tf.name], check=True)
                os.unlink(tf.name)
                # verify the model actually works with a quick test
                ollama.chat(model=LOCAL_MODEL_NAME, messages=[{"role": "user", "content": "hi"}])
                print("Local GGUF imported and verified.")
                log_lines.append(f"Model source: imported local GGUF (no mmproj — VLM limited)\n")
            except Exception as e_local:
                # clean up broken local model if it was created
                try: subprocess.run(["ollama", "rm", LOCAL_MODEL_NAME], capture_output=True)
                except: pass
                # --- Fall back to pulling from Ollama hub ---
                print(f"Local import failed: {e_local}")
                print(f"Falling back to hub model '{HUB_MODEL_NAME}'...")
                model_name = HUB_MODEL_NAME
                ollama.pull(HUB_MODEL_NAME)
                print("Hub model pulled successfully.")
                log_lines.append(f"Model source: pulled from Ollama hub ('{HUB_MODEL_NAME}')\n")

        t_load = time.perf_counter() - t_load
        print("Model ready.\n")
        log_lines.append(f"Model used: {model_name}\n")
        log_lines.append(f"Model load time: {t_load:.2f}s\n")

        messages = [{"role": "system", "content": "You are a helpful assistant. Answer concisely."}]
    except Exception as e:
        msg = f"SETUP FAILED: {e}\n{traceback.format_exc()}"
        print(msg)
        log_lines.append(msg)
        for p in PROMPTS:
            results.append({"name": p["name"], "status": "FAILED", "reason": "setup error"})
            log_lines.append(f"\n--- {p['name']}: FAILED (setup error) ---\n")
        with open(out_file, "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        return

    # --- Test Prompt Loop ---
    for p in PROMPTS:
        print(f"\n{'='*60}")
        print(f"[{p['name']}] {p['desc']}")
        print(f"Prompt: {p['prompt']}")
        print("-" * 60)

        log_lines.append(f"\n--- {p['name']}: {p['desc']} ---\n")
        log_lines.append(f"Prompt: {p['prompt']}\n")

        try:
            msg = {"role": "user", "content": p["prompt"]}
            if p["image"]:
                msg["images"] = [load_image_b64(p["image"])]
            messages.append(msg)

            t_prompt = time.perf_counter()
            result = ollama.chat(model=model_name, messages=messages)
            t_prompt = time.perf_counter() - t_prompt
            response = result["message"]["content"]
            print(response)

            messages.append({"role": "assistant", "content": response})

            passed = check(response, p["pattern"])
            status = "PASS" if passed else "FAIL"
            results.append({"name": p["name"], "status": status, "time": t_prompt})

            log_lines.append(f"Response: {response}\n")
            log_lines.append(f"Pattern: {p['pattern']}\n")
            log_lines.append(f"Result: {status} ({t_prompt:.2f}s)\n")

            print(f"[{status}] pattern=/{p['pattern']}/")
        except Exception as e:
            tb = traceback.format_exc()
            results.append({"name": p["name"], "status": "FAILED", "reason": str(e)})
            log_lines.append(f"ERROR: {e}\n{tb}\n")
            log_lines.append("Result: FAILED (exception)\n")
            print(f"[FAILED] {e}")

    # --- Summary ---
    log_lines.append("\n" + "=" * 60 + "\n")
    log_lines.append("SUMMARY\n")
    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    errors = sum(1 for r in results if r["status"] == "FAILED")

    for r in results:
        t = f" ({r['time']:.2f}s)" if "time" in r else ""
        log_lines.append(f"  {r['name']}: {r['status']}{t}\n")
    t_total = time.perf_counter() - t_total
    log_lines.append(f"\nTotal: {passed}/{total} passed, {failed} failed, {errors} errors\n")
    log_lines.append(f"Total time: {t_total:.2f}s\n")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{total} passed, {failed} failed, {errors} errors")
    print(f"Results saved to: {out_file}")

    with open(out_file, "w", encoding="utf-8") as f:
        f.writelines(log_lines)

if __name__ == "__main__":
    main()
