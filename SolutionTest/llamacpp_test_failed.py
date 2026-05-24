import os, re, traceback, time, base64
from datetime import datetime

# suppress C-level stderr spam from llama.cpp during loading
_stderr_fd = os.dup(2)
_stdout_fd = os.dup(1)
_devnull = os.open(os.devnull, os.O_WRONLY)
def _suppress_stderr():  os.dup2(_devnull, 2)
def _restore_stderr():   os.dup2(_stderr_fd, 2)
def _suppress_stdout():  os.dup2(_devnull, 1)
def _restore_stdout():   os.dup2(_stdout_fd, 1)

_suppress_stderr()
from llama_cpp import Llama
_restore_stderr()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
MMPROJ = os.path.join(MODEL_DIR, "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf")
IMAGE_PATH = os.path.join(SCRIPT_DIR, "robot2.jpg")

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

def image_to_data_uri(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(SCRIPT_DIR, f"result_llamacpp_{timestamp}.txt")
    results = []

    log_lines = [f"Llama.cpp Python Test – {datetime.now().isoformat()}\n"]
    log_lines.append(f"Model: {MODEL_PATH}\n")
    log_lines.append(f"Image: {IMAGE_PATH}\n")
    log_lines.append(
        "Note: The default 'pip install llama-cpp-python' builds CPU-only.\n"
        "GPU support requires rebuilding with a backend flag (CUDA, Vulkan, ROCm).\n"
        "Unlike LangX (which ships with Vulkan support built-in), llama-cpp-python\n"
        "requires the user to manually rebuild the library for their GPU.\n"
    )
    log_lines.append("=" * 60 + "\n")

    t_total = time.perf_counter()
    try:
        # --- Preparing the model: try Qwen2.5-VL handler, fall back to Llava16 ---
        print("Loading VLM model...")
        t_load = time.perf_counter()
        chat_handler = None
        handler_name = "none"
        _suppress_stderr()
        _suppress_stdout()
        try:
            from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            chat_handler = Qwen25VLChatHandler(clip_model_path=MMPROJ, verbose=False)
            handler_name = "Qwen25VLChatHandler"
        except ImportError:
            try:
                from llama_cpp.llama_chat_format import Llava16ChatHandler
                chat_handler = Llava16ChatHandler(clip_model_path=MMPROJ, verbose=False)
                handler_name = "Llava16ChatHandler (fallback)"
            except ImportError:
                handler_name = "none (VLM not supported)"

        llm = Llama(
            model_path=MODEL_PATH,
            chat_handler=chat_handler,
            n_ctx=8192,
            n_gpu_layers=99,
            verbose=False,
        )
        _restore_stdout()
        _restore_stderr()
        t_load = time.perf_counter() - t_load
        print(f"Vision handler: {handler_name}")
        print("Model loaded.\n")
        log_lines.append(f"Vision handler: {handler_name}\n")
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
            if p["image"]:
                content = [
                    {"type": "text", "text": p["prompt"]},
                    {"type": "image_url", "image_url": {"url": image_to_data_uri(p["image"])}},
                ]
            else:
                content = p["prompt"]
            # send image only in current turn, not in history
            send_messages = messages + [{"role": "user", "content": content}]

            t_prompt = time.perf_counter()
            result = llm.create_chat_completion(messages=send_messages, max_tokens=512)
            t_prompt = time.perf_counter() - t_prompt
            response = result["choices"][0]["message"]["content"]
            print(response)

            # store text-only version in history
            messages.append({"role": "user", "content": p["prompt"]})
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
