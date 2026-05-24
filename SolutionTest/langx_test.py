import sys, os, re, traceback, time
from datetime import datetime
# local python library
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LANGX_DIR  = os.path.join(SCRIPT_DIR, "..", "llama_langx", "langXpy", "langx")
sys.path.insert(0, LANGX_DIR)
import langx_core as lx

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


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(SCRIPT_DIR, f"result_langx_py_{timestamp}.txt")
    results = []

    log_lines = [f"LangX Python Test – {datetime.now().isoformat()}\n"]
    log_lines.append(f"Model: {MODEL_PATH}\n")
    log_lines.append(f"Image: {IMAGE_PATH}\n")
    log_lines.append("=" * 60 + "\n")

    t_total = time.perf_counter()
    try:
        # --- Preparing the model: config, load and system prompt ---
        cfg = lx.Config()
        lx.initialize_langX(cfg)
        mp = lx.ModelParams(MODEL_PATH, MMPROJ)
        mp.n_ctx = 8192 # needed for image
        print("Loading VLM model...")
        t_load = time.perf_counter()
        lx.init_model(mp, "vlm")
        t_load = time.perf_counter() - t_load
        print("Model loaded.\n")
        log_lines.append(f"Model load time: {t_load:.2f}s\n")

        lx.set_model_system_prompt("You are a helpful assistant. Answer concisely.")

        # --- Preparing the conversation: settings, random seed ---
        settings = lx.InquerySettings()
        settings.seed = lx.random_seed()

        # --- Preparing the langX stack ---
        stack = lx.make_default_stack("vlm_test")
        lx.set_inquery_settings(settings, stack)
        lx.switch_model("vlm", stack)
        lx.init_conversation("test_convo", stack)
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
            files = [p["image"]] if p["image"] else []
            t_prompt = time.perf_counter()
            response = lx.inference(stack, p["prompt"], files)
            t_prompt = time.perf_counter() - t_prompt
            print(response)

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
