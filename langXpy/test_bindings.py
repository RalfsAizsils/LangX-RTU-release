"""Quick smoke test for langx_core bindings — no model needed."""

import sys, os

# Add the langx/ subfolder so langx_core.pyd can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "langx"))

import langx_core as lx

passed = 0
failed = 0

def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [OK]   {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name}")

# --- Structs ---
print("=== Struct creation ===")

cfg = lx.Config()
cfg.verbose_logs = True
check("Config.verbose_logs", cfg.verbose_logs == True)

s = lx.InquerySettings()
s.temperature = 0.5
s.episodic_context_ratio = 0.75
s.use_native_tools = True
check("InquerySettings.temperature", s.temperature == 0.5)
check("InquerySettings.episodic_context_ratio", s.episodic_context_ratio == 0.75)
check("InquerySettings.use_native_tools", s.use_native_tools == True)

mp = lx.ModelParams()
mp.n_ctx = 4096
mp.main_gpu = 1
mp.tensor_split = [0.5, 0.5]
check("ModelParams.n_ctx", mp.n_ctx == 4096)
check("ModelParams.main_gpu", mp.main_gpu == 1)
check("ModelParams.tensor_split", mp.tensor_split == [0.5, 0.5])

mp2 = lx.ModelParams("model.gguf", "vision.gguf")
check("ModelParams(path, vision)", mp2.path == "model.gguf" and mp2.vision_path == "vision.gguf")

msg = lx.ChatMessage()
msg.role = "user"
msg.content = "hello"
check("ChatMessage", msg.role == "user" and "hello" in repr(msg))

ce = lx.ContextExtra()
ce.label = "test"
ce.score = 0.9
check("ContextExtra", ce.label == "test" and ce.score > 0.8)

conv = lx.Conversation()
conv.procedural_memory = "the user likes cats"
conv.system_prompt = "You are helpful."
check("Conversation.procedural_memory", conv.procedural_memory == "the user likes cats")

pcc = lx.PromptCheckConfig()
pcc.check_injection = True
pcc.max_length = 500
check("PromptCheckConfig", pcc.check_injection and pcc.max_length == 500)

# --- Enums ---
print("\n=== Enum values ===")

check("LayerType.INIT_INFERENCE", hasattr(lx, "INIT_INFERENCE"))
check("LayerType.BUILD_CONTEXT", hasattr(lx, "BUILD_CONTEXT"))
check("LayerType.COT_GENERATE", hasattr(lx, "COT_GENERATE"))
check("LayerType.SELF_CONSISTENT_GEN", hasattr(lx, "SELF_CONSISTENT_GEN"))
check("LayerType.BERT_GENERATE", hasattr(lx, "BERT_GENERATE"))
check("LayerType.NLI_GENERATE", hasattr(lx, "NLI_GENERATE"))
check("LayerType.LOAD_PROCEDURAL_MEM", hasattr(lx, "LOAD_PROCEDURAL_MEM"))
check("LayerType.BUILD_PROCEDURAL_MEM", hasattr(lx, "BUILD_PROCEDURAL_MEM"))

check("RagFilterMode.MEMORY", lx.RagFilterMode.MEMORY is not None)
check("RagImageMode.PASSTHROUGH", lx.RagImageMode.PASSTHROUGH is not None)

# --- Layer factories (no model needed) ---
print("\n=== Layer factories ===")

layer_makers = [
    ("make_init_inference_layer",   lx.make_init_inference_layer),
    ("make_file_processing_layer",  lx.make_file_processing_layer),
    ("make_user_push_prompt_layer", lx.make_user_push_prompt_layer),
    ("make_user_push_images_layer", lx.make_user_push_images_layer),
    ("make_build_context_layer",    lx.make_build_context_layer),
    ("make_load_chat_template_layer", lx.make_load_chat_template_layer),
    ("make_cxt_win_trim_layer",     lx.make_cxt_win_trim_layer),
    ("make_load_system_prompt_layer", lx.make_load_system_prompt_layer),
    ("make_clear_kv_cache_layer",   lx.make_clear_kv_cache_layer),
    ("make_init_sampler_layer",     lx.make_init_sampler_layer),
    ("make_init_batch_layer",       lx.make_init_batch_layer),
    ("make_feed_prompt_layer",      lx.make_feed_prompt_layer),
    ("make_feed_prompt_images_layer", lx.make_feed_prompt_images_layer),
    ("make_llm_generation_layer",   lx.make_llm_generation_layer),
    ("make_free_sampler_layer",     lx.make_free_sampler_layer),
    ("make_free_batch_layer",       lx.make_free_batch_layer),
    ("make_save_to_history_layer",  lx.make_save_to_history_layer),
    ("make_use_tools_layer",        lx.make_use_tools_layer),
    ("make_wait_tools_layer",       lx.make_wait_tools_layer),
    ("make_llm_tool_gen_layer",     lx.make_llm_tool_gen_layer),
    ("make_rag_retrieval_layer",    lx.make_rag_retrieval_layer),
    ("make_semantic_memory_layer",  lx.make_semantic_memory_layer),
    ("make_episodic_memory_layer",  lx.make_episodic_memory_layer),
    ("make_async_break_layer",      lx.make_async_break_layer),
]

for name, fn in layer_makers:
    try:
        layer = fn()
        check(name, layer is not None)
    except Exception as e:
        check(name, False)
        print(f"         ^ {e}")

# Layers with args
check("make_debug_layer", lx.make_debug_layer("test msg") is not None)
check("make_filler_rand_layer", lx.make_filler_rand_layer(["a", "b", "c"]) is not None)
check("make_filler_loop_layer", lx.make_filler_loop_layer(["x", "y"]) is not None)
check("make_swap_model_layer", lx.make_swap_model_layer("m1") is not None)
check("make_swap_convo_layer", lx.make_swap_convo_layer("c1") is not None)
check("make_set_system_prompt_layer", lx.make_set_system_prompt_layer("Be nice.") is not None)
check("make_wait_time_layer", lx.make_wait_time_layer(1000) is not None)
check("make_prompt_check_layer", lx.make_prompt_check_layer() is not None)
check("make_quick_ask_layer", lx.make_quick_ask_layer("What is 2+2?") is not None)
check("make_goto_layer", lx.make_goto_layer(0) is not None)
check("make_load_procedural_mem_layer", lx.make_load_procedural_mem_layer("mem.txt") is not None)
check("make_cot_generate_layer", lx.make_cot_generate_layer() is not None)
check("make_bert_generate_layer", lx.make_bert_generate_layer() is not None)
check("make_nli_generate_layer", lx.make_nli_generate_layer() is not None)

# Branch + custom (Python callables)
branch = lx.make_branch_layer(lambda s: s.has_tool_calls, 0)
check("make_branch_layer (lambda)", branch is not None)

custom = lx.make_custom_layer(lambda s: setattr(s, 'active_prompt', 'modified'))
check("make_custom_layer (lambda)", custom is not None)

# --- Tool creation ---
print("\n=== Tool creation ===")

tool = lx.make_tool("search", "Search the web", lambda args: f"results for {args.get('query','')}")
lx.add_tool_param(tool, "query", "Search query", "string", True)
check("make_tool", tool.name == "search")
# NOTE: tool.params access crashes if .pyd and langX.dll have mismatched CRT (debug vs release).
# Rebuild both in the same config to fix. Skipping params member access for safety.

vtool = lx.make_void_tool("log", "Log a message", lambda args: None)
check("make_void_tool", vtool.name == "log")

# --- Generic make_layer ---
print("\n=== Generic make_layer ===")

gl = lx.make_layer(lx.INIT_INFERENCE)
check("make_layer(INIT_INFERENCE)", gl.type == lx.INIT_INFERENCE)

pl = lx.make_param_layer(lx.DEBUG, "hello")
check("make_param_layer(DEBUG)", pl.type == lx.DEBUG)
# NOTE: pl.params["value"] access skipped — crashes on debug/release CRT mismatch

# --- RagParams ---
print("\n=== RAG params ===")

rp = lx.RagParams()
rp.top_k = 5
rp.mmr_lambda = 0.75
rp.filter_mode = lx.RagFilterMode.MEMORY
check("RagParams fields", rp.top_k == 5 and rp.mmr_lambda == 0.75)
check("RagParams.filter_mode", rp.filter_mode == lx.RagFilterMode.MEMORY)

# --- Summary ---
print(f"\n{'='*40}")
print(f"Results: {passed} passed, {failed} failed, {passed+failed} total")
if failed == 0:
    print("All bindings OK!")
else:
    print("Some bindings FAILED - check output above.")
    sys.exit(1)
