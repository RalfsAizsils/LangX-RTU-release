#include "langX.h"
#include "LxRAG.h"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <regex>
#include <fstream>
#include <algorithm>
#include <thread>
#include <cmath>
#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Sec = std::chrono::duration<double>;

// --- Paths (relative to executable) ---
static fs::path exe_dir() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH];
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
#else
    return fs::canonical("/proc/self/exe").parent_path();
#endif
}

static const fs::path BASE_DIR = exe_dir();
static const fs::path MODELS_DIR = BASE_DIR / "models";
static const fs::path LLM_DIR = MODELS_DIR / "LLMmodel";
static const fs::path VLM_DIR = MODELS_DIR / "VLMmodel";
static const fs::path LORA_DIR = MODELS_DIR / "LORAmodels";
static const fs::path BERT_DIR = MODELS_DIR / "BERTmodels";
static const fs::path RAG_DIR = MODELS_DIR / "RAGmodel";

static fs::path GEN_MODEL;
static fs::path LLAMA_MODEL;
static fs::path EMBED_MODEL;
static fs::path VLM_MODEL;
static fs::path VLM_MMPROJ;
static fs::path LORA_ADAPTER;
static fs::path BERT_MODEL;
static fs::path NLI_MODEL;

static const fs::path TEST_DIR = BASE_DIR / "test";
static const fs::path DATA_DIR = TEST_DIR / "data";
static const fs::path DB_PATH = TEST_DIR / "all_test.lxrag";
static const fs::path PROC_MEM_PATH = TEST_DIR / "all_test_proc.txt";
static const fs::path ROBOT2_IMG = TEST_DIR / "robot2.jpg";

static const auto TOKEN_CB = [](const char* piece, int len) {
    std::cout.write(piece, len);
    std::cout.flush();
};

struct TestQuery {
    const char* prompt;
    const char* pattern;
    const char* label;
};

static fs::path find_gguf(const fs::path& dir, const std::string& substring = "") {
    if (!fs::exists(dir)) return {};
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".gguf") continue;
        if (substring.empty()) return e.path();
        std::string name = e.path().filename().string();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        if (name.find(substring) != std::string::npos) return e.path();
    }
    return {};
}

static fs::path find_gguf_excluding(const fs::path& dir, const std::string& exclude) {
    if (!fs::exists(dir)) return {};
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file() || e.path().extension() != ".gguf") continue;
        std::string name = e.path().filename().string();
        std::transform(name.begin(), name.end(), name.begin(), ::tolower);
        if (name.find(exclude) == std::string::npos) return e.path();
    }
    return {};
}

static bool check_response(const std::string& response, const char* pattern) {
    std::regex re(pattern, std::regex_constants::icase);
    return std::regex_search(response, re);
}

struct PhaseResult {
    const char* name;
    int pass = 0;
    int total = 0;
    int verify_pass = 0;
    int verify_total = 0;
    double time_s = 0.0;
};

static std::vector<PhaseResult> all_results;

static void print_phase_header(int phase, const char* title) {
    std::cout << "\n\n##################################################\n";
    std::cout << "  PHASE " << phase << ": " << title << "\n";
    std::cout << "##################################################\n";
}

static void run_queries(langX::Stack* stack, TestQuery* queries, int count, PhaseResult& result) {
    for (int i = 0; i < count; i++) {
        std::cout << "\n--- Q" << (i + 1) << " [" << queries[i].label << "]: " << queries[i].prompt << " ---\n";
        auto tq = Clock::now();
        std::string resp = langX::inference(stack, queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();
        bool hit = check_response(resp, queries[i].pattern);
        if (hit) result.pass++;
        result.total++;
        std::cout << "\n[TEST (" << queries[i].pattern << "): " << (hit ? "PASS" : "FAIL") << " | " << tq_s << "s]\n";
    }
}

static void verify(PhaseResult& r, const char* name, bool ok, bool leading_newline = true) {
    r.verify_total++;
    if (ok) r.verify_pass++;
    if (leading_newline) std::cout << "\n";
    std::cout << "[VERIFY " << name << ": " << (ok ? "PASS" : "FAIL") << "]\n";
}

int main() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    // Validate model directories
    std::vector<std::pair<std::string, fs::path>> required_dirs = {
        {"LLMmodel", LLM_DIR}, {"VLMmodel", VLM_DIR}, {"LORAmodels", LORA_DIR},
        {"BERTmodels", BERT_DIR}, {"RAGmodel", RAG_DIR}
    };
    for (auto& [name, dir] : required_dirs) {
        if (!fs::exists(dir)) {
            std::cerr << "ERROR: Missing model directory: " << dir.string() << "\n"
                << "Expected folder structure under " << MODELS_DIR.string() << "/:\n"
                << "  LLMmodel/    - 1 GGUF (generation model)\n"
                << "  VLMmodel/    - GGUF + mmproj GGUF\n"
                << "  LORAmodels/  - base GGUF + lora GGUF\n"
                << "  BERTmodels/  - bert + nli GGUFs\n"
                << "  RAGmodel/    - embedding model GGUF\n";
            return 1;
        }
        if (find_gguf(dir).empty()) {
            std::cerr << "ERROR: No .gguf files in " << dir.string() << "\n";
            return 1;
        }
    }

    // Discover model files from subdirectories
    GEN_MODEL = find_gguf(LLM_DIR);
    VLM_MMPROJ = find_gguf(VLM_DIR, "mmproj");
    VLM_MODEL = find_gguf_excluding(VLM_DIR, "mmproj");
    LORA_ADAPTER = find_gguf(LORA_DIR, "lora");
    LLAMA_MODEL = find_gguf_excluding(LORA_DIR, "lora");
    EMBED_MODEL = find_gguf(RAG_DIR);
    BERT_MODEL = find_gguf(BERT_DIR, "bert");
    NLI_MODEL = find_gguf(BERT_DIR, "nli");

    std::cout << "Models discovered:\n";
    std::cout << "  LLM:       " << GEN_MODEL.filename().string() << "\n";
    std::cout << "  VLM:       " << (VLM_MODEL.empty() ? "(not found)" : VLM_MODEL.filename().string()) << "\n";
    std::cout << "  MMPROJ:    " << (VLM_MMPROJ.empty() ? "(not found)" : VLM_MMPROJ.filename().string()) << "\n";
    std::cout << "  LoRA base: " << (LLAMA_MODEL.empty() ? "(not found)" : LLAMA_MODEL.filename().string()) << "\n";
    std::cout << "  LoRA:      " << (LORA_ADAPTER.empty() ? "(not found)" : LORA_ADAPTER.filename().string()) << "\n";
    std::cout << "  Embed:     " << EMBED_MODEL.filename().string() << "\n";
    std::cout << "  BERT:      " << (BERT_MODEL.empty() ? "(not found)" : BERT_MODEL.filename().string()) << "\n";
    std::cout << "  NLI:       " << (NLI_MODEL.empty() ? "(not found)" : NLI_MODEL.filename().string()) << "\n";
    std::cout << "\n";

    auto t_global_start = Clock::now();

    // Clean up from previous runs
    if (fs::exists(DB_PATH)) fs::remove(DB_PATH);
    if (fs::exists(PROC_MEM_PATH)) fs::remove(PROC_MEM_PATH);

    // --- LangX init ---
    langX::Config cfg{ (fs::current_path() / "langX_out").string() };
    langX::initialize_langX(cfg);

    // =========================================================================
    // PHASE 1: Init + Model Load
    // =========================================================================
    print_phase_header(1, "Init + Model Load");
    PhaseResult p1{"Init + Model Load"};
    auto t1_start = Clock::now();

    langX::ModelParams gen_params{ GEN_MODEL.string() };
    gen_params.n_ctx = 4096;

    langX::initModel(gen_params, "gen");

    langX::InquerySettings gen_settings;
    gen_settings.seed = langX::randomSeed();
    gen_settings.n_tokens_to_predict = 512;
    gen_settings.temperature = 0.7f;

    // Verify: model loaded, LangX initialized
    verify(p1, "model_loaded", langX::global_LangX.loaded_models.count("gen") > 0);
    verify(p1, "config_exists", langX::global_LangX.config != nullptr);
    verify(p1, "random_seed", gen_settings.seed != 1234);

    p1.time_s = Sec(Clock::now() - t1_start).count();
    std::cout << "\n[PHASE 1 time: " << p1.time_s << "s]\n";
    all_results.push_back(p1);

    // =========================================================================
    // PHASE 2: Basic Inference
    // =========================================================================
    print_phase_header(2, "Basic Inference");
    PhaseResult p2{"Basic Inference"};
    auto t2_start = Clock::now();

    langX::setModelSystemPrompt("You are a helpful assistant. Answer concisely.");
    langX::initConversation();

    langX::Stack* s2 = langX::makeDefaultStack("basic_test");
    std::string streamed_tokens;
    s2->on_token = [&streamed_tokens](const char* piece, int len) {
        std::cout.write(piece, len);
        std::cout.flush();
        streamed_tokens.append(piece, len);
    };
    langX::setInquerySettings(gen_settings, s2);

    TestQuery basic_queries[] = {
        { "What is 2 + 2?", "4|four", "arithmetic" },
        { "Name one planet in our solar system.", "Mercury|Venus|Earth|Mars|Jupiter|Saturn|Uranus|Neptune", "planets" },
    };
    run_queries(s2, basic_queries, std::size(basic_queries), p2);

    verify(p2, "history_saved", s2->conversation->active_messages.size() >= 4);
    verify(p2, "response_nonempty", !s2->active_response.empty());
    verify(p2, "on_token_streamed", !streamed_tokens.empty() && streamed_tokens.find(s2->active_response) != std::string::npos);

    p2.time_s = Sec(Clock::now() - t2_start).count();
    std::cout << "\n[PHASE 2 time: " << p2.time_s << "s]\n";
    all_results.push_back(p2);

    // =========================================================================
    // PHASE 3: Stack Features (DEBUG, FILLER, GOTO, BRANCH, CUSTOM)
    // =========================================================================
    print_phase_header(3, "Stack Features (DEBUG, FILLER, GOTO, BRANCH, CUSTOM)");
    PhaseResult p3{"Stack Features"};
    auto t3_start = Clock::now();

    // --- 3a: FILLER_RAND ---
    {
        langX::initConversation("filler_convo", s2);
        langX::Stack* s_filler = langX::makeStack("filler_rand_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeFillerRandLayer({"Alpha", "Beta", "Gamma"}),
            langX::makeSaveToHistoryLayer(),
        });
        s_filler->unsafe = true;
        langX::setInquerySettings(gen_settings, s_filler);
        std::string resp = langX::inference(s_filler, "test");
        bool filler_ok = (resp == "Alpha" || resp == "Beta" || resp == "Gamma");
        std::cout << "FILLER_RAND response: \"" << resp << "\"\n";
        verify(p3, "FILLER_RAND", filler_ok);
    }

    // --- 3b: FILLER_LOOP ---
    {
        langX::initConversation("loop_convo", s2);
        langX::Stack* s_loop = langX::makeStack("filler_loop_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeFillerLoopLayer({"First", "Second", "Third"}),
            langX::makeSaveToHistoryLayer(),
        });
        s_loop->unsafe = true;
        langX::setInquerySettings(gen_settings, s_loop);
        std::string r1 = langX::inference(s_loop, "a");
        std::string r2 = langX::inference(s_loop, "b");
        std::string r3 = langX::inference(s_loop, "c");
        std::string r4 = langX::inference(s_loop, "d");
        std::cout << "FILLER_LOOP responses: \"" << r1 << "\", \"" << r2 << "\", \"" << r3 << "\", \"" << r4 << "\"\n";
        verify(p3, "FILLER_LOOP_order", r1 == "First" && r2 == "Second" && r3 == "Third");
        verify(p3, "FILLER_LOOP_cycle", r4 == "First");
    }

    // --- 3c: DEBUG layer ---
    {
        langX::initConversation("debug_convo", s2);
        langX::Stack* s_debug = langX::makeStack("debug_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeDebugLayer("DEBUG_MARKER_TEST"),
            langX::makeFillerRandLayer({"ok"}),
            langX::makeSaveToHistoryLayer(),
        });
        s_debug->unsafe = true;
        langX::setInquerySettings(gen_settings, s_debug);
        langX::inference(s_debug, "test");
        bool debug_ok = s_debug->active_status.find("DEBUG_MARKER_TEST") != std::string::npos;
        std::cout << "DEBUG active_status: \"" << s_debug->active_status << "\"\n";
        verify(p3, "DEBUG_layer", debug_ok);
    }

    // --- 3d: CUSTOM layer ---
    {
        langX::initConversation("custom_convo", s2);
        bool custom_ran = false;
        langX::Stack* s_custom = langX::makeStack("custom_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeCustomLayer([&custom_ran](langX::Stack* st) {
                custom_ran = true;
                st->active_response = "CUSTOM_RESULT_" + st->active_prompt;
            }),
            langX::makeSaveToHistoryLayer(),
        });
        s_custom->unsafe = true;
        langX::setInquerySettings(gen_settings, s_custom);
        std::string resp = langX::inference(s_custom, "hello");
        std::cout << "CUSTOM response: \"" << resp << "\"\n";
        verify(p3, "CUSTOM_ran", custom_ran);
        verify(p3, "CUSTOM_result", resp == "CUSTOM_RESULT_hello");
    }

    // --- 3e: GOTO + BRANCH ---
    {
        langX::initConversation("branch_convo", s2);
        int loop_count = 0;
        langX::Stack* s_branch = langX::makeStack("branch_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeCustomLayer([&loop_count](langX::Stack* st) {
                loop_count++;
                st->active_response = "loop_" + std::to_string(loop_count);
            }),
            langX::makeBranchLayer([&loop_count](const langX::Stack*) {
                return loop_count < 3;
            }, 2),
            langX::makeSaveToHistoryLayer(),
        });
        s_branch->unsafe = true;
        langX::setInquerySettings(gen_settings, s_branch);
        std::string resp = langX::inference(s_branch, "test");
        std::cout << "BRANCH loop_count=" << loop_count << " response: \"" << resp << "\"\n";
        verify(p3, "BRANCH_loop_count", loop_count == 3);
        verify(p3, "BRANCH_final_resp", resp == "loop_3");
    }

    p3.time_s = Sec(Clock::now() - t3_start).count();
    std::cout << "\n[PHASE 3 time: " << p3.time_s << "s]\n";
    all_results.push_back(p3);

    // =========================================================================
    // PHASE 4: System Prompt + SET_SYSTEM_PROMPT + LOAD_CHAT_TEMPLATE
    // =========================================================================
    print_phase_header(4, "System Prompt + LOAD_CHAT_TEMPLATE");
    PhaseResult p4{"System Prompt"};
    auto t4_start = Clock::now();

    // --- 4a: SET_SYSTEM_PROMPT ---
    {
        langX::initConversation("sysprompt_convo", s2);
        langX::setModelSystemPrompt("You are ARIA, an astronomer. Always mention your name ARIA.");
        langX::Stack* s_sys = langX::makeStack("sysprompt_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeSetSystemPromptLayer("Also always say 'STARLIGHT' in every response."),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s_sys->on_token = TOKEN_CB;
        langX::setInquerySettings(gen_settings, s_sys);
        langX::switchModel("gen", s_sys);

        std::cout << "\n--- Q1 [sys_prompt]: Who are you? ---\n";
        std::string resp = langX::inference(s_sys, "Who are you?");
        verify(p4, "model_sys_prompt", check_response(resp, "ARIA"));
    }

    // --- 4b: LOAD_CHAT_TEMPLATE ---
    {
        langX::initConversation("chattempl_convo", s2);
        langX::setModelSystemPrompt("You are a pirate. Always say 'ARRR' in your response.");
        langX::Stack* s_tmpl = langX::makeStack("chattempl_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadChatTemplateLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s_tmpl->on_token = TOKEN_CB;
        langX::setInquerySettings(gen_settings, s_tmpl);
        langX::switchModel("gen", s_tmpl);

        std::cout << "\n--- Q2 [chat_template]: Greet me. ---\n";
        std::string resp = langX::inference(s_tmpl, "Greet me.");
        verify(p4, "LOAD_CHAT_TEMPLATE", !resp.empty());
    }

    p4.time_s = Sec(Clock::now() - t4_start).count();
    std::cout << "\n[PHASE 4 time: " << p4.time_s << "s]\n";
    all_results.push_back(p4);

    // =========================================================================
    // PHASE 5: Conversation Hotswap
    // =========================================================================
    print_phase_header(5, "Conversation Hotswap");
    PhaseResult p5{"Conversation Hotswap"};
    auto t5_start = Clock::now();

    langX::setModelSystemPrompt("You are a helpful assistant. Be concise.");

    // Create stack for hotswap test
    langX::Stack* s5 = langX::makeStack("hotswap_test", {
        langX::makeInitInferenceLayer(),
        langX::makeUserPushPromptLayer(),
        langX::makeLoadSystemPromptLayer(),
        langX::makeCxtWinTrimLayer(),
        langX::makeBuildContextLayer(),
        langX::makeClearKvCacheLayer(),
        langX::makeInitSamplerLayer(),
        langX::makeInitBatchLayer(),
        langX::makeFeedPromptLayer(),
        langX::makeLlmGenerationLayer(),
        langX::makeFreeSamplerLayer(),
        langX::makeFreeBatchLayer(),
        langX::makeSaveToHistoryLayer(),
    });
    langX::setInquerySettings(gen_settings, s5);
    langX::switchModel("gen", s5);

    // --- 5a: Plant conversations ---
    langX::initConversation("convo_A", s5);
    std::cout << "\n--- Plant A: My name is Viktor. Remember that. ---\n";
    std::string plant_a = langX::inference(s5, "My name is Viktor. Remember that.");
    std::cout << plant_a << "\n";

    langX::initConversation("convo_B", s5);
    std::cout << "\n--- Plant B: My name is Elena. Remember that. ---\n";
    std::string plant_b = langX::inference(s5, "My name is Elena. Remember that.");
    std::cout << plant_b << "\n";

    // --- 5b: Recall via swapConversations ---
    langX::swapConversations("convo_A", s5);
    std::cout << "\n--- Recall A: What is my name? ---\n";
    std::string resp_a = langX::inference(s5, "What is my name?");
    std::cout << resp_a << "\n";
    verify(p5, "hotswap_A_recall", check_response(resp_a, "Viktor"));

    langX::swapConversations("convo_B", s5);
    std::cout << "\n--- Recall B: What is my name? ---\n";
    std::string resp_b = langX::inference(s5, "What is my name?");
    std::cout << resp_b << "\n";
    verify(p5, "hotswap_B_recall", check_response(resp_b, "Elena"));

    // --- 5c: SWAP_CONVO layer ---
    {
        langX::Stack* s_swap = langX::makeStack("swap_convo_layer_test", {
            langX::makeInitInferenceLayer(),
            langX::makeSwapConvoLayer("convo_A"),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        langX::setInquerySettings(gen_settings, s_swap);
        langX::switchModel("gen", s_swap);

        std::cout << "\n--- Recall via layer: What is my name? ---\n";
        std::string resp = langX::inference(s_swap, "What is my name?");
        std::cout << resp << "\n";
        verify(p5, "SWAP_CONVO_layer", check_response(resp, "Viktor"));
    }

    p5.time_s = Sec(Clock::now() - t5_start).count();
    std::cout << "\n[PHASE 5 time: " << p5.time_s << "s]\n";
    all_results.push_back(p5);

    // =========================================================================
    // PHASE 6: Model Hotswap (sleep/wake)
    // =========================================================================
    print_phase_header(6, "Model Hotswap (sleep/wake)");
    PhaseResult p6{"Model Hotswap"};
    auto t6_start = Clock::now();

    // --- 6a: Sleep ---
    langX::sleepModel("gen");
    verify(p6, "sleep_moved", langX::global_LangX.sleeping_models.count("gen") > 0, false);
    verify(p6, "sleep_removed_loaded", langX::global_LangX.loaded_models.count("gen") == 0, false);

    // --- 6b: Wake ---
    langX::wakeupModel("gen");
    verify(p6, "wake_loaded", langX::global_LangX.loaded_models.count("gen") > 0, false);
    verify(p6, "wake_removed_sleeping", langX::global_LangX.sleeping_models.count("gen") == 0, false);

    // --- 6c: initSleepingModel ---
    langX::initSleepingModel(gen_params, "gen_sleep_test");
    verify(p6, "initSleepingModel", langX::global_LangX.sleeping_models.count("gen_sleep_test") > 0);

    // --- 6d: Post-wake inference ---
    {
        langX::initConversation("wake_test", s2);
        langX::switchModel("gen", s2);
        std::string resp = langX::inference(s2, "Say the word 'operational'.");
        verify(p6, "post_wake_inference", !resp.empty());
    }

    p6.time_s = Sec(Clock::now() - t6_start).count();
    std::cout << "\n[PHASE 6 time: " << p6.time_s << "s]\n";
    all_results.push_back(p6);

    // =========================================================================
    // PHASE 7: Tool Calling (LLM_TOOL_GEN)
    // =========================================================================
    print_phase_header(7, "Tool Calling (LLM_TOOL_GEN)");
    PhaseResult p7{"Tool Calling"};
    auto t7_start = Clock::now();

    langX::InquerySettings tool_settings = gen_settings;
    tool_settings.temperature = 0.2f;
    tool_settings.n_tokens_to_predict = 1024;

    // --- 7a: Non-native (XML) tool calling ---
    {
        bool sync_called = false;
        bool async_called = false;

        langX::Tool* t_calc = langX::makeTool("multiply", "Multiplies two numbers and returns the result.",
            [&sync_called](std::map<std::string, std::string> args) -> std::string {
                sync_called = true;
                std::cout << "[TOOL] multiply called: a=" << args["a"] << " b=" << args["b"] << "\n";
                double a = std::stod(args.count("a") ? args["a"] : "0");
                double b = std::stod(args.count("b") ? args["b"] : "0");
                return std::to_string(static_cast<long long>(a * b));
            });
        langX::addToolParam(t_calc, "a", "First number", "number");
        langX::addToolParam(t_calc, "b", "Second number", "number");

        langX::Tool* t_async = langX::makeTool("get_hidden_code", "Returns a hidden code. Runs asynchronously.",
            [&async_called](std::map<std::string, std::string>) -> std::string {
                async_called = true;
                std::cout << "[TOOL] get_hidden_code called\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                return "HIDDEN_CODE_7749";
            }, true);

        langX::Stack* s7a = langX::makeStack("tools_xml_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmToolGenLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        langX::registerTool(s7a, t_calc);
        langX::registerTool(s7a, t_async);
        langX::setModelSystemPrompt(langX::getToolsSystemPrompt(s7a).c_str());
        s7a->on_token = TOKEN_CB;
        s7a->verbose = true;
        s7a->unsafe = true;
        langX::setInquerySettings(tool_settings, s7a);
        langX::switchModel("gen", s7a);
        langX::initConversation("tools_xml_convo", s7a);

        std::cout << "\n--- Q1 [xml_tools]: Multiply 7177 * 7919 and get the hidden code. ---\n";
        std::string resp = langX::inference(s7a, "Call multiply with a=7177 and b=7919, and also call get_hidden_code.");
        verify(p7, "xml_sync_called", sync_called);
        verify(p7, "xml_async_called", async_called);
        verify(p7, "xml_tool_result", check_response(resp, "56[,. ]?834[,. ]?663|HIDDEN_CODE_7749"));
    }

    // --- 7b: Native tool calling ---
    {
        bool sync_called = false;
        bool async_called = false;

        langX::Tool* t_calc = langX::makeTool("multiply", "Multiplies two numbers and returns the result.",
            [&sync_called](std::map<std::string, std::string> args) -> std::string {
                sync_called = true;
                std::cout << "[TOOL] multiply called: a=" << args["a"] << " b=" << args["b"] << "\n";
                double a = std::stod(args.count("a") ? args["a"] : "0");
                double b = std::stod(args.count("b") ? args["b"] : "0");
                return std::to_string(static_cast<long long>(a * b));
            });
        langX::addToolParam(t_calc, "a", "First number", "number");
        langX::addToolParam(t_calc, "b", "Second number", "number");

        langX::Tool* t_async = langX::makeTool("get_hidden_code", "Returns a hidden code. Runs asynchronously.",
            [&async_called](std::map<std::string, std::string>) -> std::string {
                async_called = true;
                std::cout << "[TOOL] get_hidden_code called\n";
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                return "HIDDEN_CODE_7749";
            }, true);

        langX::InquerySettings native_tool_settings = tool_settings;
        native_tool_settings.use_native_tools = true;

        langX::Stack* s7b = langX::makeStack("tools_native_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmToolGenLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        langX::registerTool(s7b, t_calc);
        langX::registerTool(s7b, t_async);
        langX::setModelSystemPrompt("You are a helpful assistant. Always use available tools when possible.");
        s7b->on_token = TOKEN_CB;
        s7b->verbose = true;
        s7b->unsafe = true;
        langX::setInquerySettings(native_tool_settings, s7b);
        langX::switchModel("gen", s7b);
        langX::initConversation("tools_native_convo", s7b);

        std::cout << "\n--- Q2 [native_tools]: Multiply 7177 * 7919 ---\n";
        std::string resp_mul = langX::inference(s7b, "What is 7177 multiplied by 7919? Use the multiply tool.");
        verify(p7, "native_sync_called", sync_called);
        verify(p7, "native_multiply_result", check_response(resp_mul, "56[,. ]?834[,. ]?663"));

        std::cout << "\n--- Q3 [native_tools]: Get the hidden code ---\n";
        std::string resp_code = langX::inference(s7b, "Call get_hidden_code to retrieve the hidden code.");
        verify(p7, "native_async_called", async_called);
        verify(p7, "native_hidden_result", check_response(resp_code, "HIDDEN_CODE_7749"));
    }

    p7.time_s = Sec(Clock::now() - t7_start).count();
    std::cout << "\n[PHASE 7 time: " << p7.time_s << "s]\n";
    all_results.push_back(p7);

    // =========================================================================
    // PHASE 8: Wait Layers (WAIT_TIME, WAIT_LAMBDA, WAIT_PROMISE)
    // =========================================================================
    print_phase_header(8, "Wait Layers");
    PhaseResult p8{"Wait Layers"};
    auto t8_start = Clock::now();

    // --- 8a: WAIT_TIME ---
    {
        langX::initConversation("wait_time_convo", s2);
        langX::Stack* s_wait = langX::makeStack("wait_time_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeWaitTimeLayer(500),
            langX::makeFillerRandLayer({"waited_ok"}),
            langX::makeSaveToHistoryLayer(),
        });
        s_wait->unsafe = true;
        langX::setInquerySettings(gen_settings, s_wait);
        langX::switchModel("gen", s_wait);
        auto tw = Clock::now();
        std::string resp = langX::inference(s_wait, "test");
        double elapsed = Sec(Clock::now() - tw).count();
        std::cout << "WAIT_TIME elapsed: " << elapsed << "s\n";
        verify(p8, "WAIT_TIME_paused", elapsed >= 0.4);
        verify(p8, "WAIT_TIME_result", resp == "waited_ok");
    }

    // --- 8b: WAIT_LAMBDA ---
    {
        langX::initConversation("wait_lambda_convo", s2);
        std::atomic<bool> flag{false};
        langX::Stack* s_wl = langX::makeStack("wait_lambda_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeWaitLambdaLayer([&flag](const langX::Stack*) { return flag.load(); },
                                       5000, -1, 50),
            langX::makeFillerRandLayer({"lambda_ok"}),
            langX::makeSaveToHistoryLayer(),
        });
        s_wl->unsafe = true;
        langX::setInquerySettings(gen_settings, s_wl);
        langX::switchModel("gen", s_wl);

        std::thread([&flag]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            flag.store(true);
        }).detach();

        std::string resp = langX::inference(s_wl, "test");
        verify(p8, "WAIT_LAMBDA_result", resp == "lambda_ok");
    }

    // --- 8c: WAIT_PROMISE ---
    {
        langX::initConversation("wait_promise_convo", s2);
        langX::Stack* s_wp = langX::makeStack("wait_promise_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeWaitPromiseLayer(5000),
            langX::makeCustomLayer([](langX::Stack* st) {
                st->active_response = "promise_resolved";
            }),
            langX::makeSaveToHistoryLayer(),
        });
        s_wp->unsafe = true;
        langX::setInquerySettings(gen_settings, s_wp);
        langX::switchModel("gen", s_wp);

        std::thread([s_wp]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
            langX::signalStackWait(s_wp, "signal_value");
        }).detach();

        std::string resp = langX::inference(s_wp, "test");
        verify(p8, "WAIT_PROMISE_resolved", resp == "promise_resolved");
    }

    p8.time_s = Sec(Clock::now() - t8_start).count();
    std::cout << "\n[PHASE 8 time: " << p8.time_s << "s]\n";
    all_results.push_back(p8);

    // =========================================================================
    // PHASE 9: Sub-Inference (QUICK_ASK, SUBSTACK)
    // =========================================================================
    print_phase_header(9, "Sub-Inference (QUICK_ASK, SUBSTACK)");
    PhaseResult p9{"Sub-Inference"};
    auto t9_start = Clock::now();

    // --- 9a: QUICK_ASK ---
    {
        langX::initConversation("quickask_convo", s2);
        langX::setModelSystemPrompt("You are a helpful assistant. Answer concisely using the provided context.");
        langX::Stack* s_qa = langX::makeStack("quickask_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeQuickAskLayer("What is the capital of France? Reply with just the city name.", false),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s_qa->on_token = TOKEN_CB;
        langX::setInquerySettings(gen_settings, s_qa);
        langX::switchModel("gen", s_qa);

        std::cout << "\n--- Q1 [quick_ask]: QUICK_ASK injects 'capital of France', main stack echoes it ---\n";
        std::string resp = langX::inference(s_qa, "According to the context above, what city was mentioned? Just repeat it.");
        std::cout << "\nQUICK_ASK response: \"" << resp << "\"\n";
        verify(p9, "QUICK_ASK_context", check_response(resp, "paris"));
    }

    // --- 9b: SUBSTACK ---
    {
        langX::initConversation("substack_convo", s2);
        langX::setModelSystemPrompt("You are a helpful assistant. Answer concisely.");
        langX::Stack* sub = langX::makeStack("helper_sub", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
        });
        langX::setInquerySettings(gen_settings, sub);
        langX::switchModel("gen", sub);

        langX::Stack* s_sub = langX::makeStack("substack_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeSubstackLayer("helper_sub", "What is the largest planet in our solar system? Reply with just the planet name.", "Helper Result"),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s_sub->on_token = TOKEN_CB;
        langX::setInquerySettings(gen_settings, s_sub);
        langX::switchModel("gen", s_sub);

        std::cout << "\n--- Q2 [substack]: SUBSTACK asks 'largest planet', main stack echoes it ---\n";
        std::string resp = langX::inference(s_sub, "According to the context above, what planet was mentioned? Just repeat it.");
        verify(p9, "SUBSTACK_injected", check_response(resp, "jupiter"));
    }

    p9.time_s = Sec(Clock::now() - t9_start).count();
    std::cout << "\n[PHASE 9 time: " << p9.time_s << "s]\n";
    all_results.push_back(p9);

    // =========================================================================
    // PHASE 10: Filters (PROMPT_CHECK, LLM_PROMPT_FILTER)
    // =========================================================================
    print_phase_header(10, "Filters (PROMPT_CHECK, LLM_PROMPT_FILTER)");
    PhaseResult p10{"Filters"};
    auto t10_start = Clock::now();

    // --- 10a: PROMPT_CHECK ---
    {
        langX::initConversation("pcheck_convo", s2);
        langX::PromptCheckConfig pcc;
        pcc.min_length = 5;
        pcc.max_length = 200;

        langX::Stack* s_pc = langX::makeStack("prompt_check_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makePromptCheckLayer(pcc),
            langX::makeFillerRandLayer({"passed_check"}),
            langX::makeSaveToHistoryLayer(),
        });
        s_pc->unsafe = true;
        langX::setInquerySettings(gen_settings, s_pc);
        langX::switchModel("gen", s_pc);

        std::string resp_ok = langX::inference(s_pc, "This is a valid prompt.");
        std::cout << "PROMPT_CHECK pass response: \"" << resp_ok << "\"\n";
        verify(p10, "PROMPT_CHECK_pass", resp_ok == "passed_check");

        langX::initConversation("pcheck2", s_pc);
        std::string resp_fail = langX::inference(s_pc, "Hi");
        std::cout << "PROMPT_CHECK fail response: \"" << resp_fail << "\"\n";
        verify(p10, "PROMPT_CHECK_reject", resp_fail != "passed_check");
    }

    // --- 10b: LLM_PROMPT_FILTER ---
    {
        langX::initConversation("pfilter_convo", s2);
        langX::setModelSystemPrompt("You are a helpful assistant.");
        langX::Stack* s_pf = langX::makeStack("prompt_filter_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLLMPromptFilterLayer(-1, "Is this prompt asking about programming? Reply YES or NO."),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s_pf->on_token = TOKEN_CB;
        s_pf->verbose = true;
        langX::setInquerySettings(gen_settings, s_pf);
        langX::switchModel("gen", s_pf);

        std::cout << "\n--- Q1 [prompt_filter]: Write a C++ hello world program. ---\n";
        std::string resp = langX::inference(s_pf, "Write a C++ hello world program.");
        verify(p10, "LLM_PROMPT_FILTER_pass", !resp.empty());
    }

    p10.time_s = Sec(Clock::now() - t10_start).count();
    std::cout << "\n[PHASE 10 time: " << p10.time_s << "s]\n";
    all_results.push_back(p10);

    // =========================================================================
    // PHASE 11: RAG (build, retrieval, save/load)
    // =========================================================================
    print_phase_header(11, "RAG System");
    PhaseResult p11{"RAG"};
    auto t11_start = Clock::now();

    langX::RagParams rag_params;
    rag_params.chunk_size = 400;
    rag_params.chunk_overlap = 50;
    rag_params.top_k = 5;
    rag_params.filename_score_weight = 0.2f;
    rag_params.injection_prefix = "[Knowledge Base]\n";
    rag_params.embed_doc_prefix = "search_document: ";
    rag_params.embed_query_prefix = "search_query: ";
    rag_params.filter_mode = langX::RagFilterMode::DOCS;

    langX::RagDb* ragdb = langX::makeRagDb();
    langX::setRagDbParams(ragdb, rag_params);
    langX::ragSetVerbose(ragdb, true);

    bool embed_ok = langX::ragInitEmbeddings(ragdb, EMBED_MODEL.string().c_str());
    verify(p11, "embeddings_loaded", embed_ok);

    if (embed_ok) {
        langX::ragAddDirectory(ragdb, DATA_DIR.string().c_str(), false);
        langX::saveRagDb(ragdb, DB_PATH.string().c_str());
        verify(p11, "db_saved", fs::exists(DB_PATH));

        auto results = langX::ragSearch(ragdb, "employee department", 3);
        verify(p11, "search_results", !results.empty());

        // --- 11a: RAG_RETRIEVAL layer ---
        langX::initConversation("rag_convo", s2);
        langX::setModelSystemPrompt("Use the provided context to answer. Cite specific values.");
        langX::Stack* s_rag = langX::makeStack("rag_layer_test", {
            langX::makeInitInferenceLayer(),
            langX::makeRagRetrievalLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        langX::attachRagDb(s_rag, ragdb);
        langX::setRagParams(s_rag, rag_params);
        s_rag->on_token = TOKEN_CB;
        s_rag->verbose = true;
        langX::setInquerySettings(gen_settings, s_rag);
        langX::switchModel("gen", s_rag);

        TestQuery rag_queries[] = {
            { "What is Marta Kalnina's employee ID?", "EMP-4821", "employee lookup" },
        };
        run_queries(s_rag, rag_queries, std::size(rag_queries), p11);

        // --- 11b: Save/Load round-trip ---
        langX::RagDb* ragdb2 = langX::makeRagDb();
        langX::setRagDbParams(ragdb2, rag_params);
        langX::ragInitEmbeddings(ragdb2, EMBED_MODEL.string().c_str());
        bool load_ok = langX::loadRagDb(ragdb2, DB_PATH.string().c_str());
        verify(p11, "db_load_roundtrip", load_ok);
        if (load_ok) {
            auto r2 = langX::ragSearch(ragdb2, "employee", 1);
            verify(p11, "loaded_db_search", !r2.empty());
        }
        langX::freeRagDb(ragdb2);
    }

    p11.time_s = Sec(Clock::now() - t11_start).count();
    std::cout << "\n[PHASE 11 time: " << p11.time_s << "s]\n";
    all_results.push_back(p11);

    // =========================================================================
    // PHASE 12: Episodic Memory
    // =========================================================================
    print_phase_header(12, "Episodic Memory");
    PhaseResult p12{"Episodic Memory"};
    auto t12_start = Clock::now();

    langX::setModelSystemPrompt("Remember details the user tells you. Be concise.");
    langX::initConversation("episodic_convo", s2);

    langX::Stack* s12 = langX::makeStack("episodic_test", {
        langX::makeInitInferenceLayer(),
        langX::makeUserPushPromptLayer(),
        langX::makeEpisodicMemoryLayer(),
        langX::makeLoadSystemPromptLayer(),
        langX::makeCxtWinTrimLayer(),
        langX::makeBuildContextLayer(),
        langX::makeClearKvCacheLayer(),
        langX::makeInitSamplerLayer(),
        langX::makeInitBatchLayer(),
        langX::makeFeedPromptLayer(),
        langX::makeLlmGenerationLayer(),
        langX::makeFreeSamplerLayer(),
        langX::makeFreeBatchLayer(),
        langX::makeSaveToHistoryLayer(),
        langX::makeBuildEpisodicMemLayer(),
    });
    s12->on_token = TOKEN_CB;
    s12->verbose = true;
    langX::InquerySettings ep_settings = gen_settings;
    ep_settings.episodic_context_ratio = 0.05f;
    ep_settings.episodic_tier2_ratio   = 0.05f;
    ep_settings.episodic_tier1_ratio   = 0.15f;
    langX::setInquerySettings(ep_settings, s12);
    langX::switchModel("gen", s12);

    TestQuery episodic_queries[] = {
        { "My name is Viktor and I'm a lighthouse keeper. Remember that!", "Viktor|lighthouse", "plant" },
        { "I have a pet raven named Obsidian.", "Obsidian|raven", "plant pet" },
        { "The lighthouse is on Siren's Rock island.", "Siren|lighthouse", "plant location" },
        { "What's my name and what do I do?", "Viktor|lighthouse", "recall name" },
        { "Last week I found an Amber Vault map.", "Amber|map", "plant map" },
        { "Do you remember my pet?", "Obsidian", "recall pet" },
        { "A rival named Cassandra Drake arrived.", "Cassandra|Drake", "plant rival" },
        { "Tell me about my rival.", "Cassandra", "recall rival" },
    };
    run_queries(s12, episodic_queries, std::size(episodic_queries), p12);

    auto* ep_convo = s12->conversation;
    verify(p12, "T2_populated", !ep_convo->episodic_tier2_memories.empty());
    verify(p12, "T1_populated", !ep_convo->episodic_tier1_summary.empty());

    p12.time_s = Sec(Clock::now() - t12_start).count();
    std::cout << "\n[PHASE 12 time: " << p12.time_s << "s]\n";
    all_results.push_back(p12);

    // =========================================================================
    // PHASE 13: Semantic Memory
    // =========================================================================
    print_phase_header(13, "Semantic Memory");
    PhaseResult p13{"Semantic Memory"};
    auto t13_start = Clock::now();

    if (embed_ok) {
        langX::setModelSystemPrompt("Remember all details the user tells you. Repeat back key facts to confirm.");
        langX::initConversation("semantic_convo", s2);

        langX::InquerySettings sem_settings = gen_settings;
        sem_settings.n_tokens_to_predict = 3968;

        langX::Stack* s13 = langX::makeStack("semantic_test", {
            langX::makeInitInferenceLayer(),
            langX::makeSemanticMemRetrievalLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
            langX::makeBuildSemanticMemLayer(),
        });
        langX::attachRagDb(s13, ragdb);
        langX::setRagParams(s13, rag_params);
        s13->on_token = TOKEN_CB;
        s13->verbose = true;
        langX::setInquerySettings(sem_settings, s13);
        langX::switchModel("gen", s13);

        TestQuery semantic_queries[] = {
            { "Elena Vasquez is a marine biologist on the vessel Deep Sapphire. Remember that.", "Elena|Deep.Sapphire", "plant" },
            { "She discovered a jellyfish called Aurelia profunda at 8200 meters.", "Aurelia|profunda", "plant discovery" },
            { "Her research partner is Dr. Tomoko Hayashi, a chemist.", "Tomoko|Hayashi", "plant partner" },
            { "They are funded by the Nereus Foundation grant NF-7790.", "Nereus|NF.7790", "plant grant" },
            { "What is Elena's vessel called?", "Deep.Sapphire", "recall vessel" },
        };
        run_queries(s13, semantic_queries, std::size(semantic_queries), p13);

        auto prev_f = rag_params.filter_mode;
        rag_params.filter_mode = langX::RagFilterMode::MEMORY;
        langX::setRagDbParams(ragdb, rag_params);
        auto mem_check = langX::ragSearch(ragdb, "Elena vessel", 1);
        rag_params.filter_mode = prev_f;
        langX::setRagDbParams(ragdb, rag_params);
        verify(p13, "semantic_in_RAG", !mem_check.empty());
    } else {
        std::cout << "[SKIPPED — embeddings not loaded]\n";
    }

    p13.time_s = Sec(Clock::now() - t13_start).count();
    std::cout << "\n[PHASE 13 time: " << p13.time_s << "s]\n";
    all_results.push_back(p13);

    // =========================================================================
    // PHASE 14: Procedural Memory
    // =========================================================================
    print_phase_header(14, "Procedural Memory");
    PhaseResult p14{"Procedural Memory"};
    auto t14_start = Clock::now();

    if (fs::exists(PROC_MEM_PATH)) fs::remove(PROC_MEM_PATH);
    langX::setModelSystemPrompt("Remember details the user tells you. Be concise.");
    langX::initConversation("proc_convo", s2);

    std::string proc_path = PROC_MEM_PATH.string();
    langX::Stack* s14 = langX::makeStack("procedural_test", {
        langX::makeInitInferenceLayer(),
        langX::makeLoadProceduralMemLayer(proc_path.c_str()),
        langX::makeUserPushPromptLayer(),
        langX::makeLoadSystemPromptLayer(),
        langX::makeCxtWinTrimLayer(),
        langX::makeBuildContextLayer(),
        langX::makeClearKvCacheLayer(),
        langX::makeInitSamplerLayer(),
        langX::makeInitBatchLayer(),
        langX::makeFeedPromptLayer(),
        langX::makeLlmGenerationLayer(),
        langX::makeFreeSamplerLayer(),
        langX::makeFreeBatchLayer(),
        langX::makeSaveToHistoryLayer(),
        langX::makeBuildProceduralMemLayer(proc_path.c_str()),
    });
    s14->on_token = TOKEN_CB;
    s14->verbose = true;
    langX::setInquerySettings(gen_settings, s14);
    langX::switchModel("gen", s14);

    langX::inference(s14, "My name is Adris and I'm a blacksmith in Ironhaven. Remember that!");
    langX::inference(s14, "My best sword is called Dawnbreaker, forged from meteorite iron.");

    langX::initConversation("proc_convo2", s14);
    std::string proc_resp = langX::inference(s14, "What is my name and what is my best creation?");
    verify(p14, "proc_recall", check_response(proc_resp, "Adris|Dawnbreaker"));
    verify(p14, "proc_file_exists", fs::exists(PROC_MEM_PATH));

    if (fs::exists(PROC_MEM_PATH)) {
        std::ifstream pin(PROC_MEM_PATH);
        std::string pcontent((std::istreambuf_iterator<char>(pin)), std::istreambuf_iterator<char>());
        verify(p14, "proc_file_content", pcontent.find("Adris") != std::string::npos || pcontent.find("blacksmith") != std::string::npos);
    }

    p14.time_s = Sec(Clock::now() - t14_start).count();
    std::cout << "\n[PHASE 14 time: " << p14.time_s << "s]\n";
    all_results.push_back(p14);

    // =========================================================================
    // PHASE 15: Reasoning (COT_GENERATE)
    // =========================================================================
    print_phase_header(15, "Reasoning (COT_GENERATE)");
    PhaseResult p15{"COT_GENERATE"};
    auto t15_start = Clock::now();

    langX::setModelSystemPrompt("You are a helpful assistant. Answer concisely.");
    langX::initConversation("cot_convo", s2);

    langX::Stack* s15 = langX::makeStack("cot_test", {
        langX::makeInitInferenceLayer(),
        langX::makeUserPushPromptLayer(),
        langX::makeCotGenerateLayer("Think step by step. Show your reasoning."),
        langX::makeLoadSystemPromptLayer(),
        langX::makeCxtWinTrimLayer(),
        langX::makeBuildContextLayer(),
        langX::makeClearKvCacheLayer(),
        langX::makeInitSamplerLayer(),
        langX::makeInitBatchLayer(),
        langX::makeFeedPromptLayer(),
        langX::makeLlmGenerationLayer(),
        langX::makeFreeSamplerLayer(),
        langX::makeFreeBatchLayer(),
        langX::makeSaveToHistoryLayer(),
    });
    s15->on_token = TOKEN_CB;
    s15->verbose = true;
    langX::setInquerySettings(gen_settings, s15);
    langX::switchModel("gen", s15);

    std::cout << "\n--- Q1 [cot]: If a train travels at 60 km/h for 2.5 hours, how far does it go? ---\n";
    std::string cot_resp = langX::inference(s15, "If a train travels at 60 km/h for 2.5 hours, how far does it go?");
    verify(p15, "COT_answered", check_response(cot_resp, "150|one hundred fifty"));
    verify(p15, "COT_nonempty", !cot_resp.empty());

    p15.time_s = Sec(Clock::now() - t15_start).count();
    std::cout << "\n[PHASE 15 time: " << p15.time_s << "s]\n";
    all_results.push_back(p15);

    // =========================================================================
    // PHASE 16: feedContext
    // =========================================================================
    print_phase_header(16, "Context Injection APIs");
    PhaseResult p16{"Context APIs"};
    auto t16_start = Clock::now();

    // --- 16a: feedContext ---
    {
        langX::initConversation("feedctx_convo", s2);
        langX::setModelSystemPrompt("Use the provided context to answer.");
        langX::Stack* s16 = langX::makeStack("feedctx_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s16->on_token = TOKEN_CB;
        langX::setInquerySettings(gen_settings, s16);
        langX::switchModel("gen", s16);

        langX::feedContext("The project codename is ZEPHYR-42.", "Injected Context", s16);
        std::cout << "\n--- Q1 [feedContext]: What is the project codename? ---\n";
        std::string resp = langX::inference(s16, "What is the project codename?");
        verify(p16, "feedContext", check_response(resp, "ZEPHYR.42"));
    }

    p16.time_s = Sec(Clock::now() - t16_start).count();
    std::cout << "\n[PHASE 16 time: " << p16.time_s << "s]\n";
    all_results.push_back(p16);

    // =========================================================================
    // PHASE 17: SELF_CONSISTENT_GEN
    // =========================================================================
    print_phase_header(17, "SELF_CONSISTENT_GEN");
    PhaseResult p17{"Self-Consistency"};
    auto t17_start = Clock::now();

    {
        langX::setModelSystemPrompt("You are a helpful assistant. Answer concisely with just the number.");
        langX::initConversation("sc_convo", s2);

        langX::Stack* sc_sub = langX::makeStack("sc_gen_sub", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        sc_sub->on_token = TOKEN_CB;
        langX::setInquerySettings(gen_settings, sc_sub);
        langX::switchModel("gen", sc_sub);

        langX::Stack* s17 = langX::makeStack("sc_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeSelfConsistentGenLayer("sc_gen_sub", 3, "Pick the most accurate answer from the candidates. Reply with only that answer."),
            langX::makeSaveToHistoryLayer(),
        });
        s17->on_token = TOKEN_CB;
        s17->verbose = true;
        s17->unsafe = true;
        langX::setInquerySettings(gen_settings, s17);
        langX::switchModel("gen", s17);

        std::cout << "\n--- Q1 [self_consistent]: What is 15 * 8? ---\n";
        std::string sc_resp = langX::inference(s17, "What is 15 * 8?");
        std::cout << "\nSELF_CONSISTENT_GEN response: \"" << sc_resp << "\"\n";
        verify(p17, "SC_answered", !sc_resp.empty());
        verify(p17, "SC_correct", check_response(sc_resp, "120"));
    }

    p17.time_s = Sec(Clock::now() - t17_start).count();
    std::cout << "\n[PHASE 17 time: " << p17.time_s << "s]\n";
    all_results.push_back(p17);

    // =========================================================================
    // PHASE 18: LLM_RESULT_FILTER + FACT_CHECK
    // =========================================================================
    print_phase_header(18, "LLM_RESULT_FILTER + FACT_CHECK");
    PhaseResult p18{"Result Filter"};
    auto t18_start = Clock::now();

    // --- 18a: LLM_RESULT_FILTER ---
    {
        langX::setModelSystemPrompt("You are a helpful assistant. Be concise.");
        langX::initConversation("resfilter_convo", s2);

        langX::Stack* s18a = langX::makeStack("result_filter_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(), 
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(), 
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(), 
            langX::makeLLMResultFilterLayer(-1,"Does this response mention a specific color? Reply YES or NO."),
            langX::makeSaveToHistoryLayer(),
        });
        s18a->on_token = TOKEN_CB;
        s18a->verbose = true;
        langX::setInquerySettings(gen_settings, s18a);
        langX::switchModel("gen", s18a);

        std::cout << "\n--- Q1 [result_filter]: What color is the sky on a clear day? ---\n";
        std::string resp = langX::inference(s18a, "What color is the sky on a clear day?");
        verify(p18, "RESULT_FILTER_ran", !resp.empty());
    }

    // --- 18b: FACT_CHECK ---
    // issue. this is wrong - no RAG
    {
        langX::setModelSystemPrompt("You are a helpful assistant.");
        langX::initConversation("factcheck_convo", s2);

        langX::Stack* s18b = langX::makeStack("factcheck_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeFactCheckLayer(-1, "Is this response factually consistent with the question? Reply YES or NO."),
            langX::makeSaveToHistoryLayer(),
        });
        s18b->on_token = TOKEN_CB;
        s18b->verbose = true;
        langX::setInquerySettings(gen_settings, s18b);
        langX::switchModel("gen", s18b);

        std::cout << "\n--- Q2 [fact_check]: What is the capital of France? ---\n";
        std::string resp = langX::inference(s18b, "What is the capital of France?");
        verify(p18, "FACT_CHECK_ran", !resp.empty());
        verify(p18, "FACT_CHECK_content", check_response(resp, "Paris"));
    }

    p18.time_s = Sec(Clock::now() - t18_start).count();
    std::cout << "\n[PHASE 18 time: " << p18.time_s << "s]\n";
    all_results.push_back(p18);

    // =========================================================================
    // PHASE 19: Vision (USER_PUSH_IMAGES, FEED_PROMPT_IMAGES, QUICK_LOOK)
    // =========================================================================
    print_phase_header(19, "Vision (VLM)");
    PhaseResult p19{"Vision"};
    auto t19_start = Clock::now();

    bool vlm_available = fs::exists(VLM_MODEL) && fs::exists(VLM_MMPROJ) && fs::exists(ROBOT2_IMG);
    if (vlm_available) {
        langX::sleepModel("gen");

        langX::ModelParams vlm_params{ VLM_MODEL.string(), VLM_MMPROJ.string() };
        vlm_params.n_ctx = 8192;
        vlm_params.n_gpu_layers = 99;
        langX::initLoadedModel(vlm_params, "vlm");

        // --- 19a: Direct VLM inference with USER_PUSH_IMAGES + FEED_PROMPT_IMAGES ---
        {
            langX::initConversation("vlm_convo", s2);
            langX::Stack* s19a = langX::makeStack("vlm_direct_test", {
                langX::makeInitInferenceLayer(),
                langX::makeFileProcessingLayer(),
                langX::makeUserPushImagesLayer(),
                langX::makeUserPushPromptLayer(),
                langX::makeLoadSystemPromptLayer(),
                langX::makeCxtWinTrimLayer(),
                langX::makeBuildContextLayer(),
                langX::makeClearKvCacheLayer(),
                langX::makeInitSamplerLayer(),
                langX::makeInitBatchLayer(),
                langX::makeFeedPromptLayer(),
                langX::makeLlmGenerationLayer(),
                langX::makeFreeSamplerLayer(),
                langX::makeFreeBatchLayer(),
                langX::makeSaveToHistoryLayer(),
            });
            s19a->on_token = TOKEN_CB;
            s19a->verbose = true;
            langX::switchModel("vlm", s19a);
            langX::setInquerySettings(gen_settings, s19a);

            std::cout << "\n--- Q1 [vlm_direct]: Describe what you see in this image. ---\n";
            std::string resp = langX::inference(s19a, "Describe what you see in this image.", { ROBOT2_IMG.string() });
            verify(p19, "VLM_described", !resp.empty());
            verify(p19, "VLM_content", check_response(resp, "robot|mech|blue|armor|machine"));
        }

        // --- 19b: QUICK_LOOK layer ---
        {
            langX::wakeupModel("gen");
            langX::initConversation("quicklook_convo", s2);
            langX::switchModel("gen", s2);
            langX::Stack* s19b = langX::makeStack("quicklook_test", {
                langX::makeInitInferenceLayer(),
                langX::makeUserPushPromptLayer(),
                langX::makeQuickLookLayer("Describe the image briefly.", ROBOT2_IMG.string().c_str(), "vlm"),
                langX::makeLoadSystemPromptLayer(),
                langX::makeCxtWinTrimLayer(),
                langX::makeBuildContextLayer(),
                langX::makeClearKvCacheLayer(),
                langX::makeInitSamplerLayer(),
                langX::makeInitBatchLayer(),
                langX::makeFeedPromptLayer(),
                langX::makeLlmGenerationLayer(),
                langX::makeFreeSamplerLayer(),
                langX::makeFreeBatchLayer(),
                langX::makeSaveToHistoryLayer(),
            });
            s19b->on_token = TOKEN_CB;
            s19b->verbose = true;
            langX::setInquerySettings(gen_settings, s19b);
            langX::switchModel("gen", s19b);

            std::cout << "\n--- Q2 [quick_look]: Based on the image description, what is the image about? ---\n";
            std::string resp = langX::inference(s19b, "Based on the image description, what is the image about?");
            verify(p19, "QUICK_LOOK_injected", !resp.empty());
        }

        // --- 19c: countVisionTokens ---
        {
            size_t vtokens = langX::countVisionTokens(ROBOT2_IMG.string().c_str(), "vlm");
            std::cout << "countVisionTokens: " << vtokens << "\n";
            verify(p19, "countVisionTokens", vtokens > 0);
        }

        langX::unloadModel("vlm");
        langX::switchModel("gen", s2);
    } else {
        std::cout << "[SKIPPED — VLM model or test image not found]\n";
    }

    p19.time_s = Sec(Clock::now() - t19_start).count();
    std::cout << "\n[PHASE 19 time: " << p19.time_s << "s]\n";
    all_results.push_back(p19);

    // =========================================================================
    // PHASE 20: ASYNC_BREAK (background post-gen layers)
    // =========================================================================
    print_phase_header(20, "ASYNC_BREAK");
    PhaseResult p20{"ASYNC_BREAK"};
    auto t20_start = Clock::now();

    {
        langX::setModelSystemPrompt("You are a helpful assistant. Be concise.");
        langX::initConversation("async_convo", s2);

        std::atomic<bool> background_ran{false};

        langX::Stack* s20 = langX::makeStack("async_break_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
            langX::makeAsyncBreakLayer(),
            langX::makeCustomLayer([&background_ran](langX::Stack*) {
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
                background_ran.store(true);
            }),
        });
        s20->on_token = TOKEN_CB;
        langX::setInquerySettings(gen_settings, s20);
        langX::switchModel("gen", s20);

        std::cout << "\n--- Q1 [async_break]: Say hello. ---\n";
        auto t_async = Clock::now();
        std::string resp = langX::inference(s20, "Say hello.");
        double t_return = Sec(Clock::now() - t_async).count();

        verify(p20, "ASYNC_response", !resp.empty());

        langX::initConversation("async_convo2", s20);
        langX::inference(s20, "Say hi.");

        std::cout << "ASYNC_BREAK return time: " << t_return << "s\n";
        verify(p20, "ASYNC_bg_completed", background_ran.load());
    }

    p20.time_s = Sec(Clock::now() - t20_start).count();
    std::cout << "\n[PHASE 20 time: " << p20.time_s << "s]\n";
    all_results.push_back(p20);

    // =========================================================================
    // PHASE 21: Mixed Memory (episodic + semantic + procedural + RAG)
    // =========================================================================
    print_phase_header(21, "Mixed Memory (all types combined)");
    PhaseResult p21{"Mixed Memory"};
    auto t21_start = Clock::now();

    if (embed_ok) {
        fs::path proc_path21 = TEST_DIR / "all_test_proc_p21.txt";
        if (fs::exists(proc_path21)) fs::remove(proc_path21);
        std::string pp21 = proc_path21.string();

        langX::setModelSystemPrompt(
            "You are a helpful assistant with access to a knowledge base and conversation memory. "
            "Use all available context. Be concise.");
        langX::initConversation("mixed_convo", s2);

        langX::Stack* s21 = langX::makeStack("mixed_mem_test", {
            langX::makeInitInferenceLayer(),
            langX::makeLoadProceduralMemLayer(pp21.c_str()),
            langX::makeRagRetrievalLayer(),
            langX::makeSemanticMemRetrievalLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeEpisodicMemoryLayer(),
            langX::makeLoadSystemPromptLayer(),
            langX::makeCxtWinTrimLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitSamplerLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeLlmGenerationLayer(),
            langX::makeFreeSamplerLayer(),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
            langX::makeBuildEpisodicMemLayer(),
            langX::makeBuildSemanticMemLayer(),
            langX::makeBuildProceduralMemLayer(pp21.c_str()),
        });
        langX::attachRagDb(s21, ragdb);
        langX::setRagParams(s21, rag_params);
        s21->on_token = TOKEN_CB;
        s21->verbose = true;
        langX::InquerySettings mix_settings = gen_settings;
        mix_settings.episodic_context_ratio = 0.15f;
        mix_settings.episodic_tier2_ratio   = 0.05f;
        mix_settings.episodic_tier1_ratio   = 0.15f;
        langX::setInquerySettings(mix_settings, s21);
        langX::switchModel("gen", s21);

        TestQuery mixed_queries[] = {
            { "My codename is Phantom and I operate from Station Omega-12. Remember this.", "Phantom|Omega.12", "plant identity" },
            { "What is Marta Kalnina's employee ID?", "EMP-4821", "rag lookup" },
            { "I detected an energy pulse from reactor AXIOM-7 in Sector 8443.", "AXIOM.7|8443", "plant anomaly" },
            { "What is my codename?", "Phantom", "recall procedural" },
            { "What anomaly did I detect?", "AXIOM.7|8443", "recall anomaly" },
        };
        run_queries(s21, mixed_queries, std::size(mixed_queries), p21);

        auto* mconv = s21->conversation;
        verify(p21, "mixed_episodic", !mconv->episodic_tier2_memories.empty() || !mconv->episodic_tier1_summary.empty());

        auto prev_f = rag_params.filter_mode;
        rag_params.filter_mode = langX::RagFilterMode::MEMORY;
        langX::setRagDbParams(ragdb, rag_params);
        auto mcheck = langX::ragSearch(ragdb, "Phantom codename station", 1);
        rag_params.filter_mode = prev_f;
        langX::setRagDbParams(ragdb, rag_params);
        verify(p21, "mixed_semantic", !mcheck.empty());

        verify(p21, "proc_file_exists", fs::exists(proc_path21));
        if (fs::exists(proc_path21)) {
            std::ifstream pin(pp21);
            std::string pc((std::istreambuf_iterator<char>(pin)), std::istreambuf_iterator<char>());
            bool has_content = pc.find("Phantom") != std::string::npos || pc.find("AXIOM") != std::string::npos;
            verify(p21, "proc_file_content", has_content);
        }
    } else {
        std::cout << "[SKIPPED — embeddings not loaded]\n";
    }

    p21.time_s = Sec(Clock::now() - t21_start).count();
    std::cout << "\n[PHASE 21 time: " << p21.time_s << "s]\n";
    all_results.push_back(p21);

    // =========================================================================
    // PHASE 22: LoRA Adapters
    // =========================================================================
    print_phase_header(22, "LoRA Adapters");
    PhaseResult p22{"LoRA"};
    auto t22_start = Clock::now();

    bool lora_available = fs::exists(LORA_ADAPTER) && fs::exists(LLAMA_MODEL);
    if (lora_available) {
        langX::ModelParams lora_params{ LLAMA_MODEL.string() };
        lora_params.n_ctx = 4096;
        lora_params.lora_path = LORA_ADAPTER.string();
        lora_params.lora_scale = 1.0f;

        langX::initLoadedModel(lora_params, "gen_lora");
        verify(p22, "lora_loaded", langX::global_LangX.loaded_models.count("gen_lora") > 0);

        if (langX::global_LangX.loaded_models.count("gen_lora")) {
            langX::initConversation("lora_convo", s2);
            langX::switchModel("gen_lora", s2);
            std::cout << "\n--- Q1 [lora]: Draw me a cat ---\n";
            std::string resp = langX::inference(s2, "Draw me a cat");
            verify(p22, "lora_inference", !resp.empty());
            verify(p22, "lora_ascii_art", check_response(resp, "[\\\\|/()\\-_]"));
            langX::unloadModel("gen_lora");
            langX::switchModel("gen", s2);
        }
    } else {
        std::cout << "[SKIPPED — no LoRA adapter found at " << LORA_ADAPTER.string() << "]\n";
        std::cout << "To test: place a .gguf LoRA adapter at that path.\n";
    }

    p22.time_s = Sec(Clock::now() - t22_start).count();
    std::cout << "\n[PHASE 22 time: " << p22.time_s << "s]\n";
    all_results.push_back(p22);

    // =========================================================================
    // PHASE 23: BERT_GENERATE + NLI_GENERATE
    // =========================================================================
    print_phase_header(23, "BERT_GENERATE + NLI_GENERATE");
    PhaseResult p23{"BERT/NLI"};
    auto t23_start = Clock::now();

    bool bert_available = fs::exists(BERT_MODEL);
    bool nli_available  = fs::exists(NLI_MODEL);

    if (bert_available) {
        langX::ModelParams bert_params{ BERT_MODEL.string() };
        bert_params.n_ctx = 512;
        langX::initLoadedModel(bert_params, "bert");

        langX::initConversation("bert_convo", s2);
        langX::Stack* s_bert = langX::makeStack("bert_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeBertGenerateLayer("bert"),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s_bert->unsafe = true;
        langX::setInquerySettings(gen_settings, s_bert);
        langX::switchModel("bert", s_bert);

        std::string resp = langX::inference(s_bert, "This is a test sentence for classification.");
        std::cout << "BERT label: \"" << resp << "\"\n";
        verify(p23, "BERT_classified", !resp.empty());

        langX::unloadModel("bert");
    } else {
        std::cout << "[BERT SKIPPED — no model at " << BERT_MODEL.string() << "]\n";
    }

    if (nli_available) {
        langX::ModelParams nli_params{ NLI_MODEL.string() };
        nli_params.n_ctx = 512;
        langX::initLoadedModel(nli_params, "nli");

        langX::initConversation("nli_convo", s2);
        langX::Stack* s_nli = langX::makeStack("nli_test", {
            langX::makeInitInferenceLayer(),
            langX::makeUserPushPromptLayer(),
            langX::makeBuildContextLayer(),
            langX::makeClearKvCacheLayer(),
            langX::makeInitBatchLayer(),
            langX::makeFeedPromptLayer(),
            langX::makeNliGenerateLayer("nli", "The weather is sunny today."),
            langX::makeFreeBatchLayer(),
            langX::makeSaveToHistoryLayer(),
        });
        s_nli->unsafe = true;
        langX::setInquerySettings(gen_settings, s_nli);
        langX::switchModel("nli", s_nli);

        std::string resp = langX::inference(s_nli, "It is a bright and clear day outside.");
        std::cout << "NLI label: \"" << resp << "\"\n";
        verify(p23, "NLI_classified", !resp.empty());
        verify(p23, "NLI_label_valid", check_response(resp, "entailment|contradiction|neutral"));

        langX::unloadModel("nli");
    } else {
        std::cout << "[NLI SKIPPED — no model at " << NLI_MODEL.string() << "]\n";
    }

    if (!bert_available && !nli_available) {
        std::cout << "To test: convert BERT/NLI models to GGUF and place them in models/\n";
    }

    p23.time_s = Sec(Clock::now() - t23_start).count();
    std::cout << "\n[PHASE 23 time: " << p23.time_s << "s]\n";
    all_results.push_back(p23);

    // =========================================================================
    // END — tally
    // =========================================================================
    double t_total = Sec(Clock::now() - t_global_start).count();

    std::cout << "\n\n##################################################\n";
    std::cout << "  RESULTS\n";
    std::cout << "##################################################\n\n";

    int total_pass = 0, total_queries = 0, total_vpass = 0, total_vtotal = 0;
    std::cout << "  Phase | Test                    | Score  | Verify  | Time (s)\n";
    std::cout << "  ------|-------------------------|--------|---------|----------\n";
    for (auto& r : all_results) {
        total_pass += r.pass;
        total_queries += r.total;
        total_vpass += r.verify_pass;
        total_vtotal += r.verify_total;
        printf("  P%-4d | %-23s | %2d/%-3d | %2d/%-3d  | %7.1f\n", (int)(&r - &all_results[0]) + 1, r.name, r.pass, r.total, r.verify_pass, r.verify_total, r.time_s);
    }
    std::cout << "  ------|-------------------------|--------|---------|----------\n";
    printf("  TOTAL |                         | %2d/%-3d | %2d/%-3d  | %7.1f\n", total_pass, total_queries, total_vpass, total_vtotal, t_total);
    std::cout << "\n";

    // --- cleanup ---
    if (ragdb) langX::freeRagDb(ragdb);
    langX::unloadModel("gen");

    std::cout << "\nPress Enter to exit...";
    std::cin.get();
    return 0;
}
