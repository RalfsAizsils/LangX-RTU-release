#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <regex>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>
#include <random>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Sec = std::chrono::duration<double>;

static const fs::path SCRIPT_DIR = "d:/School/RTU/LangX/SolutionTest";
static const fs::path MODEL_DIR = SCRIPT_DIR / "model";
static const fs::path MODEL_PATH = MODEL_DIR / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf";
static const fs::path MMPROJ = MODEL_DIR / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf";
static const fs::path IMAGE_PATH = SCRIPT_DIR / "robot2.jpg";

static const std::string INTRO_FACT = "My name is Jimmy";
static const std::string INTRO_PROMPT =
    "Hello! I want to have a conversation with you. "
    "Here is an important fact to remember: " + INTRO_FACT + ". "
    "Now, what color is the sky on a clear day?";

struct TestPrompt {
    std::string name;
    std::string prompt;
    std::string image;
    std::string pattern;
    std::string desc;
};

struct ChatMessage {
    std::string role;
    std::string content;
};

static bool check(const std::string& response, const std::string& pattern) {
    return std::regex_search(response, std::regex(pattern, std::regex_constants::icase));
}

static std::string timestamp_str() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm;
    localtime_s(&tm, &t);
    std::ostringstream o;
    o << std::put_time(&tm, "%Y%m%d_%H%M%S");
    return o.str();
}

static std::string iso_now() {
    auto t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm tm;
    localtime_s(&tm, &t);
    std::ostringstream o;
    o << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
    return o.str();
}

// tokenize a string into llama tokens
static std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    std::vector<llama_token> tokens(text.length() + 2);
    int n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);
    }
    tokens.resize(n);
    return tokens;
}

// add a token to a batch
static void batch_add(llama_batch& batch, llama_token id, int pos, bool logits) {
    int idx = batch.n_tokens;
    batch.token[idx] = id;
    batch.pos[idx] = pos;
    batch.n_seq_id[idx] = 1;
    batch.seq_id[idx][0] = 0;
    batch.logits[idx] = logits;
    batch.n_tokens++;
}

// apply the chat template to a message list, returns formatted prompt string
static std::string apply_template(const char* tmpl, const std::vector<ChatMessage>& msgs) {
    std::vector<llama_chat_message> chat_msgs;
    for (auto& m : msgs)
        chat_msgs.push_back({ m.role.c_str(), m.content.c_str() });
    int len = llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(), true, nullptr, 0);
    std::vector<char> buf(len + 1);
    llama_chat_apply_template(tmpl, chat_msgs.data(), chat_msgs.size(), true, buf.data(), buf.size());
    return std::string(buf.data(), len);
}

int main() {
    const std::vector<TestPrompt> prompts = {
        {"introduction", INTRO_PROMPT, "", "blue", "Basic question about sky color"},
        {"image_describe", "Describe what you see in this image in detail.", IMAGE_PATH.string(), "robot|mech|machine|armor|metal|figure|character", "Describe the robot2.jpg image"},
        {"fact_recall", "Earlier I told you an important fact. What was it? Repeat it.", "", "Jimmy", "Recall fact from the introduction"},
    };

    fs::path out_file = SCRIPT_DIR / ("result_llamacpp_" + timestamp_str() + ".txt");
    std::ostringstream log;

    struct Result { std::string name, status, reason; double time = 0; };
    std::vector<Result> results;

    log << "Llama.cpp C++ Test \xe2\x80\x93 " << iso_now() << "\n";
    log << "Model: " << MODEL_PATH.string() << "\n";
    log << "Image: " << IMAGE_PATH.string() << "\n";
    log << std::string(60, '=') << "\n";

    auto t_total = Clock::now();

    // --- Setup: load model, context, vision projector ---
    llama_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 99;

    std::cout << "Loading VLM model...\n";
    auto t_load = Clock::now();

    llama_model* model = llama_load_model_from_file(MODEL_PATH.string().c_str(), model_params);
    if (!model) {
        std::cerr << "Failed to load model: " << MODEL_PATH.string() << "\n";
        return 1;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 8192;
    ctx_params.n_batch = 512;
    llama_context* ctx = llama_new_context_with_model(model, ctx_params);
    if (!ctx) {
        std::cerr << "Failed to create context.\n";
        llama_model_free(model);
        return 1;
    }

    mtmd_context_params vp = mtmd_context_params_default();
    mtmd_helper_log_set([](enum ggml_log_level, const char*, void*) {}, nullptr);
    mtmd_context* mtmd_ctx = mtmd_init_from_file(MMPROJ.string().c_str(), model, vp);
    if (!mtmd_ctx) {
        std::cerr << "Failed to load mmproj: " << MMPROJ.string() << "\n";
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    double load_s = Sec(Clock::now() - t_load).count();
    std::cout << "Model loaded.\n\n";
    log << "Model load time: " << std::fixed << std::setprecision(2) << load_s << "s\n";

    const char* chat_tmpl = llama_model_chat_template(model, nullptr);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    int seed = std::random_device{}();

    // conversation history
    std::vector<ChatMessage> messages = {
        {"system", "You are a helpful assistant. Answer concisely."}
    };

    // --- Test prompt loop ---
    for (const auto& p : prompts) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "[" << p.name << "] " << p.desc << "\n";
        std::cout << "Prompt: " << p.prompt << "\n";
        std::cout << std::string(60, '-') << "\n";

        log << "\n--- " << p.name << ": " << p.desc << " ---\n";
        log << "Prompt: " << p.prompt << "\n";

        // prepare user message — prepend image marker if needed
        std::string user_content = p.prompt;
        mtmd_bitmap* bmp = nullptr;
        if (!p.image.empty()) {
            bmp = mtmd_helper_bitmap_init_from_file(mtmd_ctx, p.image.c_str());
            if (bmp) {
                user_content = std::string(mtmd_default_marker()) + "\n" + p.prompt;
            } else {
                std::cerr << "Warning: failed to load image " << p.image << "\n";
            }
        }
        messages.push_back({"user", user_content});

        // apply chat template to full conversation
        std::string formatted = apply_template(chat_tmpl, messages);

        // clear KV cache (rebuild full context each turn)
        llama_memory_t mem = llama_get_memory(ctx);
        llama_memory_seq_rm(mem, 0, 0, -1);

        auto t_prompt = Clock::now();
        int n_past = 0;

        if (bmp) {
            // VLM path: tokenize with image chunks and eval
            std::vector<const mtmd_bitmap*> bitmaps = { bmp };
            mtmd_input_text input_text = { formatted.c_str(), true, true };
            mtmd_input_chunks* chunks = mtmd_input_chunks_init();
            int32_t ret = mtmd_tokenize(mtmd_ctx, chunks, &input_text, bitmaps.data(), bitmaps.size());
            if (ret != 0) {
                std::cerr << "mtmd_tokenize failed (" << ret << ")\n";
                mtmd_input_chunks_free(chunks);
                mtmd_bitmap_free(bmp);
                results.push_back({p.name, "FAILED", "mtmd_tokenize failed", 0});
                log << "Result: FAILED (mtmd_tokenize)\n";
                continue;
            }
            llama_pos pos = 0;
            ret = mtmd_helper_eval_chunks(mtmd_ctx, ctx, chunks, pos, 0, 512, true, &pos);
            n_past = pos;
            mtmd_input_chunks_free(chunks);
            mtmd_bitmap_free(bmp);
            if (ret != 0) {
                std::cerr << "Image eval failed\n";
                results.push_back({p.name, "FAILED", "image eval failed", 0});
                log << "Result: FAILED (image eval)\n";
                continue;
            }
            // replace marker in history with a text label
            messages.back().content = "[Image: robot2.jpg]\n" + p.prompt;
        } else {
            // text-only path: tokenize and feed in batches
            auto tokens = tokenize(vocab, formatted);
            llama_batch batch = llama_batch_init(512, 0, 1);
            int n = (int)tokens.size();
            for (int start = 0; start < n; start += 512) {
                batch.n_tokens = 0;
                int end = std::min(start + 512, n);
                for (int i = start; i < end; i++)
                    batch_add(batch, tokens[i], i, i == n - 1);
                if (llama_decode(ctx, batch) != 0) {
                    std::cerr << "llama_decode failed during prompt feeding\n";
                    break;
                }
            }
            n_past = n;
            llama_batch_free(batch);
        }

        // init sampler (same settings as langX defaults)
        llama_sampler_chain_params sp = llama_sampler_chain_default_params();
        llama_sampler* sampler = llama_sampler_chain_init(sp);
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.8f));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));

        // generation loop
        std::string response;
        llama_batch gen_batch = llama_batch_init(1, 0, 1);
        for (int i = 0; i < 500; i++) {
            llama_token tok = llama_sampler_sample(sampler, ctx, -1);
            llama_sampler_accept(sampler, tok);
            if (llama_vocab_is_eog(vocab, tok)) break;

            char piece[256];
            int pn = llama_token_to_piece(vocab, tok, piece, sizeof(piece), 0, false);
            if (pn > 0) response.append(piece, pn);

            gen_batch.n_tokens = 0;
            batch_add(gen_batch, tok, n_past, true);
            if (llama_decode(ctx, gen_batch) != 0) break;
            n_past++;
        }
        
        llama_batch_free(gen_batch);
        llama_sampler_free(sampler);

        double prompt_s = Sec(Clock::now() - t_prompt).count();
        std::cout << response << "\n";

        messages.push_back({"assistant", response});

        bool passed = check(response, p.pattern);
        std::string status = passed ? "PASS" : "FAIL";
        results.push_back({p.name, status, "", prompt_s});

        log << "Response: " << response << "\n";
        log << "Pattern: " << p.pattern << "\n";
        log << "Result: " << status << " (" << std::fixed << std::setprecision(2) << prompt_s << "s)\n";

        std::cout << "[" << status << "] pattern=/" << p.pattern << "/\n";
    }

    // --- Summary ---
    log << "\n" << std::string(60, '=') << "\n";
    log << "SUMMARY\n";
    int total = (int)results.size();
    int n_pass = 0, n_fail = 0, n_err = 0;
    for (auto& r : results) {
        if (r.status == "PASS") n_pass++;
        else if (r.status == "FAIL") n_fail++;
        else n_err++;
    }

    for (auto& r : results) {
        log << "  " << r.name << ": " << r.status;
        if (r.time > 0) log << " (" << std::fixed << std::setprecision(2) << r.time << "s)";
        log << "\n";
    }

    double total_s = Sec(Clock::now() - t_total).count();
    log << "\nTotal: " << n_pass << "/" << total << " passed, " << n_fail << " failed, " << n_err << " errors\n";
    log << "Total time: " << std::fixed << std::setprecision(2) << total_s << "s\n";

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "SUMMARY: " << n_pass << "/" << total << " passed, " << n_fail << " failed, " << n_err << " errors\n";
    std::cout << "Results saved to: " << out_file.string() << "\n";

    std::ofstream(out_file) << log.str();

    // cleanup
    mtmd_free(mtmd_ctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
