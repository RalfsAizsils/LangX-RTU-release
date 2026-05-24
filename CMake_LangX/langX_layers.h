#pragma once
#include "langX.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "chat.h"
#include <vector>
#include <string>
#include <future>

// --- InferenceState ---

struct langX::InferenceState {
    // llama.cpp internals
    std::vector<llama_token> tokens;
    llama_sampler* sampler = nullptr;
    llama_batch batch = {};
    const llama_vocab* vocab = nullptr;
    bool has_images = false;
    std::vector<mtmd_bitmap*> image_bitmaps;
    std::vector<std::string> bitmap_names;
    std::string user_message_content;
    std::string formatted_prompt;
    std::string injected_sys_prompt;
    langX::Stack* stack; 
    langX::Model* model;
    langX::Conversation* conversation;
    langX::InquerySettings* settings;

    struct PendingAsyncTool {
        std::string name;
        std::future<std::string> future;
    };
    std::vector<PendingAsyncTool> pending_async_tools;
    common_chat_params native_chat_params;
};

void apply_chat_template(langX::InferenceState* state, const std::vector<langX::ChatMessage>& msgs, const std::string& sys_content);
void apply_chat_template_with_memory(langX::InferenceState* state);

// --- layer_* declarations ---

bool layer_init_inference (langX::InferenceState* s, const langX::Layer* sl);
bool layer_file_processing (langX::InferenceState* s, const langX::Layer* sl);
bool layer_user_push_images (langX::InferenceState* s, const langX::Layer* sl);
bool layer_user_push_prompt (langX::InferenceState* s, const langX::Layer* sl);
bool layer_build_context (langX::InferenceState* s, const langX::Layer* sl);
bool layer_load_chat_template(langX::InferenceState* s, const langX::Layer* sl);
bool layer_cxt_win_trim (langX::InferenceState* s, const langX::Layer* sl);
bool layer_clear_kv_cache (langX::InferenceState* s, const langX::Layer* sl);
bool layer_init_sampler (langX::InferenceState* s, const langX::Layer* sl);
bool layer_init_batch (langX::InferenceState* s, const langX::Layer* sl);
bool layer_feed_prompt (langX::InferenceState* s, const langX::Layer* sl);
bool layer_feed_prompt_images(langX::InferenceState* s, const langX::Layer* sl);
bool layer_llm_generation (langX::InferenceState* s, const langX::Layer* sl);
bool layer_swap_model (langX::InferenceState* s, const langX::Layer* sl);
bool layer_swap_convo (langX::InferenceState* s, const langX::Layer* sl);
bool layer_set_system_prompt (langX::InferenceState* s, const langX::Layer* sl);
bool layer_load_system_prompt(langX::InferenceState* s, const langX::Layer* sl);
bool layer_use_tools (langX::InferenceState* s, const langX::Layer* sl);
bool layer_wait_tools (langX::InferenceState* s, const langX::Layer* sl);
bool layer_llm_tool_gen (langX::InferenceState* s, const langX::Layer* sl);
bool layer_free_sampler (langX::InferenceState* s, const langX::Layer* sl);
bool layer_free_batch (langX::InferenceState* s, const langX::Layer* sl);
bool layer_save_to_history (langX::InferenceState* s, const langX::Layer* sl);
bool layer_rag_retrieval (langX::InferenceState* s, const langX::Layer* sl);
bool layer_filler_rand (langX::InferenceState* s, const langX::Layer* sl);
bool layer_filler_loop (langX::InferenceState* s, const langX::Layer* sl);
bool layer_quick_look (langX::InferenceState* s, const langX::Layer* sl);
bool layer_prompt_check (langX::InferenceState* s, const langX::Layer* sl);
bool layer_quick_ask (langX::InferenceState* s, const langX::Layer* sl);
bool layer_substack (langX::InferenceState* s, const langX::Layer* sl);
bool layer_wait_time (langX::InferenceState* s, const langX::Layer* sl);
bool layer_wait_lambda (langX::InferenceState* s, const langX::Layer* sl);
bool layer_wait_promise (langX::InferenceState* s, const langX::Layer* sl);
bool layer_llm_prompt_filter (langX::InferenceState* s, const langX::Layer* sl);
bool layer_llm_result_filter (langX::InferenceState* s, const langX::Layer* sl);
bool layer_fact_check (langX::InferenceState* s, const langX::Layer* sl);
bool layer_semantic_memory (langX::InferenceState* s, const langX::Layer* sl);
bool layer_semantic_mem_retrieval (langX::InferenceState* s, const langX::Layer* sl);
bool layer_episodic_memory (langX::InferenceState* s, const langX::Layer* sl);
bool layer_episodic_tiered_memory (langX::InferenceState* s, const langX::Layer* sl);
bool layer_build_episodic_mem (langX::InferenceState* s, const langX::Layer* sl);
bool layer_build_semantic_mem (langX::InferenceState* s, const langX::Layer* sl);
bool layer_load_procedural_mem (langX::InferenceState* s, const langX::Layer* sl);
bool layer_build_procedural_mem (langX::InferenceState* s, const langX::Layer* sl);
