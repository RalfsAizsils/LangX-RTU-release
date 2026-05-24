#include "langX_layers.h"
#include "LxRAG.h"
#include "chat.h"
#include <nlohmann/json.hpp>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <random>
#include <regex>
#include <thread>
#include <chrono>

LANGX_API langX::LangX langX::global_LangX;

std::filesystem::path langX::getDefaultDataPath() {
#ifdef _WIN32
    const char* appData = getenv("LOCALAPPDATA");
    if (appData) return std::filesystem::path(appData) / "LangX";
#elif __APPLE__
    const char* home = getenv("HOME");
    if (home) return std::filesystem::path(home) / "Library/Application Support/LangX";
#else
    const char* xdgData = getenv("XDG_DATA_HOME");
    if (xdgData) return std::filesystem::path(xdgData) / "langx";
    const char* home = getenv("HOME");
    if (home) return std::filesystem::path(home) / ".local/share/langx";
#endif
    return std::filesystem::current_path();
}

langX::LangX* langX::initialize_langX(langX::Config config) {
    llama_backend_init();

    // TODO: add verbos check for global langX
    std::cout <<"LangX: Llama backend initialized." << std::endl;
    std::cout <<"LangX: System Info: " << llama_print_system_info() << std::endl;
    std::cout <<"LangX: Initializing...\n";

    if (config.user_output_path.empty()) {
        config.user_output_path = getDefaultDataPath().string();
    }

    std::filesystem::create_directories(config.user_output_path);

    langX::Config* persistent_config = new langX::Config(config);
    langX::global_LangX.config = persistent_config;
    langX::global_LangX.verbose_logs = config.verbose_logs;

    auto lx_log_cb = [](enum ggml_log_level level, const char* text, void*) {
        if (langX::global_LangX.verbose_logs && level >= GGML_LOG_LEVEL_WARN)
            fprintf(stderr, "%s", text);
    };
    llama_log_set(lx_log_cb, nullptr);
    mtmd_helper_log_set(lx_log_cb, nullptr);

    langX::makeDefaultStack("default");
    langX::global_LangX.last_refrenced_stack_id = "default";

    std::cout << "LangX: Init done.\n";
    return &langX::global_LangX;
}

static langX::Stack* getLastStack() {
    auto it = langX::global_LangX.hotswap_stacks.find(langX::global_LangX.last_refrenced_stack_id);
    return (it != langX::global_LangX.hotswap_stacks.end()) ? it->second : nullptr;
}

static langX::Stack* get_or_last_stack(langX::Stack* s) {
    return s ? s : getLastStack();
}

static std::string gen_convo_id() {
    static int counter = 0;
    return "conv_" + std::to_string(counter++);
}

static void free_model_resources(langX::Model* m) {
    if (!m) return;
    if (m->mtmd_ctx) mtmd_free(m->mtmd_ctx);
    if (m->lora_adapter) llama_adapter_lora_free(m->lora_adapter);
    if (m->ctx) llama_free(m->ctx);
    if (m->model) llama_free_model(m->model);
    delete m->config_reference;
    delete m->params;
    delete m->ctx_params;
}

static langX::Model* load_model(const langX::ModelParams& params) {
    if (params.path.empty()) {
        std::cerr << "LangX Error [LM-00]: Model path is empty.\n";
        return nullptr;
    }
    if (!std::filesystem::exists(params.path)) {
        std::cerr << "LangX Error [LM-01]: Model file not found: " << params.path << "\n";
        return nullptr;
    }
    if (!params.vission_path.empty() && !std::filesystem::exists(params.vission_path)) {
        std::cerr << "LangX Error [LM-02]: Vision projector (mmproj) file not found: " << params.vission_path << "\n";
        return nullptr;
    }

    llama_model_params* persistent_model_params = new llama_model_params(llama_model_default_params());
    llama_context_params* persistent_ctx_params = new llama_context_params(llama_context_default_params());
    langX::Model* model = new langX::Model();
    model->params = persistent_model_params;
    model->ctx_params = persistent_ctx_params;
    model->config_reference = new langX::ModelParams(params);

    model->params->n_gpu_layers = params.n_gpu_layers;
    model->params->main_gpu = params.main_gpu;
    if (!params.tensor_split.empty()) {
        model->params->tensor_split = params.tensor_split.data();
    }

    // TODO: more settings from params

    model->model = llama_load_model_from_file(params.path.string().c_str(), *model->params);
    if (!model->model) {
        std::cerr << "LangX Error [LM-03]: Failed to load model from " << params.path << "\n";
        free_model_resources(model); delete model;
        return nullptr;
    }

    model->chat_template = llama_model_chat_template(model->model, nullptr);

    // TODO: auto-calculate n_ctx based on model size and available VRAM (could use llama_print_system_info to get VRAM info)
    model->ctx_params->n_ctx    = params.n_ctx;
    model->ctx_params->n_threads = params.n_threads;
    model->ctx_params->n_threads_batch = params.n_threads_batch;
    if (params.n_batch  > 0) model->ctx_params->n_batch  = params.n_batch;
    if (params.n_ubatch > 0) model->ctx_params->n_ubatch = params.n_ubatch;
    // TODO: more settings from params

    model->ctx = llama_new_context_with_model(model->model, *model->ctx_params);
    if (!model->ctx) {
        std::cerr << "LangX Error [LM-04]: Failed to create context.\n";
        free_model_resources(model); delete model;
        return nullptr;
    }

    if (params.vission_path != "") {
        mtmd_context_params vp = mtmd_context_params_default();
        model->mtmd_ctx = mtmd_init_from_file(params.vission_path.string().c_str(), model->model, vp);
        if (!model->mtmd_ctx) {
            std::cerr << "LangX Error [LM-05]: Failed to load mmproj.\n";
            free_model_resources(model); delete model;
            return nullptr;
        }
    }

    if (!params.lora_path.empty()) {
        if (!std::filesystem::exists(params.lora_path)) {
            std::cerr << "LangX Error [LM-06]: LoRA file not found: " << params.lora_path << "\n";
            free_model_resources(model); delete model;
            return nullptr;
        }
        model->lora_adapter = llama_adapter_lora_init(model->model, params.lora_path.string().c_str());
        if (!model->lora_adapter) {
            std::cerr << "LangX Error [LM-07]: Failed to load LoRA adapter: " << params.lora_path << "\n";
            free_model_resources(model); delete model;
            return nullptr;
        }
        if (llama_set_adapter_lora(model->ctx, model->lora_adapter, params.lora_scale) != 0) {
            std::cerr << "LangX Error [LM-08]: Failed to apply LoRA to context.\n";
            free_model_resources(model); delete model;
            return nullptr;
        }
    }

    return model;
}

void langX::initModel(langX::ModelParams params, const char* id, langX::Stack* target_stack) {
    langX::Stack* stack = get_or_last_stack(target_stack);

    if (stack && stack->model) {
        bool in_map = false;
        for (auto& [k, v] : langX::global_LangX.loaded_models)
            if (v == stack->model) { in_map = true; break; }
        if (!in_map) { free_model_resources(stack->model); delete stack->model; }
        stack->model = nullptr;
    }

    langX::Model* model = load_model(params);
    if (!model) return;

    if (stack) {
        stack->model = model;
        stack->active_model_id = (id && *id) ? id : "";
        if (!stack->conversation) stack->conversation = new langX::Conversation();
        if (!stack->settings) stack->settings = new langX::InquerySettings();
    }

    if (id && *id) langX::global_LangX.loaded_models[id] = model;
    std::cout << "LangX: Model loaded successfully.\n";
}

void langX::unloadModel(langX::Stack* target_stack) {
    langX::Stack* stack = get_or_last_stack(target_stack);
    if (!stack || !stack->model) return;
    auto* m = stack->model;

    for (auto it = langX::global_LangX.loaded_models.begin(); it != langX::global_LangX.loaded_models.end(); ++it)
        if (it->second == m) { langX::global_LangX.loaded_models.erase(it); break; }

    for (auto& [sid, s] : langX::global_LangX.hotswap_stacks)
        if (s && s->model == m) s->model = nullptr;

    free_model_resources(m);
    delete m;
}

// --- Model management ---

void langX::unloadModel(const char* model_id) {
    auto it = langX::global_LangX.loaded_models.find(model_id);
    if (it == langX::global_LangX.loaded_models.end()) {
        std::cerr << "LangX Error [UM-00]: Model not found in loaded_models: " << model_id << "\n";
        return;
    }
    langX::Model* m = it->second;
    for (auto& [sid, s] : langX::global_LangX.hotswap_stacks)
        if (s && s->model == m) s->model = nullptr;
    free_model_resources(m);
    delete m;
    langX::global_LangX.loaded_models.erase(it);
}

void langX::switchModel(const char* model_id, langX::Stack* target_stack) {
    auto it = langX::global_LangX.loaded_models.find(model_id);
    if (it == langX::global_LangX.loaded_models.end()) {
        std::cerr << "LangX Error [SW-00]: Model not found in loaded_models: " << model_id << "\n";
        return;
    }
    langX::Stack* stack = get_or_last_stack(target_stack);
    if (stack) {
        stack->model = it->second;
        stack->active_model_id = model_id;
    }
}

void langX::sleepModel(const char* model_id) {
    auto it = langX::global_LangX.loaded_models.find(model_id);
    if (it == langX::global_LangX.loaded_models.end()) {
        std::cerr << "LangX Error [SL-00]: Model not found in loaded_models: " << model_id << "\n";
        return;
    }
    langX::Model* m = it->second;
    langX::ModelParams saved = *m->config_reference;
    for (auto& [sid, s] : langX::global_LangX.hotswap_stacks)
        if (s && s->model == m) s->model = nullptr;
    free_model_resources(m);
    delete m;
    langX::global_LangX.loaded_models.erase(it);
    langX::global_LangX.sleeping_models[model_id] = saved;
    std::cout << "LangX: Model [" << model_id << "] moved to sleeping.\n";
}

void langX::wakeupModel(const char* model_id) {
    auto it = langX::global_LangX.sleeping_models.find(model_id);
    if (it == langX::global_LangX.sleeping_models.end()) {
        std::cerr << "LangX Error [WU-00]: Model not found in sleeping_models: " << model_id << "\n";
        return;
    }
    langX::Model* model = load_model(it->second);
    if (!model) return;
    langX::global_LangX.loaded_models[model_id] = model;
    langX::global_LangX.sleeping_models.erase(it);
    std::cout <<"LangX: Model [" << model_id << "] woken up into loaded_models.\n";
}

void langX::initSleepingModel(langX::ModelParams params, const char* id) {
    langX::global_LangX.sleeping_models[id] = params;
}

void langX::initLoadedModel(langX::ModelParams params, const char* id) {
    langX::Model* model = load_model(params);
    if (!model) return;
    auto it = langX::global_LangX.loaded_models.find(id);
    if (it != langX::global_LangX.loaded_models.end()) {
        bool in_use = false;
        for (auto& [sid, st] : langX::global_LangX.hotswap_stacks)
            if (st->model == it->second) { in_use = true; break; }
        if (!in_use) {
            std::cout << "LangX: Freeing model with the same id [" << id << "].\n";
            free_model_resources(it->second);
            delete it->second;
        }
    }
    langX::global_LangX.loaded_models[id] = model;
    std::cout <<"LangX: Model [" << id << "] loaded into loaded_models.\n";
}

// --- Conversation management ---

void langX::initConversation(const char* id, langX::Stack* target_stack) {
    langX::Stack* stack = get_or_last_stack(target_stack);
    if (stack && stack->conversation) {
        if (!stack->active_convo_id.empty())
            langX::global_LangX.hotswap_conversation[stack->active_convo_id] = stack->conversation;
        else
            delete stack->conversation;
        stack->conversation = nullptr;
    }
    auto* new_conv = new langX::Conversation();
    if (stack) {
        stack->conversation    = new_conv;
        stack->active_convo_id = (id && *id) ? id : "";
    }
}

void langX::swapConversations(const char* conversation_id, langX::Stack* target_stack) {
    auto it = langX::global_LangX.hotswap_conversation.find(conversation_id);
    if (it == langX::global_LangX.hotswap_conversation.end() || !it->second) {
        std::cerr << "LangX Error [SC-00]: Conversation not found: " << conversation_id << "\n";
        return;
    }
    langX::Stack* stack = get_or_last_stack(target_stack);
    if (!stack) return;

    if (stack->conversation) {
        std::string save_id = stack->active_convo_id.empty() ? gen_convo_id() : stack->active_convo_id;
        langX::global_LangX.hotswap_conversation[save_id] = stack->conversation;
    }
    stack->conversation    = it->second;
    stack->active_convo_id = conversation_id;
    langX::global_LangX.hotswap_conversation.erase(it);
}

void langX::loadConversations(langX::Conversation* conversation, const char* id) {
    std::string key = (id && *id) ? id : gen_convo_id();
    langX::global_LangX.hotswap_conversation[key] = conversation;
}

void langX::unloadConversations(const char* conversation_id) {
    auto it = langX::global_LangX.hotswap_conversation.find(conversation_id);
    if (it == langX::global_LangX.hotswap_conversation.end()) {
        std::cerr << "LangX Error [UC-00]: Conversation not found: " << conversation_id << "\n";
        return;
    }
    delete it->second;
    langX::global_LangX.hotswap_conversation.erase(it);
}

// --- Inference helper functions ---

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text) {
    std::vector<llama_token> tokens(text.length() + 2);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);

    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.length(), tokens.data(), tokens.size(), false, true);
    }

    tokens.resize(n_tokens);
    return tokens;
}

void langX_batch_add(llama_batch& batch, llama_token token_id, int pos, const std::vector<int>& seq_ids, bool logits) {
    if (!batch.seq_id) return;
    int idx = batch.n_tokens;
    batch.token[idx] = token_id;
    batch.pos[idx] = pos;
    batch.n_seq_id[idx] = seq_ids.size();
    for (size_t s = 0; s < seq_ids.size(); s++) {
        batch.seq_id[idx][s] = seq_ids[s];
    }
    batch.logits[idx] = logits;
    batch.n_tokens++;
}

static bool is_image_file(const std::filesystem::path& p) {
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    static const char* img_exts[] = { ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tga", nullptr };
    for (int i = 0; img_exts[i]; i++) {
        if (ext == img_exts[i]) return true;
    }
    return false;
}

static bool is_file_in_context(const langX::Conversation* conv, const std::string& filename) {
    if (!conv) return false;
    std::string label = "[File: " + filename + "]";
    for (const auto& msg : conv->active_messages)
        if (msg.content.find(label) != std::string::npos) return true;
    return false;
}

// --- Layer Dependacies ---

struct LayerDepInfo {
    std::vector<langX::LayerType> pre;
    std::vector<langX::LayerType> post;
    std::vector<langX::LayerType> global;
};

static const LayerDepInfo LAYER_DEPS[] = {
    { {}, {}, {} }, // INIT_INFERENCE
    { { langX::INIT_INFERENCE }, {}, {} }, // FILE_PROCESSING
    { { langX::INIT_INFERENCE }, {}, {} }, // USER_PUSH_PROMT
    { { langX::INIT_INFERENCE }, {}, {} }, // USER_PUSH_IMAGES
    { { langX::USER_PUSH_PROMT }, {}, {} }, // BUILD_CONTEXT
    { { langX::USER_PUSH_PROMT }, {}, {} }, // LOAD_CHAT_TEMPLATE
    { { langX::USER_PUSH_PROMT }, {}, {} },// CXT_WIN_TRIM
    { {}, {}, {} }, // CLEAR_KV_CACHE
    { { langX::INIT_INFERENCE }, { langX::FREE_SAMPLER }, {} }, // INIT_SAMPLER
    { { langX::INIT_INFERENCE }, { langX::FREE_BATCH }, {} }, // INIT_BATCH
    { { langX::CLEAR_KV_CACHE, langX::INIT_BATCH }, {}, {} }, // FEED_PROMPT
    { { langX::CLEAR_KV_CACHE, langX::INIT_BATCH, langX::USER_PUSH_IMAGES }, {}, {} },// FEED_PROMPT_IMAGES
    { { langX::INIT_SAMPLER, langX::INIT_BATCH }, {}, {} }, // LLM_GENERATION
    { { langX::INIT_SAMPLER, langX::LLM_GENERATION }, {}, {} }, // LLM_SAMPLE
    { {}, {}, {} }, // FREE_SAMPLER
    { {}, {}, {} }, // FREE_BATCH
    { {}, {}, {} }, // SAVE_TO_HISTORY
    { {}, {}, {} }, // GOTO
    { {}, {}, {} }, // BRANCH
    { {}, {}, {} }, // CUSTOM
    { {}, {}, {} }, // DEBUG
    { {}, {}, {} }, // SWAP_MODEL
    { {}, {}, {} }, // SWAP_CONVO
    { {}, {}, {} }, // SET_SYSTEM_PROMPT
    { {}, { langX::CXT_WIN_TRIM }, {} }, // LOAD_SYSTEM_PROMPT
    { { langX::LLM_GENERATION }, {}, {} }, // USE_TOOLS
    { { langX::USE_TOOLS }, {}, {} }, // WAIT_TOOLS
    { { langX::INIT_SAMPLER, langX::INIT_BATCH }, {}, {} }, // LLM_TOOL_GEN
    { { langX::INIT_INFERENCE }, {}, { langX::USER_PUSH_PROMT } }, // RAG_RETRIEVAL
    { {}, {}, {} }, // FILLER_RAND
    { {}, {}, {} }, // FILLER_LOOP
    { {}, {}, {} }, // QUICK_LOOK
    { {}, {}, {} }, // PROMPT_CHECK
    { {}, {}, {} }, // QUICK_ASK
    { {}, {}, {} }, // SUBSTACK
    { {}, {}, {} }, // WAIT_TIME
    { {}, {}, {} }, // WAIT_LAMBDA
    { {}, {}, {} }, // WAIT_PROMISE
    { { langX::USER_PUSH_PROMT }, {}, {} }, // EPISODIC_MEMORY
    { { langX::INIT_INFERENCE }, {}, { langX::USER_PUSH_PROMT } }, // SEMANTIC_MEMORY
    { { langX::INIT_INFERENCE }, {}, { langX::USER_PUSH_PROMT } }, // SEMANTIC_MEM_RETRIEVAL
    { { langX::USER_PUSH_PROMT }, {}, {} }, // EPISODIC_TIERED_MEMORY
    { {}, {}, {} }, // LLM_PROMPT_FILTER
    { { langX::LLM_GENERATION }, {}, {} }, // LLM_RESULT_FILTER
    { { langX::LLM_GENERATION }, {}, {} }, // FACT_CHECK
    { {}, {}, {} }, // BUILD_EPISODIC_MEM
    { { langX::SAVE_TO_HISTORY }, {}, {} }, // BUILD_SEMANTIC_MEM
    { { langX::SAVE_TO_HISTORY }, {}, {} }, // ASYNC_BREAK
    { {}, { langX::BUILD_CONTEXT }, {} }, // LOAD_PROCEDURAL_MEM
    { { langX::SAVE_TO_HISTORY }, {}, {} }, // BUILD_PROCEDURAL_MEM
    { { langX::FEED_PROMPT }, {}, {} }, // BERT_GENERATE
    { { langX::FEED_PROMPT }, {}, {} }, // NLI_GENERATE
    { { langX::USER_PUSH_PROMT }, { langX::BUILD_CONTEXT }, {} }, // COT_GENERATE
    { {}, {}, {} }, // SELF_CONSISTENT_GEN
};

static const char* layer_name(langX::LayerType t) {
    switch (t) {
        case langX::INIT_INFERENCE: return "INIT_INFERENCE";
        case langX::FILE_PROCESSING: return "FILE_PROCESSING";
        case langX::USER_PUSH_PROMT: return "USER_PUSH_PROMT";
        case langX::USER_PUSH_IMAGES: return "USER_PUSH_IMAGES";
        case langX::BUILD_CONTEXT: return "BUILD_CONTEXT";
        case langX::LOAD_CHAT_TEMPLATE: return "LOAD_CHAT_TEMPLATE";
        case langX::CXT_WIN_TRIM: return "CXT_WIN_TRIM";
        case langX::CLEAR_KV_CACHE: return "CLEAR_KV_CACHE";
        case langX::INIT_SAMPLER: return "INIT_SAMPLER";
        case langX::INIT_BATCH: return "INIT_BATCH";
        case langX::FEED_PROMPT: return "FEED_PROMPT";
        case langX::FEED_PROMPT_IMAGES: return "FEED_PROMPT_IMAGES";
        case langX::LLM_GENERATION: return "LLM_GENERATION";
        case langX::LLM_SAMPLE: return "LLM_SAMPLE";
        case langX::FREE_SAMPLER: return "FREE_SAMPLER";
        case langX::FREE_BATCH: return "FREE_BATCH";
        case langX::SAVE_TO_HISTORY: return "SAVE_TO_HISTORY";
        case langX::GOTO: return "GOTO";
        case langX::BRANCH: return "BRANCH";
        case langX::CUSTOM: return "CUSTOM";
        case langX::DEBUG: return "DEBUG";
        case langX::SWAP_MODEL: return "SWAP_MODEL";
        case langX::SWAP_CONVO: return "SWAP_CONVO";
        case langX::SET_SYSTEM_PROMPT: return "SET_SYSTEM_PROMPT";
        case langX::USE_TOOLS: return "USE_TOOLS";
        case langX::WAIT_TOOLS: return "WAIT_TOOLS";
        case langX::LLM_TOOL_GEN: return "LLM_TOOL_GEN";
        case langX::LOAD_SYSTEM_PROMPT: return "LOAD_SYSTEM_PROMPT";
        case langX::RAG_RETRIEVAL: return "RAG_RETRIEVAL";
        case langX::FILLER_RAND: return "FILLER_RAND";
        case langX::FILLER_LOOP: return "FILLER_LOOP";
        case langX::QUICK_LOOK: return "QUICK_LOOK";
        case langX::PROMPT_CHECK: return "PROMPT_CHECK";
        case langX::QUICK_ASK: return "QUICK_ASK";
        case langX::SUBSTACK: return "SUBSTACK";
        case langX::WAIT_TIME: return "WAIT_TIME";
        case langX::WAIT_LAMBDA: return "WAIT_LAMBDA";
        case langX::WAIT_PROMISE: return "WAIT_PROMISE";
        case langX::LLM_PROMPT_FILTER: return "LLM_PROMPT_FILTER";
        case langX::LLM_RESULT_FILTER: return "LLM_RESULT_FILTER";
        case langX::FACT_CHECK: return "FACT_CHECK";
        case langX::EPISODIC_MEMORY: return "EPISODIC_MEMORY";
        case langX::SEMANTIC_MEMORY: return "SEMANTIC_MEMORY";
        case langX::SEMANTIC_MEM_RETRIEVAL: return "SEMANTIC_MEM_RETRIEVAL";
        case langX::EPISODIC_TIERED_MEMORY: return "EPISODIC_TIERED_MEMORY (deprecated)";
        case langX::BUILD_EPISODIC_MEM: return "BUILD_EPISODIC_MEM";
        case langX::BUILD_SEMANTIC_MEM: return "BUILD_SEMANTIC_MEM";
        case langX::ASYNC_BREAK: return "ASYNC_BREAK";
        case langX::LOAD_PROCEDURAL_MEM: return "LOAD_PROCEDURAL_MEM";
        case langX::BUILD_PROCEDURAL_MEM: return "BUILD_PROCEDURAL_MEM";
        case langX::BERT_GENERATE: return "BERT_GENERATE";
        case langX::NLI_GENERATE: return "NLI_GENERATE";
        case langX::COT_GENERATE: return "COT_GENERATE";
        case langX::SELF_CONSISTENT_GEN: return "SELF_CONSISTENT_GEN";
        default: return "UNKNOWN";
    }
}

static bool validate_stack(const langX::Stack* stack, std::string& error_out) {
    std::vector<langX::LayerType> all_types;
    for (const auto* l : stack->layers)
        all_types.push_back(l->type);

    for (int i = 0; i < (int)stack->layers.size(); i++) {
        langX::LayerType type = stack->layers[i]->type;
        const LayerDepInfo& deps = LAYER_DEPS[type];

        for (auto dep : deps.pre) {
            bool found = false;
            for (int j = 0; j < i; j++)
                if (stack->layers[j]->type == dep) { found = true; break; }
            if (!found) {
                error_out = std::string(layer_name(type)) + " requires " + layer_name(dep) + " before it";
                return false;
            }
        }

        for (auto dep : deps.post) {
            bool found = false;
            for (int j = i + 1; j < (int)stack->layers.size(); j++)
                if (stack->layers[j]->type == dep) { found = true; break; }
            if (!found) {
                error_out = std::string(layer_name(type)) + " requires " + layer_name(dep) + " after it";
                return false;
            }
        }

        for (auto dep : deps.global) {
            bool found = false;
            for (auto t : all_types) if (t == dep) { found = true; break; }
            if (!found) {
                error_out = std::string(layer_name(type)) + " requires " + layer_name(dep) + " anywhere in the stack";
                return false;
            }
        }
    }
    return true;
}

// --- Tool-call parsing ---

// Escapes a string for safe embedding as a JSON string value.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default: out += (char)c; break;
        }
    }
    return out;
}

struct ParsedToolCall {
    std::string name;
    std::string id; // native tool call id (empty for XML none-native)
    std::map<std::string, std::string> args;
};

static std::vector<common_chat_tool> tools_to_common_chat_tools(const std::vector<langX::Tool*>& tools);

static std::vector<ParsedToolCall> parse_tool_calls(const std::string& text) {
    std::vector<ParsedToolCall> result;
    const std::string open_tag  = "<tool_call>";
    const std::string close_tag = "</tool_call>";
    size_t pos = 0;
    while (true) {
        size_t start = text.find(open_tag, pos);
        if (start == std::string::npos) break;
        size_t end = text.find(close_tag, start);
        if (end == std::string::npos) break;
        std::string block = text.substr(start + open_tag.size(), end - start - open_tag.size());
        pos = end + close_tag.size();

        ParsedToolCall call;

        size_t np = block.find("\"name\"");
        if (np == std::string::npos) continue;
        size_t nc = block.find(':', np);
        if (nc == std::string::npos) continue;
        size_t nq1 = block.find('"', nc);
        if (nq1 == std::string::npos) continue;
        size_t nq2 = block.find('"', nq1 + 1);
        if (nq2 == std::string::npos) continue;
        call.name = block.substr(nq1 + 1, nq2 - nq1 - 1);

        size_t ap = block.find("\"args\"");
        if (ap != std::string::npos) {
            size_t brace = block.find('{', ap);
            if (brace != std::string::npos) {
                size_t brace_end = block.find('}', brace);
                if (brace_end != std::string::npos) {
                    std::string args_str = block.substr(brace + 1, brace_end - brace - 1);
                    size_t apos = 0;
                    while (true) {
                        size_t kq1 = args_str.find('"', apos);
                        if (kq1 == std::string::npos) break;
                        size_t kq2 = args_str.find('"', kq1 + 1);
                        if (kq2 == std::string::npos) break;
                        std::string key = args_str.substr(kq1 + 1, kq2 - kq1 - 1);
                        size_t ac = args_str.find(':', kq2);
                        if (ac == std::string::npos) break;
                        size_t vs = args_str.find_first_not_of(" \t", ac + 1);
                        if (vs == std::string::npos) break;
                        if (args_str[vs] == '"') {
                            size_t vq2 = args_str.find('"', vs + 1);
                            if (vq2 == std::string::npos) break;
                            call.args[key] = args_str.substr(vs + 1, vq2 - vs - 1);
                            apos = vq2 + 1;
                        } else {
                            size_t ve = args_str.find_first_of(",}", vs);
                            std::string val = args_str.substr(vs, ve == std::string::npos ? std::string::npos : ve - vs);
                            while (!val.empty() && std::isspace((unsigned char)val.back())) val.pop_back();
                            call.args[key] = val;
                            apos = ve == std::string::npos ? args_str.size() : ve;
                        }
                    }
                }
            }
        }
        result.push_back(std::move(call));
    }
    return result;
}

static std::vector<ParsedToolCall> parse_tool_calls_native(const std::string& text, const common_chat_params& chat_params) {
    common_chat_parser_params pp(chat_params);
    if (!chat_params.parser.empty())
        pp.parser.load(chat_params.parser);
    pp.parse_tool_calls = true;

    common_chat_msg parsed = common_chat_parse(text, false, pp);

    std::vector<ParsedToolCall> result;
    for (const auto& tc : parsed.tool_calls) {
        ParsedToolCall pc;
        pc.name = tc.name;
        pc.id   = tc.id;
        if (!tc.arguments.empty()) {
            try {
                auto args = nlohmann::json::parse(tc.arguments);
                if (args.is_object()) {
                    for (auto& [k, v] : args.items())
                        pc.args[k] = v.is_string() ? v.get<std::string>() : v.dump();
                }
            } catch (...) {}
        }
        result.push_back(std::move(pc));
    }
    return result;
}

// --- Tokenasation and templates ---

static int estimate_tokens(const llama_vocab* vocab, const std::string& text) {
    return (int)tokenize(vocab, text).size();
}

static constexpr int TEMPLATE_OVERHEAD = 50;

static int estimate_context_tokens(const llama_vocab* vocab, const std::string& injected_sys_prompt, const langX::Conversation* conv) {
    int total = TEMPLATE_OVERHEAD;
    total += estimate_tokens(vocab, injected_sys_prompt);
    total += estimate_tokens(vocab, conv->episodic_tier1_summary);
    for (const auto& t2 : conv->episodic_tier2_memories)
        total += estimate_tokens(vocab, t2);
    for (const auto& msg : conv->active_messages)
        total += estimate_tokens(vocab, msg.content);
    for (const auto& ex : conv->extra_data) {
        if (!ex.label.empty()) total += estimate_tokens(vocab, ex.label);
        total += estimate_tokens(vocab, ex.content);
    }
    return total;
}

void apply_chat_template(langX::InferenceState* state, const std::vector<langX::ChatMessage>& msgs, const std::string& sys_content) {
    auto* model = state->model;
    bool native = state->settings && state->settings->use_native_tools && state->stack && !state->stack->tools.empty();

    if (native) {
        std::string tmpl_override = model->chat_template ? model->chat_template : "";
        auto tmpls = common_chat_templates_init(model->model, tmpl_override);

        std::vector<common_chat_msg> chat_msgs;
        if (!sys_content.empty())
            chat_msgs.push_back({ "system", sys_content });
        for (const auto& msg : msgs) {
            common_chat_msg cm;
            cm.role = msg.role;
            cm.content = msg.content;
            chat_msgs.push_back(cm);
        }

        common_chat_templates_inputs inputs;
        inputs.messages = chat_msgs;
        inputs.tools = tools_to_common_chat_tools(state->stack->tools);
        inputs.add_generation_prompt = true;
        inputs.tool_choice = COMMON_CHAT_TOOL_CHOICE_AUTO;

        state->native_chat_params = common_chat_templates_apply(tmpls.get(), inputs);
        state->formatted_prompt = state->native_chat_params.prompt;
        state->tokens = tokenize(state->vocab, state->formatted_prompt);
    } else {
        std::vector<llama_chat_message> chat_msgs;
        if (!sys_content.empty())
            chat_msgs.push_back({ "system", sys_content.c_str() });
        for (const auto& msg : msgs)
            chat_msgs.push_back({ msg.role.c_str(), msg.content.c_str() });

        int req_len = llama_chat_apply_template(model->chat_template, chat_msgs.data(), chat_msgs.size(), true, nullptr, 0);
        std::vector<char> tmpl_buf(req_len + 1);
        llama_chat_apply_template(model->chat_template, chat_msgs.data(), chat_msgs.size(), true, tmpl_buf.data(), tmpl_buf.size());
        state->formatted_prompt = std::string(tmpl_buf.data(), req_len);
        state->tokens = tokenize(state->vocab, state->formatted_prompt);
    }

    const int n_ctx = model->ctx_params->n_ctx;
    if (state->stack && !state->stack->unsafe && (int)state->tokens.size() > n_ctx)
        std::cerr << "LangX Warning [ACT-00]: formatted prompt (" << state->tokens.size() << " tokens) exceeds n_ctx (" << n_ctx << "). Reduce context or increase n_ctx.\n";
}

void apply_chat_template_with_memory(langX::InferenceState* state) {
    auto* conv = state->conversation;
    if (!conv) {
        std::cerr << "LangX Error [ACT-00]: No active conversation in apply_chat_template_with_memory.\n";
        return;
    }

    std::string sys_content = state->injected_sys_prompt;
    if (!conv->episodic_tier1_summary.empty()) {
        if (!sys_content.empty()) sys_content += "\n\n";
        sys_content += "[Conversation summary]\n" + conv->episodic_tier1_summary;
    }
    if (!conv->episodic_tier2_memories.empty()) {
        if (!sys_content.empty()) sys_content += "\n\n";
        sys_content += "[Conversation highlights]\n";
        for (const auto& m : conv->episodic_tier2_memories)
            sys_content += "- " + m + "\n";
    }
    if (!conv->procedural_memory.empty()) {
        if (!sys_content.empty()) sys_content += "\n\n";
        sys_content += "[Core Memories]\n" + conv->procedural_memory;
    }
    apply_chat_template(state, conv->active_messages, sys_content);
}

// --- Layer implementations ---

bool layer_init_inference(langX::InferenceState* s, const langX::Layer*) {
    auto* model = s->model;
    if (!model || !model->ctx || !model->model) {
        std::cerr << "LangX Error [INF-L00]: Model not initialized.\n";
        return false;
    }
    if (!s->conversation) {
        std::cerr << "LangX Error [INF-L01]: No active conversation.\n";
        return false;
    }
    s->vocab = llama_model_get_vocab(model->model);
    s->stack->has_tool_calls = false;
    s->conversation->extra_data.clear();
    return true;
}

bool layer_file_processing(langX::InferenceState* s, const langX::Layer*) {
    auto* model = s->model;
    for (const auto& f : s->stack->active_files) {
        std::filesystem::path p(f);
        if (!std::filesystem::exists(p)) {
            std::cerr << "LangX [INF-L10]: File not found: " << f << "\n";
            continue;
        }
        if (is_image_file(p)) {
            if (!model->mtmd_ctx) {
                std::cerr << "LangX [INF-L11]: Image provided but no mmproj loaded.\n";
                continue;
            }
            mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(model->mtmd_ctx, p.string().c_str());
            if (!bmp) {
                std::cerr << "LangX [INF-L12]: Failed to load image: " << f << "\n";
                continue;
            }
            s->image_bitmaps.push_back(bmp);
            s->bitmap_names.push_back(p.filename().string());
        } else {
            std::string fname = p.filename().string();
            if (!is_file_in_context(s->conversation, fname)) langX::feedTextFile(f.c_str(), nullptr, s->stack);
        }
    }
    s->has_images = !s->image_bitmaps.empty();
    return true;
}

bool layer_user_push_images(langX::InferenceState* s, const langX::Layer*) {
    for (size_t i = 0; i < s->image_bitmaps.size(); i++)
        s->user_message_content += std::string(mtmd_default_marker()) + "\n";
    return true;
}

bool layer_user_push_prompt(langX::InferenceState* s, const langX::Layer*) {
    s->user_message_content += s->stack->active_prompt;
    langX::ChatMessage msg{ "user", s->user_message_content };
    s->conversation->messages.push_back(msg);
    s->conversation->active_messages.push_back(msg);
    return true;
}

bool layer_build_context(langX::InferenceState* s, const langX::Layer*) {
    auto* conv = s->conversation;

    if (!conv->extra_data.empty()) {
        std::string extra_text;
        for (const auto& ex : conv->extra_data) {
            if (!ex.label.empty()) extra_text += "[" + ex.label + "]\n";
            extra_text += ex.content + "\n\n";
        }
        if (!s->injected_sys_prompt.empty()) s->injected_sys_prompt += "\n\n";
        s->injected_sys_prompt += extra_text;
    }

    apply_chat_template_with_memory(s);
    return true;
}

bool layer_load_chat_template(langX::InferenceState* s, const langX::Layer*) {
    apply_chat_template(s, s->conversation->active_messages, s->injected_sys_prompt);
    return true;
}

bool layer_cxt_win_trim(langX::InferenceState* s, const langX::Layer*) {
    auto* model = s->model;
    auto* settings = s->settings;
    auto* conv = s->conversation;
    const auto* vocab = s->vocab;
    const int budget = model->ctx_params->n_ctx - settings->n_tokens_to_predict;

    std::sort(conv->extra_data.begin(), conv->extra_data.end(),
        [](const langX::ContextExtra& a, const langX::ContextExtra& b) {
            return a.score > b.score;
        });

    int trimmed_turns  = 0;
    int trimmed_extras = 0;

    while (estimate_context_tokens(vocab, s->injected_sys_prompt, conv) > budget && conv->active_messages.size() > 2) {
        size_t to_erase = std::min((size_t)2, conv->active_messages.size() - 1);
        conv->active_messages.erase(conv->active_messages.begin(), conv->active_messages.begin() + to_erase);
        trimmed_turns++;
    }

    while (estimate_context_tokens(vocab, s->injected_sys_prompt, conv) > budget && !conv->extra_data.empty()) {
        conv->extra_data.pop_back();
        trimmed_extras++;
    }

    if (s->stack->verbose) {
        int final_est = estimate_context_tokens(vocab, s->injected_sys_prompt, conv);
        if (trimmed_turns > 0 || trimmed_extras > 0)
            std::cout << "[CXT_WIN_TRIM] Trimmed " << trimmed_turns << " turn(s), " << trimmed_extras << " extra(s). Est. tokens: " << final_est << " / " << budget << "\n";
        else
            std::cout << "[CXT_WIN_TRIM] No trim needed. Est. tokens: " << final_est << " / " << budget << "\n";
    }
    return true;
}

bool layer_clear_kv_cache(langX::InferenceState* s, const langX::Layer*) {
    llama_memory_t memory = llama_get_memory(s->model->ctx);
    llama_memory_seq_rm(memory, 0, 0, -1);
    s->conversation->n_past = 0;
    return true;
}

bool layer_init_sampler(langX::InferenceState* s, const langX::Layer*) {
    auto* settings = s->settings;
    if (s->sampler) { llama_sampler_free(s->sampler); s->sampler = nullptr; }
    llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    s->sampler = llama_sampler_chain_init(sp);
    if (!settings->grammar.empty()) {
        auto* vocab = llama_model_get_vocab(s->model->model);
        llama_sampler_chain_add(s->sampler, llama_sampler_init_grammar(vocab, settings->grammar.c_str(), "root"));
    }
    if (settings->top_k > 0)
        llama_sampler_chain_add(s->sampler, llama_sampler_init_top_k(settings->top_k));
    if (settings->top_p > 0.0f && settings->top_p < 1.0f)
        llama_sampler_chain_add(s->sampler, llama_sampler_init_top_p(settings->top_p, 1));
    if (settings->temperature <= 0.0f)
        llama_sampler_chain_add(s->sampler, llama_sampler_init_greedy());
    else {
        llama_sampler_chain_add(s->sampler, llama_sampler_init_temp(settings->temperature));
        llama_sampler_chain_add(s->sampler, llama_sampler_init_dist(settings->seed));
    }
    return true;
}

bool layer_init_batch(langX::InferenceState* s, const langX::Layer*) {
    auto* settings = s->settings;
    if (s->batch.token) { llama_batch_free(s->batch); s->batch = {}; }
    s->batch = llama_batch_init(settings->batch_size, 0, settings->n_sequences);
    return true;
}

bool layer_feed_prompt(langX::InferenceState* s, const langX::Layer*) {
    auto* model = s->model;
    auto* settings = s->settings;
    auto* conv = s->conversation;
    if (s->has_images) {
        std::vector<const mtmd_bitmap*> bitmaps_c(s->image_bitmaps.begin(), s->image_bitmaps.end());
        mtmd_input_text input_text = { s->formatted_prompt.c_str(), true, true };
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        int32_t ret = mtmd_tokenize(model->mtmd_ctx, chunks, &input_text, bitmaps_c.data(), bitmaps_c.size());
        if (ret != 0) {
            std::cerr << "LangX Error [INF-L40]: mtmd_tokenize failed (" << ret << ").\n";
            mtmd_input_chunks_free(chunks);
            return false;
        }
        llama_pos n_past = 0;
        ret = mtmd_helper_eval_chunks(model->mtmd_ctx, model->ctx, chunks, n_past, 0, settings->batch_size, true, &n_past);
        conv->n_past = n_past;
        mtmd_input_chunks_free(chunks);
        if (ret != 0) {
            std::cerr << "LangX Error [INF-L41]: Image eval failed.\n";
            return false;
        }
        std::string history_label;
        for (const auto& name : s->bitmap_names) history_label += "[Image: " + name + "] ";
        std::string labeled = history_label + "\n" + s->stack->active_prompt;
        if (!conv->messages.empty()) conv->messages.back().content = labeled;
        if (!conv->active_messages.empty()) conv->active_messages.back().content = labeled;
        for (auto* b : s->image_bitmaps) mtmd_bitmap_free(b);
        s->image_bitmaps.clear();
        s->has_images = false;
    } else {
        int n = (int)s->tokens.size();
        for (int start = 0; start < n; start += settings->batch_size) {
            s->batch.n_tokens = 0;
            int end = std::min(start + settings->batch_size, n);
            for (int i = start; i < end; i++)
                langX_batch_add(s->batch, s->tokens[i], i, {0}, i == n - 1);
            if (llama_decode(model->ctx, s->batch) != 0) {
                std::cerr << "LangX Error [INF-L42]: llama_decode failed.\n";
                return false;
            }
        }
        conv->n_past = n;
    }
    return true;
}

bool layer_feed_prompt_images(langX::InferenceState* s, const langX::Layer*) {
    auto* model = s->model;
    auto* settings = s->settings;
    auto* conv = s->conversation;
    std::vector<const mtmd_bitmap*> bitmaps_c(s->image_bitmaps.begin(), s->image_bitmaps.end());
    mtmd_input_text input_text = { s->formatted_prompt.c_str(), true, true };
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t ret = mtmd_tokenize(model->mtmd_ctx, chunks, &input_text, bitmaps_c.data(), bitmaps_c.size());
    if (ret != 0) {
        std::cerr << "LangX Error [INF-L50]: mtmd_tokenize failed (" << ret << ").\n";
        mtmd_input_chunks_free(chunks);
        return false;
    }
    llama_pos n_past = 0;
    ret = mtmd_helper_eval_chunks(model->mtmd_ctx, model->ctx, chunks, n_past, 0, settings->batch_size, true, &n_past);
    conv->n_past = n_past;
    mtmd_input_chunks_free(chunks);
    if (ret != 0) {
        std::cerr << "LangX Error [INF-L51]: Image eval failed.\n";
        return false;
    }
    std::string history_label;
    for (const auto& name : s->bitmap_names) history_label += "[Image: " + name + "] ";
    std::string labeled = history_label + "\n" + s->stack->active_prompt;
    if (!conv->messages.empty()) conv->messages.back().content = labeled;
    if (!conv->active_messages.empty()) conv->active_messages.back().content = labeled;
    for (auto* b : s->image_bitmaps) mtmd_bitmap_free(b);
    s->image_bitmaps.clear();
    return true;
}

bool layer_llm_generation(langX::InferenceState* s, const langX::Layer*) {
    auto* model = s->model;
    auto* settings = s->settings;
    auto* conv = s->conversation;
    for (int i = 0; i < settings->n_tokens_to_predict; i++) {
        llama_token new_token = llama_sampler_sample(s->sampler, model->ctx, -1);
        llama_sampler_accept(s->sampler, new_token);
        if (llama_vocab_is_eog(s->vocab, new_token)) break;
        char piece_buf[256];
        int n = llama_token_to_piece(s->vocab, new_token, piece_buf, sizeof(piece_buf), 0, false);
        if (n >= 0) {
            s->stack->active_response.append(piece_buf, n);
            if (s->stack->on_token) s->stack->on_token(piece_buf, n);
        }
        s->batch.n_tokens = 0;
        langX_batch_add(s->batch, new_token, conv->n_past, {0}, true);
        if (llama_decode(model->ctx, s->batch) != 0) return false;
        conv->n_past++;
    }
    return true;
}

static std::string get_param(const langX::Layer* sl, const std::string& key, const std::string& def = "") {
    if (!sl) return def;
    auto it = sl->params.find(key);
    return (it != sl->params.end()) ? it->second : def;
}

bool layer_swap_model(langX::InferenceState* s, const langX::Layer* sl) {
    langX::switchModel(get_param(sl, "model_id").c_str(), s->stack);
    s->model = s->stack->model;
    return true;
}

bool layer_swap_convo(langX::InferenceState* s, const langX::Layer* sl) {
    langX::swapConversations(get_param(sl, "convo_id").c_str(), s->stack);
    s->conversation = s->stack->conversation;
    return true;
}

bool layer_set_system_prompt(langX::InferenceState* s, const langX::Layer* sl) {
    std::string val = get_param(sl, "value");
    if (s->conversation) s->conversation->system_prompt = val;
    s->injected_sys_prompt = val;
    return true;
}

bool layer_load_system_prompt(langX::InferenceState* s, const langX::Layer*) {
    std::string combined = s->model->system_prompt;
    if (s->conversation && !s->conversation->system_prompt.empty()) {
        if (!combined.empty()) combined += "\n";
        combined += s->conversation->system_prompt;
    }
    s->injected_sys_prompt = combined;
    return true;
}

static void push_tool_result(langX::Conversation* conv, bool native, const std::string& tool_name, const std::string& tool_call_id, const std::string& result_val) {
    if (native) {
        langX::ChatMessage m{ "tool", result_val };
        conv->messages.push_back(m);
        conv->active_messages.push_back(m);
    } else {
        langX::ChatMessage m{ "system", "<tool_result>{\"name\": \"" + tool_name + "\", \"result\": " + result_val + "}</tool_result>" };
        conv->messages.push_back(m);
        conv->active_messages.push_back(m);
    }
}

bool layer_use_tools(langX::InferenceState* s, const langX::Layer*) {
    auto* stack = s->stack;
    auto* conv  = s->conversation;
    bool native = s->settings && s->settings->use_native_tools;
    stack->has_tool_calls = false;

    auto calls = native
        ? parse_tool_calls_native(stack->active_response, s->native_chat_params)
        : parse_tool_calls(stack->active_response);
    if (calls.empty()) return true;

    auto push_both = [&](const std::string& role, const std::string& content) {
        langX::ChatMessage m{role, content};
        conv->messages.push_back(m);
        conv->active_messages.push_back(m);
    };
    push_both("assistant", stack->active_response);
    stack->active_response.clear();

    for (auto& call : calls) {
        langX::Tool* tool = nullptr;
        for (auto* t : stack->tools)
            if (t->name == call.name) { tool = t; break; }

        if (!tool) {
            std::cerr << "[USE_TOOLS] Unknown tool: " << call.name << "\n";
            push_tool_result(conv, native, call.name, call.id, "\"Error: tool not found\"");
            continue;
        }
        if (stack->verbose) std::cout << "[USE_TOOLS] " << (tool->is_async ? "Async" : "Sync") << ": " << call.name << "\n";
        stack->has_tool_calls = true;

        if (tool->is_async) {
            langX::InferenceState::PendingAsyncTool pending;
            pending.name = call.name;
            pending.future = std::async(std::launch::async,
                [tool, args = call.args]() -> std::string {
                    return tool->handler ? tool->handler(args) : std::string("");
                });
            s->pending_async_tools.push_back(std::move(pending));
        } else {
            std::string result = tool->handler ? tool->handler(call.args) : "";
            std::string result_val = tool->is_void ? "null" : ("\"" + json_escape(result) + "\"");
            push_tool_result(conv, native, call.name, call.id, result_val);
        }
    }
    return true;
}

bool layer_wait_tools(langX::InferenceState* s, const langX::Layer*) {
    auto* conv = s->conversation;
    bool native = s->settings && s->settings->use_native_tools;
    for (auto& pending : s->pending_async_tools) {
        std::string result = pending.future.get();
        if (s->stack->verbose) std::cout << "[WAIT_TOOLS] Async done: " << pending.name << "\n";
        bool is_void = false;
        for (auto* t : s->stack->tools)
            if (t->name == pending.name) { is_void = t->is_void; break; }
        std::string result_val = is_void ? "null" : ("\"" + json_escape(result) + "\"");
        push_tool_result(conv, native, pending.name, "", result_val);
    }
    s->pending_async_tools.clear();
    return true;
}

bool layer_llm_tool_gen(langX::InferenceState* s, const langX::Layer*) {
    static const int MAX_TOOL_ITERS = 10;
    auto* stack = s->stack;
    auto* conv = s->conversation;
    bool native = s->settings && s->settings->use_native_tools;
    bool ok = true;

    for (int iter = 0; iter < MAX_TOOL_ITERS && ok; iter++) {
        if (iter > 0 && stack->verbose)
            std::cout << "\n[LLM_TOOL_GEN] Pass " << (iter + 1) << " - feeding tool results...\n";

        ok = layer_llm_generation(s, nullptr);
        if (!ok) break;

        auto calls = native
            ? parse_tool_calls_native(stack->active_response, s->native_chat_params)
            : parse_tool_calls(stack->active_response);
        if (calls.empty()) break;
        if (stack->verbose && stack->on_token) std::cout << "\n";

        auto tg_push = [&](const std::string& role, const std::string& content) {
            langX::ChatMessage m{role, content};
            conv->messages.push_back(m);
            conv->active_messages.push_back(m);
        };
        tg_push("assistant", stack->active_response);
        stack->active_response.clear();

        std::vector<langX::InferenceState::PendingAsyncTool> async_pending;
        for (auto& call : calls) {
            langX::Tool* tool = nullptr;
            for (auto* t : stack->tools)
                if (t->name == call.name) { tool = t; break; }

            if (!tool) {
                std::cerr << "[LLM_TOOL_GEN] Unknown tool: " << call.name << "\n";
                push_tool_result(conv, native, call.name, call.id, "\"Error: tool not found\"");
                continue;
            }
            if (stack->verbose) std::cout << "[LLM_TOOL_GEN] " << (tool->is_async ? "Async" : "Sync") << ": " << call.name << "\n";

            if (tool->is_async) {
                langX::InferenceState::PendingAsyncTool pending;
                pending.name   = call.name;
                pending.future = std::async(std::launch::async,
                    [tool, args = call.args]() -> std::string {
                        return tool->handler ? tool->handler(args) : std::string("");
                    });
                async_pending.push_back(std::move(pending));
            } else {
                std::string result     = tool->handler ? tool->handler(call.args) : "";
                std::string result_val = tool->is_void ? "null" : ("\"" + json_escape(result) + "\"");
                push_tool_result(conv, native, call.name, call.id, result_val);
            }
        }

        for (auto& pending : async_pending) {
            std::string result = pending.future.get();
            if (stack->verbose) std::cout << "[LLM_TOOL_GEN] Async done: " << pending.name << "\n";
            bool is_void = false;
            for (auto* t : stack->tools)
                if (t->name == pending.name) { is_void = t->is_void; break; }
            std::string result_val = is_void ? "null" : ("\"" + json_escape(result) + "\"");
            push_tool_result(conv, native, pending.name, "", result_val);
        }

        apply_chat_template_with_memory(s);
        layer_clear_kv_cache(s, nullptr);
        layer_init_sampler(s, nullptr);
        ok = layer_feed_prompt(s, nullptr);
    }
    return ok;
}

bool layer_free_sampler(langX::InferenceState* s, const langX::Layer*) {
    if (s->sampler) { llama_sampler_free(s->sampler); s->sampler = nullptr; }
    return true;
}

bool layer_free_batch(langX::InferenceState* s, const langX::Layer*) {
    llama_batch_free(s->batch);
    s->batch = {};
    return true;
}

bool layer_save_to_history(langX::InferenceState* s, const langX::Layer*) {
    langX::ChatMessage msg{ "assistant", s->stack->active_response };
    s->conversation->messages.push_back(msg); 
    s->conversation->active_messages.push_back(msg);
    return true;
}

bool layer_filler_rand(langX::InferenceState* s, const langX::Layer* sl) {
    if (sl->param_list.empty()) return true;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, (int)sl->param_list.size() - 1);
    const std::string& text = sl->param_list[dist(gen)];
    s->stack->active_response += text;
    if (s->stack->on_token)
        s->stack->on_token(text.c_str(), (int)text.size());
    return true;
}

bool layer_filler_loop(langX::InferenceState* s, const langX::Layer* sl) {
    if (sl->param_list.empty()) return true;
    int idx = sl->_loop_index % (int)sl->param_list.size();
    sl->_loop_index++;
    const std::string& text = sl->param_list[idx];
    s->stack->active_response += text;
    if (s->stack->on_token)
        s->stack->on_token(text.c_str(), (int)text.size());
    return true;
}

// --- Sub-inference ---

static std::string resolve_image_target(const std::string& image_target, const std::vector<std::string>& active_files) {
    static const std::vector<std::string> IMG_EXTS = {
        ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"
    };
    auto is_image_path = [&](const std::string& p) {
        std::string ext = std::filesystem::path(p).extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        return std::find(IMG_EXTS.begin(), IMG_EXTS.end(), ext) != IMG_EXTS.end();
    };

    if (image_target.empty()) {
        for (const auto& f : active_files)
            if (is_image_path(f)) return f;
        return "";
    }

    if (!image_target.empty() &&
        std::all_of(image_target.begin(), image_target.end(), ::isdigit)) {
        int idx = std::stoi(image_target);
        if (idx >= 0 && idx < (int)active_files.size()) return active_files[idx];
        return "";
    }

    if (image_target.find('/') != std::string::npos ||
        image_target.find('\\') != std::string::npos) {
        return std::filesystem::exists(image_target) ? image_target : "";
    }

    if (image_target.find('.') != std::string::npos) {
        if (std::filesystem::exists(image_target)) return image_target;
    }

    for (const auto& f : active_files)
        if (std::filesystem::path(f).filename().string() == image_target) return f;

    return "";
}

static langX::Stack* make_sub_stack_vlm() {
    langX::Stack* s = new langX::Stack();
    s->unsafe = true;
    int id = 0;
    auto add = [&](langX::LayerType t) {
        auto* l = langX::makeLayer(t);
        l->id = id++;
        s->layers.push_back(l);
    };
    add(langX::INIT_INFERENCE);
    add(langX::USER_PUSH_IMAGES);
    add(langX::USER_PUSH_PROMT);
    add(langX::LOAD_CHAT_TEMPLATE);
    add(langX::CLEAR_KV_CACHE);
    add(langX::INIT_SAMPLER);
    add(langX::INIT_BATCH);
    add(langX::FEED_PROMPT);
    add(langX::LLM_GENERATION);
    add(langX::FREE_SAMPLER);
    add(langX::FREE_BATCH);
    return s;
}

static langX::Stack* make_sub_stack_text(bool save_to_history) {
    langX::Stack* s = new langX::Stack();
    s->unsafe = true;
    int id = 0;
    auto add = [&](langX::LayerType t) {
        auto* l = langX::makeLayer(t);
        l->id = id++;
        s->layers.push_back(l);
    };
    add(langX::INIT_INFERENCE);
    add(langX::USER_PUSH_PROMT);
    add(langX::LOAD_CHAT_TEMPLATE);
    add(langX::CLEAR_KV_CACHE);
    add(langX::INIT_SAMPLER);
    add(langX::INIT_BATCH);
    add(langX::FEED_PROMPT);
    add(langX::LLM_GENERATION);
    add(langX::FREE_SAMPLER);
    add(langX::FREE_BATCH);
    if (save_to_history) add(langX::SAVE_TO_HISTORY);
    return s;
}

struct SubInferenceGuard {
    langX::Stack* sub_stack;
    langX::Model* saved_model;
    std::string saved_model_id;
    bool switched_model;
    bool use_sleep;
    std::string sub_model_id;

    SubInferenceGuard(langX::Stack* sub, const std::string& model_id, bool use_sleep_): sub_stack(sub), saved_model(sub->model), switched_model(false), use_sleep(use_sleep_), sub_model_id(model_id) {
        for (auto& [k, v] : langX::global_LangX.loaded_models)
            if (v == saved_model) { saved_model_id = k; break; }

        if (!model_id.empty()) {
            if (use_sleep_ && !saved_model_id.empty())
                langX::sleepModel(saved_model_id.c_str());
            if (use_sleep_)
                langX::wakeupModel(model_id.c_str());
            langX::switchModel(model_id.c_str(), sub);
            switched_model = true;
        }
    }

    ~SubInferenceGuard() {
        if (!switched_model) return;
        if (use_sleep) {
            langX::sleepModel(sub_model_id.c_str());
            if (!saved_model_id.empty())
                langX::wakeupModel(saved_model_id.c_str());
        }
        if (!saved_model_id.empty())
            langX::switchModel(saved_model_id.c_str(), sub_stack);
        else
            sub_stack->model = saved_model;
    }
};

static std::string run_substack(
    langX::Stack* parent_stack,
    langX::Stack* sub_stack,
    const std::string& prompt,
    const std::vector<std::string>& files,
    const std::string& model_id,
    bool use_sleep,
    bool owned = true
) {
    langX::Model* orig_model = sub_stack->model;
    langX::Conversation* orig_conv = sub_stack->conversation;
    langX::InquerySettings* orig_settings = sub_stack->settings;

    if (!sub_stack->model)    sub_stack->model    = parent_stack->model;
    if (!sub_stack->settings) sub_stack->settings = parent_stack->settings;

    auto* sub_conv = new langX::Conversation();
    sub_stack->conversation = sub_conv;

    std::string result;
    {
        SubInferenceGuard guard(sub_stack, model_id, use_sleep);
        result = langX::inference(sub_stack, prompt.c_str(), files);
    }

    delete sub_conv;
    sub_stack->model = orig_model;
    sub_stack->conversation = orig_conv;
    sub_stack->settings = orig_settings;

    if (owned) {
        for (auto* l : sub_stack->layers) delete l;
        delete sub_stack;
    }
    return result;
}

// --- QUICK_LOOK layer ---

bool layer_quick_look(langX::InferenceState* s, const langX::Layer* sl) {
    if (!s->model) { std::cerr << "[QUICK_LOOK] No model loaded.\n"; return false; }

    std::string image_path = resolve_image_target(get_param(sl, "image_target"), s->stack->active_files);
    if (image_path.empty()) {
        std::cerr << "[QUICK_LOOK] No image found. Set image_target or add images to active_files.\n";
        return false;
    }
    std::string instructions = get_param(sl, "instructions");
    if (instructions.empty())
        instructions = "Describe this image in detail for a document search index.";

    if (s->stack->verbose) std::cout << "[QUICK_LOOK] Describing: " << image_path << "\n";

    std::string desc = run_substack(s->stack, make_sub_stack_vlm(), instructions, { image_path },  get_param(sl, "model_id"), get_param(sl, "use_sleep") == "true");
    langX::feedContext(desc.c_str(), "Image Description", s->stack);
    return true;
}

// --- PROMPT_CHECK layer ---

static const std::map<std::string, std::string> PROMPT_CHECK_MSGS = {
    { "EMPTY", "Your message is empty. Please enter something." },
    { "TOO_SHORT", "Message too short. Please provide more detail." },
    { "TOO_LONG", "Message too long. Please shorten your input." },
    { "TOO_MANY_FILES", "Too many files attached. Please reduce the number of files." },
    { "FILE_TOO_LARGE", "Attached files exceed the size limit. Please reduce the total size." },
    { "INVALID_FILE_TYPE", "File type not allowed. Please use only supported file types." },
    { "DUPLICATE_PROMPT", "This message was already sent. Please try a different question." },
    { "INJECTION_DETECTED", "Your message contains patterns that cannot be processed." },
    { "PII_DETECTED", "Your message appears to contain personal information. Please remove it." },
    { "BLOCKED", "Your message was blocked by a content filter." },
};

// Built-in prompt injection patterns (case-insensitive regex).
static const std::vector<std::string> BUILTIN_INJECTION_PATTERNS = {
    "(?i)ignore (all |your )?(previous|prior|above) (instructions?|prompt|rules?)",
    "(?i)forget (your |all )?(instructions?|prompt|rules?)",
    "(?i)(disregard|override) (all |your )?(previous |prior )?(instructions?|rules?|prompt)",
    "(?i)your new instructions? (are|is) ",
    "(?i)pretend (you are|to be) (now )?(a |an )?(?!helpful|concise|brief)",
    "(?i)you are now (a |an )?(?!helpful|concise|brief)",
    "(?i)(new|updated) (system |)prompt[: ]",
    "(?i)<\\|?(system|user|assistant)\\|?>",
    "(?i)\\[INST\\]|<<SYS>>|\\[/INST\\]",
};

// Built-in PII patterns.
static const std::vector<std::string> BUILTIN_PII_PATTERNS = {
    "[a-zA-Z0-9._%+\\-]+@[a-zA-Z0-9.\\-]+\\.[a-zA-Z]{2,}",
    "\\b(\\+?1[\\s.\\-]?)?(\\(?\\d{3}\\)?[\\s.\\-]?)?\\d{3}[\\s.\\-]?\\d{4}\\b",
    "\\b(?:\\d{4}[\\s\\-]?){3}\\d{4}\\b",
    "\\b\\d{3}[\\-]\\d{2}[\\-]\\d{4}\\b",
};

static bool run_regex_checks(const std::string& text, const std::vector<std::string>& patterns) {
    for (const auto& pat : patterns) {
        try {
            if (std::regex_search(text, std::regex(pat))) return true;
        } catch (const std::regex_error&) {
            std::cerr << "[PROMPT_CHECK] Invalid regex pattern (skipped): " << pat << "\n";
        }
    }
    return false;
}

bool layer_prompt_check(langX::InferenceState* s, const langX::Layer* sl) {
    const std::string& prompt = s->stack->active_prompt;
    const auto& files = s->stack->active_files;

    int min_len = std::stoi(get_param(sl, "min_length", "1"));
    int max_len = std::stoi(get_param(sl, "max_length", "0"));
    int max_files = std::stoi(get_param(sl, "max_files", "0"));
    long long max_bytes = std::stoll(get_param(sl, "max_file_bytes", "0"));
    bool check_dup = (get_param(sl, "check_duplicate") == "true");
    bool check_inj = (get_param(sl, "check_injection") == "true");
    bool check_pii = (get_param(sl, "check_pii") == "true");
    std::string allowed_ext_str = get_param(sl, "allowed_extensions");

    std::string code;

    // Text length
    if (prompt.empty())
        code = "EMPTY";
    else if (min_len > 0 && (int)prompt.size() < min_len)
        code = "TOO_SHORT";
    else if (max_len > 0 && (int)prompt.size() > max_len)
        code = "TOO_LONG";

    // File count
    if (code.empty() && max_files > 0 && (int)files.size() > max_files)
        code = "TOO_MANY_FILES";

    // Total file size
    if (code.empty() && max_bytes > 0) {
        long long total = 0;
        for (const auto& f : files) {
            std::error_code ec;
            auto sz = std::filesystem::file_size(f, ec);
            if (!ec) total += (long long)sz;
        }
        if (total > max_bytes) code = "FILE_TOO_LARGE";
    }

    // File type allowlist
    if (code.empty() && !allowed_ext_str.empty() && !files.empty()) {
        std::vector<std::string> allowed;
        std::string tok;
        for (char c : allowed_ext_str + ",") {
            if (c == ',') {
                while (!tok.empty() && tok.front() == ' ') tok.erase(tok.begin());
                while (!tok.empty() && tok.back()  == ' ') tok.pop_back();
                if (!tok.empty()) {
                    if (tok[0] != '.') tok = "." + tok;
                    std::transform(tok.begin(), tok.end(), tok.begin(), ::tolower);
                    allowed.push_back(tok);
                }
                tok.clear();
            } else {
                tok += c;
            }
        }
        for (const auto& f : files) {
            std::string ext = std::filesystem::path(f).extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (std::find(allowed.begin(), allowed.end(), ext) == allowed.end()) {
                code = "INVALID_FILE_TYPE";
                break;
            }
        }
    }

    // Duplicate detection
    if (code.empty() && check_dup && s->conversation) {
        for (const auto& msg : s->conversation->messages) {
            if (msg.role == "user" && msg.content == prompt) {
                code = "DUPLICATE_PROMPT";
                break;
            }
        }
    }

    // Prompt injection patterns
    if (code.empty() && check_inj) {
        if (run_regex_checks(prompt, BUILTIN_INJECTION_PATTERNS))
            code = "INJECTION_DETECTED";
    }

    // PII patterns
    if (code.empty() && check_pii) {
        if (run_regex_checks(prompt, BUILTIN_PII_PATTERNS))
            code = "PII_DETECTED";
    }

    // Custom / extra patterns (always active, trigger "BLOCKED")
    if (code.empty() && !sl->param_list.empty()) {
        if (run_regex_checks(prompt, sl->param_list))
            code = "BLOCKED";
    }

    if (code.empty()) return true;

    auto it = PROMPT_CHECK_MSGS.find(code);
    std::string msg = (it != PROMPT_CHECK_MSGS.end())
        ? it->second
        : ("Blocked (" + code + ").");
    s->stack->active_response = msg;
    if (s->stack->verbose)
        std::cout << "[PROMPT_CHECK] " << code << ": " << msg << "\n";
    return false;
}

// --- QUICK_ASK layer ---

bool layer_quick_ask(langX::InferenceState* s, const langX::Layer* sl) {
    if (!s->model) { std::cerr << "[QUICK_ASK] No model loaded.\n"; return false; }

    std::string prompt = get_param(sl, "prompt");
    if (prompt.empty()) {
        std::cerr << "[QUICK_ASK] No prompt set. Use makeQuickAskLayer(\"your prompt\").\n";
        return false;
    }
    bool save_to_history = (get_param(sl, "save_to_history") == "true");
    if (s->stack->verbose) std::cout << "[QUICK_ASK] Sub-inference: " << prompt << "\n";

    std::string result = run_substack(s->stack, make_sub_stack_text(false), prompt, {}, get_param(sl, "model_id"), get_param(sl, "use_sleep") == "true");
    if (!result.empty()) {
        langX::feedContext(result.c_str(), "Context", s->stack);
        if (save_to_history) {
            s->conversation->messages.push_back({ "user", prompt });
            s->conversation->messages.push_back({ "assistant", result });
            s->conversation->active_messages.push_back({ "user", prompt });
            s->conversation->active_messages.push_back({ "assistant", result });
        }
    }
    return true;
}

// --- SUBSTACK layer ---

bool layer_substack(langX::InferenceState* s, const langX::Layer* sl) {
    std::string stack_id = get_param(sl, "stack_id");
    if (stack_id.empty()) {
        std::cerr << "[SUBSTACK] No stack_id specified.\n";
        return false;
    }
    auto it = langX::global_LangX.hotswap_stacks.find(stack_id);
    if (it == langX::global_LangX.hotswap_stacks.end()) {
        std::cerr << "[SUBSTACK] Stack not found: " << stack_id << "\n";
        return false;
    }

    std::string prompt = get_param(sl, "prompt");
    if (prompt.empty()) prompt = s->stack->active_prompt;

    const std::vector<std::string>& files = sl->param_list.empty() ? s->stack->active_files : sl->param_list;

    std::string label = get_param(sl, "context_label", "Substack Result");
    bool save_to_history = (get_param(sl, "save_to_history") == "true");
    bool use_sleep = (get_param(sl, "use_sleep") == "true");
    std::string model_id = get_param(sl, "model_id");

    if (s->stack->verbose)
        std::cout << "[SUBSTACK] Running stack '" << stack_id << "'\n";

    std::string result = run_substack(s->stack, it->second, prompt, files, model_id, use_sleep, /*owned=*/false);
    if (!result.empty()) {
        langX::feedContext(result.c_str(), label.c_str(), s->stack);
        if (save_to_history) {
            s->conversation->messages.push_back({ "user", prompt });
            s->conversation->messages.push_back({ "assistant", result });
            s->conversation->active_messages.push_back({ "user", prompt });
            s->conversation->active_messages.push_back({ "assistant", result });
        }
    }
    return true;
}

// --- WAIT layers ---

bool layer_wait_time(langX::InferenceState* s, const langX::Layer* sl) {
    int ms = std::stoi(get_param(sl, "duration_ms", "0"));
    if (ms > 0) {
        if (s->stack->verbose)
            std::cout << "[WAIT_TIME] Sleeping " << ms << "ms\n";
        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    }
    return true;
}

bool layer_wait_lambda(langX::InferenceState* s, const langX::Layer* sl) {
    if (!sl->branch_condition) return true;

    int timeout_ms = std::stoi(get_param(sl, "timeout_ms", "0"));
    int poll_ms = std::stoi(get_param(sl, "poll_ms", "50"));
    bool verbose = s->stack->verbose;

    if (verbose)
        std::cout << "[WAIT_LAMBDA] Polling condition" << (timeout_ms > 0 ? " (timeout " + std::to_string(timeout_ms) + "ms)" : "") << "...\n";

    auto start = std::chrono::steady_clock::now();
    while (!sl->branch_condition(s->stack)) {
        if (timeout_ms > 0) {
            auto ms_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start).count();
            if (ms_elapsed >= timeout_ms) {
                if (verbose) std::cout << "[WAIT_LAMBDA] Timed out.\n";
                return false;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(poll_ms));
    }
    if (verbose) std::cout << "[WAIT_LAMBDA] Condition met.\n";
    return true;
}

bool layer_wait_promise(langX::InferenceState* s, const langX::Layer* sl) {
    auto* stack = s->stack;
    int timeout_ms = std::stoi(get_param(sl, "timeout_ms", "0"));
    bool verbose = stack->verbose;

    auto promise = std::make_shared<std::promise<std::string>>();
    auto future = promise->get_future();
    stack->wait_promise = promise;

    if (verbose)
        std::cout << "[WAIT_PROMISE] Waiting for signal" << (timeout_ms > 0 ? " (timeout " + std::to_string(timeout_ms) + "ms)" : "") << "...\n";

    bool signaled;
    if (timeout_ms > 0) {
        signaled = (future.wait_for(std::chrono::milliseconds(timeout_ms)) == std::future_status::ready);
    } else {
        future.wait();
        signaled = true;
    }
    stack->wait_promise.reset();

    if (verbose) std::cout << "[WAIT_PROMISE] " << (signaled ? "Signaled.\n" : "Timed out.\n");
    return signaled;
}

// --- LLM filter layers ---

static bool parse_filter_verdict(const std::string& response) {
    size_t start = response.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return false;
    char c0 = (char)::toupper((unsigned char)response[start]);
    return c0 == 'Y';
}

static std::string extract_filter_notes(const std::string& response) {
    size_t start = response.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t skip = (start + 3 <= response.size()) ? 3 : 2;
    size_t notes_start = response.find_first_not_of(" :\t\n\r-", start + skip);
    return (notes_start != std::string::npos) ? response.substr(notes_start) : "";
}

static langX::Stack* make_filter_sub_stack(const std::string& filter_prompt, bool with_rag = false) {
    langX::Stack* s = new langX::Stack();
    s->unsafe = true;
    langX::appendLayer(s, langX::makeLayer(langX::INIT_INFERENCE));
    if (with_rag) langX::appendLayer(s, langX::makeLayer(langX::RAG_RETRIEVAL));
    langX::appendLayer(s, langX::makeSetSystemPromptLayer(filter_prompt.c_str()));
    langX::appendLayer(s, langX::makeLayer(langX::USER_PUSH_PROMT));
    langX::appendLayer(s, langX::makeLayer(langX::LOAD_CHAT_TEMPLATE));
    langX::appendLayer(s, langX::makeLayer(langX::CLEAR_KV_CACHE));
    langX::appendLayer(s, langX::makeLayer(langX::INIT_SAMPLER));
    langX::appendLayer(s, langX::makeLayer(langX::INIT_BATCH));
    langX::appendLayer(s, langX::makeLayer(langX::FEED_PROMPT));
    langX::appendLayer(s, langX::makeLayer(langX::LLM_GENERATION));
    langX::appendLayer(s, langX::makeLayer(langX::FREE_SAMPLER));
    langX::appendLayer(s, langX::makeLayer(langX::FREE_BATCH));
    return s;
}

static bool run_llm_filter(langX::InferenceState* s, const langX::Layer* sl, const std::string& text_to_check, bool with_rag = false) {
    std::string filter_prompt = get_param(sl, "filter_prompt");
    bool inject_notes = (get_param(sl, "inject_notes") == "true");
    std::string notes_label = get_param(sl, "notes_label", "Filter Notes");

    langX::Stack* sub_stack = make_filter_sub_stack(filter_prompt, with_rag);
    if (with_rag) sub_stack->rag_db = s->stack->rag_db;

    int saved_n_predict = s->settings->n_tokens_to_predict;
    std::string saved_grammar = s->settings->grammar;
    if (!inject_notes) {
        s->settings->n_tokens_to_predict = 10;
        s->settings->grammar = "root ::= (\"YES\" | \"NO\") [a-zA-Z ]*";
    }

    std::string verdict_response = run_substack(s->stack, sub_stack, text_to_check, {}, get_param(sl, "model_id"), get_param(sl, "use_sleep") == "true");

    s->settings->n_tokens_to_predict = saved_n_predict;
    s->settings->grammar = saved_grammar;

    bool passed = parse_filter_verdict(verdict_response);

    if (s->stack->verbose)
        std::cout << "[LLM_FILTER] " << (passed ? "PASS" : "FAIL") << ": " << verdict_response.substr(0, 80) << (verdict_response.size() > 80 ? "..." : "") << "\n";

    if (!passed && inject_notes) {
        std::string notes = extract_filter_notes(verdict_response);
        if (!notes.empty())
            langX::feedContext(notes.c_str(), notes_label.c_str(), s->stack);
    }
    return passed;
}

bool layer_llm_prompt_filter(langX::InferenceState* s, const langX::Layer* sl) {
    return run_llm_filter(s, sl, s->stack->active_prompt);
}

bool layer_llm_result_filter(langX::InferenceState* s, const langX::Layer* sl) {
    return run_llm_filter(s, sl, s->stack->active_response);
}

bool layer_fact_check(langX::InferenceState* s, const langX::Layer* sl) {
    if (!s->stack->rag_db) {
        std::cerr << "[FACT_CHECK] No RAG database attached. Use attachRagDb() on the stack.\n";
        return true;
    }
    return run_llm_filter(s, sl, s->stack->active_response, /*with_rag=*/true);
}

// ---- MEMORY Layers ----

static const char* DEFAULT_TIER2_COMPRESS_PROMPT =
    "Compress this conversation turn into one sentence (max 30 words).\n"
    "Preserve names, decisions, and key facts. Output only the sentence, nothing else.";

static int count_tokens(const llama_vocab* vocab, const std::string& text) {
    if (text.empty()) return 0;
    return (int)tokenize(vocab, text).size();
}

static std::string format_tier2_block(const std::vector<std::string>& memories) {
    if (memories.empty()) return "";
    std::string block = "[Conversation highlights]\n";
    for (const auto& m : memories)
        block += "- " + m + "\n";
    return block;
}

bool layer_semantic_memory(langX::InferenceState* s, const langX::Layer* sl) {
    return layer_semantic_mem_retrieval(s, sl);
}

static const int MAX_COMPRESS_ITERS = 5;

static const char* DEFAULT_EPISODIC_COMPRESS_PROMPT =
    "You are a memory compression system. Your input may contain a [Previous summary] and a [New turn].\n"
    "Merge ALL facts from the previous summary with any new facts from the new turn into a single list.\n"
    "Never drop facts from the previous summary — they are already compressed and essential.\n"
    "Preserve names, numbers, decisions, key information, and important context.\n"
    "Write as a concise fact list in third-person past tense. Omit small talk and filler.\n"
    "Output only the merged summary - no preamble, no explanation.";

bool layer_episodic_memory(langX::InferenceState* s, const langX::Layer* sl) {
    auto* model = s->model;
    auto* settings = s->settings;
    auto* conv = s->conversation;

    const int total_budget = model->ctx_params->n_ctx - settings->n_tokens_to_predict;
    float context_ratio = settings->episodic_context_ratio;
    const int context_budget = (int)(total_budget * context_ratio);

    if (estimate_context_tokens(s->vocab, s->injected_sys_prompt, conv) <= context_budget) return true;

    int evicted = 0;
    while (estimate_context_tokens(s->vocab, s->injected_sys_prompt, conv) > context_budget && conv->active_messages.size() > 2) {
        conv->active_messages.erase(conv->active_messages.begin(), conv->active_messages.begin() + std::min((size_t)2, conv->active_messages.size()));
        evicted++;
    }

    if (s->stack->verbose && evicted > 0)
        std::cout << "[EPISODIC_MEMORY] Evicted " << evicted << " turn(s). " << "T2=" << conv->episodic_tier2_memories.size() << " entries, " << "T1=" << conv->episodic_tier1_summary.size() << " chars, " << "Est.total=" << estimate_context_tokens(s->vocab, s->injected_sys_prompt, conv) << "/" << total_budget << " tok.\n";

    return true;
}

bool layer_episodic_tiered_memory(langX::InferenceState* s, const langX::Layer* sl) {
    if (s->stack && s->stack->verbose)
        std::cerr << "[EPISODIC_TIERED_MEMORY] Deprecated — use EPISODIC_MEMORY with tier2_ratio/tier1_ratio.\n";
    return layer_episodic_memory(s, sl);
}

bool layer_build_episodic_mem(langX::InferenceState* s, const langX::Layer* sl) {
    auto* conv = s->conversation;
    int total = (int)conv->messages.size();
    int active = (int)conv->active_messages.size();
    int safe_upper = total - active;

    if (safe_upper <= conv->episodic_compress_idx) return true;

    auto* model = s->model;
    auto* settings = s->settings;
    const int total_budget = model->ctx_params->n_ctx - settings->n_tokens_to_predict;

    float tier2_ratio = settings->episodic_tier2_ratio;
    float tier1_ratio = settings->episodic_tier1_ratio;
    const bool use_tier2  = (tier2_ratio > 0.0f);
    const bool use_tier1  = (tier1_ratio > 0.0f);
    const int  tier2_budget = (int)(total_budget * tier2_ratio);
    const int  tier1_budget = (int)(total_budget * tier1_ratio);

    std::string tier2_prompt = get_param(sl, "tier2_compress_prompt");
    if (tier2_prompt.empty()) tier2_prompt = DEFAULT_TIER2_COMPRESS_PROMPT;
    std::string tier1_prompt = get_param(sl, "tier1_compress_prompt");
    if (tier1_prompt.empty()) tier1_prompt = DEFAULT_EPISODIC_COMPRESS_PROMPT;
    std::string model_id = get_param(sl, "model_id");
    bool use_sleep = (get_param(sl, "use_sleep") == "true");

    int new_entries = 0;
    for (int i = conv->episodic_compress_idx; i < safe_upper; ) {
        if (conv->messages[i].role != "user" || i + 1 >= safe_upper || conv->messages[i + 1].role != "assistant") {
            conv->episodic_compress_idx = ++i;
            continue;
        }
        std::string turn_text = "User: " + conv->messages[i].content + "\nAssistant: " + conv->messages[i + 1].content;

        if (use_tier2) {
            langX::Stack* t2_stack = make_filter_sub_stack(tier2_prompt);
            std::string key_point  = run_substack(s->stack, t2_stack, turn_text, {}, model_id, use_sleep);
            conv->episodic_tier2_memories.push_back(key_point);
        } else if (use_tier1) {
            std::string blob_input;
            if (!conv->episodic_tier1_summary.empty())
                blob_input += "[Previous summary]\n" + conv->episodic_tier1_summary + "\n\n";
            blob_input += "[New turn]\n" + turn_text;
            langX::Stack* t1_stack = make_filter_sub_stack(tier1_prompt);
            conv->episodic_tier1_summary = run_substack(s->stack, t1_stack, blob_input, {}, model_id, use_sleep);
        }

        conv->episodic_compress_idx = i + 2;
        i += 2;
        new_entries++;
    }

    if (use_tier2 && use_tier1 && new_entries > 0) {
        while (count_tokens(s->vocab, format_tier2_block(conv->episodic_tier2_memories)) > tier2_budget && !conv->episodic_tier2_memories.empty()) {
            std::vector<std::string> to_merge;
            while (count_tokens(s->vocab, format_tier2_block(conv->episodic_tier2_memories)) > tier2_budget && !conv->episodic_tier2_memories.empty()) {
                to_merge.push_back(conv->episodic_tier2_memories.front());
                conv->episodic_tier2_memories.erase(conv->episodic_tier2_memories.begin());
            }
            std::string blob_input;
            if (!conv->episodic_tier1_summary.empty())
                blob_input += "[Previous summary]\n" + conv->episodic_tier1_summary + "\n\n";
            blob_input += "[Key points to absorb]\n";
            for (const auto& m : to_merge) blob_input += "- " + m + "\n";

            std::string new_t1 = conv->episodic_tier1_summary;
            std::string current = blob_input;
            for (int iter = 0; iter < MAX_COMPRESS_ITERS; iter++) {
                langX::Stack* t1_stack = make_filter_sub_stack(tier1_prompt);
                new_t1 = run_substack(s->stack, t1_stack, current, {}, model_id, use_sleep);
                if (count_tokens(s->vocab, new_t1) <= tier1_budget) break;
                current = "[Too long — condense further]\n\n" + new_t1;
            }
            conv->episodic_tier1_summary = new_t1;
        }
    }

    if (s->stack->verbose && new_entries > 0)
        std::cout << "[BUILD_EPISODIC_MEM] Compressed " << new_entries << " new turn(s). " << "T2=" << conv->episodic_tier2_memories.size() << " entries, " << "T1=" << count_tokens(s->vocab, conv->episodic_tier1_summary) << " tok.\n";

    return true;
}

bool layer_build_semantic_mem(langX::InferenceState* s, const langX::Layer* sl) {
    auto* conv = s->conversation;
    auto* stack = s->stack;

    if (!stack || !stack->rag_db) {
        std::cerr << "[BUILD_SEMANTIC_MEM] No RAG database attached. Call attachRagDb first.\n";
        return false;
    }

    int total = (int)conv->messages.size();
    int active = (int)conv->active_messages.size();
    int safe_upper = total - active;

    if (safe_upper <= conv->semantic_compress_idx) return true;

    std::string compress_prompt = get_param(sl, "compress_prompt");
    if (compress_prompt.empty()) compress_prompt = DEFAULT_TIER2_COMPRESS_PROMPT;
    std::string model_id = get_param(sl, "model_id");
    bool use_sleep = (get_param(sl, "use_sleep") == "true");

    int new_count = 0;
    for (int i = conv->semantic_compress_idx; i < safe_upper; ) {
        if (conv->messages[i].role != "user" || i + 1 >= safe_upper || conv->messages[i + 1].role != "assistant") {
            conv->semantic_compress_idx = ++i;
            continue;
        }

        std::string turn_text = "User: " + conv->messages[i].content + "\nAssistant: " + conv->messages[i + 1].content;
        langX::Stack* label_stack = make_filter_sub_stack(compress_prompt);
        std::string key_point = run_substack(s->stack, label_stack, turn_text, {}, model_id, use_sleep);
        langX::ragAddMemory(stack->rag_db, turn_text.c_str(), key_point.c_str());
        conv->semantic_compress_idx = i + 2;
        i += 2;
        new_count++;
    }

    if (s->stack->verbose && new_count > 0)
        std::cout << "[BUILD_SEMANTIC_MEM] Saved " << new_count << " new turn(s) to semantic RAG.\n";

    return true;
}

bool layer_load_procedural_mem(langX::InferenceState* s, const langX::Layer* sl) {
    std::string file_path = get_param(sl, "file_path");
    if (file_path.empty()) {
        std::cerr << "[LOAD_PROCEDURAL_MEM] No file_path param set.\n";
        return false;
    }
    auto* conv = s->conversation;
    if (std::filesystem::exists(file_path)) {
        std::ifstream in(file_path);
        std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        conv->procedural_memory = content;
        if (s->stack->verbose)
            std::cout << "[LOAD_PROCEDURAL_MEM] Loaded " << content.size() << " chars from " << file_path << "\n";
    } else {
        conv->procedural_memory.clear();
        std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());
        std::ofstream touch(file_path);
        touch.close();
        if (s->stack->verbose)
            std::cout << "[LOAD_PROCEDURAL_MEM] Created empty file: " << file_path << "\n";
    }
    return true;
}

static const char* DEFAULT_PROCEDURAL_BUILD_PROMPT =
    "List every factual detail from this exchange: names, locations, numbers, relationships, "
    "preferences, goals, events, and any other concrete information. One fact per line. "
    "Be thorough — even small details matter. "
    "Only output NONE if the exchange contains absolutely no factual content.";

bool layer_build_procedural_mem(langX::InferenceState* s, const langX::Layer* sl) {
    std::string file_path = get_param(sl, "file_path");
    if (file_path.empty()) {
        std::cerr << "[BUILD_PROCEDURAL_MEM] No file_path param set.\n";
        return false;
    }

    auto* conv = s->conversation;
    auto* stack = s->stack;

    std::string exchange;
    int n = (int)conv->active_messages.size();
    int start = std::max(0, n - 2);
    for (int i = start; i < n; i++)
        exchange += conv->active_messages[i].role + ": " + conv->active_messages[i].content + "\n";

    if (exchange.empty()) return true;

    std::string build_prompt = get_param(sl, "build_prompt");
    if (build_prompt.empty()) build_prompt = DEFAULT_PROCEDURAL_BUILD_PROMPT;

    std::string sys_prompt = build_prompt;
    if (!conv->procedural_memory.empty())
        sys_prompt += "\n\n[Already known facts]\n" + conv->procedural_memory;

    std::string model_id = get_param(sl, "model_id");
    bool use_sleep = (get_param(sl, "use_sleep") == "true");

    langX::Stack* sub = make_filter_sub_stack(sys_prompt);
    std::string result = run_substack(stack, sub, exchange, {}, model_id, use_sleep);

    while (!result.empty() && (result.back() == '\n' || result.back() == '\r' || result.back() == ' '))
        result.pop_back();
    while (!result.empty() && (result.front() == '\n' || result.front() == '\r' || result.front() == ' '))
        result.erase(result.begin());

    {
        std::string lower = result;
        for (auto& c : lower) c = (char)std::tolower((unsigned char)c);
        while (!lower.empty() && (lower.back() == '.' || lower.back() == '!' || lower.back() == ','))
            lower.pop_back();
        if (lower.empty() || lower == "none" || lower == "no new facts" || lower == "no new facts to remember" || lower == "no new facts worth remembering") {
            if (stack->verbose)
                std::cout << "[BUILD_PROCEDURAL_MEM] No new facts extracted.\n";
            return true;
        }
    }

    if (!conv->procedural_memory.empty())
        conv->procedural_memory += "\n";
    conv->procedural_memory += result;

    std::filesystem::create_directories(std::filesystem::path(file_path).parent_path());
    std::ofstream out(file_path);
    out << conv->procedural_memory;
    out.close();

    if (stack->verbose)
        std::cout << "[BUILD_PROCEDURAL_MEM] Updated " << file_path << " (" << conv->procedural_memory.size() << " chars total)\n";

    return true;
}

bool layer_bert_generate(langX::InferenceState* s, const langX::Layer* sl) {
    std::string model_id = get_param(sl, "model_id");
    auto* stack = s->stack;

    langX::Model* bert = s->model;
    if (!model_id.empty()) {
        auto it = langX::global_LangX.loaded_models.find(model_id);
        if (it == langX::global_LangX.loaded_models.end() || !it->second) {
            std::cerr << "[BERT_GENERATE] Model '" << model_id << "' not found.\n";
            return false;
        }
        bert = it->second;
    }

    if (!llama_model_has_encoder(bert->model)) {
        std::cerr << "[BERT_GENERATE] Warning: model is not encoder-only, running encode anyway.\n";
    }

    if (llama_encode(bert->ctx, s->batch) != 0) {
        std::cerr << "[BERT_GENERATE] llama_encode failed.\n";
        return false;
    }

    int n_vocab = llama_vocab_n_tokens(s->vocab);
    int last_tok = s->batch.n_tokens - 1;
    const float* logits = llama_get_logits_ith(bert->ctx, last_tok);
    if (!logits) {
        std::cerr << "[BERT_GENERATE] Could not get logits.\n";
        return false;
    }

    int best_id = 0;
    float best_score = logits[0];
    for (int i = 1; i < n_vocab; i++) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_id = i;
        }
    }

    char piece_buf[256];
    int n = llama_token_to_piece(s->vocab, best_id, piece_buf, sizeof(piece_buf), 0, false);
    stack->active_response = (n > 0) ? std::string(piece_buf, n) : std::to_string(best_id);

    if (stack->verbose)
        std::cout << "[BERT_GENERATE] Classification: " << stack->active_response << " (logit=" << best_score << ")\n";
    return true;
}

bool layer_nli_generate(langX::InferenceState* s, const langX::Layer* sl) {
    std::string model_id = get_param(sl, "model_id");
    auto* stack = s->stack;

    langX::Model* nli = s->model;
    if (!model_id.empty()) {
        auto it = langX::global_LangX.loaded_models.find(model_id);
        if (it == langX::global_LangX.loaded_models.end() || !it->second) {
            std::cerr << "[NLI_GENERATE] Model '" << model_id << "' not found.\n";
            return false;
        }
        nli = it->second;
    }

    if (llama_encode(nli->ctx, s->batch) != 0) {
        std::cerr << "[NLI_GENERATE] llama_encode failed.\n";
        return false;
    }

    int last_tok = s->batch.n_tokens - 1;
    const float* logits = llama_get_logits_ith(nli->ctx, last_tok);
    if (!logits) {
        std::cerr << "[NLI_GENERATE] Could not get logits.\n";
        return false;
    }

    static const char* NLI_LABELS[] = { "entailment", "neutral", "contradiction" };
    int best = 0;
    for (int i = 1; i < 3; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    stack->active_response = NLI_LABELS[best];

    if (stack->verbose)
        std::cout << "[NLI_GENERATE] Result: " << stack->active_response << " (scores: ent=" << logits[0] << " neu=" << logits[1] << " con=" << logits[2] << ")\n";
    return true;
}

static const char* DEFAULT_COT_PROMPT = "Think through the following question step by step. Show your reasoning process clearly.";

bool layer_cot_generate(langX::InferenceState* s, const langX::Layer* sl) {
    std::string cot_prompt = get_param(sl, "cot_prompt");
    if (cot_prompt.empty()) cot_prompt = DEFAULT_COT_PROMPT;

    std::string model_id = get_param(sl, "model_id");
    bool use_sleep = (get_param(sl, "use_sleep") == "true");
    std::string prompt = s->stack->active_prompt;

    langX::Stack* sub = make_filter_sub_stack(cot_prompt);
    std::string reasoning = run_substack(s->stack, sub, prompt, {}, model_id, use_sleep);

    if (!reasoning.empty())
        langX::feedContext(reasoning.c_str(), "Chain of Thought", s->stack);

    if (s->stack->verbose)
        std::cout << "[COT_GENERATE] Reasoning (" << reasoning.size() << " chars) added to context.\n";
    return true;
}

bool layer_self_consistent_gen(langX::InferenceState* s, const langX::Layer* sl) {
    std::string stack_id = get_param(sl, "stack_id");
    if (stack_id.empty()) {
        std::cerr << "[RUN_N_TIMES] No stack_id specified.\n";
        return false;
    }
    auto it = langX::global_LangX.hotswap_stacks.find(stack_id);
    if (it == langX::global_LangX.hotswap_stacks.end()) {
        std::cerr << "[RUN_N_TIMES] Stack not found: " << stack_id << "\n";
        return false;
    }

    int n = std::max(1, std::stoi(get_param(sl, "n", "3")));
    std::string model_id = get_param(sl, "model_id");
    bool use_sleep = (get_param(sl, "use_sleep") == "true");
    std::string prompt = s->stack->active_prompt;

    std::vector<std::string> responses;
    auto* target_stack = it->second;
    auto* orig_settings = target_stack->settings ? target_stack->settings : s->stack->settings;

    for (int i = 0; i < n; i++) {
        langX::InquerySettings varied = *orig_settings;
        varied.seed = (orig_settings->seed == LLAMA_DEFAULT_SEED)
            ? (uint32_t)(i + 1)
            : orig_settings->seed + i;

        langX::InquerySettings* prev = target_stack->settings;
        target_stack->settings = &varied;

        std::string result = run_substack(s->stack, target_stack, prompt, {}, model_id, use_sleep, /*owned=*/false);
        target_stack->settings = prev;
        responses.push_back(result);

        if (s->stack->verbose)
            std::cout << "[RUN_N_TIMES] Run " << (i+1) << "/" << n << ": " << result.substr(0, 80) << (result.size() > 80 ? "..." : "") << "\n";
    }

    std::string scoring_prompt = get_param(sl, "scoring_prompt");
    if (!scoring_prompt.empty()) {
        std::string combined = scoring_prompt + "\n\n";
        for (int i = 0; i < (int)responses.size(); i++)
            combined += "--- Answer " + std::to_string(i+1) + " ---\n" + responses[i] + "\n\n";
        combined += "Reply with ONLY the best answer, exactly as written.";
        langX::Stack* judge = make_filter_sub_stack("You are a judge. Pick the best answer from the candidates.");
        s->stack->active_response = run_substack(s->stack, judge, combined, {}, model_id, use_sleep);
    } else {
        std::map<std::string, int> counts;
        for (const auto& r : responses) counts[r]++;
        std::string best;
        int best_count = 0;
        for (const auto& [resp, cnt] : counts) {
            if (cnt > best_count) { best = resp; best_count = cnt; }
        }
        s->stack->active_response = best;
    }

    if (s->stack->verbose)
        std::cout << "[RUN_N_TIMES] Winner: " << s->stack->active_response.substr(0, 120) << "\n";
    return true;
}

// --- Inference ---

std::string langX::inference(langX::Stack* stack, const char* raw_prompt, const std::vector<std::string>& files) {
    if (!stack) {
        std::cerr << "LangX Error [INF-00]: Null stack passed to inference().\n";
        return "";
    }

    if (!stack->unsafe) {
        std::string dep_error;
        if (!validate_stack(stack, dep_error)) {
            std::cerr << "LangX Error [INF-01]: " << dep_error << "\n";
            return "";
        }

        bool expected = false;
        while (!stack->is_busy.compare_exchange_weak(expected, true, std::memory_order_acquire, std::memory_order_relaxed)) {
            expected = false;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    stack->active_prompt = raw_prompt ? raw_prompt : "";
    stack->active_files = files;
    stack->active_response.clear();

    if (!stack->model) {
        std::cerr << "LangX Error [INF-02]: Stack '" << stack->stack_id << "' has no model. Call initModel() or switchModel() first.\n";
        if (!stack->unsafe)
            stack->is_busy.store(false, std::memory_order_release);
        return "";
    }
    if (!stack->conversation) {
        std::cerr << "LangX Error [INF-03]: Stack '" << stack->stack_id << "' has no conversation. Call initConversation() first.\n";
        if (!stack->unsafe)
            stack->is_busy.store(false, std::memory_order_release);
        return "";
    }

    stack->inference_state = new langX::InferenceState();
    auto* state = stack->inference_state;
    state->stack = stack;
    state->model = stack->model;
    state->conversation = stack->conversation;
    state->settings = stack->settings;

    auto cleanup = [&]() -> std::string {
        if (state->sampler) { llama_sampler_free(state->sampler); state->sampler = nullptr; }
        for (auto* b : state->image_bitmaps) mtmd_bitmap_free(b);
        state->image_bitmaps.clear();
        std::string result = stack->active_response;
        delete stack->inference_state;
        stack->inference_state = nullptr;
        if (!stack->unsafe)
            stack->is_busy.store(false, std::memory_order_release);
        return result;
    };

    bool ok = true;
    int layer_idx = 0;
    while (ok && layer_idx < (int)stack->layers.size()) {
        const langX::Layer* sl  = stack->layers[layer_idx];
        int next_idx = layer_idx + 1;

        switch (sl->type) {
            case langX::GOTO: next_idx = sl->target; break;
            case langX::BRANCH: if (sl->branch_condition && sl->branch_condition(stack)) next_idx = sl->target; break;
            case langX::CUSTOM: if (sl->custom_logic) sl->custom_logic(stack); break;
            case langX::DEBUG: { auto msg = get_param(sl, "message"); stack->active_status = msg; std::cout << "[DEBUG] " << msg << "\n"; } break;
            case langX::LLM_SAMPLE: break; 
            case langX::INIT_INFERENCE: ok = layer_init_inference(state, sl); break;
            case langX::FILE_PROCESSING: ok = layer_file_processing(state, sl); break;
            case langX::USER_PUSH_IMAGES: ok = layer_user_push_images(state, sl); break;
            case langX::USER_PUSH_PROMT: ok = layer_user_push_prompt(state, sl); break;
            case langX::BUILD_CONTEXT: ok = layer_build_context(state, sl); break;
            case langX::LOAD_CHAT_TEMPLATE: ok = layer_load_chat_template(state, sl); break;
            case langX::CXT_WIN_TRIM: ok = layer_cxt_win_trim(state, sl); break;
            case langX::CLEAR_KV_CACHE: ok = layer_clear_kv_cache(state, sl); break;
            case langX::INIT_SAMPLER: ok = layer_init_sampler(state, sl); break;
            case langX::INIT_BATCH: ok = layer_init_batch(state, sl); break;
            case langX::FEED_PROMPT: ok = layer_feed_prompt(state, sl); break;
            case langX::FEED_PROMPT_IMAGES: ok = layer_feed_prompt_images(state, sl); break;
            case langX::LLM_GENERATION: ok = layer_llm_generation(state, sl); break;
            case langX::SWAP_MODEL: ok = layer_swap_model(state, sl); break;
            case langX::SWAP_CONVO: ok = layer_swap_convo(state, sl); break;
            case langX::SET_SYSTEM_PROMPT: ok = layer_set_system_prompt(state, sl); break;
            case langX::LOAD_SYSTEM_PROMPT: ok = layer_load_system_prompt(state, sl); break;
            case langX::USE_TOOLS: ok = layer_use_tools(state, sl); break;
            case langX::WAIT_TOOLS: ok = layer_wait_tools(state, sl); break;
            case langX::LLM_TOOL_GEN: ok = layer_llm_tool_gen(state, sl); break;
            case langX::FREE_SAMPLER: ok = layer_free_sampler(state, sl); break;
            case langX::FREE_BATCH: ok = layer_free_batch(state, sl); break;
            case langX::SAVE_TO_HISTORY: ok = layer_save_to_history(state, sl); break;
            case langX::RAG_RETRIEVAL: ok = layer_rag_retrieval(state, sl); break;
            case langX::FILLER_RAND: ok = layer_filler_rand(state, sl); break;
            case langX::FILLER_LOOP: ok = layer_filler_loop(state, sl); break;
            case langX::QUICK_LOOK: ok = layer_quick_look(state, sl); break;
            case langX::PROMPT_CHECK: ok = layer_prompt_check(state, sl); break;
            case langX::QUICK_ASK: ok = layer_quick_ask(state, sl); break;
            case langX::SUBSTACK: ok = layer_substack(state, sl); break;
            case langX::WAIT_TIME: ok = layer_wait_time(state, sl); break;
            case langX::WAIT_LAMBDA:
            case langX::WAIT_PROMISE: {
                bool done = (sl->type == langX::WAIT_LAMBDA)
                    ? layer_wait_lambda(state, sl)
                    : layer_wait_promise(state, sl);
                if (!done) {
                    if (sl->target >= 0) next_idx = sl->target;
                    else ok = false;
                }
                break;
            }
            case langX::LLM_PROMPT_FILTER:
            case langX::LLM_RESULT_FILTER:
            case langX::FACT_CHECK: {
                bool passed;
                switch (sl->type) {
                    case langX::LLM_PROMPT_FILTER: passed = layer_llm_prompt_filter(state, sl); break;
                    case langX::LLM_RESULT_FILTER: passed = layer_llm_result_filter(state, sl); break;
                    default: passed = layer_fact_check(state, sl); break;
                }
                if (!passed) {
                    if (sl->target >= 0) next_idx = sl->target;
                    else ok = false;
                }
                break;
            }
            case langX::EPISODIC_MEMORY: ok = layer_episodic_memory(state, sl); break;
            case langX::SEMANTIC_MEMORY: ok = layer_semantic_memory(state, sl); break;
            case langX::SEMANTIC_MEM_RETRIEVAL: ok = layer_semantic_mem_retrieval(state, sl); break;
            case langX::EPISODIC_TIERED_MEMORY: ok = layer_episodic_tiered_memory(state, sl); break;
            case langX::BUILD_EPISODIC_MEM: ok = layer_build_episodic_mem(state, sl); break;
            case langX::BUILD_SEMANTIC_MEM: ok = layer_build_semantic_mem(state, sl); break;
            case langX::LOAD_PROCEDURAL_MEM: ok = layer_load_procedural_mem(state, sl); break;
            case langX::BUILD_PROCEDURAL_MEM: ok = layer_build_procedural_mem(state, sl); break;
            case langX::BERT_GENERATE: ok = layer_bert_generate(state, sl); break;
            case langX::NLI_GENERATE: ok = layer_nli_generate(state, sl); break;
            case langX::COT_GENERATE: ok = layer_cot_generate(state, sl); break;
            case langX::SELF_CONSISTENT_GEN: ok = layer_self_consistent_gen(state, sl); break;

            case langX::ASYNC_BREAK: {
                std::string response = stack->active_response;
                int async_start = layer_idx + 1;
                if (async_start >= (int)stack->layers.size()) {
                    return cleanup();
                }
                auto* bg_state  = state;
                auto* bg_stack  = stack;
                std::thread([bg_state, bg_stack, async_start]() {
                    bool bg_ok = true;
                    int bg_idx = async_start;
                    while (bg_ok && bg_idx < (int)bg_stack->layers.size()) {
                        const langX::Layer* sl = bg_stack->layers[bg_idx];
                        bg_idx++;
                        switch (sl->type) {
                        case langX::BUILD_EPISODIC_MEM:
                            bg_ok = layer_build_episodic_mem(bg_state, sl); break;
                        case langX::BUILD_SEMANTIC_MEM:
                            bg_ok = layer_build_semantic_mem(bg_state, sl); break;
                        case langX::BUILD_PROCEDURAL_MEM:
                            bg_ok = layer_build_procedural_mem(bg_state, sl); break;
                        case langX::CUSTOM:
                            if (sl->custom_logic) sl->custom_logic(bg_stack); break;
                        case langX::DEBUG: {
                            auto msg = get_param(sl, "message");
                            bg_stack->active_status = msg;
                            std::cout << "[DEBUG/async] " << msg << "\n"; break;
                        }
                        case langX::WAIT_TIME:
                            layer_wait_time(bg_state, sl); break;
                        case langX::FREE_SAMPLER:
                            layer_free_sampler(bg_state, sl); break;
                        case langX::FREE_BATCH:
                            layer_free_batch(bg_state, sl);  break;
                        case langX::ASYNC_BREAK: break;

                        default:
                            if (bg_stack->verbose)
                                std::cerr << "[ASYNC] Layer '" << layer_name(sl->type)  << "' is not supported in async section — skipped.\n";
                            break;
                        }
                    }
                    if (bg_state->sampler) {
                        llama_sampler_free(bg_state->sampler);
                        bg_state->sampler = nullptr;
                    }
                    for (auto* b : bg_state->image_bitmaps) mtmd_bitmap_free(b);
                    bg_state->image_bitmaps.clear();
                    delete bg_state;
                    bg_stack->inference_state = nullptr;
                    bg_stack->is_busy.store(false, std::memory_order_release);

                    if (bg_stack->verbose)
                        std::cout << "[ASYNC] Background layers complete.\n";
                }).detach();
                return response;
            }
        }
        layer_idx = next_idx;
    }

    return cleanup();
}

std::string langX::inference(const char* raw_prompt, const std::vector<std::string>& files) {
    langX::Stack* stack = getLastStack();
    if (!stack) {
        std::cerr << "LangX Error [INF-00]: No active stack. Call makeStack + swapStack first.\n";
        return "";
    }
    return langX::inference(stack, raw_prompt, files);
}

static void inject_system_context(langX::Conversation* conv, const std::string& content) {
    if (!conv) {
        std::cerr << "LangX Error [ISC-00]: No active conversation.\n";
        return;
    }
    conv->messages.insert(conv->messages.begin(), { "system", content });
    conv->active_messages.insert(conv->active_messages.begin(), { "system", content });
}

void langX::feedTextFile(const char* filePath, const char* label, langX::Stack* target_stack) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "LangX Error [FTF-01]: Cannot open file: " << filePath << "\n";
        return;
    }
    std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    std::string lbl = label ? label : std::filesystem::path(filePath).filename().string();
    langX::Stack* stack = get_or_last_stack(target_stack);
    inject_system_context(stack ? stack->conversation : nullptr, "[File: " + lbl + "]\n" + content);
}

void langX::feedContext(const char* content, const char* label, langX::Stack* target_stack) {
    std::string msg = label ? (std::string("[") + label + "]\n" + content) : content;
    langX::Stack* stack = get_or_last_stack(target_stack);
    inject_system_context(stack ? stack->conversation : nullptr, msg);
}

void langX::setSystemPrompt(const char* prompt, langX::Stack* target_stack) {
    langX::Stack* stack = get_or_last_stack(target_stack);
    if (stack && stack->conversation)
        stack->conversation->system_prompt = prompt ? prompt : "";
}

void langX::setModelSystemPrompt(const char* prompt, langX::Stack* target_stack) {
    langX::Stack* stack = get_or_last_stack(target_stack);
    if (stack && stack->model)
        stack->model->system_prompt = prompt ? prompt : "";
}

void langX::setInquerySettings(langX::InquerySettings settings, langX::Stack* target_stack) {
    langX::Stack* stack = get_or_last_stack(target_stack);
    if (!stack) return;
    if (!stack->settings) stack->settings = new langX::InquerySettings();
    *stack->settings = settings;
}

LANGX_API int langX::randomSeed() {
    std::random_device rd;
    std::uniform_int_distribution<int> dist(1, INT_MAX);
    return dist(rd);
}

size_t langX::countVisionTokens(const char* image_path, const char* model_id) {
    langX::Model* m = nullptr;
    if (!model_id || model_id[0] == '\0') {
        langX::Stack* stack = getLastStack();
        m = stack ? stack->model : nullptr;
    } else {
        auto it = langX::global_LangX.loaded_models.find(model_id);
        if (it != langX::global_LangX.loaded_models.end()) m = it->second;
    }
    if (!m || !m->mtmd_ctx) return 0;

    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(m->mtmd_ctx, image_path);
    if (!bmp) return 0;

    uint32_t nx = mtmd_bitmap_get_nx(bmp);
    uint32_t ny = mtmd_bitmap_get_ny(bmp);
    mtmd_bitmap_free(bmp);

    const uint32_t patch_stride = 14;
    size_t tx = (nx + patch_stride - 1) / patch_stride;
    size_t ty = (ny + patch_stride - 1) / patch_stride;
    return tx * ty;
}

// --- Stack management ---

langX::Layer* langX::makeLayer(langX::LayerType type, int target) {
    langX::Layer* l = new langX::Layer();
    l->type = type;
    l->id = -1;
    l->target = target;
    return l;
}

langX::Layer* langX::makeInitInferenceLayer() { return makeLayer(INIT_INFERENCE); }
langX::Layer* langX::makeFileProcessingLayer() { return makeLayer(FILE_PROCESSING); }
langX::Layer* langX::makeUserPushPromptLayer() { return makeLayer(USER_PUSH_PROMT); }
langX::Layer* langX::makeUserPushImagesLayer() { return makeLayer(USER_PUSH_IMAGES); }
langX::Layer* langX::makeBuildContextLayer() { return makeLayer(BUILD_CONTEXT); }
langX::Layer* langX::makeLoadChatTemplateLayer() { return makeLayer(LOAD_CHAT_TEMPLATE); }
langX::Layer* langX::makeCxtWinTrimLayer() { return makeLayer(CXT_WIN_TRIM); }
langX::Layer* langX::makeLoadSystemPromptLayer() { return makeLayer(LOAD_SYSTEM_PROMPT); }
langX::Layer* langX::makeClearKvCacheLayer() { return makeLayer(CLEAR_KV_CACHE); }
langX::Layer* langX::makeInitSamplerLayer() { return makeLayer(INIT_SAMPLER); }
langX::Layer* langX::makeInitBatchLayer() { return makeLayer(INIT_BATCH); }
langX::Layer* langX::makeFeedPromptLayer() { return makeLayer(FEED_PROMPT); }
langX::Layer* langX::makeFeedPromptImagesLayer() { return makeLayer(FEED_PROMPT_IMAGES); }
langX::Layer* langX::makeLlmGenerationLayer() { return makeLayer(LLM_GENERATION); }
langX::Layer* langX::makeLlmSampleLayer() { return makeLayer(LLM_SAMPLE); }
langX::Layer* langX::makeFreeSamplerLayer() { return makeLayer(FREE_SAMPLER); }
langX::Layer* langX::makeFreeBatchLayer() { return makeLayer(FREE_BATCH); }
langX::Layer* langX::makeSaveToHistoryLayer() { return makeLayer(SAVE_TO_HISTORY); }
langX::Layer* langX::makeUseToolsLayer() { return makeLayer(USE_TOOLS); }
langX::Layer* langX::makeWaitToolsLayer() { return makeLayer(WAIT_TOOLS); }
langX::Layer* langX::makeLlmToolGenLayer() { return makeLayer(LLM_TOOL_GEN); }
langX::Layer* langX::makeRagRetrievalLayer() { return makeLayer(RAG_RETRIEVAL); }
langX::Layer* langX::makeSemanticMemRetrievalLayer() { return makeLayer(SEMANTIC_MEM_RETRIEVAL); }
langX::Layer* langX::makeSemanticMemoryLayer() { return makeLayer(SEMANTIC_MEMORY); }

void langX::appendLayer(langX::Stack* stack, langX::Layer* layer) {
    if (!stack || !layer) return;
    layer->id = (int)stack->layers.size();
    stack->layers.push_back(layer);
}

void langX::insertLayer(langX::Stack* stack, int idx, langX::Layer* layer) {
    if (!stack || !layer) return;
    if (idx < 0) idx = 0;
    if (idx > (int)stack->layers.size()) idx = (int)stack->layers.size();
    layer->id = idx;
    stack->layers.insert(stack->layers.begin() + idx, layer);
    for (int i = idx + 1; i < (int)stack->layers.size(); i++)
        stack->layers[i]->id = i;
}

void langX::removeLayer(langX::Stack* stack, int idx) {
    if (!stack || idx < 0 || idx >= (int)stack->layers.size()) return;
    delete stack->layers[idx];
    stack->layers.erase(stack->layers.begin() + idx);
    for (int i = idx; i < (int)stack->layers.size(); i++)
        stack->layers[i]->id = i;
}

langX::Layer* langX::makeBranchLayer(std::function<bool(const langX::Stack*)> condition, int target) {
    langX::Layer* l = langX::makeLayer(langX::BRANCH, target);
    l->branch_condition = condition;
    return l;
}

langX::Layer* langX::makeCustomLayer(std::function<void(langX::Stack*)> logic) {
    langX::Layer* l = langX::makeLayer(langX::CUSTOM);
    l->custom_logic = logic;
    return l;
}


langX::Layer* langX::makeParamLayer(langX::LayerType type, const char* param) {
    langX::Layer* l = langX::makeLayer(type);
    const std::string val = param ? param : "";
    switch (type) {
        case langX::DEBUG: l->params["message"] = val; break;
        case langX::SWAP_MODEL: l->params["model_id"] = val; break;
        case langX::SWAP_CONVO: l->params["convo_id"] = val; break;
        case langX::SET_SYSTEM_PROMPT: l->params["value"] = val; break;
        default: l->params["value"] = val; break;
    }
    return l;
}

langX::Layer* langX::makeDebugLayer(const char* message) {
    langX::Layer* l = langX::makeLayer(langX::DEBUG);
    l->params["message"] = message ? message : "";
    return l;
}

langX::Layer* langX::makeSwapModelLayer(const char* model_id) {
    langX::Layer* l = langX::makeLayer(langX::SWAP_MODEL);
    l->params["model_id"] = model_id ? model_id : "";
    return l;
}

langX::Layer* langX::makeSwapConvoLayer(const char* convo_id) {
    langX::Layer* l = langX::makeLayer(langX::SWAP_CONVO);
    l->params["convo_id"] = convo_id ? convo_id : "";
    return l;
}

langX::Layer* langX::makeSetSystemPromptLayer(const char* prompt) {
    langX::Layer* l = langX::makeLayer(langX::SET_SYSTEM_PROMPT);
    l->params["value"] = prompt ? prompt : "";
    return l;
}

langX::Layer* langX::makeFillerRandLayer(std::vector<std::string> strings) {
    langX::Layer* l = langX::makeLayer(langX::FILLER_RAND);
    l->param_list = std::move(strings);
    return l;
}

langX::Layer* langX::makeFillerLoopLayer(std::vector<std::string> strings) {
    langX::Layer* l = langX::makeLayer(langX::FILLER_LOOP);
    l->param_list = std::move(strings);
    return l;
}

langX::Layer* langX::makeQuickLookLayer(const char* instructions, const char* image_target, const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::QUICK_LOOK);
    l->params["instructions"] = instructions ? instructions : "";
    l->params["image_target"] = image_target ? image_target : "";
    l->params["model_id"] = model_id ? model_id : "";
    l->params["use_sleep"] = use_sleep ? "true" : "false";
    return l;
}

langX::Layer* langX::makePromptCheckLayer(langX::PromptCheckConfig config) {
    langX::Layer* l = langX::makeLayer(langX::PROMPT_CHECK);
    l->params["min_length"] = std::to_string(config.min_length);
    l->params["max_length"] = std::to_string(config.max_length);
    l->params["max_files"] = std::to_string(config.max_files);
    l->params["max_file_bytes"]= std::to_string(config.max_file_bytes);
    l->params["check_duplicate"] = config.check_duplicate ? "true" : "false";
    l->params["check_injection"] = config.check_injection ? "true" : "false";
    l->params["check_pii"] = config.check_pii ? "true" : "false";
    
    std::string ext_str;
    for (const auto& e : config.allowed_extensions) {
        if (!ext_str.empty()) ext_str += ",";
        std::string ext = e;
        if (!ext.empty() && ext[0] != '.') ext = "." + ext;
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        ext_str += ext;
    }
    l->params["allowed_extensions"] = ext_str;
    l->param_list = config.extra_patterns;
    return l;
}

langX::Layer* langX::makeQuickAskLayer(const char* prompt, bool save_to_history, const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::QUICK_ASK);
    l->params["prompt"]  = prompt ? prompt : "";
    l->params["save_to_history"]  = save_to_history ? "true" : "false";
    l->params["model_id"] = model_id  ? model_id : "";
    l->params["use_sleep"] = use_sleep ? "true" : "false";
    return l;
}

langX::Layer* langX::makeSubstackLayer(const char* stack_id, const char* prompt, const char* context_label, const char* model_id, bool use_sleep, bool save_to_history) {
    langX::Layer* l = langX::makeLayer(langX::SUBSTACK);
    l->params["stack_id"] = stack_id ? stack_id : "";
    l->params["prompt"] = prompt ? prompt : "";
    l->params["context_label"] = context_label ? context_label : "Substack Result";
    l->params["model_id"] = model_id ? model_id : "";
    l->params["use_sleep"] = use_sleep ? "true" : "false";
    l->params["save_to_history"] = save_to_history ? "true" : "false";
    return l;
}

langX::Layer* langX::makeWaitTimeLayer(int duration_ms) {
    langX::Layer* l = langX::makeLayer(langX::WAIT_TIME);
    l->params["duration_ms"] = std::to_string(duration_ms);
    return l;
}

langX::Layer* langX::makeWaitLambdaLayer(std::function<bool(const langX::Stack*)> condition, int timeout_ms, int on_timeout_target, int poll_ms) {
    langX::Layer* l = langX::makeLayer(langX::WAIT_LAMBDA, on_timeout_target);
    l->branch_condition = condition;
    l->params["timeout_ms"] = std::to_string(timeout_ms);
    l->params["poll_ms"] = std::to_string(poll_ms);
    return l;
}

langX::Layer* langX::makeWaitPromiseLayer(int timeout_ms, int on_timeout_target) {
    langX::Layer* l = langX::makeLayer(langX::WAIT_PROMISE, on_timeout_target);
    l->params["timeout_ms"] = std::to_string(timeout_ms);
    return l;
}

void langX::signalStackWait(langX::Stack* stack, const char* value) {
    if (!stack || !stack->wait_promise) return;
    try {
        stack->wait_promise->set_value(value ? value : "");
    } catch (const std::future_error&) {
        // Already signaled or promise already satisfied - safe to ignore.
    }
}

static langX::Layer* make_filter_layer(langX::LayerType type, int on_fail_target, const char* filter_prompt, const char* model_id, bool use_sleep, bool inject_notes, const char* notes_label) {
    langX::Layer* l = langX::makeLayer(type, on_fail_target);
    l->params["filter_prompt"] = filter_prompt ? filter_prompt : "";
    l->params["model_id"] = model_id ? model_id : "";
    l->params["use_sleep"] = use_sleep ? "true" : "false";
    l->params["inject_notes"] = inject_notes ? "true" : "false";
    l->params["notes_label"] = notes_label ? notes_label : "Filter Notes";
    return l;
}

static const char* DEFAULT_PROMPT_FILTER =
    "You are a content safety filter. Review the user's message. "
    "Reply with YES if it is safe and appropriate, "
    "or NO followed by a brief reason if it is not.";

static const char* DEFAULT_RESULT_FILTER =
    "You are a quality reviewer. Review the following AI-generated response. "
    "Reply with YES if it is accurate, helpful, and appropriate, "
    "or NO followed by a brief reason if it is not.";

static const char* DEFAULT_FACT_CHECK =
    "You are a fact-checker. Review the following AI-generated response against "
    "the provided reference material. "
    "Reply with YES if the response is factually accurate, "
    "or NO followed by a specific correction if it is not.";

langX::Layer* langX::makeLLMPromptFilterLayer(int on_fail_target, const char* filter_prompt, const char* model_id, bool use_sleep, bool inject_notes, const char* notes_label) {
    return make_filter_layer(langX::LLM_PROMPT_FILTER, on_fail_target, filter_prompt ? filter_prompt : DEFAULT_PROMPT_FILTER, model_id, use_sleep, inject_notes, notes_label);
}

langX::Layer* langX::makeLLMResultFilterLayer(int on_fail_target, const char* filter_prompt, const char* model_id, bool use_sleep, bool inject_notes, const char* notes_label) {
    return make_filter_layer(langX::LLM_RESULT_FILTER, on_fail_target,
        filter_prompt ? filter_prompt : DEFAULT_RESULT_FILTER,
        model_id, use_sleep, inject_notes, notes_label);
}

langX::Layer* langX::makeFactCheckLayer(int on_fail_target, const char* filter_prompt, const char* model_id, bool use_sleep, bool inject_notes, const char* notes_label) {
    return make_filter_layer(langX::FACT_CHECK, on_fail_target, filter_prompt ? filter_prompt : DEFAULT_FACT_CHECK, model_id, use_sleep, inject_notes, notes_label);
}

langX::Layer* langX::makeCotGenerateLayer(const char* cot_prompt, const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::COT_GENERATE);
    if (cot_prompt && *cot_prompt)
        l->params["cot_prompt"] = cot_prompt;
    if (model_id && *model_id)
        l->params["model_id"] = model_id;
    if (use_sleep)
        l->params["use_sleep"] = "true";
    return l;
}

langX::Layer* langX::makeSelfConsistentGenLayer(const char* substack_id, int n, const char* scoring_prompt, const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::SELF_CONSISTENT_GEN);
    l->params["stack_id"] = substack_id;
    l->params["n"] = std::to_string(n);
    if (scoring_prompt && *scoring_prompt)
        l->params["scoring_prompt"] = scoring_prompt;
    if (model_id && *model_id)
        l->params["model_id"] = model_id;
    if (use_sleep)
        l->params["use_sleep"] = "true";
    return l;
}

langX::Layer* langX::makeBertGenerateLayer(const char* model_id) {
    langX::Layer* l = langX::makeLayer(langX::BERT_GENERATE);
    if (model_id && *model_id)
        l->params["model_id"] = model_id;
    return l;
}

langX::Layer* langX::makeNliGenerateLayer(const char* model_id, const char* hypothesis) {
    langX::Layer* l = langX::makeLayer(langX::NLI_GENERATE);
    if (model_id && *model_id)
        l->params["model_id"] = model_id;
    if (hypothesis && *hypothesis)
        l->params["hypothesis"] = hypothesis;
    return l;
}

langX::Layer* langX::makeEpisodicMemoryLayer() {
    return langX::makeLayer(langX::EPISODIC_MEMORY);
}

langX::Layer* langX::makeEpisodicTieredMemoryLayer(const char* tier2_compress_prompt, const char* tier1_compress_prompt, const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::EPISODIC_TIERED_MEMORY);
    if (tier2_compress_prompt && *tier2_compress_prompt)
        l->params["tier2_compress_prompt"] = tier2_compress_prompt;
    if (tier1_compress_prompt && *tier1_compress_prompt)
        l->params["tier1_compress_prompt"] = tier1_compress_prompt;
    if (model_id && *model_id)
        l->params["model_id"] = model_id;
    if (use_sleep)
        l->params["use_sleep"] = "true";
    return l;
}

langX::Layer* langX::makeBuildEpisodicMemLayer(const char* tier2_compress_prompt, const char* tier1_compress_prompt, const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::BUILD_EPISODIC_MEM);
    if (tier2_compress_prompt && *tier2_compress_prompt)
        l->params["tier2_compress_prompt"] = tier2_compress_prompt;
    if (tier1_compress_prompt && *tier1_compress_prompt)
        l->params["tier1_compress_prompt"] = tier1_compress_prompt;
    if (model_id && *model_id)
        l->params["model_id"] = model_id;
    if (use_sleep)
        l->params["use_sleep"] = "true";
    return l;
}

langX::Layer* langX::makeBuildSemanticMemLayer(const char* compress_prompt, const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::BUILD_SEMANTIC_MEM);
    if (compress_prompt && *compress_prompt)
        l->params["compress_prompt"] = compress_prompt;
    if (model_id && *model_id)
        l->params["model_id"] = model_id;
    if (use_sleep)
        l->params["use_sleep"] = "true";
    return l;
}

langX::Layer* langX::makeAsyncBreakLayer() {
    return langX::makeLayer(langX::ASYNC_BREAK);
}

langX::Layer* langX::makeLoadProceduralMemLayer(const char* file_path) {
    langX::Layer* l = langX::makeLayer(langX::LOAD_PROCEDURAL_MEM);
    l->params["file_path"] = file_path;
    return l;
}

langX::Layer* langX::makeBuildProceduralMemLayer(const char* file_path, const char* build_prompt,  const char* model_id, bool use_sleep) {
    langX::Layer* l = langX::makeLayer(langX::BUILD_PROCEDURAL_MEM);
    l->params["file_path"] = file_path;
    if (build_prompt && *build_prompt) l->params["build_prompt"] = build_prompt;
    if (model_id && *model_id) l->params["model_id"] = model_id;
    if (use_sleep) l->params["use_sleep"] = "true";
    return l;
}

langX::Layer* langX::makeGotoLayer(int target) {
    return langX::makeLayer(langX::GOTO, target);
}

langX::Tool* langX::makeTool(const char* name, const char* description, std::function<std::string(std::map<std::string, std::string>)> handler,
    bool is_async) {
    langX::Tool* t = new langX::Tool();
    t->name = name ? name : "";
    t->description = description ? description : "";
    t->handler = handler;
    t->is_async = is_async;
    return t;
}

langX::Tool* langX::makeVoidTool(const char* name, const char* description, std::function<void(std::map<std::string, std::string>)> handler,
    bool is_async) {
    langX::Tool* t = new langX::Tool();
    t->name = name ? name : "";
    t->description = description ? description : "";
    t->is_void = true;
    t->is_async = is_async;
    t->handler = [handler](std::map<std::string, std::string> args) -> std::string {
        if (handler) handler(args);
        return "";
    };
    return t;
}

void langX::addToolParam(langX::Tool* tool, const char* name, const char* description, const char* type, bool required) {
    if (!tool) return;
    langX::ToolParam p;
    p.name = name ? name : "";
    p.description = description ? description : "";
    p.type = type ? type : "string";
    p.required = required;
    tool->params.push_back(p);
}

void langX::registerTool(langX::Stack* stack, langX::Tool* tool) {
    if (stack && tool) stack->tools.push_back(tool);
}

static std::vector<common_chat_tool> tools_to_common_chat_tools(const std::vector<langX::Tool*>& tools) {
    std::vector<common_chat_tool> result;
    for (const auto* t : tools) {
        nlohmann::json props = nlohmann::json::object();
        nlohmann::json required_list = nlohmann::json::array();
        for (const auto& p : t->params) {
            props[p.name] = { {"type", p.type}, {"description", p.description} };
            if (p.required) required_list.push_back(p.name);
        }
        nlohmann::json schema = {
            {"type", "object"},
            {"properties", props},
            {"required", required_list}
        };
        result.push_back({ t->name, t->description, schema.dump() });
    }
    return result;
}

std::string langX::getToolsSystemPrompt(const langX::Stack* stack) {
    if (!stack || stack->tools.empty()) return "";
    if (stack->settings && stack->settings->use_native_tools) return "";
    // TODO: make this prompt configurable by the dev
    std::string prompt =
        "You have access to tools. You MUST call tools whenever possible instead of answering directly.\n"
        "To call a tool, output ONLY <tool_call> XML on its own line — no other text before or after:\n"
        "<tool_call>{\"name\": \"tool_name\", \"args\": {\"param\": \"value\"}}</tool_call>\n"
        "Always wrap argument values in double quotes, even numbers.\n"
        "For tools with no parameters: <tool_call>{\"name\": \"tool_name\", \"args\": {}}</tool_call>\n"
        "You MUST include both the opening <tool_call> and closing </tool_call> tags.\n"
        "You may output multiple <tool_call> blocks, one per line.\n\n"
        "Available tools:\n";
    for (const auto* tool : stack->tools) {
        prompt += "- " + tool->name + ": " + tool->description + "\n";
        if (!tool->params.empty()) {
            prompt += "  Parameters:\n";
            for (const auto& p : tool->params)
                prompt += "  - " + p.name + " (" + p.type + ")" + (p.required ? " [required]" : " [optional]") + ": " + p.description + "\n";
        }
    }
    prompt += "\nExamples:\n";
    for (const auto* tool : stack->tools) {
        prompt += "<tool_call>{\"name\": \"" + tool->name + "\", \"args\": {";
        for (size_t i = 0; i < tool->params.size(); i++) {
            if (i > 0) prompt += ", ";
            prompt += "\"" + tool->params[i].name + "\": \"<" + tool->params[i].type + ">\"";
        }
        prompt += "}}</tool_call>\n";
    }
    return prompt;
}

static langX::Stack* create_stack(const char* stack_id, std::vector<langX::Layer*> layers) {
    langX::Stack* s = new langX::Stack();
    s->stack_id = stack_id ? stack_id : "";
    for (int i = 0; i < (int)layers.size(); i++) {
        layers[i]->id = i;
        s->layers.push_back(layers[i]);
    }

    langX::Stack* active = getLastStack();
    if (active) {
        s->model = active->model;
        if (active->settings)
            s->settings = new langX::InquerySettings(*active->settings);
    }
    if (!s->conversation)
        s->conversation = new langX::Conversation();

    langX::global_LangX.hotswap_stacks[s->stack_id] = s;
    return s;
}

langX::Stack* langX::makeStack(const char* stack_id) {
    return create_stack(stack_id, {});
}

langX::Stack* langX::makeStack(const char* stack_id, std::vector<langX::Layer*> layers) {
    return create_stack(stack_id, layers);
}

langX::Stack* langX::makeDefaultStack(const char* stack_id) {
    return create_stack(stack_id, {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLayer(langX::FILE_PROCESSING),
        langX::makeLayer(langX::USER_PUSH_IMAGES),
        langX::makeLayer(langX::USER_PUSH_PROMT),
        langX::makeLayer(langX::LOAD_SYSTEM_PROMPT),
        langX::makeLayer(langX::CXT_WIN_TRIM),
        langX::makeLayer(langX::BUILD_CONTEXT),
        langX::makeLayer(langX::CLEAR_KV_CACHE),
        langX::makeLayer(langX::INIT_SAMPLER),
        langX::makeLayer(langX::INIT_BATCH),
        langX::makeLayer(langX::FEED_PROMPT),
        langX::makeLayer(langX::LLM_GENERATION),
        langX::makeLayer(langX::FREE_SAMPLER),
        langX::makeLayer(langX::FREE_BATCH),
        langX::makeLayer(langX::SAVE_TO_HISTORY),
    });
}

void langX::swapStack(const char* stack_id) {
    auto it = langX::global_LangX.hotswap_stacks.find(stack_id);
    if (it == langX::global_LangX.hotswap_stacks.end()) {
        std::cerr << "LangX Error [SS-01]: Stack not found: " << stack_id << "\n";
        return;
    }
    langX::global_LangX.last_refrenced_stack_id = stack_id;
}

void langX::deleteStack(const char* stack_id) {
    auto it = langX::global_LangX.hotswap_stacks.find(stack_id);
    if (it == langX::global_LangX.hotswap_stacks.end()) return;

    if (langX::global_LangX.last_refrenced_stack_id == stack_id)
        langX::global_LangX.last_refrenced_stack_id.clear();

    langX::Stack* s = it->second;

    if (s->conversation) {
        std::string cid = s->active_convo_id.empty() ? gen_convo_id() : s->active_convo_id;
        langX::global_LangX.hotswap_conversation[cid] = s->conversation;
        s->conversation = nullptr;
    }

    for (auto* l : s->layers) delete l;
    for (auto* t : s->tools)  delete t;
    delete s->settings;
    delete s;
    langX::global_LangX.hotswap_stacks.erase(it);
}

