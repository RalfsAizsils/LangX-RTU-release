#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <pybind11/stl/filesystem.h>

#include "langX.h"
#include "LxRAG.h"

namespace py = pybind11;
using namespace langX;

struct PyRagDb {
    RagDb* ptr = nullptr;
    explicit PyRagDb(RagDb* p) : ptr(p) {}
};

PYBIND11_MODULE(langx_core, m) {
    m.doc() = "LangX Python bindings (pybind11)";

    // =========================================================================
    // Config
    // =========================================================================
    py::class_<Config>(m, "Config")
        .def(py::init<>())
        .def_readwrite("user_output_path", &Config::user_output_path)
        .def_readwrite("verbose_logs", &Config::verbose_logs);

    // =========================================================================
    // InquerySettings
    // =========================================================================
    py::class_<InquerySettings>(m, "InquerySettings")
        .def(py::init<>())
        .def_readwrite("n_tokens_to_predict", &InquerySettings::n_tokens_to_predict)
        .def_readwrite("batch_size", &InquerySettings::batch_size)
        .def_readwrite("n_sequences", &InquerySettings::n_sequences)
        .def_readwrite("temperature", &InquerySettings::temperature)
        .def_readwrite("top_k", &InquerySettings::top_k)
        .def_readwrite("top_p", &InquerySettings::top_p)
        .def_readwrite("seed", &InquerySettings::seed)
        .def_readwrite("grammar", &InquerySettings::grammar)
        .def_readwrite("episodic_context_ratio", &InquerySettings::episodic_context_ratio)
        .def_readwrite("episodic_tier2_ratio", &InquerySettings::episodic_tier2_ratio)
        .def_readwrite("episodic_tier1_ratio", &InquerySettings::episodic_tier1_ratio)
        .def_readwrite("use_native_tools", &InquerySettings::use_native_tools);

    // =========================================================================
    // ModelParams
    // =========================================================================
    py::class_<ModelParams>(m, "ModelParams")
        .def(py::init<>())
        .def(py::init([](const std::string& path, const std::string& vision_path) {
            return ModelParams(path, vision_path);
        }), py::arg("path"), py::arg("vision_path") = "")
        .def_property("path",
            [](const ModelParams& self){ return self.path.string(); },
            [](ModelParams& self, const std::string& v){ self.path = v; })
        .def_property("vision_path",
            [](const ModelParams& self){ return self.vission_path.string(); },
            [](ModelParams& self, const std::string& v){ self.vission_path = v; })
        .def_property("lora_path",
            [](const ModelParams& self){ return self.lora_path.string(); },
            [](ModelParams& self, const std::string& v){ self.lora_path = v; })
        .def_readwrite("lora_scale", &ModelParams::lora_scale)
        .def_readwrite("n_gpu_layers", &ModelParams::n_gpu_layers)
        .def_readwrite("n_ctx", &ModelParams::n_ctx)
        .def_readwrite("n_batch", &ModelParams::n_batch)
        .def_readwrite("n_ubatch", &ModelParams::n_ubatch)
        .def_readwrite("n_threads", &ModelParams::n_threads)
        .def_readwrite("n_threads_batch", &ModelParams::n_threads_batch)
        .def_readwrite("main_gpu", &ModelParams::main_gpu)
        .def_readwrite("tensor_split", &ModelParams::tensor_split);

    // =========================================================================
    // Model (read-only inspection of loaded model state)
    // =========================================================================
    py::class_<Model>(m, "Model")
        .def_readwrite("system_prompt", &Model::system_prompt)
        .def_property_readonly("has_vision", [](const Model& self){ return self.mtmd_ctx != nullptr; })
        .def_property_readonly("has_lora",   [](const Model& self){ return self.lora_adapter != nullptr; })
        .def_property_readonly("chat_template", [](const Model& self) -> py::object {
            if (self.chat_template) return py::str(self.chat_template);
            return py::none();
        });

    // =========================================================================
    // ChatMessage / ContextExtra / Conversation
    // =========================================================================
    py::class_<ChatMessage>(m, "ChatMessage")
        .def(py::init<>())
        .def_readwrite("role", &ChatMessage::role)
        .def_readwrite("content", &ChatMessage::content)
        .def("__repr__", [](const ChatMessage& self){
            std::string snip = self.content.size() > 40
                ? self.content.substr(0,40) + "..." : self.content;
            return "<ChatMessage role='" + self.role + "' content='" + snip + "'>";
        });

    py::class_<ContextExtra>(m, "ContextExtra")
        .def(py::init<>())
        .def_readwrite("label", &ContextExtra::label)
        .def_readwrite("content", &ContextExtra::content)
        .def_readwrite("score", &ContextExtra::score);

    py::class_<Conversation>(m, "Conversation")
        .def(py::init<>())
        .def_readwrite("n_past", &Conversation::n_past)
        .def_readwrite("messages", &Conversation::messages)
        .def_readwrite("active_messages", &Conversation::active_messages)
        .def_readwrite("extra_data", &Conversation::extra_data)
        .def_readwrite("system_prompt", &Conversation::system_prompt)
        .def_readwrite("procedural_memory", &Conversation::procedural_memory)
        .def_readwrite("episodic_tier1_summary", &Conversation::episodic_tier1_summary)
        .def_readwrite("episodic_tier2_memories", &Conversation::episodic_tier2_memories)
        .def_readwrite("episodic_compress_idx", &Conversation::episodic_compress_idx)
        .def_readwrite("semantic_compress_idx", &Conversation::semantic_compress_idx);

    // =========================================================================
    // LayerType enum
    // =========================================================================
    py::enum_<LayerType>(m, "LayerType")
        .value("INIT_INFERENCE", INIT_INFERENCE)
        .value("FILE_PROCESSING", FILE_PROCESSING)
        .value("USER_PUSH_PROMT", USER_PUSH_PROMT)
        .value("USER_PUSH_IMAGES", USER_PUSH_IMAGES)
        .value("BUILD_CONTEXT", BUILD_CONTEXT)
        .value("LOAD_CHAT_TEMPLATE", LOAD_CHAT_TEMPLATE)
        .value("CXT_WIN_TRIM", CXT_WIN_TRIM)
        .value("CLEAR_KV_CACHE", CLEAR_KV_CACHE)
        .value("INIT_SAMPLER", INIT_SAMPLER)
        .value("INIT_BATCH", INIT_BATCH)
        .value("FEED_PROMPT", FEED_PROMPT)
        .value("FEED_PROMPT_IMAGES", FEED_PROMPT_IMAGES)
        .value("LLM_GENERATION", LLM_GENERATION)
        .value("LLM_SAMPLE", LLM_SAMPLE)
        .value("FREE_SAMPLER", FREE_SAMPLER)
        .value("FREE_BATCH", FREE_BATCH)
        .value("SAVE_TO_HISTORY", SAVE_TO_HISTORY)
        .value("GOTO", GOTO)
        .value("BRANCH", BRANCH)
        .value("CUSTOM", CUSTOM)
        .value("DEBUG", DEBUG)
        .value("SWAP_MODEL", SWAP_MODEL)
        .value("SWAP_CONVO", SWAP_CONVO)
        .value("SET_SYSTEM_PROMPT", SET_SYSTEM_PROMPT)
        .value("LOAD_SYSTEM_PROMPT", LOAD_SYSTEM_PROMPT)
        .value("USE_TOOLS", USE_TOOLS)
        .value("WAIT_TOOLS", WAIT_TOOLS)
        .value("LLM_TOOL_GEN", LLM_TOOL_GEN)
        .value("RAG_RETRIEVAL", RAG_RETRIEVAL)
        .value("FILLER_RAND", FILLER_RAND)
        .value("FILLER_LOOP", FILLER_LOOP)
        .value("QUICK_LOOK", QUICK_LOOK)
        .value("PROMPT_CHECK", PROMPT_CHECK)
        .value("QUICK_ASK", QUICK_ASK)
        .value("SUBSTACK", SUBSTACK)
        .value("WAIT_TIME", WAIT_TIME)
        .value("WAIT_LAMBDA", WAIT_LAMBDA)
        .value("WAIT_PROMISE", WAIT_PROMISE)
        .value("EPISODIC_MEMORY", EPISODIC_MEMORY)
        .value("SEMANTIC_MEMORY", SEMANTIC_MEMORY)
        .value("SEMANTIC_MEM_RETRIEVAL", SEMANTIC_MEM_RETRIEVAL)
        .value("EPISODIC_TIERED_MEMORY", EPISODIC_TIERED_MEMORY)
        .value("LLM_PROMPT_FILTER", LLM_PROMPT_FILTER)
        .value("LLM_RESULT_FILTER", LLM_RESULT_FILTER)
        .value("FACT_CHECK", FACT_CHECK)
        .value("BUILD_EPISODIC_MEM", BUILD_EPISODIC_MEM)
        .value("BUILD_SEMANTIC_MEM", BUILD_SEMANTIC_MEM)
        .value("ASYNC_BREAK", ASYNC_BREAK)
        .value("LOAD_PROCEDURAL_MEM", LOAD_PROCEDURAL_MEM)
        .value("BUILD_PROCEDURAL_MEM", BUILD_PROCEDURAL_MEM)
        .value("BERT_GENERATE", BERT_GENERATE)
        .value("NLI_GENERATE", NLI_GENERATE)
        .value("COT_GENERATE", COT_GENERATE)
        .value("SELF_CONSISTENT_GEN", SELF_CONSISTENT_GEN)
        .export_values();

    // =========================================================================
    // ToolParam / Tool
    // =========================================================================
    py::class_<ToolParam>(m, "ToolParam")
        .def(py::init<>())
        .def_readwrite("name", &ToolParam::name)
        .def_readwrite("description", &ToolParam::description)
        .def_readwrite("type", &ToolParam::type)
        .def_readwrite("required", &ToolParam::required);

    py::class_<Tool>(m, "Tool")
        .def(py::init<>())
        .def_readwrite("name", &Tool::name)
        .def_readwrite("description", &Tool::description)
        .def_readwrite("params", &Tool::params)
        .def_readwrite("is_async", &Tool::is_async)
        .def_readwrite("is_void", &Tool::is_void);

    // =========================================================================
    // PromptCheckConfig
    // =========================================================================
    py::class_<PromptCheckConfig>(m, "PromptCheckConfig")
        .def(py::init<>())
        .def_readwrite("min_length", &PromptCheckConfig::min_length)
        .def_readwrite("max_length", &PromptCheckConfig::max_length)
        .def_readwrite("max_files", &PromptCheckConfig::max_files)
        .def_readwrite("max_file_bytes", &PromptCheckConfig::max_file_bytes)
        .def_readwrite("allowed_extensions", &PromptCheckConfig::allowed_extensions)
        .def_readwrite("check_duplicate", &PromptCheckConfig::check_duplicate)
        .def_readwrite("check_injection", &PromptCheckConfig::check_injection)
        .def_readwrite("check_pii", &PromptCheckConfig::check_pii)
        .def_readwrite("extra_patterns", &PromptCheckConfig::extra_patterns);

    // =========================================================================
    // Layer
    // =========================================================================
    py::class_<Layer>(m, "Layer")
        .def(py::init<>())
        .def_readwrite("type", &Layer::type)
        .def_readwrite("id", &Layer::id)
        .def_readwrite("target", &Layer::target)
        .def_readwrite("params", &Layer::params)
        .def_readwrite("param_list", &Layer::param_list)
        .def_property("branch_condition",
            [](const Layer& self) -> py::object {
                if (self.branch_condition) return py::cpp_function(self.branch_condition);
                return py::none();
            },
            [](Layer& self, py::object cb) {
                if (cb.is_none()) { self.branch_condition = nullptr; return; }
                py::function fn(cb);
                self.branch_condition = [fn](const Stack* s) -> bool {
                    py::gil_scoped_acquire gil;
                    return fn(s).cast<bool>();
                };
            })
        .def_property("custom_logic",
            [](const Layer& self) -> py::object {
                if (self.custom_logic) return py::cpp_function(self.custom_logic);
                return py::none();
            },
            [](Layer& self, py::object cb) {
                if (cb.is_none()) { self.custom_logic = nullptr; return; }
                py::function fn(cb);
                self.custom_logic = [fn](Stack* s) {
                    py::gil_scoped_acquire gil;
                    fn(s);
                };
            });

    // =========================================================================
    // Stack
    // =========================================================================
    py::class_<Stack>(m, "Stack")
        .def(py::init<>())
        .def_readwrite("stack_id", &Stack::stack_id)
        .def_readwrite("unsafe", &Stack::unsafe)
        .def_readwrite("active_prompt", &Stack::active_prompt)
        .def_readwrite("active_response", &Stack::active_response)
        .def_readwrite("active_status", &Stack::active_status)
        .def_readwrite("active_files", &Stack::active_files)
        .def_readwrite("active_model_id", &Stack::active_model_id)
        .def_readwrite("active_convo_id", &Stack::active_convo_id)
        .def_property("on_token",
            [](const Stack& self) -> py::object {
                if (self.on_token) return py::cpp_function(self.on_token);
                return py::none();
            },
            [](Stack& self, py::object cb) {
                if (cb.is_none()) { self.on_token = nullptr; return; }
                py::function fn(cb);
                self.on_token = [fn](const char* tok, int len) {
                    py::gil_scoped_acquire gil;
                    fn(std::string(tok, static_cast<size_t>(len)), len);
                };
            })
        .def_readwrite("has_tool_calls", &Stack::has_tool_calls)
        .def_readwrite("verbose", &Stack::verbose)
        .def_property("is_busy",
            [](const Stack& self) { return self.is_busy.load(); },
            [](Stack& self, bool v){ self.is_busy.store(v); })
        .def_property_readonly("model", [](const Stack& self) -> py::object {
            if (self.model) return py::cast(self.model, py::return_value_policy::reference);
            return py::none();
        })
        .def_property_readonly("conversation", [](const Stack& self) -> py::object {
            if (self.conversation) return py::cast(self.conversation, py::return_value_policy::reference);
            return py::none();
        })
        .def_property_readonly("settings", [](const Stack& self) -> py::object {
            if (self.settings) return py::cast(self.settings, py::return_value_policy::reference);
            return py::none();
        })
        .def_property_readonly("layers", [](const Stack& self){ return self.layers; })
        .def_property_readonly("tools", [](const Stack& self){ return self.tools; });

    // =========================================================================
    // LangX (global handle)
    // =========================================================================
    py::class_<LangX>(m, "LangX")
        .def(py::init<>())
        .def_readwrite("verbose_logs", &LangX::verbose_logs)
        .def_readwrite("last_refrenced_stack_id", &LangX::last_refrenced_stack_id)
        .def_readonly("loaded_models", &LangX::loaded_models)
        .def_readonly("sleeping_models", &LangX::sleeping_models)
        .def_readonly("hotswap_conversation", &LangX::hotswap_conversation)
        .def_readonly("hotswap_stacks", &LangX::hotswap_stacks);

    // =========================================================================
    // RAG types
    // =========================================================================
    py::enum_<RagFilterMode>(m, "RagFilterMode")
        .value("ALL", RagFilterMode::ALL)
        .value("DOCS", RagFilterMode::DOCS)
        .value("MEMORY", RagFilterMode::MEMORY);

    py::enum_<RagImageMode>(m, "RagImageMode")
        .value("DESCRIBE", RagImageMode::DESCRIBE)
        .value("PASSTHROUGH", RagImageMode::PASSTHROUGH)
        .value("BOTH", RagImageMode::BOTH);

    py::class_<RagChunk>(m, "RagChunk")
        .def(py::init<>())
        .def_readwrite("text", &RagChunk::text)
        .def_readwrite("score", &RagChunk::score)
        .def_readwrite("source", &RagChunk::source)
        .def_readwrite("is_image", &RagChunk::is_image)
        .def_readwrite("image_path", &RagChunk::image_path)
        .def_readwrite("is_memory", &RagChunk::is_memory)
        .def("__repr__", [](const RagChunk& c){
            return "<RagChunk score=" + std::to_string(c.score) + " source='" + c.source + "'>";
        });

    py::class_<RagParams>(m, "RagParams")
        .def(py::init<>())
        .def_readwrite("top_k", &RagParams::top_k)
        .def_readwrite("mmr_lambda", &RagParams::mmr_lambda)
        .def_readwrite("max_chunks_per_source", &RagParams::max_chunks_per_source)
        .def_readwrite("filename_score_weight", &RagParams::filename_score_weight)
        .def_readwrite("chunk_size", &RagParams::chunk_size)
        .def_readwrite("chunk_overlap", &RagParams::chunk_overlap)
        .def_readwrite("embed_ctx_size", &RagParams::embed_ctx_size)
        .def_readwrite("embed_doc_prefix", &RagParams::embed_doc_prefix)
        .def_readwrite("embed_query_prefix", &RagParams::embed_query_prefix)
        .def_readwrite("injection_prefix", &RagParams::injection_prefix)
        .def_readwrite("image_mode", &RagParams::image_mode)
        .def_readwrite("filter_mode", &RagParams::filter_mode)
        .def_readwrite("supported_extensions", &RagParams::supported_extensions)
        .def_readwrite("supported_image_extensions", &RagParams::supported_image_extensions);

    py::class_<PyRagDb>(m, "RagDb");

    // =========================================================================
    // Core API
    // =========================================================================
    m.def("get_default_data_path", [](){ return getDefaultDataPath().string(); });
    m.def("initialize_langX", &initialize_langX, py::arg("config"), py::return_value_policy::reference);
    m.def("inference",
          [](Stack* stack, const char* prompt, std::vector<std::string> files) {
              return ::langX::inference(stack, prompt, files);
          },
          py::arg("stack"), py::arg("prompt"), py::arg("files") = std::vector<std::string>{},
          py::call_guard<py::gil_scoped_release>());
    m.def("inference",
          [](const char* prompt, std::vector<std::string> files) {
              return ::langX::inference(prompt, files);
          },
          py::arg("prompt"), py::arg("files") = std::vector<std::string>{},
          py::call_guard<py::gil_scoped_release>());
    m.def("set_inquery_settings", &setInquerySettings, py::arg("settings"), py::arg("target_stack") = nullptr);
    m.def("random_seed", &randomSeed);
    m.def("count_vision_tokens", &countVisionTokens, py::arg("image_path"), py::arg("model_id") = "");
    m.def("feed_text_file", &feedTextFile, py::arg("file_path"), py::arg("label") = nullptr, py::arg("target_stack") = nullptr);
    m.def("feed_context", &feedContext, py::arg("content"), py::arg("label") = nullptr, py::arg("target_stack") = nullptr);
    m.def("set_system_prompt", &setSystemPrompt, py::arg("prompt"), py::arg("target_stack") = nullptr);
    m.def("set_model_system_prompt", &setModelSystemPrompt, py::arg("prompt"), py::arg("target_stack") = nullptr);

    // --- Conversation management ---
    m.def("init_conversation", &initConversation, py::arg("id") = "", py::arg("target_stack") = nullptr);
    m.def("swap_conversations", &swapConversations, py::arg("conversation_id"), py::arg("target_stack") = nullptr);
    m.def("load_conversations", &loadConversations, py::arg("conversation"), py::arg("id") = "");
    m.def("unload_conversations", &unloadConversations, py::arg("conversation_id"));

    // --- Model management ---
    m.def("init_model", &initModel, py::arg("params"), py::arg("id") = "", py::arg("target_stack") = nullptr, py::call_guard<py::gil_scoped_release>());
    m.def("unload_model", py::overload_cast<Stack*>(&unloadModel), py::arg("target_stack") = nullptr);
    m.def("unload_model", py::overload_cast<const char*>(&unloadModel), py::arg("model_id"));
    m.def("switch_model", &switchModel, py::arg("model_id"), py::arg("target_stack") = nullptr, py::call_guard<py::gil_scoped_release>());
    m.def("sleep_model",  &sleepModel,  py::arg("model_id"));
    m.def("wakeup_model", &wakeupModel, py::arg("model_id"),  py::call_guard<py::gil_scoped_release>());
    m.def("init_sleeping_model", &initSleepingModel, py::arg("params"), py::arg("id"));
    m.def("init_loaded_model", &initLoadedModel, py::arg("params"), py::arg("id"), py::call_guard<py::gil_scoped_release>());

    // --- Stack management ---
    m.def("make_default_stack", &makeDefaultStack, py::arg("stack_id") = "default", py::return_value_policy::reference);
    m.def("make_stack", py::overload_cast<const char*>(&makeStack), py::arg("stack_id"), py::return_value_policy::reference);
    m.def("make_stack", py::overload_cast<const char*, std::vector<Layer*>>(&makeStack), py::arg("stack_id"), py::arg("layers"), py::return_value_policy::reference);
    m.def("swap_stack",   &swapStack,   py::arg("stack_id"));
    m.def("delete_stack", &deleteStack, py::arg("stack_id"));
    
    m.def("append_layer", &appendLayer, py::arg("stack"), py::arg("layer"));
    m.def("insert_layer", &insertLayer, py::arg("stack"), py::arg("idx"), py::arg("layer"));
    m.def("remove_layer", &removeLayer, py::arg("stack"), py::arg("idx"));

    // --- Generic layer factories ---
    m.def("make_layer", &makeLayer, py::arg("type"), py::arg("target") = -1, py::return_value_policy::reference);
    m.def("make_param_layer", &makeParamLayer, py::arg("type"), py::arg("param"), py::return_value_policy::reference);
    m.def("make_goto_layer", &makeGotoLayer, py::arg("target"), py::return_value_policy::reference);
    m.def("make_branch_layer",
          [](py::function condition, int target) -> Layer* {
              return makeBranchLayer(
                  [condition](const Stack* s) -> bool {
                      py::gil_scoped_acquire gil;
                      return condition(s).cast<bool>();
                  }, target);
          },
          py::arg("condition"), py::arg("target"),
          py::return_value_policy::reference);
    m.def("make_custom_layer",
          [](py::function logic) -> Layer* {
              return makeCustomLayer([logic](Stack* s) {
                  py::gil_scoped_acquire gil;
                  logic(s);
              });
          },
          py::arg("logic"),
          py::return_value_policy::reference);

    // --- Initialization & input layers ---
    m.def("make_init_inference_layer", &makeInitInferenceLayer, py::return_value_policy::reference);
    m.def("make_file_processing_layer", &makeFileProcessingLayer, py::return_value_policy::reference);
    m.def("make_user_push_prompt_layer", &makeUserPushPromptLayer, py::return_value_policy::reference);
    m.def("make_user_push_images_layer", &makeUserPushImagesLayer, py::return_value_policy::reference);

    // --- Context building layers ---
    m.def("make_build_context_layer", &makeBuildContextLayer, py::return_value_policy::reference);
    m.def("make_load_chat_template_layer", &makeLoadChatTemplateLayer, py::return_value_policy::reference);
    m.def("make_cxt_win_trim_layer", &makeCxtWinTrimLayer, py::return_value_policy::reference);
    m.def("make_load_system_prompt_layer", &makeLoadSystemPromptLayer, py::return_value_policy::reference);

    // --- Decoding pipeline layers ---
    m.def("make_clear_kv_cache_layer", &makeClearKvCacheLayer, py::return_value_policy::reference);
    m.def("make_init_sampler_layer", &makeInitSamplerLayer, py::return_value_policy::reference);
    m.def("make_init_batch_layer", &makeInitBatchLayer, py::return_value_policy::reference);
    m.def("make_feed_prompt_layer", &makeFeedPromptLayer, py::return_value_policy::reference);
    m.def("make_feed_prompt_images_layer", &makeFeedPromptImagesLayer, py::return_value_policy::reference);
    m.def("make_llm_generation_layer", &makeLlmGenerationLayer, py::return_value_policy::reference);
    m.def("make_llm_sample_layer", &makeLlmSampleLayer, py::return_value_policy::reference);
    m.def("make_free_sampler_layer", &makeFreeSamplerLayer, py::return_value_policy::reference);
    m.def("make_free_batch_layer", &makeFreeBatchLayer, py::return_value_policy::reference);

    // --- Post-generation layers ---
    m.def("make_save_to_history_layer", &makeSaveToHistoryLayer, py::return_value_policy::reference);

    // --- Flow control layers ---
    m.def("make_debug_layer", &makeDebugLayer, py::arg("message"), py::return_value_policy::reference);
    m.def("make_filler_rand_layer", &makeFillerRandLayer, py::arg("strings"), py::return_value_policy::reference);
    m.def("make_filler_loop_layer", &makeFillerLoopLayer, py::arg("strings"), py::return_value_policy::reference);

    // --- Hotswap layers ---
    m.def("make_swap_model_layer", &makeSwapModelLayer, py::arg("model_id"), py::return_value_policy::reference);
    m.def("make_swap_convo_layer", &makeSwapConvoLayer, py::arg("convo_id"), py::return_value_policy::reference);
    m.def("make_set_system_prompt_layer", &makeSetSystemPromptLayer, py::arg("prompt"), py::return_value_policy::reference);

    // --- Tool calling layers ---
    m.def("make_use_tools_layer", &makeUseToolsLayer, py::return_value_policy::reference);
    m.def("make_wait_tools_layer", &makeWaitToolsLayer, py::return_value_policy::reference);
    m.def("make_llm_tool_gen_layer", &makeLlmToolGenLayer,  py::return_value_policy::reference);

    // --- RAG & memory retrieval layers ---
    m.def("make_rag_retrieval_layer", &makeRagRetrievalLayer, py::return_value_policy::reference);
    m.def("make_semantic_mem_retrieval_layer", &makeSemanticMemRetrievalLayer, py::return_value_policy::reference);
    m.def("make_semantic_memory_layer", &makeSemanticMemoryLayer, py::return_value_policy::reference);
    m.def("make_build_semantic_mem_layer", &makeBuildSemanticMemLayer, py::arg("compress_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);

    // --- Episodic memory layers ---
    m.def("make_episodic_memory_layer", &makeEpisodicMemoryLayer, py::return_value_policy::reference);
    m.def("make_episodic_tiered_memory_layer", &makeEpisodicTieredMemoryLayer, py::arg("tier2_compress_prompt") = nullptr, py::arg("tier1_compress_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);
    m.def("make_build_episodic_mem_layer", &makeBuildEpisodicMemLayer, py::arg("tier2_compress_prompt") = nullptr, py::arg("tier1_compress_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);

    // --- Procedural memory layers ---
    m.def("make_load_procedural_mem_layer", &makeLoadProceduralMemLayer, py::arg("file_path"), py::return_value_policy::reference);
    m.def("make_build_procedural_mem_layer", &makeBuildProceduralMemLayer, py::arg("file_path"), py::arg("build_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);

    // --- Async layer ---
    m.def("make_async_break_layer", &makeAsyncBreakLayer, py::return_value_policy::reference);

    // --- Waiting layers ---
    m.def("make_wait_time_layer", &makeWaitTimeLayer, py::arg("duration_ms"), py::return_value_policy::reference);
    m.def("make_wait_lambda_layer",
          [](py::function condition, int timeout_ms, int on_timeout_target, int poll_ms) -> Layer* {
              return makeWaitLambdaLayer(
                  [condition](const Stack* s) -> bool {
                      py::gil_scoped_acquire gil;
                      return condition(s).cast<bool>();
                  }, timeout_ms, on_timeout_target, poll_ms);
          },
          py::arg("condition"), py::arg("timeout_ms") = 0,
          py::arg("on_timeout_target") = -1, py::arg("poll_ms") = 50,
          py::return_value_policy::reference);
    m.def("make_wait_promise_layer", &makeWaitPromiseLayer, py::arg("timeout_ms") = 0, py::arg("on_timeout_target") = -1, py::return_value_policy::reference);
    m.def("signal_stack_wait", &signalStackWait, py::arg("stack"), py::arg("value") = "");

    // --- Filters & safety layers ---
    m.def("make_prompt_check_layer", &makePromptCheckLayer, py::arg("config") = PromptCheckConfig{}, py::return_value_policy::reference);
    m.def("make_llm_prompt_filter_layer", &makeLLMPromptFilterLayer, py::arg("on_fail_target") = -1, py::arg("filter_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::arg("inject_notes") = false, py::arg("notes_label") = "Filter Notes", py::return_value_policy::reference);
    m.def("make_llm_result_filter_layer", &makeLLMResultFilterLayer, py::arg("on_fail_target") = -1, py::arg("filter_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::arg("inject_notes") = false, py::arg("notes_label") = "Filter Notes", py::return_value_policy::reference);
    m.def("make_fact_check_layer", &makeFactCheckLayer, py::arg("on_fail_target") = -1, py::arg("filter_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::arg("inject_notes") = false, py::arg("notes_label") = "Fact Check Notes", py::return_value_policy::reference);

    // --- Sub-inference & compound layers ---
    m.def("make_quick_look_layer", &makeQuickLookLayer, py::arg("instructions") = "", py::arg("image_target") = "", py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);
    m.def("make_quick_ask_layer", &makeQuickAskLayer, py::arg("prompt"), py::arg("save_to_history") = false, py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);
    m.def("make_substack_layer", &makeSubstackLayer, py::arg("stack_id"), py::arg("prompt") = nullptr, py::arg("context_label") = "Substack Result", py::arg("model_id") = "", py::arg("use_sleep") = false, py::arg("save_to_history") = false, py::return_value_policy::reference);

    // --- Reasoning layers ---
    m.def("make_cot_generate_layer", &makeCotGenerateLayer, py::arg("cot_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);
    m.def("make_self_consistent_gen_layer", &makeSelfConsistentGenLayer, py::arg("substack_id"), py::arg("n") = 3, py::arg("scoring_prompt") = nullptr, py::arg("model_id") = "", py::arg("use_sleep") = false, py::return_value_policy::reference);

    // --- Specialized generation layers ---
    m.def("make_bert_generate_layer", &makeBertGenerateLayer, py::arg("model_id") = "",  py::return_value_policy::reference);
    m.def("make_nli_generate_layer", &makeNliGenerateLayer, py::arg("model_id") = "", py::arg("hypothesis") = nullptr, py::return_value_policy::reference);

    // =========================================================================
    // Tool management
    // =========================================================================
    m.def("make_tool",
          [](const char* name, const char* description,
             py::function handler, bool is_async) -> Tool* {
              return makeTool(name, description,
                  [handler](std::map<std::string,std::string> args) -> std::string {
                      py::gil_scoped_acquire gil;
                      return handler(args).cast<std::string>();
                  }, is_async);
          },
          py::arg("name"), py::arg("description"), py::arg("handler"),
          py::arg("is_async") = false,
          py::return_value_policy::reference);

    m.def("make_void_tool",
          [](const char* name, const char* description,
             py::function handler, bool is_async) -> Tool* {
              return makeVoidTool(name, description,
                  [handler](std::map<std::string,std::string> args) {
                      py::gil_scoped_acquire gil;
                      handler(args);
                  }, is_async);
          },
          py::arg("name"), py::arg("description"), py::arg("handler"),
          py::arg("is_async") = false,
          py::return_value_policy::reference);

    m.def("add_tool_param", &addToolParam, py::arg("tool"), py::arg("name"), py::arg("description"), py::arg("type") = "string", py::arg("required") = true);
    m.def("register_tool", &registerTool, py::arg("stack"), py::arg("tool"));
    m.def("get_tools_system_prompt", &getToolsSystemPrompt, py::arg("stack"));
    m.def("get_global_langX", [](){ return &global_LangX; }, py::return_value_policy::reference);

    // =========================================================================
    // RAG functions (unwrap PyRagDb → RagDb* for each call)
    // =========================================================================
    m.def("make_rag_db", [](){ return PyRagDb(makeRagDb()); });
    m.def("free_rag_db", [](PyRagDb& w){ freeRagDb(w.ptr); w.ptr = nullptr; }, py::arg("db"));
    m.def("rag_init_embeddings",
          [](PyRagDb& w, const char* path, int gl) {
              py::gil_scoped_release rel;
              return ragInitEmbeddings(w.ptr, path, gl);
          }, py::arg("db"), py::arg("embed_model_path"), py::arg("n_gpu_layers") = 99);
    m.def("rag_init_vlm",
          [](PyRagDb& w, const char* mp, const char* mm, int eb, int gl) {
              py::gil_scoped_release rel;
              return ragInitVlm(w.ptr, mp, mm, eb, gl);
          }, py::arg("db"), py::arg("model_path"), py::arg("mmproj_path"),
          py::arg("encoder_batch") = 16384, py::arg("n_gpu_layers") = 99);
    m.def("rag_free_vlm",         [](PyRagDb& w){ ragFreeVlm(w.ptr); },         py::arg("db"));
    m.def("rag_set_vlm_settings", [](PyRagDb& w, InquerySettings s){ ragSetVlmSettings(w.ptr, s); }, py::arg("db"), py::arg("settings"));
    m.def("save_rag_db", [](PyRagDb& w, const char* p){ return saveRagDb(w.ptr, p); }, py::arg("db"), py::arg("path"));
    m.def("load_rag_db", [](PyRagDb& w, const char* p){ return loadRagDb(w.ptr, p); }, py::arg("db"), py::arg("path"));
    m.def("rag_add_text", [](PyRagDb& w, const char* t, const char* s){ ragAddText(w.ptr, t, s); }, py::arg("db"), py::arg("text"), py::arg("source") = nullptr);
    m.def("rag_add_file",
          [](PyRagDb& w, const char* p) {
              py::gil_scoped_release rel;
              ragAddFile(w.ptr, p);
          }, py::arg("db"), py::arg("path"));
    m.def("rag_add_directory",
          [](PyRagDb& w, const char* d, bool r) {
              py::gil_scoped_release rel;
              ragAddDirectory(w.ptr, d, r);
          }, py::arg("db"), py::arg("dir_path"), py::arg("recursive") = false);
    m.def("rag_add_supported_extension",
          [](PyRagDb& w, const char* e){ ragAddSupportedExtension(w.ptr, e); },
          py::arg("db"), py::arg("ext"));
    m.def("rag_remove_supported_extension",
          [](PyRagDb& w, const char* e){ ragRemoveSupportedExtension(w.ptr, e); },
          py::arg("db"), py::arg("ext"));
    m.def("rag_add_image",
          [](PyRagDb& w, const char* ip, const char* desc) {
              py::gil_scoped_release rel;
              ragAddImage(w.ptr, ip, desc);
          }, py::arg("db"), py::arg("image_path"), py::arg("description") = nullptr);
    m.def("rag_add_memory", [](PyRagDb& w, const char* t, const char* s){ ragAddMemory(w.ptr, t, s); }, py::arg("db"), py::arg("text"), py::arg("source") = nullptr);
    m.def("rag_search", [](PyRagDb& w, const char* q, int k){ return ragSearch(w.ptr, q, k); }, py::arg("db"), py::arg("query"), py::arg("top_k") = 3);
    m.def("attach_rag_db", [](Stack* s, PyRagDb& w){ attachRagDb(s, w.ptr); }, py::arg("stack"), py::arg("db"));
    m.def("set_rag_db_params", [](PyRagDb& w, const RagParams& p){ setRagDbParams(w.ptr, p); }, py::arg("db"), py::arg("params"));
    m.def("rag_set_verbose", [](PyRagDb& w, bool v){ ragSetVerbose(w.ptr, v); }, py::arg("db"), py::arg("verbose"));
    m.def("set_rag_params", &setRagParams, py::arg("stack"), py::arg("params"));
}
