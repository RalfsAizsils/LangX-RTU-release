#pragma once
#ifdef _WIN32
	#ifdef BUILDING_MY_DLL
		#define LANGX_API __declspec(dllexport)
	#else
		#define LANGX_API __declspec(dllimport)
	#endif
#else
	#define LANGX_API __attribute__((visibility("default")))
#endif

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <future>
#include <memory>
#include <atomic>

struct llama_model;
struct llama_context;
struct llama_model_params;
struct llama_context_params;
struct llama_adapter_lora;
struct mtmd_context;
typedef int32_t llama_token;

namespace langX {
	struct InferenceState;
	struct Stack;
	struct RagDb;

	LANGX_API struct Config {
		std::string user_output_path;
		bool verbose_logs = false;
	};

	LANGX_API struct InquerySettings {
		int n_tokens_to_predict = 500;
		int batch_size = 512;
		int n_sequences = 1;
		float temperature = 0.8f;
		int top_k = 40;
		float top_p = 0.95f;
		int seed = 1234;
		std::string grammar;
		float episodic_context_ratio = 1.0f;
		float episodic_tier2_ratio = 0.0f;
		float episodic_tier1_ratio = 0.0f;
		bool use_native_tools = false;
	};

	LANGX_API struct ModelParams {
		std::filesystem::path path;
		std::filesystem::path vission_path = "";
		std::filesystem::path lora_path = "";
		float lora_scale = 1.0f;
		int n_gpu_layers = 99;
		int n_ctx = 2048;
		int n_batch = 0;
		int n_ubatch = 0;
		int n_threads = 4;
		int n_threads_batch = 4;
		int main_gpu = 0;
		std::vector<float> tensor_split;

		ModelParams() = default;
		ModelParams(const char* p, const char* vp = "") : path(p), vission_path(vp) {}
		ModelParams(const std::string& p, const std::string& vp = "") : path(p), vission_path(vp) {}
	};

	LANGX_API struct Model {
		ModelParams* config_reference = nullptr;
		llama_model_params* params = nullptr;
		llama_context_params* ctx_params = nullptr;
		llama_model* model = nullptr;
		llama_context* ctx = nullptr;
		const char* chat_template = nullptr;
		mtmd_context* mtmd_ctx = nullptr;
		llama_adapter_lora* lora_adapter = nullptr;
		std::string system_prompt;
	};

	LANGX_API struct ChatMessage {
		std::string role;
		std::string content;
	};

	LANGX_API struct ContextExtra {
		std::string label;
		std::string content;
		float score = 0.0f;
	};

	LANGX_API struct Conversation {
		int n_past = 0;
		std::vector<ChatMessage> messages;
		std::vector<ChatMessage> active_messages;
		std::vector<ContextExtra> extra_data;
		std::string system_prompt;
		std::string procedural_memory;
		std::string episodic_tier1_summary;
		std::vector<std::string> episodic_tier2_memories;
		int episodic_compress_idx = 0;
		int semantic_compress_idx = 0;
	};

	LANGX_API enum LayerType {
		INIT_INFERENCE,
		FILE_PROCESSING,
		USER_PUSH_PROMT,
		USER_PUSH_IMAGES,
		BUILD_CONTEXT,
		LOAD_CHAT_TEMPLATE,
		CXT_WIN_TRIM,
		CLEAR_KV_CACHE,
		INIT_SAMPLER,
		INIT_BATCH,
		FEED_PROMPT,
		FEED_PROMPT_IMAGES,
		LLM_GENERATION,
		LLM_SAMPLE,
		FREE_SAMPLER,
		FREE_BATCH,
		SAVE_TO_HISTORY,
		GOTO,
		BRANCH,
		CUSTOM,
		DEBUG,
		SWAP_MODEL,
		SWAP_CONVO,
		SET_SYSTEM_PROMPT,
		LOAD_SYSTEM_PROMPT,
		USE_TOOLS,
		WAIT_TOOLS,
		LLM_TOOL_GEN,
		RAG_RETRIEVAL,
		FILLER_RAND,
		FILLER_LOOP,
		QUICK_LOOK,
		PROMPT_CHECK,
		QUICK_ASK,
		SUBSTACK,
		WAIT_TIME,
		WAIT_LAMBDA,
		WAIT_PROMISE,
		EPISODIC_MEMORY,
		SEMANTIC_MEMORY,
		SEMANTIC_MEM_RETRIEVAL,
		EPISODIC_TIERED_MEMORY,
		LLM_PROMPT_FILTER,
		LLM_RESULT_FILTER,
		FACT_CHECK,
		BUILD_EPISODIC_MEM,
		BUILD_SEMANTIC_MEM,
		ASYNC_BREAK,
		LOAD_PROCEDURAL_MEM,
		BUILD_PROCEDURAL_MEM,
		BERT_GENERATE,
		NLI_GENERATE,
		COT_GENERATE,
		SELF_CONSISTENT_GEN,
	};

	LANGX_API struct ToolParam {
		std::string name;
		std::string description;
		std::string type = "string"; // "string", "number", "bool"
		bool required = true;
	};

	LANGX_API struct Tool {
		std::string name;
		std::string description;
		std::vector<ToolParam> params;
		bool is_async = false;
		bool is_void = false;
		std::function<std::string(std::map<std::string, std::string>)> handler;
	};

	LANGX_API struct Layer {
		LayerType type;
		int id = -1;
		int target = -1;
		std::map<std::string, std::string> params;
		std::vector<std::string> param_list;
		mutable int _loop_index = 0;
		std::function<bool(const Stack*)> branch_condition;
		std::function<void(Stack*)> custom_logic;
	};

	LANGX_API struct Stack {
		std::vector<Layer*> layers;
		std::string stack_id = "";
		bool unsafe = false;
		std::string active_prompt;
		std::string active_response;
		std::string active_status;
		std::vector<std::string> active_files;
		std::string active_model_id = "";
		std::string active_convo_id = "";
		Model* model = nullptr;
		Conversation* conversation = nullptr;
		InquerySettings* settings = nullptr;
		std::function<void(const char*, int)> on_token;
		std::vector<Tool*> tools;
		bool has_tool_calls = false;
		RagDb* rag_db = nullptr;
		bool verbose = false;
		std::shared_ptr<std::promise<std::string>> wait_promise;
		std::atomic<bool> is_busy{false};
		InferenceState* inference_state = nullptr;
	};

	LANGX_API struct LangX {
		Config* config = nullptr;
		bool verbose_logs = false;
		std::string last_refrenced_stack_id = "";
		std::map<std::string, Model*> loaded_models;
		std::map<std::string, ModelParams> sleeping_models;
		std::map<std::string, Conversation*> hotswap_conversation;
		std::map<std::string, Stack*> hotswap_stacks;
	};

	LANGX_API struct PromptCheckConfig {
		int min_length = 1;
		int max_length = 0;
		int max_files = 0;
		long long max_file_bytes = 0;
		std::vector<std::string> allowed_extensions;
		bool check_duplicate = false;
		bool check_injection = false;
		bool check_pii = false;
		std::vector<std::string> extra_patterns;
	};

	extern LANGX_API LangX global_LangX;

	LANGX_API std::filesystem::path getDefaultDataPath();
	LANGX_API LangX* initialize_langX(Config config);
	LANGX_API std::string inference(Stack* stack, const char* prompt, const std::vector<std::string>& files = {});
	LANGX_API std::string inference(const char* prompt, const std::vector<std::string>& files = {});

	LANGX_API void setInquerySettings(InquerySettings settings, Stack* target_stack = nullptr);
	LANGX_API int randomSeed();
	LANGX_API size_t countVisionTokens(const char* image_path, const char* model_id = "");
	LANGX_API void feedTextFile(const char* filePath, const char* label = nullptr, Stack* target_stack = nullptr);
	LANGX_API void feedContext(const char* content, const char* label = nullptr, Stack* target_stack = nullptr);
	LANGX_API void setSystemPrompt(const char* prompt, Stack* target_stack = nullptr);
	LANGX_API void setModelSystemPrompt(const char* prompt, Stack* target_stack = nullptr);

	LANGX_API void initConversation(const char* id = "", Stack* target_stack = nullptr);
	LANGX_API void swapConversations(const char* conversation_id, Stack* target_stack = nullptr);
	LANGX_API void loadConversations(Conversation* conversation, const char* id = "");
	LANGX_API void unloadConversations(const char* conversation_id);

	LANGX_API void initModel(ModelParams params, const char* id = "", Stack* target_stack = nullptr);
	LANGX_API void unloadModel(Stack* target_stack = nullptr);
	LANGX_API void unloadModel(const char* model_id);
	LANGX_API void switchModel(const char* model_id, Stack* target_stack = nullptr);
	LANGX_API void sleepModel(const char* model_id);
	LANGX_API void wakeupModel(const char* model_id);
	LANGX_API void initSleepingModel(ModelParams params, const char* id);
	LANGX_API void initLoadedModel(ModelParams params, const char* id);

	LANGX_API Stack* makeDefaultStack(const char* stack_id = "default");
	LANGX_API Stack* makeStack(const char* stack_id);
	LANGX_API Stack* makeStack(const char* stack_id, std::vector<Layer*> layers);
	LANGX_API void swapStack(const char* stack_id);
	LANGX_API void deleteStack(const char* stack_id);
	LANGX_API void appendLayer(Stack* stack, Layer* layer);
	LANGX_API void insertLayer(Stack* stack, int idx, Layer* layer);
	LANGX_API void removeLayer(Stack* stack, int idx);
	LANGX_API Layer* makeLayer(LayerType type, int target = -1);
	LANGX_API Layer* makeParamLayer(LayerType type, const char* param);
	LANGX_API Layer* makeGotoLayer(int target);
	LANGX_API Layer* makeBranchLayer(std::function<bool(const Stack*)> condition, int target);
	LANGX_API Layer* makeCustomLayer(std::function<void(Stack*)> logic);
	
	LANGX_API Layer* makeInitInferenceLayer();
	LANGX_API Layer* makeFileProcessingLayer();
	LANGX_API Layer* makeUserPushPromptLayer();
	LANGX_API Layer* makeUserPushImagesLayer();
	LANGX_API Layer* makeBuildContextLayer();
	LANGX_API Layer* makeLoadChatTemplateLayer();
	LANGX_API Layer* makeCxtWinTrimLayer();
	LANGX_API Layer* makeLoadSystemPromptLayer();
	LANGX_API Layer* makeClearKvCacheLayer();
	LANGX_API Layer* makeInitSamplerLayer();
	LANGX_API Layer* makeInitBatchLayer();
	LANGX_API Layer* makeFeedPromptLayer();
	LANGX_API Layer* makeFeedPromptImagesLayer();
	LANGX_API Layer* makeLlmGenerationLayer();
	LANGX_API Layer* makeLlmSampleLayer();
	LANGX_API Layer* makeFreeSamplerLayer();
	LANGX_API Layer* makeFreeBatchLayer();
	LANGX_API Layer* makeSaveToHistoryLayer();
	LANGX_API Layer* makeDebugLayer(const char* message);
	LANGX_API Layer* makeFillerRandLayer(std::vector<std::string> strings);
	LANGX_API Layer* makeFillerLoopLayer(std::vector<std::string> strings);
	LANGX_API Layer* makeSwapModelLayer(const char* model_id);
	LANGX_API Layer* makeSwapConvoLayer(const char* convo_id);
	LANGX_API Layer* makeSetSystemPromptLayer(const char* prompt);
	LANGX_API Layer* makeUseToolsLayer();
	LANGX_API Layer* makeWaitToolsLayer();
	LANGX_API Layer* makeLlmToolGenLayer();
	LANGX_API Layer* makeRagRetrievalLayer();
	LANGX_API Layer* makeSemanticMemRetrievalLayer();
	LANGX_API Layer* makeSemanticMemoryLayer();
	LANGX_API Layer* makeEpisodicMemoryLayer();
	LANGX_API Layer* makeEpisodicTieredMemoryLayer(const char* tier2_compress_prompt = nullptr, const char* tier1_compress_prompt = nullptr, const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeBuildEpisodicMemLayer(const char* tier2_compress_prompt = nullptr, const char* tier1_compress_prompt = nullptr, const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeBuildSemanticMemLayer(const char* compress_prompt = nullptr, const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeLoadProceduralMemLayer(const char* file_path);
	LANGX_API Layer* makeBuildProceduralMemLayer(const char* file_path, const char* build_prompt = nullptr, const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeAsyncBreakLayer();
	LANGX_API Layer* makeWaitTimeLayer(int duration_ms);
	LANGX_API Layer* makeWaitLambdaLayer(std::function<bool(const Stack*)> condition, int timeout_ms = 0, int on_timeout_target = -1, int poll_ms = 50);
	LANGX_API Layer* makeWaitPromiseLayer(int timeout_ms = 0, int on_timeout_target = -1);
	LANGX_API void signalStackWait(Stack* stack, const char* value = "");
	LANGX_API Layer* makePromptCheckLayer(PromptCheckConfig config = {});
	LANGX_API Layer* makeLLMPromptFilterLayer(int on_fail_target = -1, const char* filter_prompt = nullptr, const char* model_id = "", bool use_sleep = false, bool inject_notes = false, const char* notes_label = "Filter Notes");
	LANGX_API Layer* makeLLMResultFilterLayer(int on_fail_target = -1, const char* filter_prompt = nullptr, const char* model_id = "", bool use_sleep = false, bool inject_notes = false, const char* notes_label = "Filter Notes");
	LANGX_API Layer* makeFactCheckLayer(int on_fail_target = -1, const char* filter_prompt = nullptr, const char* model_id = "", bool use_sleep = false, bool inject_notes = false, const char* notes_label = "Fact Check Notes");
	LANGX_API Layer* makeQuickLookLayer(const char* instructions = "", const char* image_target = "", const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeQuickAskLayer(const char* prompt, bool save_to_history = false, const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeSubstackLayer(const char* stack_id, const char* prompt = nullptr, const char* context_label = "Substack Result", const char* model_id = "", bool use_sleep = false, bool save_to_history = false);
	LANGX_API Layer* makeCotGenerateLayer(const char* cot_prompt = nullptr, const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeSelfConsistentGenLayer(const char* substack_id, int n = 3, const char* scoring_prompt = nullptr, const char* model_id = "", bool use_sleep = false);
	LANGX_API Layer* makeBertGenerateLayer(const char* model_id = "");
	LANGX_API Layer* makeNliGenerateLayer(const char* model_id = "", const char* hypothesis = nullptr);
	LANGX_API Tool* makeTool(const char* name, const char* description, std::function<std::string(std::map<std::string, std::string>)> handler, bool is_async = false);
	LANGX_API Tool* makeVoidTool(const char* name, const char* description, std::function<void(std::map<std::string, std::string>)> handler, bool is_async = false);
	LANGX_API void addToolParam(Tool* tool, const char* name, const char* description, const char* type = "string", bool required = true);
	LANGX_API void registerTool(Stack* stack, Tool* tool);
	LANGX_API std::string getToolsSystemPrompt(const Stack* stack);

}
