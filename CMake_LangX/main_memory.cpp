#include "langX.h"
#include "LxRAG.h"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <regex>
#include <fstream>
#include <windows.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
using Sec   = std::chrono::duration<double>;

// ---- Paths (laptop) ------------------------------------------------------------------
// static const fs::path MODELS_DIR    = "C:/RTU/LangX/llama_langx/models";
// static const fs::path GEN_MODEL     = MODELS_DIR / "Llama-3.2-3B-Instruct-Q4_0.gguf";
// static const fs::path EMBED_MODEL   = MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf";
// static const fs::path VLM_MODEL     = MODELS_DIR / "Qwen2.5-VL-7B-Instruct-Q3_K_M.gguf";
// static const fs::path VLM_MMPROJ    = MODELS_DIR / "mmproj-F16.gguf";

// static const fs::path TEST_DIR      = "C:/RTU/LangX/llama_langx/test";
// static const fs::path DATA_DIR      = TEST_DIR / "data";
// static const fs::path DB_PATH       = TEST_DIR / "memory_test.lxrag";
// -----------------------------------------------------------------------------

// ---- Paths (PC) ------------------------------------------------------------------
static const fs::path MODELS_DIR    = "D:/School/RTU/LangX/llama_langx/models";
static const fs::path GEN_MODEL     = MODELS_DIR / "Llama-3.2-3B-Instruct-Q8_0.gguf";
static const fs::path EMBED_MODEL   = MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf";
static const fs::path VLM_MODEL     = MODELS_DIR / "Qwen_Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf";
static const fs::path VLM_MMPROJ    = MODELS_DIR / "mmproj-Qwen_Qwen2.5-VL-7B-Instruct-f16.gguf";

static const fs::path TEST_DIR      = "D:/School/RTU/LangX/llama_langx/test";
static const fs::path DATA_DIR      = TEST_DIR / "data";
static const fs::path ROBOT2_IMG    = TEST_DIR / "robot2.jpg";
static const fs::path DB_PATH       = TEST_DIR / "memory_test.lxrag";
static const fs::path PROC_MEM_PATH = TEST_DIR / "proc_memory.txt";
// -----------------------------------------------------------------------------

static const auto TOKEN_CB = [](const char* piece, int len) {
    std::cout.write(piece, len);
    std::cout.flush();
};

struct TestQuery {
    const char* prompt;
    const char* pattern;
    const char* label;
};

static bool check_response(const std::string& response, const char* pattern) {
    std::regex re(pattern, std::regex_constants::icase);
    return std::regex_search(response, re);
}

int main() {
    SetConsoleOutputCP(CP_UTF8);

    if (fs::exists(DB_PATH)) fs::remove(DB_PATH);
    if (fs::exists(PROC_MEM_PATH)) fs::remove(PROC_MEM_PATH);

    // --- LangX init ---
    langX::Config cfg{ (fs::current_path() / "langX_out").string() };
    langX::initialize_langX(cfg);

    langX::ModelParams gen_model_params{ GEN_MODEL.string() };
    gen_model_params.n_ctx = 4096;
    langX::InquerySettings gen_settings;
    gen_settings.seed = langX::randomSeed();
    gen_settings.n_tokens_to_predict = 512;
    gen_settings.temperature = 0.7f;

    // =========================================================================
    // PASHE 1 — Prepare the RAG database
    // =========================================================================
    std::cout << "\n\n##################################################\n";
    std::cout << "  PASHE 1: Prepare RAG Database\n";
    std::cout << "##################################################\n";

    langX::RagParams rag_params;
    rag_params.chunk_size = 400;
    rag_params.chunk_overlap = 50;
    rag_params.top_k = 5;
    rag_params.filename_score_weight = 0.2f;
    rag_params.injection_prefix = "[Knowledge Base]\n";
    rag_params.image_mode = langX::RagImageMode::DESCRIBE;
    rag_params.embed_doc_prefix = "search_document: ";
    rag_params.embed_query_prefix = "search_query: ";
    rag_params.filter_mode = langX::RagFilterMode::DOCS;

    langX::RagDb* ragdb = langX::makeRagDb();
    langX::setRagDbParams(ragdb, rag_params);
    langX::ragSetVerbose(ragdb, true);

    if (!langX::ragInitEmbeddings(ragdb, EMBED_MODEL.string().c_str())) {
        std::cout << "Test ended early.";
        return 1;
    }

    bool vlm_loaded = langX::ragInitVlm(ragdb, VLM_MODEL.string().c_str(), VLM_MMPROJ.string().c_str(), 65536);
    if (!vlm_loaded) {
        std::cout << "[PASHE 1] VLM failed to load — images will use filename fallback.\n";
    }

    std::cout << "\n[Building RAG database...]\n";
    auto t1_build_start = Clock::now();

    std::cout << "[RAG - loading data directory...]\n";
    langX::ragAddDirectory(ragdb, DATA_DIR.string().c_str(), false);

    std::cout << "[RAG - adding robot2 with manual description...]\n";
    langX::ragAddImage(ragdb, ROBOT2_IMG.string().c_str(),
        "Concept art of a large bipedal combat mech called 'Scarab Azul' from "
        "'Project Chitin' by Mechanical Blue. The mech is predominantly blue and white "
        "with gray mechanical details and small red accents. It stands 10.3 meters tall, "
        "has a speed of 120 KPH, and is armed with internal gatling guns x2, a core rail "
        "gun, cluster missiles x200, mounted blades x2, and 2 option slots. It carries "
        "two large rectangular shield-like modules on its back marked with alpha and omega "
        "symbols. A pilot is shown beside it — a young person with small horns wearing a "
        "white and blue uniform.");

    if (vlm_loaded) langX::ragFreeVlm(ragdb);

    langX::saveRagDb(ragdb, DB_PATH.string().c_str());
    double t1_build = Sec(Clock::now() - t1_build_start).count();
    std::cout << "\n[PASHE 1 RAG build time: " << t1_build << "s]\n";

    auto t1_model_start = Clock::now();
    langX::initModel(gen_model_params);
    langX::setInquerySettings(gen_settings);
    double t1_model = Sec(Clock::now() - t1_model_start).count();
    std::cout << "[PASHE 1 model load time: " << t1_model << "s]\n";

    // =========================================================================
    // PASHE 2 — small RAG test
    // =========================================================================
    std::cout << "\n\n##################################################\n";
    std::cout << "  PASHE 2: RAG Retrieval Test\n";
    std::cout << "##################################################\n";

    langX::setModelSystemPrompt(
        "You are a helpful assistant with access to a knowledge base. "
        "Use the provided context to answer questions accurately and specifically. "
        "When referencing data, cite exact values from the context."
    );
    langX::initConversation();

    langX::Stack* s2 = langX::makeStack("rag_test", {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLayer(langX::RAG_RETRIEVAL),
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
    langX::attachRagDb(s2, ragdb);
    langX::setRagParams(s2, rag_params);
    s2->on_token = TOKEN_CB;
    s2->verbose = true;
    langX::swapStack("rag_test");

    TestQuery rag_queries[] = {
        { "What is Marta Kalnina's employee ID and what department does she work in?", "EMP-4821", "user lookup" },
        { "What is the domain name of the Vortex Solutions website and what is their registry number?", "vortexsolutions\\.lv","html domain" },
        { "Who is the leader of the Embersmiths faction and where do they live?", "Dunric", "lore faction" },
        { "What exact string does the secret_message.py Python script output when run?", "Zephyr.Delta.9.Kilo", "py output" },
        { "What combat robots or mechs do we have images of in our database? List their names.", "Scarab.Azul", "robot recall" },
    };
    int n_rag_queries = std::size(rag_queries);
    int p2_pass = 0;

    auto t2_start = Clock::now();
    for (int i = 0; i < n_rag_queries; i++) {
        std::cout << "\n--- P2 Q" << (i + 1) << " [" << rag_queries[i].label << "]: " << rag_queries[i].prompt << " ---\n";

        auto tq = Clock::now();
        std::string resp = langX::inference(rag_queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();

        bool hit = check_response(resp, rag_queries[i].pattern);
        if (hit) p2_pass++;

        std::cout << "\n[" << (hit ? "PASS" : "FAIL") << " | pattern: " << rag_queries[i].pattern << " | " << tq_s << "s]\n";
    }

    double t2_total = Sec(Clock::now() - t2_start).count();
    std::cout << "\n[PASHE 2 score: " << p2_pass << "/" << n_rag_queries << " | total time: " << t2_total << "s]\n";

    bool p2_rag_ok = false;
    {
        auto check = langX::ragSearch(ragdb, "employee department", 1);
        p2_rag_ok = !check.empty();
        std::cout << "[VERIFY RAG: " << (p2_rag_ok ? "PASS" : "FAIL") << " | " << check.size() << " chunk(s) found]\n";
    }

    // =========================================================================
    // PASHE 3 — episodic memory
    // =========================================================================
    std::cout << "\n\n##################################################\n";
    std::cout << "  PASHE 3: Episodic Memory Test\n";
    std::cout << "##################################################\n";

    langX::setModelSystemPrompt(
        "You are a friendly assistant. Remember details the user tells you and recall them accurately when asked. Be concise in your responses."
    );
    langX::initConversation();

    langX::Stack* s3 = langX::makeStack("episodic_test", {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLayer(langX::USER_PUSH_PROMT),
        langX::makeEpisodicMemoryLayer(),
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
        langX::makeBuildEpisodicMemLayer(),
    });
    s3->on_token = TOKEN_CB;
    s3->verbose = true;
    langX::swapStack("episodic_test");
    gen_settings.episodic_context_ratio = 0.05f;
    gen_settings.episodic_tier2_ratio   = 0.05f;
    gen_settings.episodic_tier1_ratio   = 0.15f;
    langX::setInquerySettings(gen_settings);

    TestQuery episodic_queries[] = {
        { "My name is Viktor and I work as a lighthouse keeper on Siren's Rock island. Remember that!", "Viktor|lighthouse|Siren", "plant name+job" },
        { "I have a pet raven named Obsidian who can speak exactly 12 words. Pretty cool, right?", "Obsidian|raven|12", "plant pet" },
        { "The lighthouse on Siren's Rock is over 200 years old. There's a ghost named Helena who appears during storms — the other keepers have all seen her too.", "Helena|ghost|200|storm", "plant ghost" },
        { "What's my name and where do I work?", "Viktor|lighthouse", "recall name+job" },
        { "Last week I found a sealed bottle on the shore containing a map to something called the Amber Vault.", "bottle|Amber.Vault|map", "plant bottle" },
        { "Do you remember my pet? Also, have I told you about the ghost at the lighthouse?", "Obsidian|Helena", "recall pet+ghost" },
        { "I decoded part of the map — the Amber Vault is hidden beneath the old windmill in Greymoor.", "Greymoor|windmill|Amber.Vault", "plant vault location" },
        { "A rival treasure hunter named Cassandra Drake arrived yesterday on a black-hulled ship called the Nightcurrent. She's after the Amber Vault too.", "Cassandra|Drake|Nightcurrent", "plant rival" },
        { "What did I find on the shore and where is the vault hidden?", "Amber.Vault|Greymoor|bottle", "recall bottle+vault" },
        { "Obsidian has started repeating a strange phrase: 'beneath the stone where tides don't reach'. That's 7 of his 12 words — I think it's a clue.", "beneath.the.stone|tides", "plant raven clue" },
        { "Tell me about my rival — what's her name and her ship?", "Cassandra|Nightcurrent", "recall rival" },
        { "Summarize everything you know about me — my name, job, pet, the ghost, and all my discoveries.", "Viktor[\\s\\S]*Obsidian|Obsidian[\\s\\S]*Viktor", "full recall" },
    };
    int n_episodic_queries = std::size(episodic_queries);
    int p3_pass = 0;

    auto t3_start = Clock::now();
    for (int i = 0; i < n_episodic_queries; i++) {
        std::cout << "\n--- P3 Q" << (i + 1) << " [" << episodic_queries[i].label << "]: " << episodic_queries[i].prompt << " ---\n";

        auto tq = Clock::now();
        std::string resp = langX::inference(episodic_queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();

        bool hit = check_response(resp, episodic_queries[i].pattern);
        if (hit) p3_pass++;

        std::cout << "\n[" << (hit ? "PASS" : "FAIL") << " | pattern: " << episodic_queries[i].pattern << " | " << tq_s << "s]\n";
    }

    double t3_total = Sec(Clock::now() - t3_start).count();
    std::cout << "\n[PASHE 3 score: " << p3_pass << "/" << n_episodic_queries << " | total time: " << t3_total << "s]\n";

    bool p3_episodic_ok = false;
    {
        auto* convo = s3->conversation;
        bool has_t2 = !convo->episodic_tier2_memories.empty();
        bool has_t1 = !convo->episodic_tier1_summary.empty();
        p3_episodic_ok = has_t2 && has_t1;
        std::cout << "[VERIFY EPISODIC: " << (p3_episodic_ok ? "PASS" : "FAIL")
                  << " | T2=" << convo->episodic_tier2_memories.size() << " entries, T1="
                  << (has_t1 ? std::to_string(convo->episodic_tier1_summary.size()) + " chars" : std::string("empty")) << "]\n";
    }

    {
        langX::Conversation* convo = s3->conversation;
        std::ofstream dump((TEST_DIR / "p3_episodic_dump.txt").string());
        dump << "=== EPISODIC MEMORY DUMP (Phase 3) ===\n\n";

        dump << "--- T1 Summary ---\n";
        dump << (convo->episodic_tier1_summary.empty() ? "(empty)" : convo->episodic_tier1_summary) << "\n\n";

        dump << "--- T2 Memories (" << convo->episodic_tier2_memories.size() << ") ---\n";
        for (size_t i = 0; i < convo->episodic_tier2_memories.size(); i++) {
            dump << "[" << i << "] " << convo->episodic_tier2_memories[i] << "\n";
        }

        dump << "\n--- T3 Active Messages (" << convo->active_messages.size() << ") ---\n";
        for (auto& msg : convo->active_messages) {
            dump << "[" << msg.role << "] " << msg.content << "\n";
        }

        dump << "\n--- Compress Index: " << convo->episodic_compress_idx << " ---\n";
        dump.close();
        std::cout << "[Episodic memory state saved to p3_episodic_dump.txt]\n";
    }

    // =========================================================================
    // PASHE 4 — semantic memory
    // =========================================================================
    std::cout << "\n\n##################################################\n";
    std::cout << "  PASHE 4: Semantic Memory Test\n";
    std::cout << "##################################################\n";

    gen_settings.n_tokens_to_predict = 3850;
    gen_settings.episodic_context_ratio = 1.0f;
    gen_settings.episodic_tier2_ratio   = 0.0f;
    gen_settings.episodic_tier1_ratio   = 0.0f;
    langX::setInquerySettings(gen_settings);

    langX::setModelSystemPrompt(
        "You are a helpful assistant. Remember details the user tells you and recall them accurately when asked. Be concise in your responses."
    );
    langX::initConversation();

    langX::Stack* s4 = langX::makeStack("semantic_test", {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLayer(langX::SEMANTIC_MEM_RETRIEVAL),
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
        langX::makeBuildSemanticMemLayer(),
    });
    langX::attachRagDb(s4, ragdb);
    langX::setRagParams(s4, rag_params);
    s4->on_token = TOKEN_CB;
    s4->verbose = true;
    langX::swapStack("semantic_test");

    TestQuery semantic_queries[] = {
        { "I need to tell you about my colleague Elena Vasquez. She's a marine biologist who has spent eight years studying bioluminescent deep-sea organisms in the western Pacific. Her research vessel is a modified submersible called the 'Deep Sapphire', operated by the Oceanic Research Institute in Yokohama. Remember these details.", "Elena|Deep.Sapphire|Yokohama", "plant researcher" },
        { "Elena made a groundbreaking discovery last month — a previously unknown jellyfish species at 8200 meters depth near the Challenger Deep. It produces blue-green bioluminescence at exactly 475 nanometers. She named it Aurelia profunda and submitted her paper to Nature Marine Biology.", "Aurelia|profunda|475", "plant discovery" },
        { "What is Elena's research vessel called and what organization operates it?", "Deep.Sapphire", "recall vessel" },
        { "Elena's research partner is Dr. Kenji Nakamura from Kyoto University, an expert in hydrothermal vent chemistry. Their joint research is funded by a three-year grant from the Poseidon Foundation, grant number PF-7741, totaling 2.4 million euros.", "Kenji|PF-7741|Poseidon", "plant partner" },
        { "What species did Elena discover and what is the grant number funding their research?", "Aurelia.*profunda|PF-7741", "recall discovery+grant" },
    };
    int n_semantic_queries = std::size(semantic_queries);
    int p4_pass = 0;

    auto t4_start = Clock::now();
    for (int i = 0; i < n_semantic_queries; i++) {
        std::cout << "\n--- P4 Q" << (i + 1) << " [" << semantic_queries[i].label << "]: " << semantic_queries[i].prompt << " ---\n";

        auto tq = Clock::now();
        std::string resp = langX::inference(semantic_queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();

        bool hit = check_response(resp, semantic_queries[i].pattern);
        if (hit) p4_pass++;

        std::cout << "\n[" << (hit ? "PASS" : "FAIL") << " | pattern: " << semantic_queries[i].pattern << " | " << tq_s << "s]\n";
    }

    double t4_total = Sec(Clock::now() - t4_start).count();
    std::cout << "\n[PASHE 4 score: " << p4_pass << "/" << n_semantic_queries << " | total time: " << t4_total << "s]\n";

    bool p4_semantic_ok = false;
    {
        auto prev_f = rag_params.filter_mode;
        rag_params.filter_mode = langX::RagFilterMode::MEMORY;
        langX::setRagDbParams(ragdb, rag_params);
        auto check = langX::ragSearch(ragdb, "Elena research vessel", 1);
        rag_params.filter_mode = prev_f;
        langX::setRagDbParams(ragdb, rag_params);
        p4_semantic_ok = !check.empty();
        std::cout << "[VERIFY SEMANTIC: " << (p4_semantic_ok ? "PASS" : "FAIL") << " | " << check.size() << " memory chunk(s) in RAG]\n";
    }

    // Dump semantic memory state
    {
        langX::Conversation* convo = s4->conversation;
        std::ofstream dump((TEST_DIR / "p4_semantic_dump.txt").string());
        dump << "=== SEMANTIC MEMORY DUMP (Phase 4) ===\n\n";
        dump << "Full history:     " << convo->messages.size() << " messages\n";
        dump << "Active window:    " << convo->active_messages.size() << " messages\n";
        dump << "Compress index:   " << convo->semantic_compress_idx << "\n\n";

        dump << "--- RAG memory chunks (search: 'Elena research vessel discovery') ---\n";
        auto prev_filter = rag_params.filter_mode;
        rag_params.filter_mode = langX::RagFilterMode::MEMORY;
        langX::setRagDbParams(ragdb, rag_params);
        auto mem_chunks = langX::ragSearch(ragdb, "Elena research vessel discovery jellyfish", 10);
        rag_params.filter_mode = prev_filter;
        langX::setRagDbParams(ragdb, rag_params);

        for (size_t i = 0; i < mem_chunks.size(); i++) {
            dump << "\n[Memory " << i << " | score: " << mem_chunks[i].score << " | source: " << mem_chunks[i].source << "]\n";
            dump << mem_chunks[i].text << "\n";
        }
        dump.close();
        std::cout << "[Semantic memory state saved to p4_semantic_dump.txt]\n";
    }

    gen_settings.n_tokens_to_predict = 512;
    langX::setInquerySettings(gen_settings);

    // =========================================================================
    // PASHE 5 — mixed (episodic + semantic + RAG)
    // =========================================================================
    std::cout << "\n\n##################################################\n";
    std::cout << "  PASHE 5: Mixed Memory + RAG Test\n";
    std::cout << "##################################################\n";

    langX::setModelSystemPrompt(
        "You are a helpful assistant with access to a knowledge base and conversation memory. "
        "Use all available context to answer questions accurately. Be concise."
    );
    langX::initConversation();

    langX::Stack* s5 = langX::makeStack("mixed_test", {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLayer(langX::RAG_RETRIEVAL),
        langX::makeLayer(langX::SEMANTIC_MEM_RETRIEVAL),
        langX::makeLayer(langX::USER_PUSH_PROMT),
        langX::makeEpisodicMemoryLayer(),
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
        langX::makeBuildEpisodicMemLayer(),
        langX::makeBuildSemanticMemLayer(),
    });
    langX::attachRagDb(s5, ragdb);
    langX::setRagParams(s5, rag_params);
    s5->on_token = TOKEN_CB;
    s5->verbose = true;
    langX::swapStack("mixed_test");
    gen_settings.episodic_context_ratio = 0.15f;
    gen_settings.episodic_tier2_ratio   = 0.05f;
    gen_settings.episodic_tier1_ratio   = 0.15f;
    langX::setInquerySettings(gen_settings);

    TestQuery mixed_queries[] = {
        { "I'm a security researcher at CyberNova with clearance code SIGMA-9. I report directly to Director Hayes. Remember this.", "SIGMA-9|CyberNova|Hayes", "plant identity" },
        { "What department does Tomass Berzins work in and what is his employee ID?", "EMP-5037", "rag employee" },
        { "Yesterday I discovered a critical vulnerability in the VortexFlow event bus — a deserialization flaw in the message serializer, tracked as CVE-2026-4417.", "CVE-2026-4417|VortexFlow|serializ", "plant vuln" },
        { "What is my clearance code? Also, who designed the Vortex Solutions website?", "SIGMA-9", "recall+rag identity" },
        { "What is the Heart of Kaelthar according to the Ashenveil chronicles?", "gemstone|dead.god|magma", "rag lore" },
        { "What vulnerability did I discover and what's the CVE number?", "CVE-2026-4417", "recall vuln" },
        { "What is Annika Tamm's employee ID and what was her career before tech?", "EMP-4402", "rag employee2" },
        { "Summarize everything about me — my role, clearance, who I report to, and the vulnerability I found.", "SIGMA-9[\\s\\S]*CVE-2026-4417|CVE-2026-4417[\\s\\S]*SIGMA-9", "full mixed recall" },
    };
    int n_mixed_queries = std::size(mixed_queries);
    int p5_pass = 0;

    auto t5_start = Clock::now();
    for (int i = 0; i < n_mixed_queries; i++) {
        std::cout << "\n--- P5 Q" << (i + 1) << " [" << mixed_queries[i].label << "]: " << mixed_queries[i].prompt << " ---\n";

        auto tq = Clock::now();
        std::string resp = langX::inference(mixed_queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();

        bool hit = check_response(resp, mixed_queries[i].pattern);
        if (hit) p5_pass++;

        std::cout << "\n[" << (hit ? "PASS" : "FAIL") << " | pattern: " << mixed_queries[i].pattern << " | " << tq_s << "s]\n";
    }

    double t5_total = Sec(Clock::now() - t5_start).count();
    std::cout << "\n[PASHE 5 score: " << p5_pass << "/" << n_mixed_queries << " | total time: " << t5_total << "s]\n";

    bool p5_rag_ok = false, p5_episodic_t2_ok = false, p5_episodic_t1_ok = false, p5_semantic_ok = false;
    {
        auto* convo = s5->conversation;
        auto prev_f = rag_params.filter_mode;
        rag_params.filter_mode = langX::RagFilterMode::DOCS;
        langX::setRagDbParams(ragdb, rag_params);
        auto doc_check = langX::ragSearch(ragdb, "employee department", 1);
        p5_rag_ok = !doc_check.empty();
        rag_params.filter_mode = langX::RagFilterMode::MEMORY;
        langX::setRagDbParams(ragdb, rag_params);
        auto mem_check = langX::ragSearch(ragdb, "security clearance SIGMA", 1);
        p5_semantic_ok = !mem_check.empty();
        rag_params.filter_mode = prev_f;
        langX::setRagDbParams(ragdb, rag_params);
        p5_episodic_t2_ok = !convo->episodic_tier2_memories.empty();
        p5_episodic_t1_ok = !convo->episodic_tier1_summary.empty();
        int p5_v = (int)p5_rag_ok + (int)p5_episodic_t2_ok + (int)p5_episodic_t1_ok + (int)p5_semantic_ok;
        std::cout << "[VERIFY RAG: " << (p5_rag_ok ? "PASS" : "FAIL")
                  << " | EPISODIC_T2: " << (p5_episodic_t2_ok ? "PASS" : "FAIL")
                  << " (" << convo->episodic_tier2_memories.size() << " entries)"
                  << " | EPISODIC_T1: " << (p5_episodic_t1_ok ? "PASS" : "FAIL")
                  << " (" << convo->episodic_tier1_summary.size() << " chars)"
                  << " | SEMANTIC: " << (p5_semantic_ok ? "PASS" : "FAIL")
                  << " — " << p5_v << "/4]\n";
    }

    {
        langX::Conversation* convo = s5->conversation;
        std::ofstream dump((TEST_DIR / "p5_mixed_dump.txt").string());
        dump << "=== MIXED MEMORY DUMP (Phase 5) ===\n\n";

        dump << "--- Episodic ---\n";
        dump << "T1 Summary: " << (convo->episodic_tier1_summary.empty() ? "(empty)" : convo->episodic_tier1_summary) << "\n";
        dump << "T2 Memories (" << convo->episodic_tier2_memories.size() << "):\n";
        for (size_t i = 0; i < convo->episodic_tier2_memories.size(); i++)
            dump << "  [" << i << "] " << convo->episodic_tier2_memories[i] << "\n";

        dump << "\n--- Semantic ---\n";
        dump << "Full history:   " << convo->messages.size() << " messages\n";
        dump << "Active window:  " << convo->active_messages.size() << " messages\n";
        dump << "Episodic idx:   " << convo->episodic_compress_idx << "\n";
        dump << "Semantic idx:   " << convo->semantic_compress_idx << "\n";

        dump << "\n--- RAG memory chunks (search: 'security clearance vulnerability') ---\n";
        auto prev_filter = rag_params.filter_mode;
        rag_params.filter_mode = langX::RagFilterMode::MEMORY;
        langX::setRagDbParams(ragdb, rag_params);
        auto mem_chunks = langX::ragSearch(ragdb, "security clearance vulnerability CyberNova", 10);
        rag_params.filter_mode = prev_filter;
        langX::setRagDbParams(ragdb, rag_params);

        for (size_t i = 0; i < mem_chunks.size(); i++) {
            dump << "\n[Memory " << i << " | score: " << mem_chunks[i].score << " | source: " << mem_chunks[i].source << "]\n";
            dump << mem_chunks[i].text << "\n";
        }

        dump << "\n--- T3 Active Messages (" << convo->active_messages.size() << ") ---\n";
        for (auto& msg : convo->active_messages)
            dump << "[" << msg.role << "] " << msg.content << "\n";

        dump.close();
        std::cout << "[Mixed memory state saved to p5_mixed_dump.txt]\n";
    }

    // =========================================================================
    // PASHE 6 — Procedural Memory (Core Memories)
    // =========================================================================
    std::cout << "\n\n##################################################\n";
    std::cout << "  PASHE 6: Procedural Memory Test\n";
    std::cout << "##################################################\n";

    langX::setModelSystemPrompt(
        "You are a friendly assistant. Remember details the user tells you and recall them accurately when asked. Be concise in your responses."
    );
    langX::initConversation();

    std::string proc_path = PROC_MEM_PATH.string();

    langX::Stack* s6 = langX::makeStack("procedural_test", {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLoadProceduralMemLayer(proc_path.c_str()),
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
        langX::makeBuildProceduralMemLayer(proc_path.c_str()),
    });
    s6->on_token = TOKEN_CB;
    s6->verbose = true;
    langX::swapStack("procedural_test");

    // Phase 6a: plant facts across several exchanges
    TestQuery proc_plant_queries[] = {
        { "My name is Adris and I'm a blacksmith in the city of Ironhaven. I've been forging weapons for 15 years. Remember that!", "Adris|blacksmith|Ironhaven|15", "plant identity" },
        { "My best creation is a legendary sword called Dawnbreaker, forged from meteorite iron. It took me 3 months to complete.", "Dawnbreaker|meteorite|3.month", "plant creation" },
        { "I have an apprentice named Lira who is exceptionally talented with silver alloys. She's been with me for 2 years.", "Lira|silver|apprentice|2.year", "plant apprentice" },
    };
    int n_proc_plant = std::size(proc_plant_queries);
    int p6_pass = 0;

    auto t6_start = Clock::now();
    for (int i = 0; i < n_proc_plant; i++) {
        std::cout << "\n--- P6a Q" << (i + 1) << " [" << proc_plant_queries[i].label << "]: " << proc_plant_queries[i].prompt << " ---\n";
        auto tq = Clock::now();
        std::string resp = langX::inference(proc_plant_queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();
        bool hit = check_response(resp, proc_plant_queries[i].pattern);
        if (hit) p6_pass++;
        std::cout << "\n[" << (hit ? "PASS" : "FAIL") << " | pattern: " << proc_plant_queries[i].pattern << " | " << tq_s << "s]\n";
    }

    // Phase 6b: start a FRESH conversation — procedural memory must survive via file
    std::cout << "\n--- Resetting conversation (procedural memory must persist via file) ---\n";
    langX::initConversation();

    TestQuery proc_recall_queries[] = {
        { "Do you know who I am and what I do?", "Adris|blacksmith|Ironhaven", "recall identity" },
        { "What is the name of my best creation and what is it made from?", "Dawnbreaker|meteorite", "recall creation" },
        { "Tell me about my apprentice.", "Lira|silver", "recall apprentice" },
    };
    int n_proc_recall = std::size(proc_recall_queries);

    for (int i = 0; i < n_proc_recall; i++) {
        std::cout << "\n--- P6b Q" << (i + 1) << " [" << proc_recall_queries[i].label << "]: " << proc_recall_queries[i].prompt << " ---\n";
        auto tq = Clock::now();
        std::string resp = langX::inference(proc_recall_queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();
        bool hit = check_response(resp, proc_recall_queries[i].pattern);
        if (hit) p6_pass++;
        std::cout << "\n[" << (hit ? "PASS" : "FAIL") << " | pattern: " << proc_recall_queries[i].pattern << " | " << tq_s << "s]\n";
    }

    int n_proc_queries = n_proc_plant + n_proc_recall;
    double t6_total = Sec(Clock::now() - t6_start).count();
    std::cout << "\n[PASHE 6 score: " << p6_pass << "/" << n_proc_queries << " | total time: " << t6_total << "s]\n";

    bool p6_file_ok = false, p6_content_ok = false;
    {
        p6_file_ok = fs::exists(PROC_MEM_PATH);
        if (p6_file_ok) {
            std::ifstream in(PROC_MEM_PATH);
            std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            p6_content_ok = content.find("Adris") != std::string::npos || content.find("blacksmith") != std::string::npos;
        }
        std::cout << "[VERIFY PROCEDURAL: file=" << (p6_file_ok ? "PASS" : "FAIL") << " content=" << (p6_content_ok ? "PASS" : "FAIL") << "]\n";
    }

    // Dump procedural memory state
    {
        std::ofstream dump((TEST_DIR / "p6_procedural_dump.txt").string());
        dump << "=== PROCEDURAL MEMORY DUMP (Phase 6) ===\n\n";
        dump << "--- File content (" << PROC_MEM_PATH.string() << ") ---\n";
        if (fs::exists(PROC_MEM_PATH)) {
            std::ifstream in(PROC_MEM_PATH);
            dump << std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        } else {
            dump << "(file not found)";
        }
        dump << "\n\n--- conv->procedural_memory ---\n";
        dump << s6->conversation->procedural_memory << "\n";
        dump.close();
        std::cout << "[Procedural memory state saved to p6_procedural_dump.txt]\n";
    }

    // =========================================================================
    // PASHE 7 — All Memory Types (episodic + semantic + procedural + RAG)
    // =========================================================================
    std::cout << "\n\n##################################################\n";
    std::cout << "  PASHE 7: All Memory Types Combined\n";
    std::cout << "##################################################\n";

    if (fs::exists(PROC_MEM_PATH)) fs::remove(PROC_MEM_PATH);

    langX::setModelSystemPrompt(
        "You are a helpful assistant with access to a knowledge base and conversation memory. "
        "Use all available context to answer questions accurately. Be concise."
    );
    langX::initConversation();

    std::string proc_path7 = (TEST_DIR / "proc_memory_p7.txt").string();
    if (fs::exists(proc_path7)) fs::remove(proc_path7);

    langX::Stack* s7 = langX::makeStack("all_memory_test", {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLoadProceduralMemLayer(proc_path7.c_str()),
        langX::makeLayer(langX::RAG_RETRIEVAL),
        langX::makeLayer(langX::SEMANTIC_MEM_RETRIEVAL),
        langX::makeLayer(langX::USER_PUSH_PROMT),
        langX::makeEpisodicMemoryLayer(),
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
        langX::makeBuildEpisodicMemLayer(),
        langX::makeBuildSemanticMemLayer(),
        langX::makeBuildProceduralMemLayer(proc_path7.c_str()),
    });
    langX::attachRagDb(s7, ragdb);
    langX::setRagParams(s7, rag_params);
    s7->on_token = TOKEN_CB;
    s7->verbose = true;
    langX::swapStack("all_memory_test");
    gen_settings.episodic_context_ratio = 0.15f;
    gen_settings.episodic_tier2_ratio   = 0.1f;
    gen_settings.episodic_tier1_ratio   = 0.15f;
    langX::setInquerySettings(gen_settings);

    TestQuery all_mem_queries[] = {
        { "My codename is Phantom and I operate from Station Omega-12, a deep-sea research platform. My handler's call sign is Warden. Remember all of this.", "Phantom|Omega.12|Warden", "plant identity" },
        { "What is the main product that Vortex Solutions offers?", "event.bus|VortexFlow|event.driven", "rag product" },
        { "Last night my sensors detected an anomalous energy pulse from a prototype reactor called AXIOM-7 in Sector 8443. The pulse was caused by a modified fusion cascade sequence. This is top priority.", "AXIOM.7|8443|fusion|cascade", "plant anomaly" },
        { "What is Marta Kalnina's employee ID?", "EMP-4821", "rag employee" },
        { "What is my codename and where do I operate from?", "Phantom|Omega.12", "recall identity (procedural)" },
        { "What anomaly did I detect and what sector was it in?", "AXIOM.7|8443", "recall anomaly" },
        { "Tell me about the Embersmiths faction from the Ashenveil chronicles. Also, what is my handler's call sign?", "Embersmiths|Warden", "rag+procedural combo" },
        { "Summarize everything you know about me — my codename, station, handler, and the anomaly I detected.", "Phantom[\\s\\S]*AXIOM|AXIOM[\\s\\S]*Phantom", "full combined recall" },
    };
    int n_all_queries = std::size(all_mem_queries);
    int p7_pass = 0;

    auto t7_start = Clock::now();
    for (int i = 0; i < n_all_queries; i++) {
        std::cout << "\n--- P7 Q" << (i + 1) << " [" << all_mem_queries[i].label << "]: " << all_mem_queries[i].prompt << " ---\n";
        auto tq = Clock::now();
        std::string resp = langX::inference(all_mem_queries[i].prompt);
        double tq_s = Sec(Clock::now() - tq).count();
        bool hit = check_response(resp, all_mem_queries[i].pattern);
        if (hit) p7_pass++;
        std::cout << "\n[" << (hit ? "PASS" : "FAIL") << " | pattern: " << all_mem_queries[i].pattern << " | " << tq_s << "s]\n";
    }

    double t7_total = Sec(Clock::now() - t7_start).count();
    std::cout << "\n[PASHE 7 score: " << p7_pass << "/" << n_all_queries << " | total time: " << t7_total << "s]\n";

    bool p7_rag_ok = false, p7_episodic_ok = false, p7_semantic_ok = false, p7_procedural_ok = false;
    {
        auto* convo = s7->conversation;
        auto prev_f = rag_params.filter_mode;
        rag_params.filter_mode = langX::RagFilterMode::DOCS;
        langX::setRagDbParams(ragdb, rag_params);
        auto doc_check = langX::ragSearch(ragdb, "employee department", 1);
        p7_rag_ok = !doc_check.empty();
        rag_params.filter_mode = langX::RagFilterMode::MEMORY;
        langX::setRagDbParams(ragdb, rag_params);
        auto mem_check = langX::ragSearch(ragdb, "codename Phantom station", 1);
        p7_semantic_ok = !mem_check.empty();
        rag_params.filter_mode = prev_f;
        langX::setRagDbParams(ragdb, rag_params);
        p7_episodic_ok = !convo->episodic_tier2_memories.empty() || !convo->episodic_tier1_summary.empty();
        if (fs::exists(proc_path7)) {
            std::ifstream pin(proc_path7);
            std::string pcontent((std::istreambuf_iterator<char>(pin)), std::istreambuf_iterator<char>());
            p7_procedural_ok = pcontent.find("Phantom") != std::string::npos || pcontent.find("Omega") != std::string::npos || pcontent.find("Warden") != std::string::npos || pcontent.find("AXIOM") != std::string::npos;
        }
        int p7_v = (int)p7_rag_ok + (int)p7_episodic_ok + (int)p7_semantic_ok + (int)p7_procedural_ok;
        std::cout << "[VERIFY RAG: " << (p7_rag_ok ? "PASS" : "FAIL")
                  << " | EPISODIC: " << (p7_episodic_ok ? "PASS" : "FAIL")
                  << " | SEMANTIC: " << (p7_semantic_ok ? "PASS" : "FAIL")
                  << " | PROCEDURAL: " << (p7_procedural_ok ? "PASS" : "FAIL")
                  << " — " << p7_v << "/4]\n";
    }

    {
        langX::Conversation* convo = s7->conversation;
        std::ofstream dump((TEST_DIR / "p7_all_memory_dump.txt").string());
        dump << "=== ALL MEMORY DUMP (Phase 7) ===\n\n";

        dump << "--- Procedural ---\n";
        dump << (convo->procedural_memory.empty() ? "(empty)" : convo->procedural_memory) << "\n";

        dump << "\n--- Episodic ---\n";
        dump << "T1 Summary: " << (convo->episodic_tier1_summary.empty() ? "(empty)" : convo->episodic_tier1_summary) << "\n";
        dump << "T2 Memories (" << convo->episodic_tier2_memories.size() << "):\n";
        for (size_t i = 0; i < convo->episodic_tier2_memories.size(); i++)
            dump << "  [" << i << "] " << convo->episodic_tier2_memories[i] << "\n";

        dump << "\n--- Semantic ---\n";
        dump << "Full history:   " << convo->messages.size() << " messages\n";
        dump << "Active window:  " << convo->active_messages.size() << " messages\n";
        dump << "Episodic idx:   " << convo->episodic_compress_idx << "\n";
        dump << "Semantic idx:   " << convo->semantic_compress_idx << "\n";

        dump << "\n--- T3 Active Messages (" << convo->active_messages.size() << ") ---\n";
        for (auto& msg : convo->active_messages)
            dump << "[" << msg.role << "] " << msg.content << "\n";

        dump.close();
        std::cout << "[All memory state saved to p7_all_memory_dump.txt]\n";
    }

    // =========================================================================
    // END — tally
    // =========================================================================
    double t_total = Sec(Clock::now() - t1_build_start).count();

    std::cout << "\n\n##################################################\n";
    std::cout << "  RESULTS\n";
    std::cout << "##################################################\n\n";
    int p5_v = (int)p5_rag_ok + (int)p5_episodic_t2_ok + (int)p5_episodic_t1_ok + (int)p5_semantic_ok;
    int p6_v = (int)p6_file_ok + (int)p6_content_ok;
    int p7_v = (int)p7_rag_ok + (int)p7_episodic_ok + (int)p7_semantic_ok + (int)p7_procedural_ok;
    int total_verified = (int)p2_rag_ok + (int)p3_episodic_ok + (int)p4_semantic_ok + p5_v + p6_v + p7_v;

    std::cout << "  Phase | Test                  | Score | Verify | Time (s)\n";
    std::cout << "  ------|-----------------------|-------|--------|----------\n";
    printf("  P1    | RAG build + model     |  ---  |  ---   | %7.1f\n", t1_build + t1_model);
    printf("  P2    | RAG retrieval         | %2d/%-2d |  %d/1   | %7.1f\n", p2_pass, n_rag_queries, (int)p2_rag_ok, t2_total);
    printf("  P3    | Episodic memory       | %2d/%-2d |  %d/1   | %7.1f\n", p3_pass, n_episodic_queries, (int)p3_episodic_ok, t3_total);
    printf("  P4    | Semantic memory       | %2d/%-2d |  %d/1   | %7.1f\n", p4_pass, n_semantic_queries, (int)p4_semantic_ok, t4_total);
    printf("  P5    | Mixed (epi+sem+rag)   | %2d/%-2d |  %d/4   | %7.1f\n", p5_pass, n_mixed_queries, p5_v, t5_total);
    printf("  P6    | Procedural memory     | %2d/%-2d |  %d/2   | %7.1f\n", p6_pass, n_proc_queries, p6_v, t6_total);
    printf("  P7    | All memory types      | %2d/%-2d |  %d/4   | %7.1f\n", p7_pass, n_all_queries, p7_v, t7_total);
    std::cout << "  ------|-----------------------|-------|--------|----------\n";
    printf("  TOTAL |                       | %2d/%-2d | %2d/13  | %7.1f\n", p2_pass + p3_pass + p4_pass + p5_pass + p6_pass + p7_pass,  n_rag_queries + n_episodic_queries + n_semantic_queries + n_mixed_queries + n_proc_queries + n_all_queries, total_verified, t_total);
    std::cout << "\n";

    // ---- Cleanup ------------------------------------------------------------
    langX::freeRagDb(ragdb);
    langX::unloadModel();
    return 0;
}
