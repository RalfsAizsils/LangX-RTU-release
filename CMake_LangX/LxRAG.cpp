#include "LxRAG.h"
#include "LxPDF.h"
#include "langX_layers.h"
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;

// --- RagDb internals ---

struct langX::RagDb {
    struct Chunk {
        std::string text;
        std::string source;
        std::vector<float> embedding;
        bool is_image   = false;
        std::string image_path;
        bool is_memory  = false;
    };

    std::vector<Chunk> chunks;
    int embedding_dim = 0;
    llama_model*   embed_model = nullptr;
    llama_context* embed_ctx   = nullptr;

    std::string vlm_model_id = "";
    langX::InquerySettings   vlm_settings = []() {
        langX::InquerySettings s;
        s.temperature = 0.0f;
        s.n_tokens_to_predict = 256;
        return s;
    }();

    langX::RagParams params;

    bool verbose = false;
};

// --- Helpers ---
static std::vector<std::string> chunk_text_tokens(const llama_vocab* vocab, const std::string& text, int chunk_size, int chunk_overlap) {
    std::vector<std::string> chunks;
    if (text.empty() || !vocab) return chunks;

    std::vector<llama_token> tokens;
    tokens.resize(text.size() + 16);
    int n = llama_tokenize(vocab, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), false, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int)text.size(), tokens.data(), (int)tokens.size(), false, false);
    }
    if (n <= 0) return chunks;
    tokens.resize(n);

    const int step = std::max(1, chunk_size - chunk_overlap);
    char piece_buf[256];
    for (int start = 0; start < n; start += step) {
        int end = std::min(n, start + chunk_size);
        std::string chunk;
        for (int i = start; i < end; i++) {
            int len = llama_token_to_piece(vocab, tokens[i], piece_buf, sizeof(piece_buf), 0, false);
            if (len > 0) chunk.append(piece_buf, len);
        }
        if (!chunk.empty()) chunks.push_back(std::move(chunk));
        if (end >= n) break;
    }
    return chunks;
}

static std::vector<std::string> tokenize_words(const std::string& s) {
    std::vector<std::string> words;
    std::string cur;
    for (char c : s) {
        if (std::isalnum((unsigned char)c)) {
            cur += std::tolower((unsigned char)c);
        } else if (!cur.empty()) {
            words.push_back(cur);
            cur.clear();
        }
    }
    if (!cur.empty()) words.push_back(cur);
    return words;
}

static float filename_keyword_score(const std::string& query, const std::string& source) {
    auto q_words = tokenize_words(query);
    if (q_words.empty()) return 0.0f;
    std::filesystem::path p(source);
    auto fn_words = tokenize_words(p.stem().string());
    if (fn_words.empty()) return 0.0f;
    std::unordered_set<std::string> fn_set(fn_words.begin(), fn_words.end());
    int hits = 0;
    for (const auto& w : q_words)
        if (fn_set.count(w)) hits++;
    return (float)hits / (float)q_words.size();
}

static float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-8f ? dot / denom : 0.0f;
}

static std::vector<float> compute_embedding(langX::RagDb* db, const std::string& text, bool is_query = false) {
    if (!db->embed_model || !db->embed_ctx) {
        std::cerr << "LxRAG Error [CE-00]: No embedding model loaded. Call ragInitEmbeddings first.\n";
        return {};
    }
    const llama_vocab* vocab = llama_model_get_vocab(db->embed_model);

    const std::string& prefix = is_query ? db->params.embed_query_prefix : db->params.embed_doc_prefix;
    std::string input = prefix.empty() ? text : (prefix + text);

    std::vector<llama_token> tokens;
    tokens.resize(input.size() + 16);
    // TODO: change to LangX tokenize
    int n = llama_tokenize(vocab, input.c_str(), (int)input.size(), tokens.data(), (int)tokens.size(), true, false);
    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, input.c_str(), (int)input.size(), tokens.data(), (int)tokens.size(), true, false);
    }
    if (n <= 0) {
        std::cerr << "LxRAG Error [CE-01]: Tokenization failed for embedding.\n";
        return {};
    }
    tokens.resize(n);

    const int n_ubatch = (int)llama_n_ubatch(db->embed_ctx);
    if (n > n_ubatch) {
        if (db->verbose)
            std::cerr << "LxRAG Warning [CE-00]: " << n << " tokens exceeds n_ubatch=" << n_ubatch << ", truncating (reduce chunk_size).\n";
        n = n_ubatch;
        tokens.resize(n);
    }

    const int dim = llama_model_n_embd(db->embed_model);

    llama_memory_t memory = llama_get_memory(db->embed_ctx);
    llama_memory_seq_rm(memory, 0, 0, -1);
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1;
    }
    batch.n_tokens = n;

    if (llama_decode(db->embed_ctx, batch) != 0) {
        std::cerr << "LxRAG Error [CE-02]: llama_decode failed during embedding.\n";
        llama_batch_free(batch);
        return {};
    }

    const float* emb = llama_get_embeddings_seq(db->embed_ctx, 0);
    if (!emb) emb = llama_get_embeddings_ith(db->embed_ctx, n - 1);
    if (!emb) {
        std::cerr << "LxRAG Error [CE-03]: Could not retrieve embedding from model.\n";
        llama_batch_free(batch);
        return {};
    }

    std::vector<float> result(emb, emb + dim);
    llama_batch_free(batch);
    return result;
}

// --- Public API implementation ---

langX::RagDb* langX::makeRagDb() {
    return new langX::RagDb();
}

void langX::freeRagDb(langX::RagDb* db) {
    if (!db) return;
    if (db->embed_ctx)   { llama_free(db->embed_ctx);        db->embed_ctx   = nullptr; }
    if (db->embed_model) { llama_model_free(db->embed_model); db->embed_model = nullptr; }
    langX::ragFreeVlm(db);
    delete db;
}

bool langX::ragInitEmbeddings(langX::RagDb* db, const char* embed_model_path, int n_gpu_layers) {
    if (!db) { 
        std::cerr << "LxRAG Error [RAG-IE-00]: Missing RAG database.\n";
        return false; 
    }
    if (!fs::exists(embed_model_path)) {
        std::cerr << "LxRAG Error [RAG-IE-01]: Embedding model not found: " << embed_model_path << "\n";
        return false;
    }

    if (db->embed_ctx)   { llama_free(db->embed_ctx);        db->embed_ctx   = nullptr; }
    if (db->embed_model) { llama_model_free(db->embed_model); db->embed_model = nullptr; }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = n_gpu_layers;
    db->embed_model = llama_model_load_from_file(embed_model_path, mparams);
    if (!db->embed_model) {
        std::cerr << "LxRAG Error [RAG-IE-02]: Failed to load embedding model: " << embed_model_path << "\n";
        return false;
    }

    int n_embd = llama_model_n_embd(db->embed_model);
    int n_ctx  = db->params.embed_ctx_size;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = n_ctx;
    cparams.n_batch = n_ctx;
    cparams.n_ubatch = n_ctx;
    cparams.embeddings = true;
    cparams.pooling_type = LLAMA_POOLING_TYPE_MEAN;

    db->embed_ctx = llama_new_context_with_model(db->embed_model, cparams);
    if (!db->embed_ctx) {
        std::cerr << "LxRAG Error [RAG-IE-03]: Failed to create embedding context.\n";
        llama_model_free(db->embed_model);
        db->embed_model = nullptr;
        return false;
    }

    db->embedding_dim = n_embd;
    if (db->verbose) std::cout << "LxRAG Log [RAG-IE]: Embedding model loaded. dim=" << n_embd << "\n";
    return true;
}

// --- VLM for image description ---

static const std::string RAG_VLM_ID    = "_rag_vlm_";
static const std::string RAG_VLM_STACK = "_rag_vlm_stack_";

void langX::ragFreeVlm(langX::RagDb* db) {
    if (!db || db->vlm_model_id.empty()) return;
    langX::unloadModel(db->vlm_model_id.c_str());
    db->vlm_model_id = "";
}

void langX::ragSetVlmSettings(langX::RagDb* db, langX::InquerySettings settings) {
    if (db) db->vlm_settings = settings;
}

bool langX::ragInitVlm(langX::RagDb* db, const char* model_path, const char* mmproj_path, int encoder_batch, int n_gpu_layers) {
    if (!db || !model_path || !mmproj_path) {
        std::cerr << "LxRAG Error [VLM-00]: Missing parameters.\n";
        return false;
    }
    if (!fs::exists(model_path)) {
        std::cerr << "LxRAG Error [VLM-01]: VLM model not found: " << model_path << "\n";
        return false;
    }
    if (!fs::exists(mmproj_path)) {
        std::cerr << "LxRAG Error [VLM-02]: mmproj not found: " << mmproj_path << "\n";
        return false;
    }

    ragFreeVlm(db);

    bool loaded = false;
    for (int batch = encoder_batch; batch >= 256; batch /= 2) {
        langX::ModelParams params(model_path, mmproj_path);
        params.n_gpu_layers = n_gpu_layers;
        params.n_ctx = params.n_batch = params.n_ubatch = batch;
        langX::initLoadedModel(params, RAG_VLM_ID.c_str());

        if (langX::global_LangX.loaded_models.find(RAG_VLM_ID) != langX::global_LangX.loaded_models.end()) {
            if (batch != encoder_batch && db->verbose)
                std::cout << "LxRAG Log [VLM-FB]: Loaded with reduced encoder_batch=" << batch << " (requested " << encoder_batch << ")\n";
            loaded = true;
            break;
        }
        if (db->verbose)
            std::cerr << "LxRAG Warning [VLM-03]: Failed with encoder_batch=" << batch << ", retrying with " << batch / 2 << "...\n";
    }
    if (!loaded) {
        std::cerr << "LxRAG Error [VLM-03]: Failed to load RAG VLM after all retries.\n";
        return false;
    }
    db->vlm_model_id = RAG_VLM_ID;
    auto vlm_it = langX::global_LangX.loaded_models.find(RAG_VLM_ID);
    if (vlm_it != langX::global_LangX.loaded_models.end() && vlm_it->second && vlm_it->second->ctx) {
        uint32_t actual_ubatch = llama_n_ubatch(vlm_it->second->ctx);
        if (db->verbose) std::cout << "LxRAG Log [VLM-00]: RAG VLM loaded. n_ubatch=" << actual_ubatch << "\n";
    } else {
        if (db->verbose) std::cout << "LxRAG Log [VLM-01]: RAG VLM loaded for automatic image description.\n";
    }
    return true;
}

static std::string rag_describe_image(langX::RagDb* db, const std::string& image_path) {
    auto it = langX::global_LangX.loaded_models.find(db->vlm_model_id);
    if (it == langX::global_LangX.loaded_models.end() || !it->second) {
        std::cerr << "LxRAG Error [VLM-03]: Failed to load RAG VLM.\n";
        return "";
    }

    langX::Stack* vlm_stack = new langX::Stack();
    vlm_stack->stack_id  = RAG_VLM_STACK;
    vlm_stack->model = it->second;
    vlm_stack->unsafe = true;
    vlm_stack->conversation = new langX::Conversation();
    auto* vlm_settings = new langX::InquerySettings(db->vlm_settings);
    vlm_settings->seed = langX::randomSeed();
    vlm_stack->settings  = vlm_settings;
    vlm_stack->layers = {
        langX::makeLayer(langX::INIT_INFERENCE),
        langX::makeLayer(langX::FILE_PROCESSING),
        langX::makeLayer(langX::USER_PUSH_IMAGES),
        langX::makeLayer(langX::USER_PUSH_PROMT),
        langX::makeLayer(langX::LOAD_CHAT_TEMPLATE),
        langX::makeLayer(langX::CLEAR_KV_CACHE),
        langX::makeLayer(langX::INIT_SAMPLER),
        langX::makeLayer(langX::INIT_BATCH),
        langX::makeLayer(langX::FEED_PROMPT),
        langX::makeLayer(langX::LLM_GENERATION),
        langX::makeLayer(langX::FREE_SAMPLER),
        langX::makeLayer(langX::FREE_BATCH),
    };
    for (int i = 0; i < (int)vlm_stack->layers.size(); i++)
        vlm_stack->layers[i]->id = i;

    std::string description = langX::inference(
        vlm_stack,
        "Describe this image in detail for a document search index.",
        { image_path }
    );

    for (auto* l : vlm_stack->layers) delete l;
    delete vlm_stack->conversation;
    delete vlm_settings;
    delete vlm_stack;

    return description;
}

// --- Persistence ---

bool langX::saveRagDb(langX::RagDb* db, const char* path) {
    if (!db) {
        std::cerr << "LxRAG Error [RAG-SRDB-00]: Cannot open for write: " << path << "\n";
        return false;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) { 
        std::cerr << "LxRAG Error [RAG-SRDB-01]: Cannot open for write: " << path << "\n";
        return false;
    }

    const char magic[4] = {'L','X','R','G'};
    uint32_t version = 3;
    uint32_t dim = (uint32_t)db->embedding_dim;
    uint32_t count= (uint32_t)db->chunks.size();
    f.write(magic, 4);
    f.write(reinterpret_cast<const char*>(&version), 4);
    f.write(reinterpret_cast<const char*>(&dim), 4);
    f.write(reinterpret_cast<const char*>(&count), 4);

    for (const auto& c : db->chunks) {
        uint32_t slen = (uint32_t)c.source.size();
        uint32_t tlen = (uint32_t)c.text.size();
        uint8_t  img  = c.is_image  ? 1 : 0;
        uint8_t  mem  = c.is_memory ? 1 : 0;
        uint32_t ilen = (uint32_t)c.image_path.size();

        f.write(reinterpret_cast<const char*>(&slen), 4);
        f.write(c.source.data(), slen);
        f.write(reinterpret_cast<const char*>(&tlen), 4);
        f.write(c.text.data(), tlen);
        f.write(reinterpret_cast<const char*>(&img),  1);
        f.write(reinterpret_cast<const char*>(&ilen), 4);
        f.write(c.image_path.data(), ilen);
        f.write(reinterpret_cast<const char*>(&mem),  1);
        f.write(reinterpret_cast<const char*>(c.embedding.data()), dim * sizeof(float));
    }

    if (db->verbose) std::cout << "LxRAG Log [RAG-SRDB] Saved " << count << " chunks to " << path << "\n";
    return true;
}

bool langX::loadRagDb(langX::RagDb* db, const char* path) {
    if (!db) {
        std::cerr << "LxRAG Error [RAG-LDB-00]: Missing RAG database.\n";
        return false;
    }
    if (!fs::exists(path)) {
        std::cerr << "LxRAG Error [RAG-LDB-01]: DB file can't be found: " << path << "\n";
        return false;
    }

    std::ifstream f(path, std::ios::binary);
    if (!f) {
        std::cerr << "LxRAG Error [RAG-LDB-02]: Cannot open database file: " << path << "\n";
        return false;
    }

    char magic[4];
    uint32_t version, dim, count;
    f.read(magic, 4);
    if (std::strncmp(magic, "LXRG", 4) != 0) {
        std::cerr << "LxRAG Error [RAG-LDB-03]: Invalid .lxrag file (bad magic): " << path << "\n";
        return false;
    }
    f.read(reinterpret_cast<char*>(&version), 4);
    f.read(reinterpret_cast<char*>(&dim),     4);
    f.read(reinterpret_cast<char*>(&count),   4);

    if (version < 1 || version > 3) {
        std::cerr << "LxRAG Error [RAG-LDB-04]: Unsupported .lxrag version " << version << " (supported: 1–3). Rebuild the database.\n";
        return false;
    }
    if (db->embedding_dim != 0 && (int)dim != db->embedding_dim) {
        std::cerr << "LxRAG Error [RAG-LDB-05]: Embedding dim mismatch — file=" << dim << " model=" << db->embedding_dim << ". Rebuild the database with the current embedding model.\n";
        return false;
    }
    db->embedding_dim = (int)dim;
    db->chunks.reserve(db->chunks.size() + count);

    for (uint32_t i = 0; i < count; i++) {
        langX::RagDb::Chunk c;
        uint32_t slen, tlen;
        f.read(reinterpret_cast<char*>(&slen), 4);
        c.source.resize(slen); f.read(c.source.data(), slen);
        f.read(reinterpret_cast<char*>(&tlen), 4);
        c.text.resize(tlen);   f.read(c.text.data(), tlen);

        if (version >= 2) {
            uint8_t  img  = 0;
            uint32_t ilen = 0;
            f.read(reinterpret_cast<char*>(&img),  1);
            f.read(reinterpret_cast<char*>(&ilen), 4);
            c.is_image = (img != 0);
            c.image_path.resize(ilen);
            f.read(c.image_path.data(), ilen);
        }
        if (version >= 3) {
            uint8_t mem = 0;
            f.read(reinterpret_cast<char*>(&mem), 1);
            c.is_memory = (mem != 0);
        }

        c.embedding.resize(dim);
        f.read(reinterpret_cast<char*>(c.embedding.data()), dim * sizeof(float));
        db->chunks.push_back(std::move(c));
    }
    if (db->verbose) std::cout << "LxRAG Log [RAG-LDB]: Loaded " << count << " chunks (v" << version << ") from " << path << "\n";
    return true;
}

void langX::ragAddText(langX::RagDb* db, const char* text, const char* source) {
    if (!db || !text) {
        std::cerr << "LxRAG Error [RAG-AT-00]: Missing parameters.\n";
        return;
    }
    std::string src = source ? source : "";
    const llama_vocab* vocab = db->embed_model ? llama_model_get_vocab(db->embed_model) : nullptr;
    if (!vocab) {
        std::cerr << "LxRAG Error [RAG-AT-01]: No embedding model loaded. Call ragInitEmbeddings first.\n";
        return;
    }
    auto raw_chunks = chunk_text_tokens(vocab, std::string(text), db->params.chunk_size, db->params.chunk_overlap);
    int added = 0;
    for (const auto& chunk : raw_chunks) {
        auto emb = compute_embedding(db, chunk);
        if (emb.empty()) continue;
        if (db->embedding_dim == 0) db->embedding_dim = (int)emb.size();
        db->chunks.push_back({ chunk, src, std::move(emb) });
        added++;
    }
    if (db->verbose) std::cout << "LxRAG Log [RAG-AT]: Added " << added << " chunks from " << (src.empty() ? "text" : src) << "\n";
}

void langX::ragAddMemory(langX::RagDb* db, const char* text, const char* source) {
    if (!db || !text) {
        std::cerr << "LxRAG Error [RAG-AM-00]: Missing parameters.\n";
        return;
    }
    const llama_vocab* vocab = db->embed_model ? llama_model_get_vocab(db->embed_model) : nullptr;
    if (!vocab) {
        std::cerr << "LxRAG Error [RAG-AM-01]: No embedding model loaded. Call ragInitEmbeddings first.\n";
        return;
    }
    std::string src = source ? source : "";
    auto raw_chunks = chunk_text_tokens(vocab, std::string(text), db->params.chunk_size, db->params.chunk_overlap);
    int added = 0;
    for (const auto& chunk : raw_chunks) {
        auto emb = compute_embedding(db, chunk);
        if (emb.empty()) continue;
        if (db->embedding_dim == 0) db->embedding_dim = (int)emb.size();
        langX::RagDb::Chunk c;
        c.text = chunk;
        c.source = src;
        c.embedding = std::move(emb);
        c.is_memory = true;
        db->chunks.push_back(std::move(c));
        added++;
    }
    if (db->verbose) std::cout << "LxRAG Log [RAG-AM]: Added " << added << " memory chunk(s) — source: " << (src.empty() ? "(none)" : src) << "\n";
}

void langX::ragAddImage(langX::RagDb* db, const char* image_path, const char* description) {
    if (!db || !image_path) {
        std::cerr << "LxRAG Error [RAG-AI-00]: Missing parameters.\n";
        return;
    }
    fs::path p(image_path);
    if (!fs::exists(p)) {
        std::cerr << "LxRAG Error [RAG-AI-01]: Image not found: " << image_path << "\n";
        return;
    }

    std::string desc;
    if (description) {
        desc = description;
    } else if (!db->vlm_model_id.empty()) {
        size_t img_tokens = langX::countVisionTokens(p.string().c_str(), db->vlm_model_id.c_str());
        uint32_t n_ubatch = 0;
        auto it = langX::global_LangX.loaded_models.find(db->vlm_model_id);
        if (it != langX::global_LangX.loaded_models.end() && it->second && it->second->ctx)
            n_ubatch = llama_n_ubatch(it->second->ctx);

        bool token_check_ok = (img_tokens == 0 || n_ubatch == 0);
        if (!token_check_ok && img_tokens > n_ubatch) {
            if (db->verbose) std::cout << "LxRAG Warning Log [RAG-IA-00]: Skipping VLM description for " << p.filename().string() << ": estimated " << img_tokens << " raw patches but VLM n_ubatch=" << n_ubatch << ". Increase encoder_batch in ragInitVlm to at least " << img_tokens << ".\n";
        } else {
            if (db->verbose) std::cout << "LxRAG Log [RAG-IA-00]: Describing image with VLM: " << p.filename().string() << " (" << img_tokens << " vision tokens)\n";
            
            desc = rag_describe_image(db, p.string());

            if (desc.empty()) {
                if (db->verbose) std::cout << "LxRAG Warning Log [RAG-IA-02]: VLM description failed for " << p.filename().string() << " — falling back to filename.\n";
            } else if (db->verbose) {
                std::cout << "LxRAG Log [RAG-IA-01]: Description: " << desc << "\n";
            }
        }
    }
    
    if (desc.empty()) desc = "[image: " + p.filename().string() + "]";

    auto emb = compute_embedding(db, desc);
    if (emb.empty()) return;
    if (db->embedding_dim == 0) db->embedding_dim = (int)emb.size();

    langX::RagDb::Chunk chunk;
    chunk.text       = desc;
    chunk.source     = p.filename().string();
    chunk.embedding  = std::move(emb);
    chunk.is_image   = true;
    chunk.image_path = p.string();

    db->chunks.push_back(std::move(chunk));
    if (db->verbose) std::cout << "LxRAG Log [RAG-IA-02]: Added image chunk: " << p.filename().string() << "\n";
}

void langX::ragAddFile(langX::RagDb* db, const char* path) {
    if (!db || !path) {
        std::cerr << "LxRAG Error [RAG-AF-00]: Missing paramaters.\n";
        return;
    }
    fs::path p(path);
    if (!fs::exists(p)) {
        std::cerr << "LxRAG Error [RAG-AF-01]: File not found: " << path << "\n";
        return;
    }
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    // TODO: move to a utility function
    bool supported = false;
    for (const auto& s : db->params.supported_extensions) {
        if (ext == s) {
            supported = true;
            break;
        }
    }

    if (!supported) {
        std::cerr << "LxRAG Error [RAG-AF-02]: Unsupported file type: " << ext << "\n";
        return;
    }

    if (ext == ".pdf") { // WIP
        std::string text = langX::pdfExtractText(p);
        if (!text.empty())
            ragAddText(db, text.c_str(), p.filename().string().c_str());
    } else {
        std::ifstream f(path);
        if (!f) {
            std::cerr << "LxRAG Error [RAG-AF-03]: Cannot open: " << path << "\n";
            return;
        }
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        ragAddText(db, text.c_str(), p.filename().string().c_str());
    }
}

void langX::ragAddDirectory(langX::RagDb* db, const char* dir_path, bool recursive) {
    if (!db || !dir_path) {
        std::cerr << "LxRAG Error [RAG-AD-00]: Missing paramaters.\n";
        return;
    }
    fs::path dir(dir_path);
    if (!fs::is_directory(dir)) {
        std::cerr << "LxRAG Error [RAG-AD-01]: Not a directory: " << dir_path << "\n"; 
        return;
    }

    int added = 0;
    auto process = [&](const fs::path& file) {
        if (!fs::is_regular_file(file)) return;
        std::string ext = file.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        for (const auto& s : db->params.supported_image_extensions) {
            if (ext == s) { 
                ragAddImage(db, file.string().c_str(), nullptr); added++;
                return;
            }
        }
        for (const auto& s : db->params.supported_extensions) {
            if (ext == s) {
                ragAddFile(db, file.string().c_str()); added++;
                break;
            }
        }
    };

    std::error_code ec;
    if (recursive) {
        for (const auto& entry : fs::recursive_directory_iterator(dir, ec))
            process(entry.path());
    } else {
        for (const auto& entry : fs::directory_iterator(dir, ec))
            process(entry.path());
    }
    if (ec) std::cerr << "LxRAG Error [RAG-AD-02]: Directory iteration error: " << ec.message() << "\n";
    if (db->verbose) std::cout << "LxRAG Log [RAG-AD]: Added " << added << " file(s) from " << dir_path << "\n";
}

void langX::ragAddSupportedExtension(langX::RagDb* db, const char* ext) {
    if (!db || !ext) return;
    for (const auto& s : db->params.supported_extensions)
        if (s == ext) return;
    db->params.supported_extensions.emplace_back(ext);
}

void langX::ragRemoveSupportedExtension(langX::RagDb* db, const char* ext) {
    if (!db || !ext) return;
    auto& v = db->params.supported_extensions;
    v.erase(std::remove(v.begin(), v.end(), std::string(ext)), v.end());
}

// --- RAG Search ---

std::vector<langX::RagChunk> langX::ragSearch(langX::RagDb* db, const char* query, int top_k) {
    std::vector<langX::RagChunk> results;
    if (!db || !query || db->chunks.empty()) return results;

    auto query_emb = compute_embedding(db, std::string(query), /*is_query=*/true);
    if (query_emb.empty()) return results;

    const float lambda = db->params.mmr_lambda;
    const int src_cap = db->params.max_chunks_per_source;
    const float fn_w = db->params.filename_score_weight;
    const int n = (int)db->chunks.size();
    const std::string query_str(query);

    const langX::RagFilterMode filter = db->params.filter_mode;
    struct Candidate { float query_score; int idx; bool selected; };
    std::vector<Candidate> cands;
    cands.reserve(n);
    for (int i = 0; i < n; i++) {
        const auto& chunk = db->chunks[i];
        if (filter == langX::RagFilterMode::DOCS   &&  chunk.is_memory) continue;
        if (filter == langX::RagFilterMode::MEMORY && !chunk.is_memory) continue;

        float cosine = cosine_similarity(query_emb, chunk.embedding);
        float score  = cosine;
        if (fn_w > 0.0f) {
            float fn = filename_keyword_score(query_str, chunk.source);
            score = (1.0f - fn_w) * cosine + fn_w * fn;
        }
        cands.push_back({ score, i, false });
    }

    std::unordered_map<std::string, int> source_count;
    const int k = std::min(top_k, (int)cands.size());
    std::vector<std::vector<float>*> selected_embs;
    selected_embs.reserve(k);

    for (int pick = 0; pick < k; pick++) {
        float best_mmr = -1e9f;
        int   best_i   = -1;

        for (int i = 0; i < (int)cands.size(); i++) {
            if (cands[i].selected) continue;

            if (src_cap > 0) {
                auto it = source_count.find(db->chunks[cands[i].idx].source);
                if (it != source_count.end() && it->second >= src_cap) continue;
            }

            float relevance = cands[i].query_score;
            float redundancy = 0.0f;
            if (!selected_embs.empty()) {
                for (auto* emb : selected_embs)
                    redundancy = std::max(redundancy,
                        cosine_similarity(*emb, db->chunks[cands[i].idx].embedding));
            }

            float mmr = lambda * relevance - (1.0f - lambda) * redundancy;
            if (mmr > best_mmr) { best_mmr = mmr; best_i = i; }
        }

        if (best_i < 0) break;
        cands[best_i].selected = true;
        selected_embs.push_back(&db->chunks[cands[best_i].idx].embedding);
        source_count[db->chunks[cands[best_i].idx].source]++;

        const auto& c = db->chunks[cands[best_i].idx];
        results.push_back({ c.text, cands[best_i].query_score, c.source, c.is_image, c.image_path, c.is_memory });
    }
    return results;
}

// --- Stack integration ---

void langX::attachRagDb(langX::Stack* stack, langX::RagDb* db) {
    if (stack) stack->rag_db = db;
}

void langX::setRagDbParams(langX::RagDb* db, const langX::RagParams& params) {
    if (db) db->params = params;
}

void langX::ragSetVerbose(langX::RagDb* db, bool verbose) {
    if (db) db->verbose = verbose;
}

void langX::setRagParams(langX::Stack* stack, const langX::RagParams& params) {
    if (!stack || !stack->rag_db) return;
    stack->rag_db->params = params;
}

bool layer_rag_retrieval(langX::InferenceState* s, const langX::Layer*) {
    auto* stack = s->stack;
    langX::RagDb* db = stack->rag_db;

    if (!db) {
        std::cerr << "[RAG_RETRIEVAL]: No RagDb attached to stack. Use attachRagDb().\n";
        return false;
    }
    if (db->chunks.empty()) {
        if (stack->verbose) std::cout << "[RAG_RETRIEVAL]: Database is empty — skipping retrieval.\n";
        return true;
    }

    const std::string& query = stack->active_prompt;
    int top_k = db->params.top_k;

    auto results = langX::ragSearch(db, query.c_str(), top_k);
    if (results.empty()) {
        if (stack->verbose) std::cout << "[RAG_RETRIEVAL]: No results for query.\n";
        return true;
    }

    auto* conv = s->conversation;
    const langX::RagImageMode img_mode = db->params.image_mode;

    for (const auto& chunk : results) {
        if (stack->verbose) {
            if (chunk.is_memory) {
                std::cout << "[RAG_RETRIEVAL]: memory score=" << chunk.score << " src=" << chunk.source << "\n";
            } else if (chunk.is_image) {
                const char* mode_str = img_mode == langX::RagImageMode::DESCRIBE    ? "DESCRIBE(text-only)" :
                                       img_mode == langX::RagImageMode::PASSTHROUGH ? "PASSTHROUGH(image-to-VLM)" : "BOTH";
                std::cout << "[RAG_RETRIEVAL]: image(" << mode_str << ") score=" << chunk.score << " src=" << chunk.source << "\n";
            } else {
                std::cout << "[RAG_RETRIEVAL]: text score=" << chunk.score << " src=" << chunk.source << "\n";
            }
        }

        if (chunk.is_image) {
            if (img_mode == langX::RagImageMode::PASSTHROUGH || img_mode == langX::RagImageMode::BOTH) {
                if (!s->model || !s->model->mtmd_ctx) {
                    std::cerr << "[RAG_RETRIEVAL]: PASSTHROUGH: active model has no mmproj. Use RagImageMode::DESCRIBE instead.\n";
                } else {
                    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(s->model->mtmd_ctx, chunk.image_path.c_str());
                    if (bmp) {
                        s->image_bitmaps.push_back(bmp);
                        s->bitmap_names.push_back(chunk.source);
                        s->has_images = true;
                        if (stack->verbose)
                            std::cout << "[RAG_RETRIEVAL] Loaded image for VLM: " << chunk.source << "\n";
                    } else {
                        std::cerr << "[RAG_RETRIEVAL]: Failed to load image bitmap: " << chunk.image_path << "\n";
                    }
                }
            }
            
            if (img_mode == langX::RagImageMode::DESCRIBE || img_mode == langX::RagImageMode::BOTH) {
                conv->extra_data.push_back({ "Image: " + chunk.source, chunk.text, chunk.score });
            }
        } else if (chunk.is_memory) {
            conv->extra_data.push_back({ "Memory: " + chunk.source, chunk.text, chunk.score });
        } else {
            conv->extra_data.push_back({ chunk.source, chunk.text, chunk.score });
        }
    }
    return true;
}


bool layer_semantic_mem_retrieval(langX::InferenceState* s, const langX::Layer* sl) {
    auto* stack = s->stack;
    if (!stack || !stack->rag_db) {
        std::cerr << "[SEMANTIC_MEM_RETRIEVAL] No RAG database attached. Call attachRagDb first.\n";
        return false;
    }
    const langX::RagFilterMode prev_filter = stack->rag_db->params.filter_mode;
    stack->rag_db->params.filter_mode = langX::RagFilterMode::MEMORY;
    bool ok = layer_rag_retrieval(s, sl);
    stack->rag_db->params.filter_mode = prev_filter;
    return ok;
}
