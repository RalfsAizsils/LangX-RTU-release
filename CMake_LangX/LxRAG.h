#pragma once
#include "langX.h"
#include <string>
#include <vector>

namespace langX {

    struct RagChunk {
        std::string text;
        float score = 0.0f;
        std::string source;
        bool is_image = false;
        std::string image_path;
        bool is_memory = false;
    };

    enum class RagFilterMode {
        ALL,
        DOCS,
        MEMORY,
    };

    enum class RagImageMode {
        DESCRIBE,
        PASSTHROUGH,
        BOTH,
    };

    struct RagParams {
        int top_k = 3;
        float mmr_lambda = 1.0f;
        int max_chunks_per_source = 0;
        float filename_score_weight = 0.0f;
        int chunk_size = 400;
        int chunk_overlap = 50;
        int embed_ctx_size = 512;
        std::string embed_doc_prefix = "";
        std::string embed_query_prefix = "";
        std::string injection_prefix = "[World Context]\n";
        RagImageMode image_mode = RagImageMode::DESCRIBE;
        RagFilterMode filter_mode = RagFilterMode::ALL;
        std::vector<std::string> supported_extensions = {
            ".txt", ".md", ".rst",
            ".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml",
            ".xml", ".html", ".htm",
            ".py", ".js", ".ts", ".cpp", ".c", ".h", ".hpp", ".java", ".cs",
            ".go", ".rs", ".rb", ".sh", ".bat", ".ps1",
            ".log", ".ini", ".cfg", ".conf",
            ".pdf"
        };
        std::vector<std::string> supported_image_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"
        };
    };

    struct RagDb;

    LANGX_API RagDb* makeRagDb();
    LANGX_API void freeRagDb(RagDb* db);
    LANGX_API bool ragInitEmbeddings(RagDb* db, const char* embed_model_path, int n_gpu_layers = 99);
    LANGX_API bool ragInitVlm(RagDb* db, const char* model_path, const char* mmproj_path, int encoder_batch = 16384, int n_gpu_layers = 99);
    LANGX_API void ragFreeVlm(RagDb* db);
    LANGX_API void ragSetVlmSettings(RagDb* db, InquerySettings settings);

    LANGX_API bool saveRagDb(RagDb* db, const char* path);
    LANGX_API bool loadRagDb(RagDb* db, const char* path);

    LANGX_API void ragAddText(RagDb* db, const char* text, const char* source = nullptr);
    LANGX_API void ragAddFile(RagDb* db, const char* path);
    LANGX_API void ragAddDirectory(RagDb* db, const char* dir_path, bool recursive = false);
    LANGX_API void ragAddSupportedExtension(RagDb* db, const char* ext);
    LANGX_API void ragRemoveSupportedExtension(RagDb* db, const char* ext);
    LANGX_API void ragAddImage(RagDb* db, const char* image_path, const char* description = nullptr);
    LANGX_API void ragAddMemory(RagDb* db, const char* text, const char* source = nullptr);

    LANGX_API std::vector<RagChunk> ragSearch(RagDb* db, const char* query, int top_k = 3);

    LANGX_API void attachRagDb(Stack* stack, RagDb* db);
    LANGX_API void setRagDbParams(RagDb* db, const RagParams& params);
    LANGX_API void ragSetVerbose(RagDb* db, bool verbose);
    LANGX_API void setRagParams(Stack* stack, const RagParams& params);

}
