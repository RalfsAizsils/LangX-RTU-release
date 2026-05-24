#include "langX.h"

#include <iostream>
#include <filesystem>
#include <chrono>
#include <regex>
#include <fstream>
#include <iomanip>
#include <sstream>

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

int main() {
    const std::vector<TestPrompt> prompts = {
        {"introduction", INTRO_PROMPT, "", "blue", "Basic question about sky color"},
        {"image_describe", "Describe what you see in this image in detail.", IMAGE_PATH.string(), "robot|mech|machine|armor|metal|figure|character", "Describe the robot2.jpg image"},
        {"fact_recall", "Earlier I told you an important fact. What was it? Repeat it.", "", "Jimmy", "Recall fact from the introduction"},
    };

    fs::path out_file = SCRIPT_DIR / ("result_langx_cpp_" + timestamp_str() + ".txt");
    std::ostringstream log;

    struct Result { std::string name, status, reason; double time = 0; };
    std::vector<Result> results;

    log << "LangX C++ Test \xe2\x80\x93 " << iso_now() << "\n";
    log << "Model: " << MODEL_PATH.string() << "\n";
    log << "Image: " << IMAGE_PATH.string() << "\n";
    log << std::string(60, '=') << "\n";

    auto t_total = Clock::now();

    // --- Setup: config, model, stack ---
    langX::Config cfg;
    langX::initialize_langX(cfg);

    langX::ModelParams mp(MODEL_PATH.string(), MMPROJ.string());
    mp.n_ctx = 8192;

    std::cout << "Loading VLM model...\n";
    auto t_load = Clock::now();
    langX::initModel(mp, "vlm");
    double load_s = Sec(Clock::now() - t_load).count();
    std::cout << "Model loaded.\n\n";
    log << "Model load time: " << std::fixed << std::setprecision(2) << load_s << "s\n";

    langX::setModelSystemPrompt("You are a helpful assistant. Answer concisely.");

    langX::InquerySettings settings;
    settings.seed = langX::randomSeed();

    langX::Stack* stack = langX::makeDefaultStack("vlm_test");
    langX::setInquerySettings(settings, stack);
    langX::switchModel("vlm", stack);
    langX::initConversation("test_convo", stack);

    // --- Test prompt loop ---
    for (const auto& p : prompts) {
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "[" << p.name << "] " << p.desc << "\n";
        std::cout << "Prompt: " << p.prompt << "\n";
        std::cout << std::string(60, '-') << "\n";

        log << "\n--- " << p.name << ": " << p.desc << " ---\n";
        log << "Prompt: " << p.prompt << "\n";

        std::vector<std::string> files;
        if (!p.image.empty()) files.push_back(p.image);

        auto t_prompt = Clock::now();
        std::string response = langX::inference(stack, p.prompt.c_str(), files);
        double prompt_s = Sec(Clock::now() - t_prompt).count();

        std::cout << response << "\n";

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
    log << "\nTotal: " << n_pass << "/" << total << " passed, "
        << n_fail << " failed, " << n_err << " errors\n";
    log << "Total time: " << std::fixed << std::setprecision(2) << total_s << "s\n";

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "SUMMARY: " << n_pass << "/" << total << " passed, " << n_fail << " failed, " << n_err << " errors\n";
    std::cout << "Results saved to: " << out_file.string() << "\n";

    std::ofstream(out_file) << log.str();
    return 0;
}
