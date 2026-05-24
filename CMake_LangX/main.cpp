#include "langX.h"
#include <iostream>
#include <windows.h>
namespace fs = std::filesystem;

int main() {
    SetConsoleOutputCP(CP_UTF8);

    // Init System
    langX::Config config = langX::Config{ (fs::current_path() / "langX_out").string() };
    langX::LangX* lx = langX::initialize_langX(config);

    std::string modelPath = "FULL_PATH_TO_MODEL.gguf";

    // Check if file exists for safety
    if (!fs::exists(modelPath)) {
        std::cerr << "Model file not found at: " << modelPath << std::endl;
        return 1;
    }

    langX::ModelParams model_params = langX::ModelParams{ 
        modelPath
    };

    // Load
    std::cout << "Loading model...\n\n";
    langX::initModel(model_params);

    langX::InquerySettings settings = langX::InquerySettings{};
    settings.seed = 5000;
    langX::setInquerySettings(settings);

    // Ask Loop
    std::string reply = "";
    std::string userInput;

    while (true) {
        std::cout << "Enter prompt: ";
        std::getline(std::cin, userInput);
        if (userInput == "exit") {
            std::cout << "Exiting program. Goodbye!" << std::endl;
            break;
        }

        std::cout << "\nSending prompt... [\"" << userInput << "\"]\n";

        reply = langX::inference(userInput.c_str());
        std::cout << "Full Reply:\n" << reply << std::endl;
    }

    // Cleanup
    langX::unloadModel();

    return 0;
}