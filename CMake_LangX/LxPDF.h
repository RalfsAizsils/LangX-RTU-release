#pragma once

#include <string>
#include <filesystem>

namespace langX {
    std::string pdfExtractText(const std::filesystem::path& pdf_path);

}
