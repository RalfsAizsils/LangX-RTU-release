cmake_minimum_required(VERSION 3.14)
project(SolutionTest)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- LangX paths (pre-built DLL) ---
set(LANGX_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../llama_langx/CMake_LangX")
set(LANGX_BUILD_DIR "${LANGX_SRC_DIR}/out/build/x64-release" CACHE PATH
    "Path to LangX build directory containing langX.lib and langX.dll")

set(LANGX_HEADER "${LANGX_SRC_DIR}/langX.h")
set(LANGX_LIB    "${LANGX_BUILD_DIR}/langX.lib")
set(LANGX_DLL    "${LANGX_BUILD_DIR}/langX.dll")

if(NOT EXISTS "${LANGX_LIB}")
    message(FATAL_ERROR "langX.lib not found at ${LANGX_LIB}\n"
            "Set LANGX_BUILD_DIR to the directory containing langX.lib/dll.")
endif()

# Import langX as a shared library target
add_library(langX SHARED IMPORTED)
set_target_properties(langX PROPERTIES
    IMPORTED_LOCATION "${LANGX_DLL}"
    IMPORTED_IMPLIB   "${LANGX_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${LANGX_SRC_DIR}"
)

# --- langx_test ---
add_executable(langx_test "langx_test.cpp")
target_link_libraries(langx_test PRIVATE langX)

# Copy langX.dll next to the executable
file(GLOB LANGX_RUNTIME_DLLS "${LANGX_BUILD_DIR}/*.dll")
add_custom_command(TARGET langx_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${LANGX_RUNTIME_DLLS}
        $<TARGET_FILE_DIR:langx_test>
)

# =============================================================================
# llamacpp_test — raw llama.cpp built from source with Vulkan
# =============================================================================

include(FetchContent)

add_definitions(-w)

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Zc:char8_t->")
endif()

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG bcb43163aed6a8986cf3d66e90848c9c258d4936
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
set(LLAMA_HTTPLIB OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN ON CACHE BOOL "" FORCE)

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(llama)

set(LLAMA_INSTALL_VERSION "0.0.0" CACHE STRING "" FORCE)
if(NOT DEFINED LLAMA_BUILD_NUMBER)
    set(LLAMA_BUILD_NUMBER 0)
endif()
if(NOT DEFINED LLAMA_BUILD_COMMIT)
    set(LLAMA_BUILD_COMMIT "unknown")
endif()
if(NOT DEFINED BUILD_COMPILER)
    set(BUILD_COMPILER "${CMAKE_CXX_COMPILER_ID}")
endif()
if(NOT DEFINED BUILD_TARGET)
    set(BUILD_TARGET "${CMAKE_SYSTEM_PROCESSOR}")
endif()

# common library needed by mtmd CLI targets (we don't use it, but cmake requires it)
add_subdirectory(${llama_SOURCE_DIR}/common ${llama_BINARY_DIR}/common)
target_sources(common PRIVATE "${LANGX_SRC_DIR}/stubs.cpp")
if(MSVC)
    target_compile_definitions(common PRIVATE JSON_HAS_CPP_17 JSON_HAS_CPP_14)
endif()

add_subdirectory(${llama_SOURCE_DIR}/tools/mtmd ${llama_BINARY_DIR}/tools/mtmd)

# --- llamacpp_test ---
add_executable(llamacpp_test "llamacpp_test.cpp")
target_link_libraries(llamacpp_test PRIVATE llama mtmd)
target_include_directories(llamacpp_test PRIVATE
    ${llama_SOURCE_DIR}/include
    ${llama_SOURCE_DIR}/tools/mtmd
)

# =============================================================================
# llamalib_test — LlamaLib pre-built DLL (Vulkan)
# =============================================================================

set(LLAMALIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/LlamaLib-v2.0.5")
set(LLAMALIB_INCLUDE "${LLAMALIB_DIR}/include")
set(LLAMALIB_RUNTIME "${LLAMALIB_DIR}/runtimes/win-x64/native")
set(LLAMALIB_DLL "${LLAMALIB_RUNTIME}/llamalib_win-x64_vulkan.dll")
set(LLAMALIB_LIB "${LLAMALIB_RUNTIME}/llamalib_win-x64_vulkan.lib")

if(NOT EXISTS "${LLAMALIB_LIB}")
    message(FATAL_ERROR "llamalib .lib not found at ${LLAMALIB_LIB}")
endif()

add_library(llamalib SHARED IMPORTED)
set_target_properties(llamalib PROPERTIES
    IMPORTED_LOCATION "${LLAMALIB_DLL}"
    IMPORTED_IMPLIB "${LLAMALIB_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${LLAMALIB_INCLUDE}"
)

add_executable(llamalib_test "llamalib_test.cpp")
target_link_libraries(llamalib_test PRIVATE llamalib)

add_custom_command(TARGET llamalib_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LLAMALIB_DLL}"
        $<TARGET_FILE_DIR:llamalib_test>
)
