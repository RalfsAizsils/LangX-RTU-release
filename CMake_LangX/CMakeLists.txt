cmake_minimum_required (VERSION 3.14)

add_definitions(-w)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "ProgramDatabase")
endif()

project ("LangX")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Zc:char8_t->")
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/FS>")
endif()

# =============================================================================
# Dependencies
# =============================================================================

include(FetchContent)

FetchContent_Declare(
    llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG        bcb43163aed6a8986cf3d66e90848c9c258d4936
)

FetchContent_Declare(
    miniz
    GIT_REPOSITORY https://github.com/richgel999/miniz
    GIT_TAG        3.0.2
)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libs" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "Build with native instructions" FORCE)
set(LLAMA_HTTPLIB OFF CACHE BOOL "Disable httplib" FORCE)

set(LANGX_BACKEND "" CACHE STRING "Force a single GPU backend (vulkan, cuda, cpu, metal). Leave empty for all detected.")

if(LANGX_BACKEND)
    string(TOLOWER "${LANGX_BACKEND}" _lb)
    if(_lb STREQUAL "cuda")
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
        set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    elseif(_lb STREQUAL "vulkan")
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    elseif(_lb STREQUAL "metal")
        set(GGML_METAL ON CACHE BOOL "" FORCE)
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
        set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
    elseif(_lb STREQUAL "cpu")
        set(GGML_CUDA OFF CACHE BOOL "" FORCE)
        set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
        set(GGML_METAL OFF CACHE BOOL "" FORCE)
    else()
        message(FATAL_ERROR "Unknown LANGX_BACKEND '${LANGX_BACKEND}'. Use: vulkan, cuda, cpu, metal")
    endif()
    message(STATUS "LangX: Forced single backend: ${_lb}")
else()
    if(APPLE)
        set(GGML_METAL ON CACHE BOOL "" FORCE)
        message(STATUS "LangX: Enabling Metal backend.")
    endif()
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        set(GGML_CUDA ON CACHE BOOL "" FORCE)
        message(STATUS "LangX: Enabling CUDA backend.")
    endif()
    find_package(Vulkan QUIET)
    if(Vulkan_FOUND)
        set(GGML_VULKAN ON CACHE BOOL "" FORCE)
        message(STATUS "LangX: Enabling Vulkan backend.")
    endif()
    if(NOT CUDAToolkit_FOUND AND NOT Vulkan_FOUND AND NOT APPLE)
        message(STATUS "LangX: No GPU SDK found. CPU-only build.")
    endif()
endif()

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
FetchContent_MakeAvailable(llama miniz)

set(LLAMA_INSTALL_VERSION "0.0.0" CACHE STRING "" FORCE)
add_subdirectory(${llama_SOURCE_DIR}/tools/mtmd ${llama_BINARY_DIR}/tools/mtmd)

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
add_subdirectory(${llama_SOURCE_DIR}/common ${llama_BINARY_DIR}/common)
target_sources(common PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/stubs.cpp")
if(MSVC)
    target_compile_definitions(common PRIVATE JSON_HAS_CPP_17 JSON_HAS_CPP_14)
endif()

if(MSVC)
    target_compile_options(llama PRIVATE /wd4244 /wd4267)
    target_compile_options(llama PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/Zc:char8_t->")
endif()

# =============================================================================
# langX shared library (langX.dll / liblangX.so)
# =============================================================================

add_library(CMake_LangX SHARED "langX.cpp" "LxRAG.cpp" "LxPDF.cpp")
set_target_properties(CMake_LangX PROPERTIES OUTPUT_NAME "langX")
target_compile_definitions(CMake_LangX PRIVATE BUILDING_MY_DLL JSON_HAS_CPP_17 JSON_HAS_CPP_14)
target_link_libraries(CMake_LangX PRIVATE llama mtmd miniz common)
target_include_directories(CMake_LangX PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(CMake_LangX PRIVATE ${llama_SOURCE_DIR}/tools/mtmd)
target_include_directories(CMake_LangX PRIVATE ${llama_SOURCE_DIR}/include)
target_include_directories(CMake_LangX PRIVATE ${llama_SOURCE_DIR}/common)
target_include_directories(CMake_LangX PRIVATE ${llama_SOURCE_DIR}/vendor)
target_include_directories(CMake_LangX PRIVATE ${miniz_SOURCE_DIR})

# =============================================================================
# Test executables
# =============================================================================

add_executable(LangX_Test "main.cpp")
target_link_libraries(LangX_Test PRIVATE CMake_LangX)

add_executable(LangX_Memory_Test "main_memory.cpp")
target_link_libraries(LangX_Memory_Test PRIVATE CMake_LangX)

add_executable(LangX_All_Tests "main_all_tests.cpp")
target_link_libraries(LangX_All_Tests PRIVATE CMake_LangX)


