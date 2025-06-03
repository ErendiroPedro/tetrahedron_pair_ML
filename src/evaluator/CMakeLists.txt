cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(tetra_net)

# Use C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set path to LibTorch CPU (adjust if needed)
set(Torch_DIR "/opt/libtorch_cpu/share/cmake/Torch")

add_executable(tetra_net CppWrapper.cpp)


# Find LibTorch
find_package(Torch REQUIRED)

# Link LibTorch
target_link_libraries(tetra_net "${TORCH_LIBRARIES}")

# Include LibTorch headers
target_include_directories(tetra_net PRIVATE "${TORCH_INCLUDE_DIRS}")

# Ensure proper ABI if needed (usually required for GCC-based builds)
target_compile_definitions(tetra_net PRIVATE _GLIBCXX_USE_CXX11_ABI=1)
