cmake_minimum_required(VERSION 3.12.0)
set (CMAKE_CXX_STANDARD 14)

if(POLICY CMP0091)
    # https://stackoverflow.com/a/56490614
    cmake_policy(SET CMP0091 NEW)
endif()

option(STATIC_WINDOWS_RUNTIME "Use static (MT/MTd) Windows runtime" ON)
# This needs cmake_policy(SET CMP0091 NEW)
if (STATIC_WINDOWS_RUNTIME)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
else()
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

project(ICP LANGUAGES C CXX)

# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
target_sources(Open3D PRIVATE
    Open3DConfig.cpp
)

#link_directories(${CMAKE_BINARY_DIR}/lib)

#target_link_libraries(LibsModule -C:/Users/afiqa/ICP/lib/)
add_executable(ICP testvisualizer.cpp)
#target_include_directories(ICP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/open3d)
#target_link_libraries(ICP LibsModule)