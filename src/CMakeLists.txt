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

project(ICP)


#file(GLOB_RECURSE SOURCE_FILE_TOP "*.c" "*.h" "*.cpp")
#file(GLOB_RECURSE SOURCE_FILE_NESTED 
#	"Eigen/*.c" "*.h"
#	"fmt/*.c *.h"
#	"GL/*.c *.h"
#	"GLFW/*.c *.h"
#	"math/ *.c *.h"
#	"open3d/*.c *.h"
#	"open3d/camera/*.cpp *.h"
#	"open3d/core/*.cpp *.h"
#	"open3d/data/*.cpp *.h"
#	"open3d/geometry/*.cpp *.h"
#	"open3d/io/*.cpp *.h"
#	"open3d/ml/*.cpp *.h"
#	"open3d/pipelines/*.cpp *.h"
#	"open3d/pipelines/color_map/*.cpp *.h"
#	"open3d/pipelines/integration/*.cpp *.h"
#	"open3d/pipelines/odometry/*.cpp *.h"
#	"open3d/pipelines/registration/*.cpp *.h"
#	"open3d/t/*.cpp *.h"
#	"open3d/utility/*.cpp *.h"
#	"open3d/visualization/*.cpp *.h"
#	"open3d/visualization/gui/*.cpp *.h"
#	"open3d/visualization/rendering/*.cpp *.h"
#	"open3d/visualization/shader/*.cpp *.h"
#	"open3d/visualization/utility/*.cpp *.h"
#	"open3d/visualization/visualizer/*.cpp *.h")
#set(SOURCE_FILES ${SOURCE_FILES_TOP} ${SOURCE_FILES_NESTED})


#add_executable(ICP testvisualizer.cpp ${SOURCE_FILES})
#target_include_directories(ICP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/open3d)
#target_link_libraries(ICP Open3D)


#add_executable( ICP testvisualizer.cpp )
#target_include_directories(ICP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#ADD_SUBDIRECTORY(include)

add_executable(ICP Main.cpp TestVisualizer.cpp)
#target_include_directories(ICP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/open3d)
#target_link_libraries(ICP Open3D)


#add_executable( ICP main.cpp )
#target_include_directories(ICP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#ADD_SUBDIRECTORY(include)
