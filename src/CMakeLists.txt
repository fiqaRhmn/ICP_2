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


file(GLOB_RECURSE SOURCE_FILE_TOP "*.c" "*.h")
file(GLOB_RECURSE SOURCE_FILE_NESTED 
	"Eigen/*.c" "*.h"
	"fmt/*.c *.h"
	"GL/*.c *.h"
	"GLFW/*.c *.h"
	"math/ *.c *.h"
	"open3d/*.c *.h"
	"open3d/camera/*.cpp *.h"
	"open3d/core/*.cpp *.h"
	"open3d/data/*.cpp *.h"
	"open3d/geometry/*.cpp *.h"
	"open3d/io/*.cpp *.h"
	"open3d/ml/*.cpp *.h"
	"open3d/pipelines/*.cpp *.h"
	"open3d/pipelines/color_map/*.cpp *.h"
	"open3d/pipelines/integration/*.cpp *.h"
	"open3d/pipelines/odometry/*.cpp *.h"
	"open3d/pipelines/registration/*.cpp *.h"
	"open3d/t/*.cpp *.h"
	"open3d/utility/*.cpp *.h"
	"open3d/visualization/*.cpp *.h"
	"open3d/visualization/gui/*.cpp *.h"
	"open3d/visualization/rendering/*.cpp *.h"
	"open3d/visualization/shader/*.cpp *.h"
	"open3d/visualization/utility/*.cpp *.h"
	"open3d/visualization/visualizer/*.cpp *.h")
set(SOURCE_FILES ${SOURCE_FILES_TOP} ${SOURCE_FILES_NESTED})

#add_library(
#	include-files
#	${SOURCE_FILES}
#	)

# function(open3d_set_open3d_lib_properties target)
    # cmake_parse_arguments(arg "HIDDEN" "" "" ${ARGN})
    # if(NOT BUILD_SHARED_LIBS)
        # target_compile_definitions(${target} PUBLIC OPEN3D_STATIC)
    # endif()
    # if (arg_HIDDEN)
        # set_target_properties(${target} PROPERTIES
                        # CXX_VISIBILITY_PRESET hidden
                        # VISIBILITY_INLINES_HIDDEN ON
                        # )
    # else ()
        # target_compile_definitions(${target} PRIVATE OPEN3D_ENABLE_DLL_EXPORTS)
    # endif()

# endfunction()

# # Configure a header file to pass the version settings to the source code
# configure_file("${PROJECT_SOURCE_DIR}/include/open3d/Open3D.h.in"
               # "${PROJECT_SOURCE_DIR}/include/Open3DNew.h")
# configure_file("${PROJECT_SOURCE_DIR}/include/open3d/Open3DConfig.h.in"
               # "${PROJECT_SOURCE_DIR}/include/open3d/Open3DConfig.h")


# add_library(Open3D)

# add_subdirectory(camera)
# add_subdirectory(core)
# add_subdirectory(data)
# add_subdirectory(geometry)
# add_subdirectory(io)
# add_subdirectory(ml)
# add_subdirectory(pipelines)
# add_subdirectory(t/geometry)
# add_subdirectory(t/io)
# add_subdirectory(t/pipelines)
# add_subdirectory(utility)
# add_subdirectory(visualization)

# if (BUILD_GUI)
    # add_subdirectory(visualization/gui)
# endif()
# s

# note: adding at least one real source file to any target that references
# reference: https://cmake.org/cmake/help/v3.12/command/add_library.html#object-libraries
# target_sources(Open3D PRIVATE
    # Open3DConfig.cpp
# )

# open3d_ispc_target_sources_TARGET_OBJECTS(Open3D PRIVATE
    # camera
    # core
    # data
    # geometry
    # tgeometry
    # tgeometry_kernel
    # io
    # tio
    # ml_contrib
    # pipelines
    # tpipelines
    # tpipelines_kernel
    # utility
    # visualization
# )

# if (BUILD_GUI)
    # open3d_ispc_target_sources_TARGET_OBJECTS(Open3D PRIVATE
        # GUI
    # )
# endif()

# if (BUILD_WEBRTC)
    # open3d_ispc_target_sources_TARGET_OBJECTS(Open3D PRIVATE
        # webrtc_server
    # )
# endif()

# Source group for Visual Studio
# add_source_group(camera)
# add_source_group(core)
# add_source_group(data)
# add_source_group(geometry)
# add_source_group(tgeometry)
# add_source_group(io)
# add_source_group(tio)
# add_source_group(ml)
# add_source_group(pipelines)
# add_source_group(tpipelines)
# add_source_group(utility)
# add_source_group(visualization)

# open3d_show_and_abort_on_warning(Open3D)
# open3d_set_global_properties(Open3D)
#open3d_set_open3d_lib_properties(Open3D)

add_executable(ICP testvisualizer.cpp ${SOURCE_FILES})
target_include_directories(ICP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/open3d)
#target_link_libraries(ICP Open3D)


#add_executable( ICP testvisualizer.cpp )
#target_include_directories(ICP PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
#ADD_SUBDIRECTORY(include)

