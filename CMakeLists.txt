cmake_minimum_required(VERSION 2.8)
project(gSLICr)

set(CMAKE_CXX_STANDARD 11)

IF(MSVC_IDE)
  set(OpenCV_STATIC OFF)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(ORUtils)

set(GSLICR_LIB
gSLICr_Lib/engines/gSLICr_core_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine.h
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.h
gSLICr_Lib/engines/gSLICr_seg_engine_shared.h
gSLICr_Lib/engines/gSLICr_core_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine.cpp
gSLICr_Lib/engines/gSLICr_seg_engine_GPU.cu
gSLICr_Lib/objects/gSLICr_settings.h
gSLICr_Lib/objects/gSLICr_spixel_info.h
gSLICr_Lib/gSLICr_defines.h
gSLICr_Lib/gSLICr.h
)

list(APPEND "-ftree-vectorize")
SOURCE_GROUP(engines FILES ${GSLICR_LIB})

cuda_add_library(gSLICr_lib
			${GSLICR_LIB}
			NVTimer.h
			OPTIONS -gencode arch=compute_30,code=compute_30)
target_link_libraries(gSLICr_lib ${CUDA_LIBRARY})

add_executable(demo demo.cpp)
target_link_libraries(demo gSLICr_lib ${OpenCV_LIBS})

include(GNUInstallDirs)

# install library
install(TARGETS gSLICr_lib EXPORT ${PROJECT_NAME}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# install header file hierarchy
file(GLOB_RECURSE HEADER_FILES RELATIVE ${CMAKE_SOURCE_DIR} *.h)
foreach(HEADER ${HEADER_FILES})
    string(REGEX MATCH "(.*)[/\\]" DIR ${HEADER})
    install(FILES ${HEADER} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/${PROJECT_NAME}/${DIR})
endforeach()

target_include_directories(gSLICr_lib PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>"
    "$<INSTALL_INTERFACE:$<INSTALL_PREFIX>/${CMAKE_INSTALL_INCLUDEDIR}>/${PROJECT_NAME}/gSLICr_Lib/")

# export library
install(EXPORT ${PROJECT_NAME}
    DESTINATION share/${PROJECT_NAME}/cmake
    FILE ${PROJECT_NAME}Config.cmake
)
