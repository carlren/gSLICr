cmake_minimum_required(VERSION 2.8)
project(gSLIC2)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

IF(MSVC_IDE)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-DUSING_CMAKE=1)
ELSE(MSVC_IDE)
  set(CFLAGS_WARN "-Wall -Wextra -Wno-unused-parameter -Wno-strict-aliasing")
  set(CMAKE_CXX_FLAGS "-fPIC -O3 -march=native ${CFLAGS_WARN} ${CMAKE_CXX_FLAGS}")
ENDIF(MSVC_IDE)

if(APPLE)
  set(CUDA_HOST_COMPILER /usr/bin/clang)
endif(APPLE)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(ORUtils)

set(GSLIC_ENGINES
engines/gSLIC_core_engine.h
engines/gSLIC_seg_engine.h
engines/gSLIC_seg_engine_GPU.h
engines/gSLIC_seg_engine_shared.h
engines/gSLIC_core_engine.cpp
engines/gSLIC_seg_engine.cpp
engines/gSLIC_seg_engine_GPU.cu
)

set(GSLIC_OBJECTS
objects/gSLIC_settings.h
objects/gSLIC_spixel_info.h
)

set(GSLIC_UTILS
gSLIC_defines.h
NVTimer.h
)

SOURCE_GROUP(engines FILES ${GSLIC_ENGINES})
SOURCE_GROUP(objects FILES ${GSLIC_OBJECTS})
SOURCE_GROUP(utils FILES ${GSLIC_UTILS})

cuda_add_library(gSLIC2_lib
			${GSLIC_ENGINES}
			${GSLIC_OBJECTS}
			${ORUTILS_HEADERS}
			${GSLIC_UTILS}
			OPTIONS -gencode arch=compute_30,code=compute_30)
target_link_libraries(gSLIC2_lib ${CUDA_LIBRARY})

add_executable(demo demo.cpp)
target_link_libraries(demo gSLIC2_lib ${OpenCV_LIBS})
