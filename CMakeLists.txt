cmake_minimum_required(VERSION 3.16)
project(Kahu VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(KAHU_BUILD_GUI "Build the Qt GUI" ON)
option(KAHU_BUILD_TRAINER "Build the NNUE trainer" ON)
option(KAHU_NATIVE_ARCH "Optimize for host CPU (-march=native)" ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(ENGINE_SOURCES
  src/main.cpp
  src/engine.cpp
  src/position.cpp
  src/zobrist.cpp
  src/protocol.cpp
  src/datagen.cpp
  src/nnue.cpp
)

find_package(Threads REQUIRED)

function(kahu_set_arch_flags)
  if(KAHU_NATIVE_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
      set(KAHU_ARCH_FLAGS "-mcpu=native" PARENT_SCOPE)
      set(KAHU_SIMD_LABEL "NEON" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      set(KAHU_ARCH_FLAGS "-march=native;-mavx2;-mfma;-mbmi2" PARENT_SCOPE)
      set(KAHU_SIMD_LABEL "AVX2" PARENT_SCOPE)
    else()
      set(KAHU_ARCH_FLAGS "-march=native" PARENT_SCOPE)
      set(KAHU_SIMD_LABEL "native" PARENT_SCOPE)
    endif()
  else()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      set(KAHU_ARCH_FLAGS "-mavx2;-mfma;-mbmi2" PARENT_SCOPE)
      set(KAHU_SIMD_LABEL "AVX2" PARENT_SCOPE)
    else()
      set(KAHU_ARCH_FLAGS "" PARENT_SCOPE)
      set(KAHU_SIMD_LABEL "portable" PARENT_SCOPE)
    endif()
  endif()
endfunction()

kahu_set_arch_flags()

function(kahu_apply_target_options target is_trainer)
  target_compile_options(${target} PRIVATE
    -Wall
    -Wextra
    ${KAHU_ARCH_FLAGS}
  )
  target_link_libraries(${target} PRIVATE Threads::Threads)

  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
      target_compile_options(${target} PRIVATE /arch:AVX2)
    endif()
    if(NOT is_trainer)
      target_compile_options(${target} PRIVATE /fp:fast)
    endif()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(${target} PRIVATE /Od /Zi)
    else()
      target_compile_options(${target} PRIVATE /O2 /DNDEBUG)
    endif()
  else()
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
      target_compile_options(${target} PRIVATE -O0 -g)
      target_link_options(${target} PRIVATE -g)
    else()
      target_compile_options(${target} PRIVATE -O3 -DNDEBUG -flto)
      target_link_options(${target} PRIVATE -flto)
      if(NOT is_trainer)
        target_compile_options(${target} PRIVATE -ffast-math)
      endif()
    endif()
  endif()
endfunction()

add_executable(kuba_engine ${ENGINE_SOURCES})
kahu_apply_target_options(kuba_engine FALSE)

if(KAHU_BUILD_TRAINER)
  add_executable(nnue_trainer nnue_trainer.cpp)
  kahu_apply_target_options(nnue_trainer TRUE)
  message(STATUS "NNUE trainer enabled (SIMD label: ${KAHU_SIMD_LABEL})")
endif()

if(EXISTS "${CMAKE_SOURCE_DIR}/kuba.nnue")
  add_custom_command(TARGET kuba_engine POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${CMAKE_SOURCE_DIR}/kuba.nnue"
      "$<TARGET_FILE_DIR:kuba_engine>/kuba.nnue"
    COMMENT "Copying kuba.nnue next to kuba_engine"
  )
endif()

if(KAHU_BUILD_GUI)
  find_package(Qt6 QUIET COMPONENTS Core Gui Widgets)
  if(Qt6_FOUND)
    add_subdirectory(gui)
  else()
    message(STATUS "Qt6 not found; skipping GUI (install Qt6 or pass -DKAHU_BUILD_GUI=OFF)")
  endif()
endif()
