cmake_minimum_required(VERSION 3.20)

project(inference_engine_mvp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(kv_tree STATIC
	core/kv_tree.cpp
)

target_compile_features(kv_tree PUBLIC cxx_std_20)

target_include_directories(kv_tree
	PUBLIC
		${CMAKE_CURRENT_SOURCE_DIR}/core
)

if(MSVC)
	target_compile_options(kv_tree PRIVATE /W4 /WX /permissive-)
else()
	target_compile_options(kv_tree PRIVATE
		-Wall
		-Wextra
		-Wpedantic
		-Wconversion
		-Wsign-conversion
		-Wshadow
		-Werror
	)
endif()

add_executable(memory_bench
	bench/main_bench.cpp
)

target_compile_features(memory_bench PRIVATE cxx_std_20)

target_include_directories(memory_bench
	PRIVATE
		${CMAKE_CURRENT_SOURCE_DIR}/bench
)

target_link_libraries(memory_bench PRIVATE kv_tree)

if(MSVC)
	target_compile_options(memory_bench PRIVATE /W4 /WX /permissive-)
else()
	target_compile_options(memory_bench PRIVATE
		-Wall
		-Wextra
		-Wpedantic
		-Wconversion
		-Wsign-conversion
		-Wshadow
		-Werror
	)
endif()
