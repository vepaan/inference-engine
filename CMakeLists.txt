cmake_minimum_required(VERSION 3.20)

project(inference_engine_mvp LANGUAGES CXX)

option(IE_WITH_LLAMA "Build the real llama.cpp backend and smoke app" OFF)

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

if(IE_WITH_LLAMA)
	set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
	set(GGML_CUDA OFF CACHE BOOL "" FORCE)
	set(GGML_METAL OFF CACHE BOOL "" FORCE)
	add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

	add_library(ie_llm STATIC
		src/llm/llama_wrap.cpp
		src/llm/llama_backend.cpp
	)

	target_compile_features(ie_llm PUBLIC cxx_std_20)
	target_include_directories(ie_llm
		PUBLIC
			${CMAKE_CURRENT_SOURCE_DIR}/core
			${CMAKE_CURRENT_SOURCE_DIR}/src/llm
	)
	target_link_libraries(ie_llm PUBLIC llama)

	if(MSVC)
		target_compile_options(ie_llm PRIVATE /W4 /WX /permissive-)
	else()
		target_compile_options(ie_llm PRIVATE
			-Wall
			-Wextra
			-Wpedantic
			-Wconversion
			-Wsign-conversion
			-Wshadow
			-Werror
		)
	endif()

	add_executable(ie_smoke
		apps/ie_smoke.cpp
	)
	target_compile_features(ie_smoke PRIVATE cxx_std_20)
	target_link_libraries(ie_smoke PRIVATE ie_llm)

	if(MSVC)
		target_compile_options(ie_smoke PRIVATE /W4 /WX /permissive-)
	else()
		target_compile_options(ie_smoke PRIVATE
			-Wall
			-Wextra
			-Wpedantic
			-Wconversion
			-Wsign-conversion
			-Wshadow
			-Werror
		)
	endif()
endif()
