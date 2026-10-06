#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace minitest {

using Test = std::pair<const char*, std::function<void()>>;

inline std::vector<Test>& registry() {
    static std::vector<Test> tests;
    return tests;
}

struct Registrar {
    Registrar(const char* name, std::function<void()> test) {
        registry().emplace_back(name, std::move(test));
    }
};

inline void check(bool condition, const char* expression, const char* file,
                  int line) {
    if (!condition) {
        throw std::runtime_error(std::string(file) + ":" +
                                 std::to_string(line) + ": CHECK(" +
                                 expression + ") failed");
    }
}

inline int run_all() {
    int failures = 0;
    for (const auto& [name, test] : registry()) {
        try {
            test();
            std::cout << "PASS " << name << "\n";
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << "FAIL " << name << ": " << error.what() << "\n";
        }
    }
    return failures == 0 ? 0 : 1;
}

}  // namespace minitest

#define IE_TEST_CASE(name)                                                    \
    static void name();                                                        \
    static ::minitest::Registrar name##_registrar(#name, name);               \
    static void name()

#define IE_CHECK(expression)                                                    \
    ::minitest::check((expression), #expression, __FILE__, __LINE__)
