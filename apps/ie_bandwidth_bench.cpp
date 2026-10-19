#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Config {
    std::size_t size_mb{256};
    std::size_t threads{1};
    std::size_t trials{5};
    std::size_t warmup{1};
    bool sweep{false};
};

[[nodiscard]] Config parse_args(int argc, char** argv) {
    Config config;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto require_val = [&](const std::string& name) {
            if (i + 1 >= argc) {
                throw std::invalid_argument("missing value for " + name);
            }
            return std::string(argv[++i]);
        };
        if (arg == "--size-mb") {
            config.size_mb = std::stoull(require_val(arg));
        } else if (arg == "--threads") {
            config.threads = std::stoull(require_val(arg));
        } else if (arg == "--trials") {
            config.trials = std::stoull(require_val(arg));
        } else if (arg == "--warmup") {
            config.warmup = std::stoull(require_val(arg));
        } else if (arg == "--sweep") {
            config.sweep = true;
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (config.size_mb == 0 || config.threads == 0 || config.trials == 0) {
        throw std::invalid_argument("invalid configuration parameters");
    }
    return config;
}

// Prevents compiler from optimizing away memory reads
volatile std::uint64_t g_sink = 0;

void read_chunk(const std::uint64_t* data, std::size_t count, std::uint64_t& out_sum) {
    std::uint64_t sum = 0;
    for (std::size_t i = 0; i < count; ++i) {
        sum += data[i];
    }
    out_sum = sum;
}

double run_benchmark_pass(const std::vector<std::uint64_t>& buffer, std::size_t thread_count) {
    const std::size_t total_elements = buffer.size();
    const std::size_t chunk_size = total_elements / thread_count;
    std::vector<std::uint64_t> thread_sums(thread_count, 0);
    std::vector<std::thread> workers;
    workers.reserve(thread_count);

    const auto start = std::chrono::steady_clock::now();

    for (std::size_t t = 0; t < thread_count; ++t) {
        const std::size_t offset = t * chunk_size;
        const std::size_t count = (t + 1 == thread_count) ? (total_elements - offset) : chunk_size;
        workers.emplace_back(read_chunk, buffer.data() + offset, count, std::ref(thread_sums[t]));
    }

    for (auto& worker : workers) {
        worker.join();
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed_sec = std::chrono::duration<double>(end - start).count();

    std::uint64_t total_sum = 0;
    for (const auto s : thread_sums) {
        total_sum += s;
    }
    g_sink = total_sum;

    const double bytes_read = static_cast<double>(total_elements * sizeof(std::uint64_t));
    const double gb_read = bytes_read / 1.0e9; // decimal gigabytes
    return gb_read / elapsed_sec;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Config config = parse_args(argc, argv);
        const std::size_t total_bytes = config.size_mb * 1024ULL * 1024ULL;
        const std::size_t element_count = total_bytes / sizeof(std::uint64_t);

        std::cout << "=== Memory Bandwidth Calibration Benchmark ===\n"
                  << "Buffer size: " << config.size_mb << " MiB ("
                  << total_bytes << " bytes)\n"
                  << "Threads: " << config.threads << "\n"
                  << "Warm-up passes: " << config.warmup << "\n"
                  << "Measured trials: " << config.trials << "\n\n";

        // Allocate and initialize buffer
        std::vector<std::uint64_t> buffer(element_count);
        for (std::size_t i = 0; i < element_count; ++i) {
            buffer[i] = static_cast<std::uint64_t>(i ^ 0x5a5a5a5aULL);
        }

        const std::vector<std::size_t> thread_counts =
            config.sweep ? std::vector<std::size_t>{1, 2, 4, 8}
                         : std::vector<std::size_t>{config.threads};

        for (const auto threads : thread_counts) {
            std::cout << "--- Benchmarking with " << threads << " thread(s) ---\n";
            // Warm-up
            for (std::size_t w = 0; w < config.warmup; ++w) {
                static_cast<void>(run_benchmark_pass(buffer, threads));
            }

            // Measured trials
            std::vector<double> bandwidths;
            bandwidths.reserve(config.trials);
            for (std::size_t trial = 0; trial < config.trials; ++trial) {
                const double bw = run_benchmark_pass(buffer, threads);
                bandwidths.push_back(bw);
                std::cout << "  Trial " << (trial + 1) << ": "
                          << std::fixed << std::setprecision(2) << bw << " GB/s\n";
            }

            std::sort(bandwidths.begin(), bandwidths.end());
            const double median_bw = bandwidths[bandwidths.size() / 2];
            const double min_bw = bandwidths.front();
            const double max_bw = bandwidths.back();

            std::cout << "  Median: " << std::fixed << std::setprecision(2) << median_bw
                      << " GB/s (Min: " << min_bw << ", Max: " << max_bw << ")\n\n";

            std::cout << "CSV," << config.size_mb << "," << threads << ","
                      << std::fixed << std::setprecision(4)
                      << median_bw << "," << min_bw << "," << max_bw << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "ie_bandwidth_bench: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
