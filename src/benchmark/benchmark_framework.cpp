#include "benchmark/benchmark_framework.h"


#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace renderer::benchmark {

namespace {

constexpr std::array<std::uint32_t, 64> kSha256Constants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

class Sha256 {
public:
    void update(const unsigned char* data, std::size_t size) {
        total_bytes_ += size;
        while (size > 0) {
            const std::size_t chunk = std::min(size, block_.size() - block_size_);
            std::copy_n(data, chunk, block_.begin() + static_cast<std::ptrdiff_t>(block_size_));
            data += chunk;
            size -= chunk;
            block_size_ += chunk;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::string finish() {
        const std::uint64_t bit_count = static_cast<std::uint64_t>(total_bytes_) * 8U;
        block_[block_size_++] = 0x80U;
        if (block_size_ > 56) {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0U);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(
            block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
            block_.begin() + 56,
            0U);
        for (int index = 0; index < 8; ++index) {
            block_[63 - index] = static_cast<unsigned char>(bit_count >> (index * 8));
        }
        transform(block_.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (const std::uint32_t value : state_) {
            output << std::setw(8) << value;
        }
        return output.str();
    }

private:
    static std::uint32_t choose(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (~x & z);
    }

    static std::uint32_t majority(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
        return (x & y) ^ (x & z) ^ (y & z);
    }

    void transform(const unsigned char* data) {
        std::array<std::uint32_t, 64> words{};
        for (int index = 0; index < 16; ++index) {
            const int offset = index * 4;
            words[static_cast<std::size_t>(index)] =
                (static_cast<std::uint32_t>(data[offset]) << 24U) |
                (static_cast<std::uint32_t>(data[offset + 1]) << 16U) |
                (static_cast<std::uint32_t>(data[offset + 2]) << 8U) |
                static_cast<std::uint32_t>(data[offset + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const std::uint32_t previous = words[static_cast<std::size_t>(index - 15)];
            const std::uint32_t recent = words[static_cast<std::size_t>(index - 2)];
            const std::uint32_t small0 =
                std::rotr(previous, 7) ^ std::rotr(previous, 18) ^ (previous >> 3U);
            const std::uint32_t small1 =
                std::rotr(recent, 17) ^ std::rotr(recent, 19) ^ (recent >> 10U);
            words[static_cast<std::size_t>(index)] =
                words[static_cast<std::size_t>(index - 16)] + small0 +
                words[static_cast<std::size_t>(index - 7)] + small1;
        }

        std::uint32_t a = state_[0];
        std::uint32_t b = state_[1];
        std::uint32_t c = state_[2];
        std::uint32_t d = state_[3];
        std::uint32_t e = state_[4];
        std::uint32_t f = state_[5];
        std::uint32_t g = state_[6];
        std::uint32_t h = state_[7];
        for (int index = 0; index < 64; ++index) {
            const std::uint32_t large1 =
                std::rotr(e, 6) ^ std::rotr(e, 11) ^ std::rotr(e, 25);
            const std::uint32_t first =
                h + large1 + choose(e, f, g) +
                kSha256Constants[static_cast<std::size_t>(index)] +
                words[static_cast<std::size_t>(index)];
            const std::uint32_t large0 =
                std::rotr(a, 2) ^ std::rotr(a, 13) ^ std::rotr(a, 22);
            const std::uint32_t second = large0 + majority(a, b, c);
            h = g;
            g = f;
            f = e;
            e = d + first;
            d = c;
            c = b;
            b = a;
            a = first + second;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_{
        0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
        0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
    std::array<unsigned char, 64> block_{};
    std::size_t block_size_ = 0;
    std::size_t total_bytes_ = 0;
};

Vec3 parse_vec3(const nlohmann::json& value, const char* name) {
    if (!value.is_array() || value.size() != 3) {
        throw std::runtime_error(std::string(name) + " must contain three numbers");
    }
    Vec3 result(
        value.at(0).get<float>(),
        value.at(1).get<float>(),
        value.at(2).get<float>());
    if (!result.allFinite()) {
        throw std::runtime_error(std::string(name) + " must be finite");
    }
    return result;
}

std::filesystem::path normalized_absolute(const std::filesystem::path& path) {
    return std::filesystem::absolute(path).lexically_normal();
}

std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) {
        return {};
    }
    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

void add_dependency(
    std::set<std::filesystem::path>& dependencies,
    const std::filesystem::path& path) {
    const std::filesystem::path resolved = normalized_absolute(path);
    if (!std::filesystem::is_regular_file(resolved)) {
        throw std::runtime_error("benchmark input is missing: " + resolved.string());
    }
    dependencies.insert(resolved);
}

void add_mtl_dependencies(
    std::set<std::filesystem::path>& dependencies,
    const std::filesystem::path& mtl_path) {
    add_dependency(dependencies, mtl_path);
    std::ifstream input(mtl_path);
    if (!input) {
        throw std::runtime_error("failed to read MTL dependency: " + mtl_path.string());
    }
    std::string line;
    while (std::getline(input, line)) {
        const std::string text = trim(line);
        if (text.empty() || text.front() == '#') {
            continue;
        }
        std::istringstream tokens(text);
        std::string directive;
        tokens >> directive;
        std::string lower = directive;
        std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char value) {
            return static_cast<char>(std::tolower(value));
        });
        if (lower.rfind("map_", 0) != 0 && lower != "bump" && lower != "disp" &&
            lower != "decal" && lower != "refl") {
            continue;
        }
        std::string token;
        std::string last;
        while (tokens >> token) {
            last = token;
        }
        if (!last.empty()) {
            add_dependency(dependencies, mtl_path.parent_path() / last);
        }
    }
}

void add_obj_dependencies(
    std::set<std::filesystem::path>& dependencies,
    const std::filesystem::path& obj_path) {
    add_dependency(dependencies, obj_path);
    std::ifstream input(obj_path);
    if (!input) {
        throw std::runtime_error("failed to read OBJ dependency: " + obj_path.string());
    }
    std::string line;
    while (std::getline(input, line)) {
        const std::string text = trim(line);
        if (text.rfind("mtllib ", 0) == 0) {
            add_mtl_dependencies(
                dependencies,
                obj_path.parent_path() / trim(text.substr(7)));
        }
    }
}

void add_gltf_dependencies(
    std::set<std::filesystem::path>& dependencies,
    const std::filesystem::path& gltf_path) {
    add_dependency(dependencies, gltf_path);
    std::ifstream input(gltf_path);
    nlohmann::json root;
    input >> root;
    for (const char* collection : {"buffers", "images"}) {
        if (!root.contains(collection)) {
            continue;
        }
        for (const auto& item : root.at(collection)) {
            const std::string uri = item.value("uri", std::string());
            if (!uri.empty() && uri.rfind("data:", 0) != 0) {
                add_dependency(dependencies, gltf_path.parent_path() / uri);
            }
        }
    }
}

nlohmann::json metric_json(const Metric& metric) {
    const char* type = !metric.histogram.is_null()
        ? "histogram"
        : (metric.samples.size() == 1 ? "scalar" : "series");
    nlohmann::json result{
        {"name", metric.name},
        {"type", type},
        {"unit", metric.unit},
        {"instrumented", metric.instrumented},
        {"tags", metric.tags},
    };
    if (!metric.samples.empty()) {
        const Summary summary = summarize(metric.samples);
        result["samples"] = metric.samples;
        result["summary"] = {
            {"count", summary.count},
            {"min", summary.minimum},
            {"mean", summary.mean},
            {"median", summary.median},
            {"p95", summary.p95},
            {"max", summary.maximum},
            {"stddev", summary.standard_deviation},
        };
    }
    if (!metric.histogram.is_null()) {
        result["histogram"] = metric.histogram;
    }
    return result;
}

const nlohmann::json* find_metric(
    const nlohmann::json& report,
    const std::string& phase_name,
    const std::string& metric_name) {
    if (!report.contains("phases")) {
        return nullptr;
    }
    for (const auto& phase : report.at("phases")) {
        if (phase.value("name", std::string()) != phase_name) {
            continue;
        }
        for (const auto& metric : phase.at("metrics")) {
            if (metric.value("name", std::string()) == metric_name) {
                return &metric;
            }
        }
    }
    return nullptr;
}

}  // namespace

Summary summarize(const std::vector<double>& samples) {
    if (samples.empty()) {
        throw std::invalid_argument("cannot summarize an empty metric series");
    }
    for (const double value : samples) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("metric samples must be finite");
        }
    }
    std::vector<double> ordered = samples;
    std::sort(ordered.begin(), ordered.end());
    Summary result;
    result.count = ordered.size();
    result.minimum = ordered.front();
    result.maximum = ordered.back();
    result.mean = std::accumulate(ordered.begin(), ordered.end(), 0.0) /
        static_cast<double>(ordered.size());
    const std::size_t middle = ordered.size() / 2;
    result.median = ordered.size() % 2 == 0
        ? (ordered[middle - 1] + ordered[middle]) * 0.5
        : ordered[middle];
    const std::size_t p95_index = std::min(
        ordered.size() - 1,
        static_cast<std::size_t>(
            std::ceil(0.95 * static_cast<double>(ordered.size()))) - 1);
    result.p95 = ordered[p95_index];
    double squared_sum = 0.0;
    for (const double value : ordered) {
        const double delta = value - result.mean;
        squared_sum += delta * delta;
    }
    result.standard_deviation = std::sqrt(
        squared_sum / static_cast<double>(ordered.size()));
    return result;
}

CaseConfig load_case(
    const std::filesystem::path& descriptor_path,
    const std::filesystem::path& source_root) {
    const std::filesystem::path absolute_descriptor = normalized_absolute(
        descriptor_path.is_absolute()
            ? descriptor_path
            : source_root / descriptor_path);
    std::ifstream input(absolute_descriptor);
    if (!input) {
        throw std::runtime_error("failed to open benchmark case: " + absolute_descriptor.string());
    }
    nlohmann::json root;
    input >> root;
    CaseConfig config;
    config.schema_version = root.value("schema_version", 0);
    if (config.schema_version != 1) {
        throw std::runtime_error("unsupported benchmark case schema version");
    }
    config.name = root.at("name").get<std::string>();
    if (config.name.empty()) {
        throw std::runtime_error("benchmark case name cannot be empty");
    }
    config.descriptor_path = absolute_descriptor;
    config.scene_path = normalized_absolute(
        absolute_descriptor.parent_path() /
        root.at("scene").get<std::string>());
    config.width = root.at("output").at("width").get<int>();
    config.height = root.at("output").at("height").get<int>();
    config.render_scale = root.at("output").value("render_scale", 1.0f);
    if (config.width <= 0 || config.height <= 0 ||
        !std::isfinite(config.render_scale) || config.render_scale < 0.25f ||
        config.render_scale > 1.0f) {
        throw std::runtime_error("benchmark output settings are invalid");
    }
    const int render_width = std::max(
        1,
        static_cast<int>(std::lround(
            static_cast<float>(config.width) * config.render_scale)));
    const int render_height = std::max(
        1,
        static_cast<int>(std::lround(
            static_cast<float>(config.height) * config.render_scale)));

    const auto& camera = root.at("camera");
    const Vec3 eye = parse_vec3(camera.at("eye"), "camera.eye");
    const Vec3 forward = parse_vec3(camera.at("forward"), "camera.forward");
    const Vec3 up = parse_vec3(camera.at("up"), "camera.up");
    if (forward.squaredNorm() < 1.0e-12f || up.squaredNorm() < 1.0e-12f ||
        forward.cross(up).squaredNorm() < 1.0e-12f) {
        throw std::runtime_error("benchmark camera basis is invalid");
    }
    const float fov = camera.at("vertical_fov_degrees").get<float>();
    config.camera = Camera(
        eye,
        eye + forward,
        up,
        fov,
        static_cast<float>(render_width) / static_cast<float>(render_height));

    const auto& display = root.at("display");
    config.exposure_ev = display.value("exposure_ev", 0.0f);
    config.tone_mapper = display.value("tone_mapper", std::string("aces"));

    config.render_settings.width = render_width;
    config.render_settings.height = render_height;
    const auto& path = root.at("render_settings").at("path");
    config.render_settings.path.samples_per_pixel = path.value("samples_per_pixel", 1);
    config.render_settings.path.max_bounces = path.value("max_bounces", 64);
    config.render_settings.path.russian_roulette_start_bounce =
        path.value("rr_start_bounce", 3);
    config.render_settings.path.russian_roulette_min_probability =
        path.value("rr_min_probability", 0.05f);
    config.render_settings.path.russian_roulette_max_probability =
        path.value("rr_max_probability", 0.95f);
    config.render_settings.path.sample_seed_offset =
        path.value("sample_seed_offset", std::uint64_t{0});
    config.render_settings.path.cuda_device =
        std::max(0, path.value("cuda_device", 0));

    const auto& phases = root.at("phases");
    config.opengl_warmup_frames = phases.value("opengl_warmup_frames", 60);
    config.opengl_measure_frames = phases.value("opengl_measure_frames", 300);
    config.cuda_full_warmup_frames = phases.value("cuda_full_warmup_frames", 1);
    config.cuda_full_measure_frames = phases.value("cuda_full_measure_frames", 5);
    config.cuda_interaction_frames = phases.value("cuda_interaction_frames", 30);
    config.cuda_native_sweeps = phases.value("cuda_native_sweeps", 10);
    config.diagnostic_samples = phases.value("diagnostic_samples", 1);
    if (config.opengl_warmup_frames < 0 || config.opengl_measure_frames <= 0 ||
        config.cuda_full_warmup_frames < 0 || config.cuda_full_measure_frames <= 0 ||
        config.cuda_interaction_frames <= 1 || config.cuda_native_sweeps <= 0 ||
        config.diagnostic_samples <= 0) {
        throw std::runtime_error("benchmark phase counts are invalid");
    }
    config.source = std::move(root);
    return config;
}

std::vector<InputFingerprint> fingerprint_case_inputs(
    const CaseConfig& config,
    const std::filesystem::path& source_root) {
    std::set<std::filesystem::path> dependencies;
    add_dependency(dependencies, config.descriptor_path);
    add_dependency(dependencies, config.scene_path);

    std::ifstream scene_input(config.scene_path);
    nlohmann::json scene;
    scene_input >> scene;
    const std::filesystem::path scene_base = config.scene_path.parent_path();
    if (scene.contains("assets")) {
        for (const auto& asset : scene.at("assets")) {
            const std::filesystem::path asset_path = scene_base /
                asset.at("path").get<std::string>();
            std::string extension = asset_path.extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char value) {
                return static_cast<char>(std::tolower(value));
            });
            if (extension == ".obj") {
                add_obj_dependencies(dependencies, asset_path);
            } else if (extension == ".gltf") {
                add_gltf_dependencies(dependencies, asset_path);
            } else {
                add_dependency(dependencies, asset_path);
            }
        }
    }
    if (scene.contains("environment")) {
        const std::string environment_path =
            scene.at("environment").value("path", std::string());
        if (!environment_path.empty()) {
            add_dependency(dependencies, scene_base / environment_path);
        }
    }

    const std::filesystem::path absolute_root = normalized_absolute(source_root);
    std::vector<InputFingerprint> result;
    result.reserve(dependencies.size());
    for (const std::filesystem::path& dependency : dependencies) {
        std::error_code relative_error;
        std::filesystem::path relative = std::filesystem::relative(
            dependency,
            absolute_root,
            relative_error);
        result.push_back(InputFingerprint{
            relative_error ? dependency.generic_string() : relative.generic_string(),
            std::filesystem::file_size(dependency),
            sha256_file(dependency)});
    }
    return result;
}

std::string sha256_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to hash input: " + path.string());
    }
    Sha256 hash;
    std::array<unsigned char, 64 * 1024> buffer{};
    while (input) {
        input.read(
            reinterpret_cast<char*>(buffer.data()),
            static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            hash.update(buffer.data(), static_cast<std::size_t>(count));
        }
    }
    return hash.finish();
}

std::string sha256_text(const std::string& text) {
    Sha256 hash;
    hash.update(
        reinterpret_cast<const unsigned char*>(text.data()),
        text.size());
    return hash.finish();
}

std::string utc_timestamp() {
    const auto now = std::chrono::system_clock::now();
    const std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#ifdef _WIN32
    gmtime_s(&utc, &time);
#else
    gmtime_r(&time, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

nlohmann::json report_json(const Report& report) {
    nlohmann::json result{
        {"schema_version", report.schema_version},
        {"kind", report.kind},
        {"generated_at_utc", report.generated_at_utc},
        {"metadata", report.metadata},
        {"case", report.case_metadata},
        {"compatibility_key", report.compatibility_key},
        {"inputs", nlohmann::json::array()},
        {"phases", nlohmann::json::array()},
        {"comparisons", report.comparisons},
    };
    for (const InputFingerprint& input : report.inputs) {
        result["inputs"].push_back({
            {"path", input.path},
            {"bytes", input.bytes},
            {"sha256", input.sha256},
        });
    }
    for (const PhaseResult& phase : report.phases) {
        nlohmann::json phase_json{
            {"name", phase.name},
            {"backend", phase.backend},
            {"status", phase.status},
            {"detail", phase.detail},
            {"metrics", nlohmann::json::array()},
        };
        for (const Metric& metric : phase.metrics) {
            phase_json["metrics"].push_back(metric_json(metric));
        }
        result["phases"].push_back(std::move(phase_json));
    }
    return result;
}

void compare_with_baseline(
    Report& report,
    const std::filesystem::path& baseline_path) {
    std::ifstream input(baseline_path);
    if (!input) {
        throw std::runtime_error("failed to open benchmark baseline: " + baseline_path.string());
    }
    nlohmann::json baseline;
    input >> baseline;
    if (baseline.value("compatibility_key", std::string()) !=
        report.compatibility_key) {
        report.comparisons.push_back({
            {"status", "incompatible"},
            {"baseline", baseline_path.generic_string()},
            {"detail", "compatibility keys differ"},
        });
        return;
    }
    const nlohmann::json current = report_json(report);
    for (const PhaseResult& phase : report.phases) {
        for (const Metric& metric : phase.metrics) {
            if (metric.samples.empty()) {
                continue;
            }
            const nlohmann::json* baseline_metric = find_metric(
                baseline,
                phase.name,
                metric.name);
            if (!baseline_metric || !baseline_metric->contains("summary")) {
                report.comparisons.push_back({
                    {"status", "new_metric"},
                    {"phase", phase.name},
                    {"metric", metric.name},
                });
                continue;
            }
            const nlohmann::json* current_metric = find_metric(
                current,
                phase.name,
                metric.name);
            const double previous =
                baseline_metric->at("summary").at("median").get<double>();
            const double present =
                current_metric->at("summary").at("median").get<double>();
            report.comparisons.push_back({
                {"status", "compared"},
                {"phase", phase.name},
                {"metric", metric.name},
                {"baseline_median", previous},
                {"current_median", present},
                {"change_percent", previous == 0.0
                    ? 0.0
                    : 100.0 * (present - previous) / previous},
            });
        }
    }
}

void write_report_files(
    const Report& report,
    const std::filesystem::path& output_directory) {
    std::filesystem::create_directories(output_directory);
    const nlohmann::json raw = report_json(report);
    {
        std::ofstream output(output_directory / "raw.json");
        if (!output) {
            throw std::runtime_error("failed to create benchmark raw.json");
        }
        output << std::setw(2) << raw << '\n';
    }
    std::ofstream markdown(output_directory / "summary.md");
    if (!markdown) {
        throw std::runtime_error("failed to create benchmark summary.md");
    }
    markdown << "# Benchmark: "
             << report.case_metadata.value("name", std::string("unknown"))
             << "\n\n"
             << "- Generated: `" << report.generated_at_utc << "`\n"
             << "- Kind: `" << report.kind << "`\n"
             << "- Compatibility key: `" << report.compatibility_key << "`\n\n"
             << "| Phase | Backend | Metric | Median | P95 | Max | Unit |\n"
             << "|---|---|---|---:|---:|---:|---|\n";
    for (const PhaseResult& phase : report.phases) {
        for (const Metric& metric : phase.metrics) {
            if (metric.samples.empty()) {
                continue;
            }
            const Summary summary = summarize(metric.samples);
            markdown << "| " << phase.name
                     << " | " << phase.backend
                     << " | `" << metric.name << "`"
                     << " | " << summary.median
                     << " | " << summary.p95
                     << " | " << summary.maximum
                     << " | " << metric.unit << " |\n";
        }
    }
    bool has_histograms = false;
    for (const PhaseResult& phase : report.phases) {
        for (const Metric& metric : phase.metrics) {
            if (!metric.histogram.is_null()) {
                if (!has_histograms) {
                    markdown << "\n## Histograms\n\n";
                    has_histograms = true;
                }
                markdown << "### `" << metric.name << "`\n\n```json\n"
                         << std::setw(2) << metric.histogram << "\n```\n\n";
            }
        }
    }
    if (!report.comparisons.empty()) {
        markdown << "\n## Baseline comparison\n\n"
                 << "| Status | Phase | Metric | Change |\n"
                 << "|---|---|---|---:|\n";
        for (const auto& comparison : report.comparisons) {
            markdown << "| " << comparison.value("status", std::string())
                     << " | " << comparison.value("phase", std::string())
                     << " | `" << comparison.value("metric", std::string()) << "`"
                     << " | ";
            if (comparison.contains("change_percent")) {
                markdown << comparison.at("change_percent").get<double>() << "%";
            }
            markdown << " |\n";
        }
    }
}

}  // namespace renderer::benchmark
