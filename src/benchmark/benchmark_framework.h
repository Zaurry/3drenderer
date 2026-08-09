#pragma once

#include "render/render_settings.h"
#include "scene/camera.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace renderer::benchmark {

struct Summary {
    std::size_t count = 0;
    double minimum = 0.0;
    double mean = 0.0;
    double median = 0.0;
    double p95 = 0.0;
    double maximum = 0.0;
    double standard_deviation = 0.0;
};

struct Metric {
    std::string name;
    std::string unit;
    bool instrumented = false;
    std::vector<double> samples;
    nlohmann::json tags = nlohmann::json::object();
    nlohmann::json histogram;
};

struct PhaseResult {
    std::string name;
    std::string backend;
    std::string status = "ok";
    std::string detail;
    std::vector<Metric> metrics;
};

struct CaseConfig {
    int schema_version = 1;
    std::string name;
    std::filesystem::path descriptor_path;
    std::filesystem::path scene_path;
    int width = 0;
    int height = 0;
    float render_scale = 1.0f;
    float exposure_ev = 0.0f;
    std::string tone_mapper = "aces";
    Camera camera{
        Vec3(0.0f, 0.0f, 1.0f),
        Vec3::Zero(),
        Vec3(0.0f, 1.0f, 0.0f),
        45.0f,
        1.0f};
    RenderSettings render_settings;
    int opengl_warmup_frames = 60;
    int opengl_measure_frames = 300;
    int cuda_full_warmup_frames = 1;
    int cuda_full_measure_frames = 5;
    int cuda_interaction_frames = 30;
    int cuda_native_sweeps = 10;
    int diagnostic_samples = 1;
    nlohmann::json source;
};

struct InputFingerprint {
    std::string path;
    std::uintmax_t bytes = 0;
    std::string sha256;
};

struct Report {
    int schema_version = 1;
    std::string kind = "timing";
    std::string generated_at_utc;
    nlohmann::json metadata = nlohmann::json::object();
    nlohmann::json case_metadata = nlohmann::json::object();
    std::string compatibility_key;
    std::vector<InputFingerprint> inputs;
    std::vector<PhaseResult> phases;
    nlohmann::json comparisons = nlohmann::json::array();
};

Summary summarize(const std::vector<double>& samples);
CaseConfig load_case(
    const std::filesystem::path& descriptor_path,
    const std::filesystem::path& source_root);
std::vector<InputFingerprint> fingerprint_case_inputs(
    const CaseConfig& config,
    const std::filesystem::path& source_root);
std::string sha256_file(const std::filesystem::path& path);
std::string sha256_text(const std::string& text);
std::string utc_timestamp();

nlohmann::json report_json(const Report& report);
void compare_with_baseline(
    Report& report,
    const std::filesystem::path& baseline_path);
void write_report_files(
    const Report& report,
    const std::filesystem::path& output_directory);

}  // namespace renderer::benchmark
