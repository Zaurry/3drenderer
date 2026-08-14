#include "test_framework.h"

#include "benchmark/benchmark_framework.h"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

bool near(double left, double right) {
    return std::abs(left - right) <= 1.0e-6;
}

RENDER_TEST(test_benchmark_summary_statistics) {
    const renderer::benchmark::Summary summary =
        renderer::benchmark::summarize({5.0, 1.0, 4.0, 2.0, 3.0});
    RENDER_CHECK(summary.count == 5);
    RENDER_CHECK(near(summary.minimum, 1.0));
    RENDER_CHECK(near(summary.mean, 3.0));
    RENDER_CHECK(near(summary.median, 3.0));
    RENDER_CHECK(near(summary.p95, 5.0));
    RENDER_CHECK(near(summary.maximum, 5.0));
}

RENDER_TEST(test_benchmark_sha256_vector) {
    RENDER_CHECK(
        renderer::benchmark::sha256_text("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
}

RENDER_TEST(test_benchmark_case_contract) {
    const std::filesystem::path source_root(RENDERER_SOURCE_DIR);
    const renderer::benchmark::CaseConfig config =
        renderer::benchmark::load_case(
            source_root / "benchmarks/cases/san_miguel_first_scene.json",
            source_root);
    RENDER_CHECK(config.name == "san_miguel_first_scene");
    RENDER_CHECK(config.width == 2418);
    RENDER_CHECK(config.height == 1343);
    RENDER_CHECK(config.render_settings.width == 2418);
    RENDER_CHECK(config.render_settings.height == 1343);
    RENDER_CHECK(config.render_settings.path.russian_roulette_start_bounce == 3);
    RENDER_CHECK(near(
        config.render_settings.path.russian_roulette_min_probability,
        0.05));
    RENDER_CHECK(near(
        config.render_settings.path.russian_roulette_max_probability,
        0.95));
    RENDER_CHECK(std::filesystem::is_regular_file(config.scene_path));
}

renderer::benchmark::Report comparison_report(
    const std::string& compatibility_key,
    double sample) {
    renderer::benchmark::Report report;
    report.compatibility_key = compatibility_key;
    renderer::benchmark::PhaseResult phase;
    phase.name = "steady";
    phase.backend = "opengl";
    phase.metrics.push_back(renderer::benchmark::Metric{
        "opengl.frame.gpu_ms", "ms", false, {sample}});
    report.phases.push_back(std::move(phase));
    return report;
}

RENDER_TEST(test_benchmark_baseline_comparison_is_report_only) {
    const auto unique = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    const std::filesystem::path baseline_path =
        std::filesystem::temp_directory_path() /
        ("renderer-benchmark-baseline-" + std::to_string(unique) + ".json");
    {
        std::ofstream output(baseline_path);
        output << renderer::benchmark::report_json(
            comparison_report("compatible", 2.0));
    }

    renderer::benchmark::Report compatible =
        comparison_report("compatible", 3.0);
    renderer::benchmark::compare_with_baseline(compatible, baseline_path);
    RENDER_CHECK(compatible.comparisons.size() == 1);
    RENDER_CHECK(compatible.comparisons.at(0).at("status") == "compared");
    RENDER_CHECK(near(
        compatible.comparisons.at(0).at("change_percent").get<double>(),
        50.0));

    renderer::benchmark::Report incompatible =
        comparison_report("different", 3.0);
    renderer::benchmark::compare_with_baseline(incompatible, baseline_path);
    RENDER_CHECK(incompatible.comparisons.size() == 1);
    RENDER_CHECK(incompatible.comparisons.at(0).at("status") == "incompatible");
    std::filesystem::remove(baseline_path);
}

RENDER_TEST(test_benchmark_report_shape) {
    renderer::benchmark::Report report;
    report.kind = "timing";
    report.generated_at_utc = "2026-08-09T00:00:00Z";
    report.case_metadata = {{"name", "test"}};
    renderer::benchmark::PhaseResult phase;
    phase.name = "steady";
    phase.backend = "opengl";
    phase.metrics.push_back(renderer::benchmark::Metric{
        "opengl.frame.gpu_ms", "ms", false, {1.0, 2.0, 3.0}});
    report.phases.push_back(std::move(phase));
    const nlohmann::json json = renderer::benchmark::report_json(report);
    RENDER_CHECK(json.at("schema_version") == 1);
    RENDER_CHECK(json.at("phases").size() == 1);
    RENDER_CHECK(
        json.at("phases").at(0).at("metrics").at(0).at("type") ==
        "series");
    RENDER_CHECK(
        near(
            json.at("phases").at(0).at("metrics").at(0)
                .at("summary").at("median").get<double>(),
            2.0));
}

}  // namespace
