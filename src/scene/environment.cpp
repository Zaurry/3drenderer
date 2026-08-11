#include "scene/environment.h"

#include <tinyexr.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <string>

#include "stb_image.h"

namespace renderer {

namespace {

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

float clamp_unit(float value) {
    return std::clamp(value, 0.0f, std::nextafter(1.0f, 0.0f));
}

float wrap_unit(float value) {
    value -= std::floor(value);
    return value < 0.0f ? value + 1.0f : value;
}

int wrap_index(int value, int size) {
    const int wrapped = value % size;
    return wrapped < 0 ? wrapped + size : wrapped;
}

float luminance(const Color& color) {
    return std::max(0.0f, color.dot(Color(0.2126f, 0.7152f, 0.0722f)));
}

float decode_srgb(unsigned char value) {
    const float encoded = static_cast<float>(value) / 255.0f;
    return encoded <= 0.04045f
        ? encoded / 12.92f
        : std::pow((encoded + 0.055f) / 1.055f, 2.4f);
}

std::size_t checked_pixel_count(int width, int height, const char* label) {
    if (width <= 0 || height <= 0) {
        throw std::runtime_error(std::string(label) + " has invalid dimensions");
    }
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (count > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(std::string(label) + " is too large");
    }
    return count;
}

std::array<float, 9> sh_basis(const Vec3& d) {
    return {
        0.2820947918f,
        0.4886025119f * d.y(),
        0.4886025119f * d.z(),
        0.4886025119f * d.x(),
        1.0925484306f * d.x() * d.y(),
        1.0925484306f * d.y() * d.z(),
        0.3153915653f * (3.0f * d.z() * d.z() - 1.0f),
        1.0925484306f * d.x() * d.z(),
        0.5462742153f * (d.x() * d.x() - d.y() * d.y())};
}

Vec3 rotate_y(const Vec3& direction, float degrees) {
    const float radians = degrees * kPi / 180.0f;
    const float cosine = std::cos(radians);
    const float sine = std::sin(radians);
    return Vec3(
        cosine * direction.x() + sine * direction.z(),
        direction.y(),
        -sine * direction.x() + cosine * direction.z());
}

std::string lowercase_extension(const std::filesystem::path& path) {
    std::string extension = path.extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return extension;
}

std::vector<Color> load_exr_pixels(const std::filesystem::path& path, int& width, int& height) {
    float* rgba = nullptr;
    const char* error = nullptr;
    const int result = LoadEXR(&rgba, &width, &height, path.string().c_str(), &error);
    if (result != TINYEXR_SUCCESS) {
        std::string message = "Failed to load EXR environment: " + path.string();
        if (error) {
            message += " (" + std::string(error) + ")";
            FreeEXRErrorMessage(error);
        }
        throw std::runtime_error(message);
    }
    const std::unique_ptr<float, decltype(&std::free)> owned_rgba(rgba, &std::free);
    std::vector<Color> pixels;
    const std::size_t count = checked_pixel_count(width, height, "EXR environment");
    pixels.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t base = static_cast<std::size_t>(index) * 4U;
        pixels.emplace_back(rgba[base], rgba[base + 1U], rgba[base + 2U]);
    }
    return pixels;
}

std::vector<Color> load_stb_pixels(
    const std::filesystem::path& path,
    int& width,
    int& height) {
    const std::string native_path = path.string();
    int channels = 0;
    std::vector<Color> pixels;
    if (stbi_is_hdr(native_path.c_str())) {
        float* data = stbi_loadf(
            native_path.c_str(), &width, &height, &channels, 3);
        if (!data) {
            std::string message = "Failed to load HDR environment: " + native_path;
            if (const char* reason = stbi_failure_reason()) {
                message += " (" + std::string(reason) + ")";
            }
            throw std::runtime_error(message);
        }
        const std::unique_ptr<float, decltype(&stbi_image_free)> owned_data(
            data,
            &stbi_image_free);
        const std::size_t count = checked_pixel_count(width, height, "HDR environment");
        pixels.reserve(count);
        for (std::size_t index = 0; index < count; ++index) {
            const std::size_t base = index * 3U;
            pixels.emplace_back(data[base], data[base + 1U], data[base + 2U]);
        }
        return pixels;
    }

    unsigned char* data = stbi_load(
        native_path.c_str(), &width, &height, &channels, 3);
    if (!data) {
        std::string message = "Failed to load LDR environment: " + native_path;
        if (const char* reason = stbi_failure_reason()) {
            message += " (" + std::string(reason) + ")";
        }
        throw std::runtime_error(message);
    }
    const std::unique_ptr<unsigned char, decltype(&stbi_image_free)> owned_data(
        data,
        &stbi_image_free);
    const std::size_t count = checked_pixel_count(width, height, "LDR environment");
    pixels.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        const std::size_t base = index * 3U;
        pixels.emplace_back(
            decode_srgb(data[base]),
            decode_srgb(data[base + 1U]),
            decode_srgb(data[base + 2U]));
    }
    return pixels;
}

}  // namespace

std::shared_ptr<const EnvironmentMap> EnvironmentMap::load(
    const std::filesystem::path& path) {
    if (path.empty()) {
        throw std::invalid_argument("Environment path is empty");
    }
    int width = 0;
    int height = 0;
    std::vector<Color> pixels = lowercase_extension(path) == ".exr"
        ? load_exr_pixels(path, width, height)
        : load_stb_pixels(path, width, height);
    return std::make_shared<const EnvironmentMap>(
        width,
        height,
        std::move(pixels),
        std::filesystem::absolute(path).lexically_normal());
}

EnvironmentMap::EnvironmentMap(
    int width,
    int height,
    std::vector<Color> pixels,
    std::filesystem::path source_path)
    : width_(width),
      height_(height),
      pixels_(std::move(pixels)),
      source_path_(std::move(source_path)) {
    if (width_ <= 0 || height_ <= 0 || width_ != height_ * 2) {
        throw std::invalid_argument("Environment image must use an exact 2:1 equirectangular layout");
    }
    if (pixels_.size() != static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_)) {
        throw std::invalid_argument("Environment pixel count does not match dimensions");
    }
    for (Color& color : pixels_) {
        if (!color.allFinite()) {
            throw std::invalid_argument("Environment image contains NaN or infinity");
        }
        color = color.cwiseMax(Color::Zero());
    }
    build_sampling_data();
}

int EnvironmentMap::width() const { return width_; }
int EnvironmentMap::height() const { return height_; }
const std::vector<Color>& EnvironmentMap::pixels() const { return pixels_; }
const std::filesystem::path& EnvironmentMap::source_path() const { return source_path_; }
const std::array<Color, 9>& EnvironmentMap::radiance_sh() const { return radiance_sh_; }
const std::vector<float>& EnvironmentMap::importance_pmf() const { return pmf_; }
const std::vector<float>& EnvironmentMap::importance_cdf() const { return cdf_; }

Vec2 EnvironmentMap::direction_to_uv(const Vec3& direction) {
    const Vec3 d = direction.normalized();
    const float u = wrap_unit(std::atan2(d.z(), d.x()) / kTwoPi + 0.5f);
    const float v = std::acos(std::clamp(d.y(), -1.0f, 1.0f)) / kPi;
    return Vec2(u, v);
}

Vec3 EnvironmentMap::uv_to_direction(const Vec2& uv) {
    const float phi = (wrap_unit(uv.x()) - 0.5f) * kTwoPi;
    const float theta = std::clamp(uv.y(), 0.0f, 1.0f) * kPi;
    const float sin_theta = std::sin(theta);
    return Vec3(
        std::cos(phi) * sin_theta,
        std::cos(theta),
        std::sin(phi) * sin_theta);
}

Color EnvironmentMap::sample_direction(const Vec3& direction) const {
    if (!direction.allFinite() || direction.squaredNorm() <= 1.0e-24f) {
        return Color::Zero();
    }
    const Vec2 uv = direction_to_uv(direction);
    const float x = uv.x() * static_cast<float>(width_) - 0.5f;
    const float y = uv.y() * static_cast<float>(height_) - 0.5f;
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = std::clamp(static_cast<int>(std::floor(y)), 0, height_ - 1);
    const int y1 = std::min(y0 + 1, height_ - 1);
    const float tx = x - std::floor(x);
    const float ty = std::clamp(y - std::floor(y), 0.0f, 1.0f);
    const Color a = pixel(wrap_index(x0, width_), y0) * (1.0f - tx) +
        pixel(wrap_index(x0 + 1, width_), y0) * tx;
    const Color b = pixel(wrap_index(x0, width_), y1) * (1.0f - tx) +
        pixel(wrap_index(x0 + 1, width_), y1) * tx;
    return a * (1.0f - ty) + b * ty;
}

Color EnvironmentMap::diffuse_irradiance(const Vec3& normal) const {
    if (!normal.allFinite() || normal.squaredNorm() <= 1.0e-24f) {
        return Color::Zero();
    }
    const auto basis = sh_basis(normal.normalized());
    Color result = Color::Zero();
    for (int index = 0; index < 9; ++index) {
        const float convolution = index == 0 ? kPi : (index <= 3 ? 2.0f * kPi / 3.0f : kPi / 4.0f);
        result += radiance_sh_[static_cast<std::size_t>(index)] *
            (basis[static_cast<std::size_t>(index)] * convolution);
    }
    return result.cwiseMax(Color::Zero());
}

float EnvironmentMap::direction_pdf(const Vec3& direction) const {
    if (!direction.allFinite() || direction.squaredNorm() <= 1.0e-24f) {
        return 0.0f;
    }
    const Vec2 uv = direction_to_uv(direction);
    const int x = std::min(static_cast<int>(uv.x() * static_cast<float>(width_)), width_ - 1);
    const int y = std::min(static_cast<int>(uv.y() * static_cast<float>(height_)), height_ - 1);
    const std::size_t index = static_cast<std::size_t>(y * width_ + x);
    return pmf_[index] / texel_solid_angle(y);
}

EnvironmentMapSample EnvironmentMap::sample(
    float select,
    float jitter_u,
    float jitter_v) const {
    const float target = clamp_unit(select);
    const auto found = std::lower_bound(cdf_.begin(), cdf_.end(), target);
    const std::size_t index = std::min(
        static_cast<std::size_t>(std::distance(cdf_.begin(), found)),
        pmf_.size() - 1U);
    const int x = static_cast<int>(index % static_cast<std::size_t>(width_));
    const int y = static_cast<int>(index / static_cast<std::size_t>(width_));
    const float theta0 = kPi * static_cast<float>(y) / static_cast<float>(height_);
    const float theta1 = kPi * static_cast<float>(y + 1) / static_cast<float>(height_);
    const float cos_theta = std::lerp(
        std::cos(theta0),
        std::cos(theta1),
        clamp_unit(jitter_v));
    const float theta = std::acos(std::clamp(cos_theta, -1.0f, 1.0f));
    const Vec2 uv(
        (static_cast<float>(x) + clamp_unit(jitter_u)) / static_cast<float>(width_),
        theta / kPi);
    EnvironmentMapSample result;
    result.direction = uv_to_direction(uv);
    result.radiance = sample_direction(result.direction);
    result.pdf = pmf_[index] / texel_solid_angle(y);
    return result;
}

void EnvironmentMap::build_sampling_data() {
    const std::size_t count = pixels_.size();
    pmf_.resize(count);
    cdf_.resize(count);
    radiance_sh_.fill(Color::Zero());
    float total = 0.0f;
    for (int y = 0; y < height_; ++y) {
        const float solid_angle = texel_solid_angle(y);
        for (int x = 0; x < width_; ++x) {
            const std::size_t index = static_cast<std::size_t>(y * width_ + x);
            const float weight = luminance(pixels_[index]) * solid_angle;
            pmf_[index] = weight;
            total += weight;
            const Vec3 direction = uv_to_direction(Vec2(
                (static_cast<float>(x) + 0.5f) / static_cast<float>(width_),
                (static_cast<float>(y) + 0.5f) / static_cast<float>(height_)));
            const auto basis = sh_basis(direction);
            for (int coefficient = 0; coefficient < 9; ++coefficient) {
                radiance_sh_[static_cast<std::size_t>(coefficient)] +=
                    pixels_[index] * (basis[static_cast<std::size_t>(coefficient)] * solid_angle);
            }
        }
    }
    if (!(total > std::numeric_limits<float>::min()) || !std::isfinite(total)) {
        total = 0.0f;
        for (int y = 0; y < height_; ++y) {
            const float solid_angle = texel_solid_angle(y);
            for (int x = 0; x < width_; ++x) {
                const std::size_t index = static_cast<std::size_t>(y * width_ + x);
                pmf_[index] = solid_angle;
                total += solid_angle;
            }
        }
    }
    float cumulative = 0.0f;
    for (std::size_t index = 0; index < count; ++index) {
        pmf_[index] /= total;
        cumulative += pmf_[index];
        cdf_[index] = cumulative;
    }
    cdf_.back() = 1.0f;
}

const Color& EnvironmentMap::pixel(int x, int y) const {
    return pixels_[static_cast<std::size_t>(y * width_ + x)];
}

float EnvironmentMap::texel_solid_angle(int y) const {
    const float theta0 = kPi * static_cast<float>(y) / static_cast<float>(height_);
    const float theta1 = kPi * static_cast<float>(y + 1) / static_cast<float>(height_);
    return (kTwoPi / static_cast<float>(width_)) * (std::cos(theta0) - std::cos(theta1));
}

Color environment_radiance(
    const Color& fallback,
    const std::shared_ptr<const EnvironmentMap>& map,
    float intensity,
    float rotation_degrees,
    const Vec3& world_direction) {
    const float safe_intensity = std::max(0.0f, intensity);
    if (!map) {
        return fallback * safe_intensity;
    }
    const Vec3 local_direction = rotate_y(world_direction, -rotation_degrees);
    return fallback.cwiseProduct(map->sample_direction(local_direction)) * safe_intensity;
}

float environment_pdf(
    const std::shared_ptr<const EnvironmentMap>& map,
    float rotation_degrees,
    const Vec3& world_direction) {
    if (!map) {
        return 1.0f / (4.0f * kPi);
    }
    return map->direction_pdf(rotate_y(world_direction, -rotation_degrees));
}

EnvironmentMapSample sample_environment(
    const Color& fallback,
    const std::shared_ptr<const EnvironmentMap>& map,
    float intensity,
    float rotation_degrees,
    float select,
    float jitter_u,
    float jitter_v) {
    if (!map) {
        const float y = 1.0f - 2.0f * clamp_unit(select);
        const float radius = std::sqrt(std::max(0.0f, 1.0f - y * y));
        const float phi = kTwoPi * clamp_unit(jitter_u);
        return EnvironmentMapSample{
            Vec3(radius * std::cos(phi), y, radius * std::sin(phi)),
            fallback * std::max(0.0f, intensity),
            1.0f / (4.0f * kPi)};
    }
    EnvironmentMapSample result = map->sample(select, jitter_u, jitter_v);
    result.direction = rotate_y(result.direction, rotation_degrees);
    result.radiance = fallback.cwiseProduct(result.radiance) * std::max(0.0f, intensity);
    return result;
}

DominantEnvironmentLight extract_dominant_environment_light(
    const std::shared_ptr<const EnvironmentMap>& map,
    float peak_threshold_ev,
    float minimum_energy_fraction) {
    DominantEnvironmentLight result;
    result.residual_map = map;
    if (!map || !std::isfinite(peak_threshold_ev) ||
        !std::isfinite(minimum_energy_fraction)) {
        return result;
    }

    const int width = map->width();
    const int height = map->height();
    const std::vector<Color>& pixels = map->pixels();
    std::vector<float> texel_luminance(pixels.size(), 0.0f);
    std::vector<float> row_solid_angle(static_cast<std::size_t>(height), 0.0f);
    float peak = 0.0f;
    double total_energy = 0.0;
    for (int y = 0; y < height; ++y) {
        const float theta0 = kPi * static_cast<float>(y) /
            static_cast<float>(height);
        const float theta1 = kPi * static_cast<float>(y + 1) /
            static_cast<float>(height);
        const float solid_angle =
            (kTwoPi / static_cast<float>(width)) *
            (std::cos(theta0) - std::cos(theta1));
        row_solid_angle[static_cast<std::size_t>(y)] = solid_angle;
        for (int x = 0; x < width; ++x) {
            const std::size_t index = static_cast<std::size_t>(y * width + x);
            const float value = luminance(pixels[index]);
            texel_luminance[index] = value;
            peak = std::max(peak, value);
            total_energy += static_cast<double>(value) * solid_angle;
        }
    }
    if (!(peak > std::numeric_limits<float>::min()) ||
        !(total_energy > std::numeric_limits<double>::min())) {
        return result;
    }

    const float threshold = peak * std::exp2(-std::max(0.0f, peak_threshold_ev));
    std::vector<std::uint8_t> visited(pixels.size(), 0U);
    std::vector<std::size_t> selected;
    double selected_energy = 0.0;
    float selected_solid_angle = 0.0f;
    double selected_direction_x = 0.0;
    double selected_direction_y = 0.0;
    double selected_direction_z = 0.0;
    Color selected_rgb = Color::Zero();

    for (int seed_y = 0; seed_y < height; ++seed_y) {
        for (int seed_x = 0; seed_x < width; ++seed_x) {
            const std::size_t seed =
                static_cast<std::size_t>(seed_y * width + seed_x);
            if (visited[seed] != 0U || texel_luminance[seed] < threshold) {
                continue;
            }
            std::queue<std::pair<int, int>> pending;
            std::vector<std::size_t> component;
            pending.emplace(seed_x, seed_y);
            visited[seed] = 1U;
            double component_energy = 0.0;
            float component_solid_angle = 0.0f;
            double direction_x = 0.0;
            double direction_y = 0.0;
            double direction_z = 0.0;
            Color integrated_rgb = Color::Zero();
            while (!pending.empty()) {
                const auto [x, y] = pending.front();
                pending.pop();
                const std::size_t index =
                    static_cast<std::size_t>(y * width + x);
                component.push_back(index);
                const float solid_angle =
                    row_solid_angle[static_cast<std::size_t>(y)];
                const double energy =
                    static_cast<double>(texel_luminance[index]) * solid_angle;
                component_energy += energy;
                component_solid_angle += solid_angle;
                integrated_rgb += pixels[index] * solid_angle;
                const Vec3 direction = EnvironmentMap::uv_to_direction(Vec2(
                    (static_cast<float>(x) + 0.5f) / static_cast<float>(width),
                    (static_cast<float>(y) + 0.5f) / static_cast<float>(height)));
                direction_x += static_cast<double>(direction.x()) * energy;
                direction_y += static_cast<double>(direction.y()) * energy;
                direction_z += static_cast<double>(direction.z()) * energy;

                for (int offset_y = -1; offset_y <= 1; ++offset_y) {
                    const int neighbor_y = y + offset_y;
                    if (neighbor_y < 0 || neighbor_y >= height) {
                        continue;
                    }
                    for (int offset_x = -1; offset_x <= 1; ++offset_x) {
                        if (offset_x == 0 && offset_y == 0) {
                            continue;
                        }
                        const int neighbor_x = wrap_index(x + offset_x, width);
                        const std::size_t neighbor = static_cast<std::size_t>(
                            neighbor_y * width + neighbor_x);
                        if (visited[neighbor] == 0U &&
                            texel_luminance[neighbor] >= threshold) {
                            visited[neighbor] = 1U;
                            pending.emplace(neighbor_x, neighbor_y);
                        }
                    }
                }
            }
            if (component_energy > selected_energy) {
                selected = std::move(component);
                selected_energy = component_energy;
                selected_solid_angle = component_solid_angle;
                selected_direction_x = direction_x;
                selected_direction_y = direction_y;
                selected_direction_z = direction_z;
                selected_rgb = integrated_rgb;
            }
        }
    }

    const float energy_fraction = static_cast<float>(
        selected_energy / total_energy);
    const double direction_length_squared =
        selected_direction_x * selected_direction_x +
        selected_direction_y * selected_direction_y +
        selected_direction_z * selected_direction_z;
    // A component covering a hemisphere or more represents broad ambient
    // illumination, not a useful virtual direct light (and rejects flat maps).
    if (selected.empty() ||
        energy_fraction < std::clamp(minimum_energy_fraction, 0.0f, 1.0f) ||
        selected_solid_angle >= kTwoPi ||
        direction_length_squared <= std::numeric_limits<double>::min()) {
        return result;
    }

    const double inverse_direction_length =
        1.0 / std::sqrt(direction_length_squared);
    result.valid = true;
    result.direction = Vec3(
        static_cast<float>(selected_direction_x * inverse_direction_length),
        static_cast<float>(selected_direction_y * inverse_direction_length),
        static_cast<float>(selected_direction_z * inverse_direction_length));
    result.integrated_radiance = selected_rgb.cwiseMax(Color::Zero());
    result.solid_angle = selected_solid_angle;
    result.energy_fraction = energy_fraction;
    result.angular_radius_radians = std::acos(std::clamp(
        1.0f - selected_solid_angle / kTwoPi,
        -1.0f,
        1.0f));

    std::vector<Color> residual_pixels = pixels;
    for (const std::size_t index : selected) {
        residual_pixels[index] = Color::Zero();
    }
    result.residual_map = std::make_shared<const EnvironmentMap>(
        width,
        height,
        std::move(residual_pixels));
    return result;
}

}  // namespace renderer
