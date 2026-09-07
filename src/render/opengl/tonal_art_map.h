#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

namespace renderer {

// Six nested tone layers. Each darker layer retains every earlier stroke.
// Build once before rendering; no random values or frame-dependent patterns.
struct TonalArtMap {
    static constexpr int size = 128;
    static constexpr int tones = 6;
    std::vector<std::vector<std::uint8_t>> mips;
};

inline TonalArtMap make_tonal_art_map() {
    TonalArtMap map;
    constexpr int size = TonalArtMap::size;
    constexpr int tones = TonalArtMap::tones;
    std::vector<std::uint8_t> pixels(size * size * tones, 0);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float ink = 0.0f;
            for (int tone = 1; tone < tones; ++tone) {
                // Periodic slanted strokes: add parallel lines, then crosshatching.
                const float u = (x + 0.5f) / size;
                const float v = (y + 0.5f) / size;
                const float direction = tone == 5 ? v : (tone < 3 ? v - u : v + u);
                const float phase = tone == 2 || tone == 4 ? 0.5f : 0.0f;
                const float frequency = 8.0f;
                const float wave = 0.055f * std::sin(6.2831853f * u * 3.0f);
                const float position = direction * frequency + phase + wave;
                const float distance = std::abs(position - std::round(position));
                const float coverage = std::clamp((0.105f - distance) / 0.065f, 0.0f, 1.0f);
                const float graphite = 0.82f + 0.14f * std::sin(6.2831853f * (u * 23.0f + v * 17.0f));
                ink = std::max(ink, coverage * graphite);
                pixels[(tone * size + y) * size + x] =
                    static_cast<std::uint8_t>(std::lround(ink * 255.0f));
            }
        }
    }
    map.mips.push_back(std::move(pixels));
    // Shared area filtering preserves tone nesting at every MIP, and the
    // same stroke identities survive minification rather than being rerolled.
    for (int width = size; width > 1; width /= 2) {
        const int next = width / 2;
        const auto& source = map.mips.back();
        std::vector<std::uint8_t> reduced(next * next * tones);
        for (int tone = 0; tone < tones; ++tone) {
            for (int y = 0; y < next; ++y) {
                for (int x = 0; x < next; ++x) {
                    int sum = 0;
                    for (int dy = 0; dy < 2; ++dy)
                        for (int dx = 0; dx < 2; ++dx)
                            sum += source[(tone * width + y * 2 + dy) * width + x * 2 + dx];
                    reduced[(tone * next + y) * next + x] =
                        static_cast<std::uint8_t>((sum + 2) / 4);
                }
            }
        }
        map.mips.push_back(std::move(reduced));
    }
    return map;
}

}  // namespace renderer
