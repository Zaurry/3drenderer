#pragma once
#include "render/ddgi/ddgi_settings.h"
#include <nlohmann/json.hpp>
#include <type_traits>
namespace renderer {
inline nlohmann::json ddgi_settings_json(const DdgiSettings &s) {
    return {{"enabled", s.enabled},
            {"paused", s.paused},
            {"auto_fit", s.auto_fit},
            {"origin", s.origin},
            {"extent", s.extent},
            {"probe_counts", s.probe_counts},
            {"rays_per_probe", s.rays_per_probe},
            {"probes_per_frame", s.probes_per_frame},
            {"hysteresis", s.hysteresis},
            {"normal_bias", s.normal_bias},
            {"view_bias", s.view_bias},
            {"intensity", s.intensity},
            {"relocation", s.relocation},
            {"classification", s.classification},
            {"show_probes", s.show_probes},
            {"debug_view", static_cast<int>(s.debug_view)}};
}
inline DdgiSettings parse_ddgi_settings(const nlohmann::json &source) {
    DdgiSettings s;
    if (!source.is_object())
        return s;
    const auto read = [&]<class T>(const char *key, T &value) {
        const auto found = source.find(key);
        if (found == source.end())
            return;
        try {
            if constexpr (std::is_same_v<T, bool>) {
                if (found->is_boolean())
                    value = found->get<bool>();
            } else if constexpr (std::is_arithmetic_v<T>) {
                if (found->is_number())
                    value = found->get<T>();
            } else {
                if (found->is_array() && found->size() == 3)
                    value = found->get<T>();
            }
        } catch (const nlohmann::json::exception &) { /* Keep the field default. */
        }
    };
    read("enabled", s.enabled);
    read("paused", s.paused);
    read("auto_fit", s.auto_fit);
    read("origin", s.origin);
    read("extent", s.extent);
    read("probe_counts", s.probe_counts);
    read("rays_per_probe", s.rays_per_probe);
    read("probes_per_frame", s.probes_per_frame);
    read("hysteresis", s.hysteresis);
    read("normal_bias", s.normal_bias);
    read("view_bias", s.view_bias);
    read("intensity", s.intensity);
    read("relocation", s.relocation);
    read("classification", s.classification);
    read("show_probes", s.show_probes);
    int debug = 0;
    read("debug_view", debug);
    s.debug_view = static_cast<DdgiDebugView>(std::clamp(debug, 0, 5));
    return normalized_ddgi_settings(s);
}
} // namespace renderer
