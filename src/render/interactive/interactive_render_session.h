#pragma once

#include "render/framebuffer.h"
#include "render/interactive/render_mode.h"
#include "render/render_settings.h"
#include "scene/camera.h"
#include "scene/instanced_scene.h"
#include "scene/scene_revision.h"

#include <cstdint>

namespace renderer {

enum class SceneChange : std::uint32_t {
    None = 0,
    Geometry = 1U << 0U,
    MaterialBindings = 1U << 1U,
    Materials = 1U << 2U,
    Textures = 1U << 3U,
    Lighting = 1U << 4U,
    InstanceTransforms = 1U << 5U,
    Environment = 1U << 6U,
    All = (1U << 7U) - 1U,
};

using SceneChangeSet = SceneChange;

constexpr SceneChange operator|(SceneChange left, SceneChange right) {
    return static_cast<SceneChange>(
        static_cast<std::uint32_t>(left) |
        static_cast<std::uint32_t>(right));
}

constexpr SceneChange operator&(SceneChange left, SceneChange right) {
    return static_cast<SceneChange>(
        static_cast<std::uint32_t>(left) &
        static_cast<std::uint32_t>(right));
}

constexpr SceneChange& operator|=(SceneChange& left, SceneChange right) {
    left = left | right;
    return left;
}

constexpr bool has_scene_change(SceneChangeSet changes, SceneChange change) {
    return (
        static_cast<std::uint32_t>(changes) &
        static_cast<std::uint32_t>(change)) != 0U;
}

inline SceneChangeSet scene_changes_between(
    const SceneRevisions& previous,
    const SceneRevisions& current) {
    SceneChangeSet changes = SceneChange::None;
    if (previous.topology != current.topology) {
        changes |= SceneChange::Geometry;
        changes |= SceneChange::MaterialBindings;
        changes |= SceneChange::InstanceTransforms;
        changes |= SceneChange::Lighting;
    }
    if (previous.geometry != current.geometry) {
        changes |= SceneChange::Geometry;
        changes |= SceneChange::MaterialBindings;
    }
    if (previous.transforms != current.transforms) {
        changes |= SceneChange::InstanceTransforms;
        changes |= SceneChange::Lighting;
    }
    if (previous.material_bindings != current.material_bindings) {
        changes |= SceneChange::MaterialBindings;
    }
    if (previous.materials != current.materials) {
        changes |= SceneChange::Materials;
    }
    if (previous.textures != current.textures) {
        changes |= SceneChange::Textures;
    }
    if (previous.lighting != current.lighting) {
        changes |= SceneChange::Lighting;
    }
    if (previous.environment != current.environment) {
        changes |= SceneChange::Environment;
    }
    return changes;
}

inline SceneChangeSet scene_changes_for_snapshot(
    std::uint64_t previous_source_id,
    const SceneRevisions& previous_revisions,
    bool has_previous_snapshot,
    const RenderSceneSnapshot& current,
    SceneChangeSet hinted_changes = SceneChange::None) {
    if (!has_previous_snapshot ||
        (previous_source_id != current.source_id &&
         (previous_source_id != 0 || current.source_id != 0))) {
        return SceneChange::All;
    }
    // Versioned snapshots are authoritative. Caller hints are retained only for
    // legacy, unversioned snapshots; otherwise stale UI bookkeeping can cause
    // unnecessary uploads and progressive resets.
    if (current.source_id != 0 || !current.revisions.empty()) {
        return scene_changes_between(previous_revisions, current.revisions);
    }
    return hinted_changes;
}

struct InteractiveFrameState {
    bool camera_cut = false;
    bool camera_changed = false;
    SceneChangeSet scene_changes = SceneChange::None;
    bool framebuffer_resized = false;
    bool reset_requested = false;
    bool automatic_interaction_quality = false;
    float delta_seconds = 0.0f;
};

class InteractiveRenderSession {
public:
    virtual ~InteractiveRenderSession() = default;
    virtual void reset(
        const RenderSceneSnapshot& snapshot,
        const RenderSettings& settings) = 0;
    virtual void render_next_frame(
        const RenderSceneSnapshot& snapshot,
        const Camera& camera,
        const RenderSettings& settings,
        const InteractiveFrameState& frame_state,
        Framebuffer& target) = 0;
};

}  // namespace renderer
