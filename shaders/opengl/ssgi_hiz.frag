#version 450 core

layout(binding = 0) uniform sampler2D u_linear_depth;
layout(binding = 1) uniform sampler2D u_hiz_source;

uniform int u_initialize;
uniform int u_source_level;
uniform ivec2 u_source_size;
uniform ivec2 u_destination_size;

in vec2 v_uv;
layout(location = 0) out vec2 out_depth_range;

bool valid_range(vec2 range_value) {
    return range_value.x <= range_value.y && range_value.y > 0.0;
}

void main() {
    ivec2 destination = ivec2(gl_FragCoord.xy);
    if (u_initialize != 0) {
        float depth = texelFetch(u_linear_depth, destination, 0).r;
        out_depth_range = depth > 0.0
            ? vec2(depth)
            : vec2(3.402823466e+38, 0.0);
        return;
    }

    // Integer coverage, rather than a fixed 2x2 footprint, keeps the final
    // row and column of odd/NPOT levels. A destination texel covers at most
    // 3x3 source texels for the halving chain used by this renderer.
    ivec2 source_begin = ivec2(floor(
        vec2(destination) * vec2(u_source_size) /
        vec2(u_destination_size)));
    ivec2 source_end = ivec2(ceil(
        vec2(destination + ivec2(1)) * vec2(u_source_size) /
        vec2(u_destination_size))) - ivec2(1);
    source_begin = clamp(source_begin, ivec2(0), u_source_size - ivec2(1));
    source_end = clamp(source_end, source_begin, u_source_size - ivec2(1));

    vec2 reduced = vec2(3.402823466e+38, 0.0);
    for (int y = 0; y < 3; ++y) {
        int source_y = source_begin.y + y;
        if (source_y > source_end.y) {
            break;
        }
        for (int x = 0; x < 3; ++x) {
            int source_x = source_begin.x + x;
            if (source_x > source_end.x) {
                break;
            }
            vec2 sample_range = texelFetch(
                u_hiz_source,
                ivec2(source_x, source_y),
                u_source_level).rg;
            if (!valid_range(sample_range)) {
                continue;
            }
            reduced.x = min(reduced.x, sample_range.x);
            reduced.y = max(reduced.y, sample_range.y);
        }
    }
    out_depth_range = reduced;
}
