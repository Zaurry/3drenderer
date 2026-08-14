#version 450 core
layout(binding = 6) uniform samplerCube u_environment_prefilter;
uniform vec3 u_camera_forward;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_viewport_width;
uniform float u_viewport_height;
uniform vec3 u_environment_color;
uniform float u_environment_intensity;
uniform float u_environment_rotation_radians;
in vec2 v_ndc;
layout(location = 0) out vec4 out_linear_color;
vec3 rotate_y(vec3 direction, float radians) {
    float c = cos(radians);
    float s = sin(radians);
    return vec3(c * direction.x + s * direction.z, direction.y, -s * direction.x + c * direction.z);
}
void main() {
    vec3 direction = normalize(
        u_camera_forward +
        v_ndc.x * 0.5 * u_viewport_width * u_camera_right +
        v_ndc.y * 0.5 * u_viewport_height * u_camera_up);
    direction = rotate_y(direction, -u_environment_rotation_radians);
    vec3 radiance = textureLod(u_environment_prefilter, direction, 0.0).rgb;
    out_linear_color = vec4(radiance * u_environment_color * u_environment_intensity, 1.0);
}
