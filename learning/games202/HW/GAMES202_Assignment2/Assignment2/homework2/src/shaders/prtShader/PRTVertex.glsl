attribute vec3 aVertexPosition;
attribute mat3 aPrecomputeLT;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;

uniform mat3 uPrecomputeLR;
uniform mat3 uPrecomputeLG;
uniform mat3 uPrecomputeLB;

varying highp vec3 vColor;

float dotSH(mat3 lighting, mat3 transport) {
    // mat3 由 3 个 vec3 列组成，下面 3 次点积正好覆盖二阶 SH 的 9 个系数。
    return dot(lighting[0], transport[0]) +
           dot(lighting[1], transport[1]) +
           dot(lighting[2], transport[2]);
}

void main(void) {
    vec3 linearColor = vec3(
        dotSH(uPrecomputeLR, aPrecomputeLT),
        dotSH(uPrecomputeLG, aPrecomputeLT),
        dotSH(uPrecomputeLB, aPrecomputeLT));

    // 低阶球谐近似可能产生轻微的负值，而实际辐射亮度不能为负。
    vColor = max(linearColor, vec3(0.0));
    gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix *
                  vec4(aVertexPosition, 1.0);
}
