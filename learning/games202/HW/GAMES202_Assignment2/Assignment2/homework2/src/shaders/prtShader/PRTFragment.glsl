#ifdef GL_ES
precision mediump float;
#endif

varying highp vec3 vColor;

void main(void) {
    // 实时 PRT 的主要计算已经在顶点着色器完成，这里只输出插值后的颜色。
    gl_FragColor = vec4(vColor, 1.0);
}
