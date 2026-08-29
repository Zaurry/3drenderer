#ifdef GL_ES
precision highp float;
#endif

/*
 * WebGL 1 中用 RGBA8 颜色纹理保存深度。pack() 把一个 [0, 1] 浮点数
 * 拆到四个 8-bit 通道；相机 pass 中的 unpack() 使用相反权重恢复它。
 */
vec4 pack(float depth) {
  const vec4 bitShift = vec4(
      1.0,
      256.0,
      256.0 * 256.0,
      256.0 * 256.0 * 256.0
  );
  const vec4 bitMask =
      vec4(1.0 / 256.0, 1.0 / 256.0, 1.0 / 256.0, 0.0);

  vec4 rgbaDepth = fract(depth * bitShift);

  /*
   * 消除低位通道向高位通道重复贡献的进位。省略这一步会让重建深度
   * 出现周期性误差，在表面上表现为条纹状的错误自遮挡。
   */
  rgbaDepth -= rgbaDepth.gbaa * bitMask;
  return rgbaDepth;
}

void main() {
  gl_FragColor = pack(gl_FragCoord.z);
}
