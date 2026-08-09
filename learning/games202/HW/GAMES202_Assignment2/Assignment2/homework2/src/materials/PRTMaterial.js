class PRTMaterial extends Material {
    constructor(vertexShader, fragmentShader) {
        super({
            // 每个 mat3 保存一个颜色通道的 9 个光照 SH 系数。
            // 自定义类型会在每帧绘制时读取当前 GUI 选中的环境光。
            'uPrecomputeLR': { type: 'precomputeL', value: 0 },
            'uPrecomputeLG': { type: 'precomputeL', value: 1 },
            'uPrecomputeLB': { type: 'precomputeL', value: 2 },
        }, [
            // WebGL 会把一个 mat3 顶点属性展开成 3 个连续的 vec3 属性槽位。
            'aPrecomputeLT'
        ], vertexShader, fragmentShader, null);
    }
}

async function buildPRTMaterial(vertexPath, fragmentPath) {
    const [vertexShader, fragmentShader] = await Promise.all([
        getShaderString(vertexPath),
        getShaderString(fragmentPath),
    ]);
    return new PRTMaterial(vertexShader, fragmentShader);
}
