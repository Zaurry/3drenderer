class ShadowMaterial extends Material {

    constructor(light, transform, vertexShader, fragmentShader) {
        let lightMVP = light.CalcLightMVP(transform);

        super({
            'uLightMVP': { type: 'matrix4fv', value: lightMVP }
        }, [], vertexShader, fragmentShader, light.fbo);

        this.light = light;
    }

    /*
     * 动态物体的深度 pass 也必须使用当前模型矩阵，否则屏幕中的物体已经
     * 移动，而 Shadow Map 里仍保留加载时的位置。
     */
    syncLightMVP(transform) {
        this.uniforms.uLightMVP.value =
            this.light.CalcLightMVP(transform);
    }
}

async function buildShadowMaterial(light, transform, vertexPath, fragmentPath) {


    let vertexShader = await getShaderString(vertexPath);
    let fragmentShader = await getShaderString(fragmentPath);

    return new ShadowMaterial(light, transform, vertexShader, fragmentShader);

}
