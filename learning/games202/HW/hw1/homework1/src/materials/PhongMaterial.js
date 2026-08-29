class PhongMaterial extends Material {

    constructor(color, specular, lights, transform, vertexShader, fragmentShader) {
        if (lights.length < 1) {
            throw new Error("PhongMaterial 至少需要一盏光源。");
        }

        /*
         * WebGL 1 的 sampler 数组兼容性较差，因此显式声明两组 uniform。
         * 不足两盏灯时让第二组复用第一盏灯的纹理，并通过 uLightCount 跳过。
         */
        const light0 = lights[0];
        const light1 = lights[1] || light0;

        super({
            // Phong
            'uSampler': { type: 'texture', value: color },
            'uKs': { type: '3fv', value: specular },
            'uLightCount': { type: '1i', value: Math.min(lights.length, 2) },

            // 第一盏光源及其独立 Shadow Map。
            'uLightPos0': { type: '3fv', value: light0.lightPos },
            'uLightIntensity0': { type: '3fv', value: light0.mat.GetIntensity() },
            'uShadowMap0': { type: 'texture', value: light0.fbo },
            'uLightMVP0': { type: 'matrix4fv', value: light0.CalcLightMVP(transform) },
            'uLightNearPlane0': { type: '1f', value: light0.nearPlane },
            'uLightFarPlane0': { type: '1f', value: light0.farPlane },
            'uLightFrustumWidth0': { type: '1f', value: light0.orthoHalfWidth * 2.0 },
            'uLightRadius0': { type: '1f', value: light0.lightRadius },

            // 第二盏光源使用完全独立的深度纹理和投影参数。
            'uLightPos1': { type: '3fv', value: light1.lightPos },
            'uLightIntensity1': { type: '3fv', value: light1.mat.GetIntensity() },
            'uShadowMap1': { type: 'texture', value: light1.fbo },
            'uLightMVP1': { type: 'matrix4fv', value: light1.CalcLightMVP(transform) },
            'uLightNearPlane1': { type: '1f', value: light1.nearPlane },
            'uLightFarPlane1': { type: '1f', value: light1.farPlane },
            'uLightFrustumWidth1': { type: '1f', value: light1.orthoHalfWidth * 2.0 },
            'uLightRadius1': { type: '1f', value: light1.lightRadius },

            /*
             * PCSS 的距离和半影计算必须与光源正交投影使用相同参数。
             * 将它们显式传入 shader，避免在 JS/GLSL 两处维护“魔数”。
             */
            'uShadowMapSize': { type: '1f', value: resolution },
            // 由 loadOBJ 按物体覆盖：1 接收阴影，0 仅显示直接光照。
            'uReceiveShadow': { type: '1f', value: 1.0 },
            // 0 = 硬阴影，1 = PCF，2 = PCSS；默认展示最终的 PCSS。
            'uShadowMode': { type: '1i', value: 2 }
        }, [], vertexShader, fragmentShader);

        this.lights = [light0, light1];
    }

    // 动态物体每帧重算两盏灯下的 Light MVP。
    syncLights(transform) {
        this.uniforms.uLightMVP0.value =
            this.lights[0].CalcLightMVP(transform);
        this.uniforms.uLightMVP1.value =
            this.lights[1].CalcLightMVP(transform);
        this.uniforms.uLightPos0.value = this.lights[0].lightPos;
        this.uniforms.uLightPos1.value = this.lights[1].lightPos;
    }
}

async function buildPhongMaterial(color, specular, lights, transform, vertexPath, fragmentPath) {


    let vertexShader = await getShaderString(vertexPath);
    let fragmentShader = await getShaderString(fragmentPath);

    return new PhongMaterial(color, specular, lights, transform, vertexShader, fragmentShader);

}
