class DirectionalLight {

    constructor(lightIntensity, lightColor, lightPos, focalPoint, lightUp, hasShadowMap, gl) {
        this.mesh = Mesh.cube(setTransform(0, 0, 0, 0.2, 0.2, 0.2, 0));
        this.mat = new EmissiveMaterial(lightIntensity, lightColor);
        this.lightPos = lightPos;
        this.focalPoint = focalPoint;
        this.lightUp = lightUp;

        /*
         * 正交投影参数同时决定阴影贴图覆盖的世界空间范围。
         *
         * 当前场景在光源视角下大约落在 [-112, 112] 的范围内，因此使用
         * 240 x 240 的投影平面，既能完整覆盖场景，又不会因范围过大浪费
         * 2048 x 2048 阴影贴图的有效分辨率。
         *
         * 这些参数也会作为 uniform 传给 PCSS。这样 CPU 侧的投影矩阵和
         * shader 中的深度/半影计算使用同一套单位，避免出现“参数虽然能
         * 调出效果，但公式没有物理意义”的问题。
         */
        this.orthoHalfWidth = 120.0;
        this.nearPlane = 0.1;
        this.farPlane = 300.0;

        // 将方向光看作具有有限面积的方形光源；这里保存其半宽（世界单位）。
        this.lightRadius = 6.0;

        this.hasShadowMap = hasShadowMap;
        this.fbo = new FBO(gl);
        if (!this.fbo) {
            console.log("无法设置帧缓冲区对象");
            return;
        }
    }

    CalcLightMVP(transformOrTranslate, legacyScale) {
        let lightMVP = mat4.create();
        let modelMatrix = mat4.create();
        let viewMatrix = mat4.create();
        let projectionMatrix = mat4.create();

        /*
         * 新代码直接传共享 TRSTransform；保留 translate/scale 两参数形式，
         * 使旧调用方式仍然可用。
         */
        const transform = transformOrTranslate instanceof TRSTransform
            ? transformOrTranslate
            : new TRSTransform(transformOrTranslate, legacyScale);

        /*
         * 模型变换必须与相机 pass 中 MeshRender.bindCameraParameters()
         * 的顺序完全一致。gl-matrix 采用列向量约定，最终矩阵为
         * T * Rx * Ry * Rz * S。
         */
        mat4.identity(modelMatrix);
        mat4.translate(modelMatrix, modelMatrix, transform.translate);
        mat4.rotateX(modelMatrix, modelMatrix, transform.rotation[0]);
        mat4.rotateY(modelMatrix, modelMatrix, transform.rotation[1]);
        mat4.rotateZ(modelMatrix, modelMatrix, transform.rotation[2]);
        mat4.scale(modelMatrix, modelMatrix, transform.scale);

        /*
         * 从光源位置朝 focalPoint 建立虚拟相机。lightUp 用来固定相机的
         * “上方”，否则 LookAt 的滚转方向不确定，阴影贴图会发生旋转。
         */
        mat4.lookAt(
            viewMatrix,
            this.lightPos,
            this.focalPoint,
            this.lightUp
        );

        /*
         * 方向光的光线互相平行，因此使用正交投影，而不是透视投影。
         * near/far 必须覆盖场景沿光照方向的完整深度范围。
         */
        mat4.ortho(
            projectionMatrix,
            -this.orthoHalfWidth,
            this.orthoHalfWidth,
            -this.orthoHalfWidth,
            this.orthoHalfWidth,
            this.nearPlane,
            this.farPlane
        );

        // 顶点依次经历 Model -> Light View -> Light Projection 变换。
        mat4.multiply(lightMVP, projectionMatrix, viewMatrix);
        mat4.multiply(lightMVP, lightMVP, modelMatrix);

        return lightMVP;
    }
}
