class WebGLRenderer {
    meshes = [];
    shadowMeshes = [];
    lights = [];
    dynamicTransforms = [];

    constructor(gl, camera) {
        this.gl = gl;
        this.camera = camera;
        // 0 = 硬阴影，1 = PCF，2 = PCSS，3 = 无阴影对照。
        this.shadowMode = 2;
        this.animationEnabled = true;
        this.animationTime = 0.0;
        this.lastFrameTime = null;
    }

    addLight(light) {
        if (this.lights.length >= 2) {
            throw new Error(
                "当前 WebGL 1 shader 最多支持两盏投影光源。"
            );
        }
        this.lights.push({
            entity: light,
            meshRender: new MeshRender(this.gl, light.mesh, light.mat)
        });
        // shadowMeshes[l] 只写入第 l 盏灯自己的 framebuffer。
        this.shadowMeshes.push([]);
    }

    addMeshRender(mesh) {
        // OBJ 是异步加载的；新加入的材质应继承 GUI 当前选择的阴影模式。
        if (mesh.material.uniforms.uShadowMode) {
            mesh.material.uniforms.uShadowMode.value = this.shadowMode;
        }
        this.meshes.push(mesh);
    }

    addShadowMeshRender(lightIndex, mesh) {
        if (!this.shadowMeshes[lightIndex]) {
            throw new Error(`不存在索引为 ${lightIndex} 的投影光源。`);
        }
        this.shadowMeshes[lightIndex].push(mesh);
    }

    addDynamicTransform(transform, animator) {
        this.dynamicTransforms.push({ transform, animator });
    }

    setAnimationEnabled(enabled) {
        this.animationEnabled = Boolean(enabled);
    }

    updateDynamicTransforms(timeSeconds) {
        if (!this.animationEnabled) {
            return;
        }
        for (let i = 0; i < this.dynamicTransforms.length; ++i) {
            const dynamic = this.dynamicTransforms[i];
            dynamic.animator(timeSeconds, dynamic.transform);
        }
    }

    setShadowMode(mode) {
        this.shadowMode = Math.max(0, Math.min(3, Number(mode)));
        for (let i = 0; i < this.meshes.length; ++i) {
            const uniform = this.meshes[i].material.uniforms.uShadowMode;
            if (uniform) {
                uniform.value = this.shadowMode;
            }
        }
    }

    render(timeSeconds = 0.0) {
        const gl = this.gl;

        /*
         * 使用累积动画时间而不是页面绝对时间。暂停期间不累加，重新播放时
         * 会从冻结位置继续，不会因为时间轴仍在前进而突然跳跃。
         */
        if (this.lastFrameTime === null) {
            this.lastFrameTime = timeSeconds;
        }
        const deltaTime = Math.min(
            Math.max(timeSeconds - this.lastFrameTime, 0.0),
            0.1
        );
        this.lastFrameTime = timeSeconds;
        if (this.animationEnabled) {
            this.animationTime += deltaTime;
        }
        this.updateDynamicTransforms(this.animationTime);

        gl.enable(gl.DEPTH_TEST);
        gl.depthFunc(gl.LEQUAL);

        console.assert(this.lights.length != 0, "No light");

        /*
         * 第一阶段：逐光源生成独立 Shadow Map。必须先完成全部深度 pass，
         * 才能在一次相机 pass 中同时读取两张纹理并累加光照。
         */
        for (let l = 0; l < this.lights.length; l++) {
            if (this.lights[l].entity.hasShadowMap == true) {
                /*
                 * 每帧都要清理光源 pass。红色通道为 1、其余为 0 时，
                 * unpack() 恰好得到深度 1.0，表示没有遮挡物。
                 */
                gl.bindFramebuffer(
                    gl.FRAMEBUFFER,
                    this.lights[l].entity.fbo
                );
                gl.viewport(0, 0, resolution, resolution);
                gl.clearColor(1.0, 0.0, 0.0, 0.0);
                gl.clearDepth(1.0);
                gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

                for (let i = 0;
                    i < this.shadowMeshes[l].length;
                    i++) {
                    const shadowMesh = this.shadowMeshes[l][i];
                    shadowMesh.material.syncLightMVP(
                        shadowMesh.mesh.transform
                    );
                    shadowMesh.draw(this.camera);
                }
            }
        }

        /*
         * 第二阶段：恢复默认 framebuffer，只清屏一次。过去在灯光循环内
         * 绘制会让后一盏灯覆盖前一盏灯；现在 fragment shader 一次累加。
         */
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
        gl.clearColor(0.0, 0.0, 0.0, 1.0);
        gl.clearDepth(1.0);
        gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

        for (let l = 0; l < this.lights.length; ++l) {
            // 用小立方体显示每盏灯的位置和颜色。
            this.lights[l].meshRender.mesh.transform.translate =
                this.lights[l].entity.lightPos;
            this.lights[l].meshRender.draw(this.camera);
        }

        for (let i = 0; i < this.meshes.length; i++) {
            if (this.meshes[i].material.syncLights) {
                this.meshes[i].material.syncLights(
                    this.meshes[i].mesh.transform
                );
            }
            this.meshes[i].draw(this.camera);
        }
    }
}
