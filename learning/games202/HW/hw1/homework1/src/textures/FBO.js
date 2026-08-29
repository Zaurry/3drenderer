class FBO {
    constructor(gl) {
        let framebuffer = null;
        let texture = null;
        let depthBuffer = null;

        // 创建过程失败时，按 WebGL 对象的真实类型释放已分配资源。
        function cleanup() {
            if (framebuffer) gl.deleteFramebuffer(framebuffer);
            if (texture) gl.deleteTexture(texture);
            if (depthBuffer) gl.deleteRenderbuffer(depthBuffer);
        }

        framebuffer = gl.createFramebuffer();
        if (!framebuffer) {
            throw new Error("无法创建阴影贴图帧缓冲。");
        }

        /*
         * WebGL 1 不能保证直接采样 depth attachment，因此用 RGBA8 纹理
         * 保存 pack() 后的深度，另配一个 DEPTH_COMPONENT16 renderbuffer
         * 负责光源 pass 的深度测试。
         */
        texture = gl.createTexture();
        if (!texture) {
            cleanup();
            throw new Error("无法创建阴影贴图颜色纹理。");
        }

        gl.bindTexture(gl.TEXTURE_2D, texture);
        gl.texImage2D(
            gl.TEXTURE_2D,
            0,
            gl.RGBA,
            resolution,
            resolution,
            0,
            gl.RGBA,
            gl.UNSIGNED_BYTE,
            null
        );

        /*
         * 深度比较必须读取相邻 texel 的原始编码值，不能让硬件在线性过滤时
         * 先混合 RGBA 字节。PCF/PCSS 的过滤由 shader 显式完成。
         */
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
        framebuffer.texture = texture;

        depthBuffer = gl.createRenderbuffer();
        if (!depthBuffer) {
            cleanup();
            throw new Error("无法创建阴影贴图深度缓冲。");
        }

        gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer);
        gl.renderbufferStorage(
            gl.RENDERBUFFER,
            gl.DEPTH_COMPONENT16,
            resolution,
            resolution
        );

        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
        gl.framebufferTexture2D(
            gl.FRAMEBUFFER,
            gl.COLOR_ATTACHMENT0,
            gl.TEXTURE_2D,
            texture,
            0
        );
        gl.framebufferRenderbuffer(
            gl.FRAMEBUFFER,
            gl.DEPTH_ATTACHMENT,
            gl.RENDERBUFFER,
            depthBuffer
        );

        const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
        if (status !== gl.FRAMEBUFFER_COMPLETE) {
            cleanup();
            throw new Error(`阴影贴图帧缓冲不完整，WebGL 状态码：${status}`);
        }

        // 保存 renderbuffer 引用，便于以后扩展显式销毁逻辑。
        framebuffer.depthBuffer = depthBuffer;

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);

        /*
         * 构造函数显式返回 WebGLFramebuffer，使现有材质可以直接将它作为
         * framebuffer 绑定，并通过 framebuffer.texture 访问深度纹理。
         */
        return framebuffer;
    }
}
