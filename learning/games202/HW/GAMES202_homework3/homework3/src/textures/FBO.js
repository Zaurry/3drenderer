class FBO{
    constructor(gl){
        //定义错误函数
        function error(message) {
            if(framebuffer && framebuffer.textures) {
                for (const target of framebuffer.textures) {
                    if (target) gl.deleteTexture(target);
                }
            }
            if(depthBuffer) gl.deleteRenderbuffer(depthBuffer);
            if(framebuffer) gl.deleteFramebuffer(framebuffer);
            throw new Error(message);
        }

        function CreateAndBindColorTargetTexture(fbo, attachment) {
            //创建纹理对象并设置其尺寸和参数
            var texture = gl.createTexture();
            if(!texture){
                return error("Unable to create a GBuffer texture.");
            }
            gl.bindTexture(gl.TEXTURE_2D, texture);
            gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, framebuffer.width, framebuffer.height, 0, gl.RGBA, gl.FLOAT, null);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
            gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

            gl.framebufferTexture2D(gl.FRAMEBUFFER, attachment, gl.TEXTURE_2D, texture, 0);
            return texture;
        };

        //创建帧缓冲区对象
        var framebuffer = gl.createFramebuffer();
        if(!framebuffer){
            return error("Unable to create a framebuffer.");
        }
        gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
		framebuffer.width = gl.drawingBufferWidth;
		framebuffer.height = gl.drawingBufferHeight;

        var GBufferNum = 5;
	    framebuffer.attachments = [];
	    framebuffer.textures = []

	    for (var i = 0; i < GBufferNum; i++) {
	    	var attachment = gl_draw_buffers['COLOR_ATTACHMENT' + i + '_WEBGL'];
	    	var texture = CreateAndBindColorTargetTexture(framebuffer, attachment);
	    	framebuffer.attachments.push(attachment);
	    	framebuffer.textures.push(texture);
	    }
	    // * Tell the WEBGL_draw_buffers extension which FBO attachments are
	    //   being used. (This extension allows for multiple render targets.)
	    gl_draw_buffers.drawBuffersWEBGL(framebuffer.attachments);

        // Create depth buffer
        var depthBuffer = gl.createRenderbuffer(); // Create a renderbuffer object
        gl.bindRenderbuffer(gl.RENDERBUFFER, depthBuffer); // Bind the object to target
        gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, framebuffer.width, framebuffer.height);
        gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, depthBuffer);

		if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
			return error('Framebuffer is incomplete.');
		}

        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        gl.bindTexture(gl.TEXTURE_2D, null);
        gl.bindRenderbuffer(gl.RENDERBUFFER, null);

        return framebuffer;
    }
}
