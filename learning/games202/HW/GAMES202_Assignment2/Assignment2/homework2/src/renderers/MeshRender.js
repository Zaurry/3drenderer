
class MeshRender {

	#vertexBuffer;
	#normalBuffer;
	#texcoordBuffer;
	#indicesBuffer;
	#prtBuffer;
	#prtEnvmapId;

	constructor(gl, mesh, material) {

		this.gl = gl;
		this.mesh = mesh;
		this.material = material;

		this.#vertexBuffer = gl.createBuffer();
		this.#normalBuffer = gl.createBuffer();
		this.#texcoordBuffer = gl.createBuffer();
		this.#indicesBuffer = gl.createBuffer();
		this.#prtBuffer = null;
		this.#prtEnvmapId = -1;

		let extraAttribs = []
		if (mesh.hasVertices) {
			extraAttribs.push(mesh.verticesName);
			gl.bindBuffer(gl.ARRAY_BUFFER, this.#vertexBuffer);
			gl.bufferData(gl.ARRAY_BUFFER, mesh.vertices, gl.STATIC_DRAW);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
		}

		if (mesh.hasNormals) {
			extraAttribs.push(mesh.normalsName);
			gl.bindBuffer(gl.ARRAY_BUFFER, this.#normalBuffer);
			gl.bufferData(gl.ARRAY_BUFFER, mesh.normals, gl.STATIC_DRAW);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
		}

		if (mesh.hasTexcoords) {
			extraAttribs.push(mesh.texcoordsName);
			gl.bindBuffer(gl.ARRAY_BUFFER, this.#texcoordBuffer);
			gl.bufferData(gl.ARRAY_BUFFER, mesh.texcoords, gl.STATIC_DRAW);
			gl.bindBuffer(gl.ARRAY_BUFFER, null);
		}

		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.#indicesBuffer);
		gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(mesh.indices), gl.STATIC_DRAW);
		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);

		this.material.setMeshAttribs(extraAttribs);
		this.shader = this.material.compile(gl);
	}

	bindGeometryInfo() {
		const gl = this.gl;

		if (this.mesh.hasVertices) {
			const location = this.shader.program.attribs[this.mesh.verticesName];
			if (location >= 0) {
				const numComponents = 3;
				const type = gl.FLOAT;
				const normalize = false;
				const stride = 0;
				const offset = 0;
				gl.bindBuffer(gl.ARRAY_BUFFER, this.#vertexBuffer);
				gl.vertexAttribPointer(
					location,
					numComponents,
					type,
					normalize,
					stride,
					offset);
				gl.enableVertexAttribArray(location);
			}
		}

		if (this.mesh.hasNormals) {
			const location = this.shader.program.attribs[this.mesh.normalsName];
			if (location >= 0) {
				const numComponents = 3;
				const type = gl.FLOAT;
				const normalize = false;
				const stride = 0;
				const offset = 0;
				gl.bindBuffer(gl.ARRAY_BUFFER, this.#normalBuffer);
				gl.vertexAttribPointer(
					location,
					numComponents,
					type,
					normalize,
					stride,
					offset);
				gl.enableVertexAttribArray(location);
			}
		}

		if (this.mesh.hasTexcoords) {
			const location = this.shader.program.attribs[this.mesh.texcoordsName];
			if (location >= 0) {
				const numComponents = 2;
				const type = gl.FLOAT;
				const normalize = false;
				const stride = 0;
				const offset = 0;
				gl.bindBuffer(gl.ARRAY_BUFFER, this.#texcoordBuffer);
				gl.vertexAttribPointer(
					location,
					numComponents,
					type,
					normalize,
					stride,
					offset);
				gl.enableVertexAttribArray(location);
			}
		}

		gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.#indicesBuffer);


	}

	bindCameraParameters(camera) {
		const gl = this.gl;

		let modelMatrix = mat4.create();
		let viewMatrix = mat4.create();
		let projectionMatrix = mat4.create();
		// Model transform
		mat4.identity(modelMatrix);
		mat4.translate(modelMatrix, modelMatrix, this.mesh.transform.translate);
		mat4.scale(modelMatrix, modelMatrix, this.mesh.transform.scale);
		// View transform
		camera.updateMatrixWorld();
		mat4.invert(viewMatrix, camera.matrixWorld.elements);
		// mat4.lookAt(viewMatrix, cameraPosition, [0,0,0], [0,1,0]);
		// Projection transform
		mat4.copy(projectionMatrix, camera.projectionMatrix.elements);

		gl.uniformMatrix4fv(
			this.shader.program.uniforms.uProjectionMatrix,
			false,
			projectionMatrix);
		gl.uniformMatrix4fv(
			this.shader.program.uniforms.uModelMatrix,
			false,
			modelMatrix);
		gl.uniformMatrix4fv(
			this.shader.program.uniforms.uViewMatrix,
			false,
			viewMatrix);
		gl.uniform3fv(
			this.shader.program.uniforms.uCameraPos,
			[camera.position.x, camera.position.y, camera.position.z]);
	}

	bindMaterialParameters() {
		const gl = this.gl;

		let textureNum = 0;
		let precomputedLightMatrices = null;
		for (let k in this.material.uniforms) {

			if (this.material.uniforms[k].type == 'matrix4fv') {
				gl.uniformMatrix4fv(
					this.shader.program.uniforms[k],
					false,
					this.material.uniforms[k].value);
			} else if (this.material.uniforms[k].type == 'matrix3fv') {
				gl.uniformMatrix3fv(
					this.shader.program.uniforms[k],
					false,
					this.material.uniforms[k].value);
			} else if (this.material.uniforms[k].type == '3fv') {
				gl.uniform3fv(
					this.shader.program.uniforms[k],
					this.material.uniforms[k].value);
			} else if (this.material.uniforms[k].type == '1f') {
				gl.uniform1f(
					this.shader.program.uniforms[k],
					this.material.uniforms[k].value);
			} else if (this.material.uniforms[k].type == '1i') {
				gl.uniform1i(
					this.shader.program.uniforms[k],
					this.material.uniforms[k].value);
			} else if (this.material.uniforms[k].type == 'precomputeL') {
				if (precomputedLightMatrices == null) {
					const lighting = precomputeL[guiParams.envmapId];
					if (lighting == null)
						throw new Error(`Missing lighting SH data for environment ${guiParams.envmapId}`);
					precomputedLightMatrices = getMat3ValueFromRGB(lighting);
				}
				gl.uniformMatrix3fv(
					this.shader.program.uniforms[k],
					false,
					precomputedLightMatrices[this.material.uniforms[k].value]);
			} else if (this.material.uniforms[k].type == 'texture') {
				gl.activeTexture(gl.TEXTURE0 + textureNum);
				gl.bindTexture(gl.TEXTURE_2D, this.material.uniforms[k].value.texture);
				gl.uniform1i(this.shader.program.uniforms[k], textureNum);
				textureNum += 1;
			} else if (this.material.uniforms[k].type == 'CubeTexture') {
				gl.activeTexture(gl.TEXTURE0 + textureNum);
				//console.log(cubeMap.texture)
				gl.bindTexture(gl.TEXTURE_CUBE_MAP, cubeMaps[guiParams.envmapId].texture);
				gl.uniform1i(this.shader.program.uniforms[k], textureNum);
				textureNum += 1;
			}
		}
	}

	draw(camera) {
		const gl = this.gl;

		gl.bindFramebuffer(gl.FRAMEBUFFER, this.material.frameBuffer);
		if (this.material.frameBuffer != null) {
			// Shadow map
			gl.viewport(0.0, 0.0, resolution, resolution);
		} else {
			gl.viewport(0.0, 0.0, window.screen.width, window.screen.height);
		}

		gl.useProgram(this.shader.program.glShaderProgram);
		
		// mat3 顶点属性会占用 3 个连续的 vec3 属性槽位。
		const prtLocation = this.shader.program.attribs['aPrecomputeLT'];
		if (prtLocation >= 0) {
			const transport = precomputeLT[guiParams.envmapId];
			const expectedValueCount = this.mesh.count * 9;
			if (transport == null || transport.length !== expectedValueCount) {
				throw new Error(`PRT transport size mismatch: expected ${expectedValueCount}, got ${transport == null ? 0 : transport.length}`);
			}

			if (this.#prtBuffer == null)
				this.#prtBuffer = gl.createBuffer();
			gl.bindBuffer(gl.ARRAY_BUFFER, this.#prtBuffer);
			if (this.#prtEnvmapId !== guiParams.envmapId) {
				// transport 在加载阶段已经转换为 Float32Array，可直接上传到 GPU。
				gl.bufferData(gl.ARRAY_BUFFER, transport, gl.STATIC_DRAW);
				this.#prtEnvmapId = guiParams.envmapId;
			}

			for (let column = 0; column < 3; ++column) {
				gl.enableVertexAttribArray(prtLocation + column);
				gl.vertexAttribPointer(prtLocation + column, 3, gl.FLOAT, false, 36, column * 12);
			}
		}

		// Bind geometry information
		this.bindGeometryInfo();

		// Bind Camera parameters
		this.bindCameraParameters(camera);

		// Bind material parameters
		this.bindMaterialParameters();

		// Draw
		{
			const vertexCount = this.mesh.count;
			const type = gl.UNSIGNED_SHORT;
			const offset = 0;
			gl.drawElements(gl.TRIANGLES, vertexCount, type, offset);
		}
	}
}
