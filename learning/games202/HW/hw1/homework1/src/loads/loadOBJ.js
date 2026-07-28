function loadOBJ(
	renderer,
	path,
	name,
	objMaterial,
	transform,
	castShadow = true,
	receiveShadow = true
) {
	/*
	 * 一个 OBJ 的所有子网格共享同一份变换。函数立即返回该对象，调用者
	 * 可以在模型异步加载完成前就注册动画。
	 */
	const sharedTransform = new TRSTransform(
		[
			transform.modelTransX,
			transform.modelTransY,
			transform.modelTransZ
		],
		[
			transform.modelScaleX,
			transform.modelScaleY,
			transform.modelScaleZ
		],
		[
			transform.modelRotateX || 0,
			transform.modelRotateY || 0,
			transform.modelRotateZ || 0
		]
	);

	const manager = new THREE.LoadingManager();
	manager.onProgress = function (item, loaded, total) {
		console.log(item, loaded, total);
	};

	function onProgress(xhr) {
		if (xhr.lengthComputable) {
			const percentComplete = xhr.loaded / xhr.total * 100;
			console.log('model ' + Math.round(percentComplete, 2) + '% downloaded');
		}
	}
	function onError() { }

	new THREE.MTLLoader(manager)
		.setPath(path)
		.load(name + '.mtl', function (materials) {
			materials.preload();
			new THREE.OBJLoader(manager)
				.setMaterials(materials)
				.setPath(path)
				.load(name + '.obj', function (object) {
					object.traverse(function (child) {
						if (child.isMesh) {
							let geo = child.geometry;
							let mat;
							if (Array.isArray(child.material)) mat = child.material[0];
							else mat = child.material;

							var indices = Array.from({ length: geo.attributes.position.count }, (v, k) => k);
							let mesh = new Mesh({ name: 'aVertexPosition', array: geo.attributes.position.array },
								{ name: 'aNormalPosition', array: geo.attributes.normal.array },
								{ name: 'aTextureCoord', array: geo.attributes.uv.array },
								indices, sharedTransform);

							let colorMap = new Texture();
							if (mat.map != null) {
								colorMap.CreateImageTexture(renderer.gl, mat.map.image);
							}
							else {
								colorMap.CreateConstantTexture(renderer.gl, mat.color.toArray());
							}

							let material;
							const lights = renderer.lights.map(
								(lightEntry) => lightEntry.entity
							);
							switch (objMaterial) {
								case 'PhongMaterial':
									/*
									 * fetch() 会独立缓存 GLSL；仅刷新 index.html 不一定重新请求
									 * shader。版本参数确保本次阴影修复立即生效。
									 */
									material = buildPhongMaterial(
										colorMap,
										mat.specular.toArray(),
										lights,
										sharedTransform,
										"./src/shaders/phongShader/phongVertex.glsl?v=multi-shadow-2",
										"./src/shaders/phongShader/phongFragment.glsl?v=multi-shadow-2"
									);
									break;
							}

							material.then((data) => {
								data.uniforms.uReceiveShadow.value =
									receiveShadow ? 1.0 : 0.0;
								let meshRender = new MeshRender(renderer.gl, mesh, data);
								renderer.addMeshRender(meshRender);
							});

							/*
							 * 每盏投影光源需要一份 ShadowMaterial，因为 framebuffer
							 * 和 uLightMVP 都属于特定光源；几何数据和动态变换仍共享。
							 */
							if (castShadow) {
								for (let lightIndex = 0;
									lightIndex < lights.length;
									++lightIndex) {
									buildShadowMaterial(
										lights[lightIndex],
										sharedTransform,
										"./src/shaders/shadowShader/shadowVertex.glsl",
										"./src/shaders/shadowShader/shadowFragment.glsl"
									).then((data) => {
										const shadowMeshRender =
											new MeshRender(renderer.gl, mesh, data);
										renderer.addShadowMeshRender(
											lightIndex,
											shadowMeshRender
										);
									});
								}
							}
						}
					});
				}, onProgress, onError);
		});

	return sharedTransform;
}
