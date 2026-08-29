var gl, gl_draw_buffers;

var bufferFBO;
var bumpMap;

const homework3Scenes = {
	cube1: {
		assetPath: 'assets/cube/',
		assetName: 'cube1',
		cameraPosition: [6, 1, 0],
		cameraTarget: [0, 0, 0],
		lightRadiance: [1, 1, 1],
		lightPosition: [-2, 4, 1],
		lightDirection: { x: 0.4, y: -0.9, z: -0.2 },
	},
	cube2: {
		assetPath: 'assets/cube/',
		assetName: 'cube2',
		cameraPosition: [6, 1, 0],
		cameraTarget: [0, 0, 0],
		lightRadiance: [1, 1, 1],
		lightPosition: [-2, 4, 1],
		lightDirection: { x: 0.4, y: -0.9, z: -0.2 },
	},
	cave: {
		assetPath: 'assets/cave/',
		assetName: 'cave',
		cameraPosition: [4.18927, 1.0313, 2.07331],
		cameraTarget: [2.92191, 0.98, 1.55037],
		lightRadiance: [20, 20, 20],
		lightPosition: [-0.45, 5.40507, 0.637043],
		lightDirection: {
			x: 0.39048811,
			y: -0.89896828,
			z: 0.19843153,
		},
	},
};

const homework3RenderModes = {
	combined: 0,
	direct: 1,
	indirect: 2,
	reflection: 3,
};

GAMES202Main();

function GAMES202Main() {
	const query = new URLSearchParams(window.location.search);
	const requestedScene = query.get('scene') || 'cube1';
	const sceneName = Object.prototype.hasOwnProperty.call(
		homework3Scenes,
		requestedScene) ? requestedScene : 'cube1';
	const requestedMode = query.get('mode') || 'combined';
	const modeName = Object.prototype.hasOwnProperty.call(
		homework3RenderModes,
		requestedMode) ? requestedMode : 'combined';
	const requestedSamples = Number.parseInt(query.get('samples') || '2', 10);
	const requestedSteps = Number.parseInt(query.get('steps') || '96', 10);
	const scene = homework3Scenes[sceneName];
	window.homework3SceneName = sceneName;
	window.homework3RenderMode = homework3RenderModes[modeName];
	window.homework3SampleCount = Math.max(
		1,
		Math.min(4, Number.isFinite(requestedSamples) ? requestedSamples : 2));
	window.homework3RayMarchSteps = Math.max(
		16,
		Math.min(128, Number.isFinite(requestedSteps) ? requestedSteps : 96));
	window.homework3Ready = false;
	document.title = `GAMES202 Homework 3 - ${sceneName} - ${modeName}`;

	// Init canvas
	const canvas = document.querySelector('#glcanvas');
	canvas.width = window.innerWidth;
	canvas.height = window.innerHeight;
	// Init gl
	gl = canvas.getContext('webgl');
	if (!gl) {
		alert('Unable to initialize WebGL. Your browser or machine may not support it.');
		return;
	}
	const floatTextures = gl.getExtension('OES_texture_float');
	const floatColorBuffer = gl.getExtension('WEBGL_color_buffer_float');
	gl_draw_buffers = gl.getExtension('WEBGL_draw_buffers');
	if (!floatTextures || !floatColorBuffer || !gl_draw_buffers) {
		throw new Error(
			'Homework 3 requires OES_texture_float, ' +
			'WEBGL_color_buffer_float and WEBGL_draw_buffers.');
	}
	var maxdb = gl.getParameter(gl_draw_buffers.MAX_DRAW_BUFFERS_WEBGL);
    console.log('MAX_DRAW_BUFFERS_WEBGL: ' + maxdb);
	if (maxdb < 5) {
		throw new Error('Homework 3 requires at least five draw buffers.');
	}

	// Add camera
	const camera = new THREE.PerspectiveCamera(75, gl.canvas.clientWidth / gl.canvas.clientHeight, 1e-3, 1000);
	const cameraPosition = scene.cameraPosition;
	const cameraTarget = scene.cameraTarget;
	camera.position.set(cameraPosition[0], cameraPosition[1], cameraPosition[2]);
	camera.fbo = new FBO(gl);

	// Add resize listener
	function setSize(width, height) {
		camera.aspect = width / height;
		camera.updateProjectionMatrix();
	}
	setSize(canvas.clientWidth, canvas.clientHeight);
	window.addEventListener('resize', () => setSize(canvas.clientWidth, canvas.clientHeight));

	// Add camera control
	const cameraControls = new THREE.OrbitControls(camera, canvas);
	cameraControls.enableZoom = true;
	cameraControls.enableRotate = true;
	cameraControls.enablePan = true;
	cameraControls.rotateSpeed = 0.3;
	cameraControls.zoomSpeed = 1.0;
	cameraControls.panSpeed = 0.8;
	cameraControls.target.set(cameraTarget[0], cameraTarget[1], cameraTarget[2]);

	// Add renderer
	const renderer = new WebGLRenderer(gl, camera);

	// Add light
	const lightPos = scene.lightPosition;
	const lightDir = scene.lightDirection;
	const lightRadiance = scene.lightRadiance;
	let lightUp = [1, 0, 0];
	const directionLight = new DirectionalLight(lightRadiance, lightPos, lightDir, lightUp, renderer.gl);
	renderer.addLight(directionLight);

	// Add shapes
	loadGLTF(renderer, scene.assetPath, scene.assetName, 'SSRMaterial');

	function createGUI() {
		const gui = new dat.gui.GUI();
		const demo = {
			scene: sceneName,
			mode: modeName,
			samples: window.homework3SampleCount,
			raySteps: window.homework3RayMarchSteps,
		};
		gui.add(demo, 'scene', Object.keys(homework3Scenes)).onChange((value) => {
			const url = new URL(window.location.href);
			url.searchParams.set('scene', value);
			url.searchParams.set('mode', demo.mode);
			url.searchParams.set('samples', String(Math.round(demo.samples)));
			url.searchParams.set('steps', String(Math.round(demo.raySteps)));
			window.location.href = url.toString();
		});
		gui.add(demo, 'mode', Object.keys(homework3RenderModes)).onChange((value) => {
			window.homework3RenderMode = homework3RenderModes[value];
			for (const mesh of renderer.meshes) {
				mesh.material.uniforms.uRenderMode.value =
					window.homework3RenderMode;
			}
		});
		gui.add(demo, 'samples', 1, 4, 1).onChange((value) => {
			window.homework3SampleCount = Math.round(value);
			for (const mesh of renderer.meshes) {
				mesh.material.uniforms.uSampleCount.value =
					window.homework3SampleCount;
			}
		});
		gui.add(demo, 'raySteps', 16, 128, 1).onChange((value) => {
			window.homework3RayMarchSteps = Math.round(value);
			for (const mesh of renderer.meshes) {
				mesh.material.uniforms.uRayMarchSteps.value =
					window.homework3RayMarchSteps;
			}
		});
		const lightPanel = gui.addFolder('Directional Light');
		lightPanel.add(renderer.lights[0].entity.lightDir, 'x', -10, 10, 0.1);
		lightPanel.add(renderer.lights[0].entity.lightDir, 'y', -10, 10, 0.1);
		lightPanel.add(renderer.lights[0].entity.lightDir, 'z', -10, 10, 0.1);
		lightPanel.open();
	}
	createGUI();

	function mainLoop(now) {
		cameraControls.update();

		renderer.render();
		window.homework3Stats = {
			meshes: renderer.meshes.length,
			shadowMeshes: renderer.shadowMeshes.length,
			bufferMeshes: renderer.bufferMeshes.length,
		};
		window.homework3Ready = renderer.meshes.length > 0 &&
			renderer.meshes.length === renderer.shadowMeshes.length &&
			renderer.meshes.length === renderer.bufferMeshes.length;
		requestAnimationFrame(mainLoop);
	}
	requestAnimationFrame(mainLoop);
}

function setTransform(t_x, t_y, t_z, s_x, s_y, s_z, r_x = 0, r_y = 0, r_z = 0) {
	return {
		modelTransX: t_x,
		modelTransY: t_y,
		modelTransZ: t_z,
		modelScaleX: s_x,
		modelScaleY: s_y,
		modelScaleZ: s_z,
		modelRotateX: r_x,
		modelRotateY: r_y,
		modelRotateZ: r_z,
	};
}
