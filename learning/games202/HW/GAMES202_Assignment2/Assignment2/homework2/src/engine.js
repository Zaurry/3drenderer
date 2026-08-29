let precomputeLT = [];
let precomputeL = [];
var cameraPosition = [50, 0, 100];

var envmap = [
	'assets/cubemap/GraceCathedral',
	'assets/cubemap/Indoor',
	'assets/cubemap/Skybox',
];

var guiParams = {
	envmapId: 0
}

var cubeMaps = [];

//生成的纹理的分辨率，纹理必须是标准的尺寸 256*256 1024*1024  2048*2048
var resolution = 2048;

let envMapPass = null;

GAMES202Main().catch(error => {
	console.error('Failed to initialize GAMES202 homework 2:', error);
});

function parseNumericText(text, filename) {
	const trimmedText = text.trim();
	if (trimmedText.length === 0) {
		throw new Error(`Empty numeric data in ${filename}`);
	}
	const values = trimmedText.split(/\s+/).map(Number);
	if (values.length === 0 || values.some(value => !Number.isFinite(value))) {
		throw new Error(`Invalid numeric data in ${filename}`);
	}
	return values;
}

async function loadPRTData(envDirectory, envIndex) {
	const transportPath = envDirectory + '/transport.txt';
	const lightPath = envDirectory + '/light.txt';
	const [transportText, lightText] = await Promise.all([
		loadShaderFile(transportPath),
		loadShaderFile(lightPath),
	]);

	const transportValues = parseNumericText(transportText, transportPath);
	const uniqueVertexCount = transportValues.shift();
	if (!Number.isInteger(uniqueVertexCount) || transportValues.length % 9 !== 0) {
		throw new Error(`Malformed PRT transport data in ${transportPath}`);
	}
	// 启动阶段一次性转换为 GPU 可直接上传的格式，切换环境时无需重复分配大数组。
	precomputeLT[envIndex] = new Float32Array(transportValues);

	const lightValues = parseNumericText(lightText, lightPath);
	if (lightValues.length !== 27) {
		throw new Error(`Expected 9 RGB SH coefficients in ${lightPath}`);
	}
	precomputeL[envIndex] = Array.from({ length: 9 }, (_, coefficientIndex) =>
		lightValues.slice(coefficientIndex * 3, coefficientIndex * 3 + 3));

	console.log(`Loaded PRT data for ${envDirectory}: ${uniqueVertexCount} unique vertices, ` +
		`${transportValues.length / 9} expanded vertices.`);
}

async function GAMES202Main() {
	// Init canvas and gl
	const canvas = document.querySelector('#glcanvas');
	canvas.width = window.screen.width;
	canvas.height = window.screen.height;
	const gl = canvas.getContext('webgl');
	if (!gl) {
		alert('Unable to initialize WebGL. Your browser or machine may not support it.');
		return;
	}

	// Add camera
	const camera = new THREE.PerspectiveCamera(75, gl.canvas.clientWidth / gl.canvas.clientHeight, 1e-2, 1000);
	camera.position.set(cameraPosition[0], cameraPosition[1], cameraPosition[2]);

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
	cameraControls.target.set(0, 0, 0);

	// Add renderer
	const renderer = new WebGLRenderer(gl, camera);

	// Add lights
	// light - is open shadow map == false
	let lightPos = [0, 10000, 0];
	let lightRadiance = [1, 0, 0];
	const pointLight = new PointLight(lightRadiance, lightPos, false, renderer.gl);
	renderer.addLight(pointLight);

	// Add shapes
	let skyBoxTransform = setTransform(0, 50, 50, 150, 150, 150);
	let boxTransform = setTransform(0, 0, 0, 200, 200, 200);
	let box2Transform = setTransform(0, -10, 0, 20, 20, 20);

	for (let i = 0; i < envmap.length; i++) {
		let urls = [
			envmap[i] + '/posx.jpg',
			envmap[i] + '/negx.jpg',
			envmap[i] + '/posy.jpg',
			envmap[i] + '/negy.jpg',
			envmap[i] + '/posz.jpg',
			envmap[i] + '/negz.jpg',
		];
		cubeMaps.push(new CubeTexture(gl, urls))
		await cubeMaps[i].init();
		await loadPRTData(envmap[i], i);
	}
	// load skybox
	loadOBJ(renderer, 'assets/testObj/', 'testObj', 'SkyBoxMaterial', skyBoxTransform);

	// 离线 PRT 场景与这里的 OBJ 使用相同的三角形/顶点顺序，传输系数可以逐顶点对应。
	// 这里只做统一缩放和平移，不会改变预计算可见性与余弦项所使用的方向关系。
	loadOBJ(renderer, 'assets/mary/', 'mary', 'PRTMaterial', box2Transform);

	function createGUI() {
		const gui = new dat.gui.GUI();
		const panelModel = gui.addFolder('Switch Environemtn Map');
		panelModel.add(guiParams, 'envmapId', { 'GraceGathedral': 0, 'Indoor': 1, 'Skybox': 2 }).name('Envmap Name');
		panelModel.open();
	}

	createGUI();

	function mainLoop(now) {
		cameraControls.update();

		renderer.render();

		requestAnimationFrame(mainLoop);
	}
	requestAnimationFrame(mainLoop);
}

function setTransform(t_x, t_y, t_z, s_x, s_y, s_z) {
	return {
		modelTransX: t_x,
		modelTransY: t_y,
		modelTransZ: t_z,
		modelScaleX: s_x,
		modelScaleY: s_y,
		modelScaleZ: s_z,
	};
}
