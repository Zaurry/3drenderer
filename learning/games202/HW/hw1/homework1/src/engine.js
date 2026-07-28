/*
 * Marry 模型经 20 倍缩放后高度约为 68 个世界单位。将相机后移并对准
 * 模型中部，确保两个模型和它们在地面上的阴影都能进入初始画面。
 */
var cameraPosition = [100, 80, 100];

//生成的纹理的分辨率，纹理必须是标准的尺寸 256*256 1024*1024  2048*2048
var resolution = 2048;
var fbo;

GAMES202Main();

function GAMES202Main() {
	// Init canvas and gl
	const canvas = document.querySelector('#glcanvas');
	const gl = canvas.getContext('webgl');
	if (!gl) {
		alert('Unable to initialize WebGL. Your browser or machine may not support it.');
		return;
	}

	// Add camera
	const camera = new THREE.PerspectiveCamera(45, gl.canvas.clientWidth / gl.canvas.clientHeight, 1e-2, 1000);
	camera.position.set(cameraPosition[0], cameraPosition[1], cameraPosition[2]);

	// Add resize listener
	function setSize(width, height) {
		/*
		 * drawing buffer 按设备像素比创建，但限制在 2 倍以内，兼顾清晰度
		 * 和 PCSS 多样本 shader 的实时性能。
		 */
		const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
		canvas.width = Math.max(1, Math.floor(width * pixelRatio));
		canvas.height = Math.max(1, Math.floor(height * pixelRatio));
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
	cameraControls.target.set(0, 25, 0);

	// Add renderer
	const renderer = new WebGLRenderer(gl, camera);

	/*
	 * 两盏方向光分别位于场景两侧，并各自拥有 FBO / Shadow Map。
	 * 暖光和冷光便于观察：某盏灯被遮挡时，阴影区域仍会保留另一盏灯的
	 * 颜色，而不是简单地统一乘上一个黑色可见度。
	 */
	const focalPoint = [0, 10, -20];
	const lightUp = [0, 1, 0];
	const warmLight = new DirectionalLight(
		3600,
		[1.0, 0.72, 0.48],
		[-55, 85, 75],
		focalPoint,
		lightUp,
		true,
		renderer.gl
	);
	const coolLight = new DirectionalLight(
		3000,
		[0.35, 0.55, 1.0],
		[85, 65, -35],
		focalPoint,
		lightUp,
		true,
		renderer.gl
	);
	renderer.addLight(warmLight);
	renderer.addLight(coolLight);

	// Add shapes
	
	let floorTransform = setTransform(0, 0, -30, 4, 4, 4);
	let obj1Transform = setTransform(0, 0, 0, 20, 20, 20);
	let obj2Transform = setTransform(40, 0, -40, 10, 10, 10);

	// 模型投射阴影；关闭自身接收以避免低精度 Shadow Map 的严重 acne。
	loadOBJ(renderer, 'assets/mary/', 'Marry', 'PhongMaterial', obj1Transform, true, false);
	const movingMaryTransform = loadOBJ(
		renderer,
		'assets/mary/',
		'Marry',
		'PhongMaterial',
		obj2Transform,
		true,
		false
	);

	/*
	 * 第二个模型沿椭圆轨迹移动并绕 Y 轴旋转。相机 pass 和两盏灯的深度
	 * pass 共享 movingMaryTransform，因此模型和两份投影始终保持一致。
	 */
	renderer.addDynamicTransform(
		movingMaryTransform,
		(timeSeconds, transform) => {
			transform.translate[0] =
				32.0 + 22.0 * Math.sin(timeSeconds * 0.65);
			transform.translate[2] =
				-35.0 + 12.0 * Math.cos(timeSeconds * 0.65);
			transform.rotation[1] = timeSeconds * 0.8;
		}
	);
	/*
	 * 地板位于所有模型下方，只需要接收阴影，不可能遮挡模型。禁止它写入
	 * Shadow Map 可从根源消除大平面的共面自遮挡（半块地板发黑）。
	 */
	loadOBJ(renderer, 'assets/floor/', 'floor', 'PhongMaterial', floorTransform, false, true);
	

	// let floorTransform = setTransform(0, 0, 0, 100, 100, 100);
	// let cubeTransform = setTransform(0, 50, 0, 10, 50, 10);
	// let sphereTransform = setTransform(30, 10, 0, 10, 10, 10);

	//loadOBJ(renderer, 'assets/basic/', 'cube', 'PhongMaterial', cubeTransform);
	// loadOBJ(renderer, 'assets/basic/', 'sphere', 'PhongMaterial', sphereTransform);
	//loadOBJ(renderer, 'assets/basic/', 'plane', 'PhongMaterial', floorTransform);


	function createGUI() {
		const gui = new dat.gui.GUI();

		/*
		 * 作业要求分别检查硬阴影、PCF 和 PCSS。运行时切换可以确保三种
		 * 算法使用完全相同的相机、模型和光源，便于观察过滤差异。
		 */
		const GUIParams = {
			shadowMode: 2,
			animateObject: true
		};
		gui.add(GUIParams, 'shadowMode', {
			'Hard Shadow Map': 0,
			'PCF': 1,
			'PCSS': 2,
			'No Shadow (reference)': 3
		}).name('Shadow mode').onChange((value) => {
			renderer.setShadowMode(value);
		});
		gui.add(GUIParams, 'animateObject')
			.name('Animate object')
			.onChange((enabled) => {
				renderer.setAnimationEnabled(enabled);
			});
	}
	createGUI();

	function mainLoop(now) {
		cameraControls.update();

		renderer.render(now * 0.001);
		requestAnimationFrame(mainLoop);
	}
	requestAnimationFrame(mainLoop);
}

function setTransform(
	t_x,
	t_y,
	t_z,
	s_x,
	s_y,
	s_z,
	r_x = 0,
	r_y = 0,
	r_z = 0
) {
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
