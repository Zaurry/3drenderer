# GAMES202 Homework 1：实时阴影

本目录实现了作业 1 要求的三种实时阴影算法：

- Two-Pass Shadow Map（硬阴影）
- PCF（Percentage Closer Filtering）
- PCSS（Percentage Closer Soft Shadow）

在此基础上还实现了：

- 双光源 Shadow Map：每盏灯拥有独立的 FBO、深度纹理、Light MVP 和
  PCSS 参数；相机 pass 分别计算可见度后累加两盏灯的直接光。
- 动态物体：第二个 Marry 模型沿椭圆轨迹运动并旋转，模型变换会同步
  更新普通渲染 pass 和两盏灯的 Shadow Map pass。

PCSS 按照“遮挡物搜索 -> 平均遮挡深度 -> 相似三角形估算半影 ->
可变半径 PCF”的顺序实现。光源的正交投影参数通过 uniform 传给 shader，
因此深度还原和半影半径计算使用一致的世界空间单位。

## 运行

浏览器不能直接通过 `file://` 加载 GLSL 和 OBJ 文件，需要在本目录启动
一个静态文件服务器。

### Python

```powershell
python -m http.server 8000
```

然后访问 <http://127.0.0.1:8000/>。

### Node.js

```powershell
npx http-server . -p 8000
```

## 页面操作

- 右键拖动：旋转相机
- 鼠标滚轮：缩放
- 左键拖动：平移相机
- 右上角 `Shadow mode`：
  - `Hard Shadow Map`：单样本硬阴影
  - `PCF`：固定过滤半径
  - `PCSS`：根据遮挡距离动态计算半影
  - `No Shadow (reference)`：无阴影基准画面，便于对照
- `Animate object`：暂停或继续动态模型及其两份实时阴影

默认模式为 PCSS。
