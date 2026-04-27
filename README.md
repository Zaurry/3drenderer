# CPU 3D Renderer

一个用于练习和验证 CPU 软件光栅化管线的 C++20 3D 渲染器项目。

## 构建与运行

```powershell
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\bin\renderer_tests.exe
.\build\bin\renderer.exe
```
