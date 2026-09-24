#include "test_framework.h"
#include "platform/opengl/cuda_opengl_interop.h"
#include "render/optix/optix_realtime_renderer.h"
#include "scene/scene_asset_loader.h"
#include <SDL3/SDL.h>
#include <glad/gl.h>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace renderer;
namespace {
struct Context {
    SDL_Window* window=nullptr;SDL_GLContext gl=nullptr;CudaDeviceContext cuda;
    Context() {
        std::string reason;
        if(!optix_realtime_available(0,&reason)) {
            if(std::getenv("RTRT_REQUIRE_OPTIX"))throw TestFailure{reason};
            RENDER_SKIP(reason);
        }
        if(!SDL_Init(SDL_INIT_VIDEO)) RENDER_SKIP(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,4);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,5);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
        window=SDL_CreateWindow("OptiX interop validation",80,64,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
        if(window) gl=SDL_GL_CreateContext(window);
        if(!gl) {reason=SDL_GetError();if(window)SDL_DestroyWindow(window);SDL_Quit();RENDER_SKIP(reason);}
        RENDER_CHECK(SDL_GL_MakeCurrent(window,gl));
        RENDER_CHECK(gladLoadGL(reinterpret_cast<GLADloadfunc>(SDL_GL_GetProcAddress)));
        auto selected=select_cuda_device_for_current_opengl_context(0,&reason);
        if(!selected) {SDL_GL_DestroyContext(gl);SDL_DestroyWindow(window);SDL_Quit();RENDER_SKIP(reason);}
        cuda=*selected;
    }
    ~Context(){SDL_GL_DestroyContext(gl);SDL_DestroyWindow(window);SDL_Quit();}
};
Camera camera(float x=0){return Camera(Vec3(x,0,0),Vec3(x,0,-3),Vec3::UnitY(),50,67.0f/51);}
void quad(Scene& s,float z,int material,bool flipped=false) {
    TriangleVertex v[4];
    v[0].position=Vec3(-3,-3,z);v[0].uv=Vec2(0,0);
    v[1].position=Vec3(3,-3,z);v[1].uv=Vec2(1,0);
    v[2].position=Vec3(3,3,z);v[2].uv=Vec2(1,1);
    v[3].position=Vec3(-3,3,z);v[3].uv=Vec2(0,1);
    s.triangles.emplace_back(v[0],v[flipped?2:1],v[flipped?1:2],material);
    s.triangles.emplace_back(v[0],v[flipped?3:2],v[flipped?2:3],material);
}
}

RENDER_TEST(test_optix_interop_matches_host_and_reuses_resources) {
    Context context;
    OptixRealtimeRenderer resident(context.cuda),host(context.cuda);
    CudaOpenGlInteropTexture output;RENDER_CHECK(output.initialize(context.cuda));
    Scene scene;Material m;m.type=MaterialType::Emissive;m.emission=Color(.2f,.4f,.6f);
    scene.materials.push_back(m);quad(scene,-3,0);
    Material marker=m;marker.emission=Color(1,.1f,.05f);scene.materials.push_back(marker);
    scene.triangles.emplace_back(Vec3(-.8f,.2f,-2),Vec3(0,.2f,-2),Vec3(-.4f,1,-2),1);
    auto snapshot=make_render_scene_snapshot(scene);
    RenderSettings settings;settings.width=67;settings.height=51;settings.path.cuda_device=context.cuda.device_id();
    InteractiveFrameState state;Framebuffer expected(1,1);
    std::uint64_t generation=0;
    for(int frame=0;frame<30;++frame) {
        CudaSurfaceHandle surface=0;const auto stream=resident.stream_handle();
        RENDER_CHECK(output.begin_frame(settings.width,settings.height,stream,surface));
        resident.render_next_frame_to_surface(snapshot,camera(),settings,state,surface);
        RENDER_CHECK(output.end_frame(stream));
        host.render_next_frame(snapshot,camera(),settings,state,expected);
        if(frame==0)generation=resident.statistics().allocation_generation;
        RENDER_CHECK(resident.statistics().allocation_generation==generation);
    }
    std::vector<float> rgba(67*51*4);
    glGetTextureImage(output.texture(),0,GL_RGBA,GL_FLOAT,GLsizei(rgba.size()*sizeof(float)),rgba.data());
    for(int y=0;y<51;++y)for(int x=0;x<67;++x) {
        const auto index=(y*67+x)*4;
        for(int c=0;c<3;++c)RENDER_CHECK(std::abs(rgba[index+c]-expected.pixel(x,y)[c])<1e-5f);
    }
    const auto& stats=resident.statistics();
    RENDER_CHECK(stats.framebuffer_downloads==0 && stats.realtime.hardware_ray_tracing_active);
    RENDER_CHECK(stats.realtime.rt_core_version>0);
    RENDER_CHECK(stats.blas_build_count==0 && stats.tlas_build_count==0 && stats.tlas_refit_count==0);
    RENDER_CHECK(stats.realtime.gas_builds==1 && stats.realtime.ias_builds==1);
    RENDER_CHECK(host.statistics().framebuffer_downloads==30);
    RENDER_CHECK(glGetError()==GL_NO_ERROR);
}

RENDER_TEST(test_optix_primary_watertight_shared_edges) {
    Context context;OptixRealtimeRenderer r(context.cuda);
    Scene scene;Material m;m.type=MaterialType::Emissive;m.emission=Color::Ones();scene.materials.push_back(m);quad(scene,-3,0);
    RenderSettings settings;settings.width=settings.height=64;settings.path.cuda_device=context.cuda.device_id();
    settings.realtime.taa=false;settings.realtime.temporal_upscale=false;settings.realtime.denoise=false;
    Framebuffer f(1,1);InteractiveFrameState state;
    r.render_next_frame(make_render_scene_snapshot(scene),Camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1),settings,state,f);
    for(int y=0;y<64;++y)for(int x=0;x<64;++x)RENDER_CHECK((f.pixel(x,y)-Color::Ones()).norm()<1e-5);
}
