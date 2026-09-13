#include "test_framework.h"
#include "platform/opengl/cuda_opengl_interop.h"
#include "platform/opengl/rtrt_primary_visibility.h"
#include "render/realtime/cuda_realtime_renderer.h"
#include "core/image.h"
#include "scene/scene_asset_loader.h"
#include <SDL3/SDL.h>
#include <glad/gl.h>
#if RENDERER_HAS_CUDA
#include <cuda_runtime_api.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

using namespace renderer;
namespace {
struct Context {
    SDL_Window* window=nullptr;SDL_GLContext gl=nullptr;CudaDeviceContext cuda;
    Context() {
        std::string reason;
        if(!cuda_path_backend_available(0,&reason)) RENDER_SKIP(reason);
        if(!SDL_Init(SDL_INIT_VIDEO)) RENDER_SKIP(SDL_GetError());
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION,4);SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION,5);
        SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,SDL_GL_CONTEXT_PROFILE_CORE);
        window=SDL_CreateWindow("RTRT hybrid validation",80,64,SDL_WINDOW_OPENGL|SDL_WINDOW_HIDDEN);
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
double difference(const Framebuffer& a,const Framebuffer& b) {
    double sum=0;
    for(int y=0;y<a.height();++y) for(int x=0;x<a.width();++x) {
        RENDER_CHECK(a.pixel(x,y).allFinite());sum+=(a.pixel(x,y)-b.pixel(x,y)).squaredNorm();
    }
    return sum/(3*a.width()*a.height());
}
void export_json(const char* file,const nlohmann::json& value) {
    if(const char* dir=std::getenv("RTRT_VALIDATION_DIR")) {
        std::filesystem::create_directories(dir);std::ofstream out(std::filesystem::path(dir)/file);out<<value.dump(2);
    }
}
}

RENDER_TEST(test_hybrid_primary_material_visibility_and_gl_state) {
    Context context;
    for(int variant=0;variant<8;++variant) {
        Scene scene;scene.environment=Color(.1f,.2f,.3f);
        Material front;front.type=MaterialType::Pbr;front.base_color=Color(.6f,.4f,.2f);front.two_sided=variant!=2 && variant!=7;
        front.emission=Color(.2f,.5f,.8f);front.roughness=.45f;
        if(variant==1) {front.alpha_mode=AlphaMode::Mask;front.base_color_texture_id=0;}
        if(variant==3) {front.alpha_mode=AlphaMode::Blend;front.opacity=.4f;}
        if(variant==4) {front.type=MaterialType::Dielectric;front.emission=Color::Zero();}
        if(variant>=5) {front.base_color_texture_id=0;front.emission=Color::Zero();}
        Material back;back.type=MaterialType::Emissive;back.emission=Color(.9f,.1f,.2f);
        scene.materials={front,back};
        scene.textures.emplace_back(2,2,std::vector<Color>{Color(1,0,0),Color(0,1,0),Color(0,0,1),Color::Ones()},std::vector<float>{0,1,1,0});
        if(variant>=5) scene.directional_lights.push_back(DirectionalLight{Vec3(0,0,-1),Color::Ones()});
        quad(scene,-3,1);quad(scene,-2,0,variant==2);
        auto snapshot=make_render_scene_snapshot(scene);
        if(variant>=6) for(auto& instance:snapshot.instances) {
            // Perspective barycentrics under a mirrored, nonuniform transform.
            instance.object_to_world(0,0)=-1.2f;instance.object_to_world(1,1)=.8f;
            instance.object_to_world(2,0)=.3f;
            instance.world_to_object=instance.object_to_world.inverse();
            instance.normal_to_world=instance.world_to_object.topLeftCorner<3,3>().transpose();
            instance.world_bounds=Bounds3(Vec3(-10,-10,-10),Vec3(10,10,10));
        }
        CudaRealtimeRenderer raster(context.cuda),ray(context.cuda);
        auto provider=make_opengl_primary_visibility(context.cuda);raster.set_primary_visibility(provider);
        RenderSettings settings;settings.width=67;settings.height=51;settings.path.cuda_device=context.cuda.device_id();
        settings.realtime.raster_primary=true;
        settings.realtime.denoise=false;settings.realtime.max_bounces=4;settings.realtime.direct_lighting=false;
        settings.realtime.samples_per_pixel=4;settings.realtime.debug_view=RealtimeDebugView::Raw;
        if(variant>=5) {settings.realtime.direct_lighting=true;settings.realtime.max_bounces=1;}
        Framebuffer a(1,1),b(1,1);InteractiveFrameState state;
        for(int frame=0;frame<3;++frame) {
            // Verify that GUI/raster state does not disable the visibility pass,
            // and that the visibility pass restores the caller's state.
            glEnable(GL_SCISSOR_TEST);glScissor(0,0,1,1);glDisable(GL_DEPTH_TEST);
            glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE);glDepthMask(GL_FALSE);
            raster.render_next_frame(snapshot,camera(.003f+.01f*frame),settings,state,a);
            if(!raster.statistics().realtime.raster_primary_active) std::cerr<<provider->reason()<<'\n';
            RENDER_CHECK(raster.statistics().realtime.raster_primary_active);
            RENDER_CHECK(glIsEnabled(GL_SCISSOR_TEST) && !glIsEnabled(GL_DEPTH_TEST));
            GLboolean mask[4];glGetBooleanv(GL_COLOR_WRITEMASK,mask);RENDER_CHECK(mask[0]==GL_FALSE);
            glDisable(GL_SCISSOR_TEST);glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);glDepthMask(GL_TRUE);
            auto software_settings=settings;software_settings.realtime.hardware_ray_tracing=false;
            ray.render_next_frame(snapshot,camera(.003f+.01f*frame),software_settings,state,b);
            const auto error=difference(a,b);
            std::cout<<"hybrid material="<<variant<<" frame="<<frame<<" mse="<<error<<'\n';
            RENDER_CHECK(error<2e-4);
        }
        RENDER_CHECK(glGetError()==GL_NO_ERROR);
    }
}

RENDER_TEST(test_hybrid_motion_resize_cut_and_geometry_refresh) {
    Context context;CudaRealtimeRenderer raster(context.cuda),ray(context.cuda);
    auto provider=make_opengl_primary_visibility(context.cuda);raster.set_primary_visibility(provider);
    Scene scene;Material m;m.type=MaterialType::Emissive;m.emission=Color(.3f,.6f,.2f);scene.materials.push_back(m);
    quad(scene,-3,0);auto snapshot=make_render_scene_snapshot(scene);
    RenderSettings settings;settings.width=67;settings.height=51;settings.path.cuda_device=context.cuda.device_id();
    settings.realtime.raster_primary=true;
    Framebuffer a(1,1),b(1,1);InteractiveFrameState state;
    for(int frame=0;frame<20;++frame) {
        if(frame==10) {
            auto& instance=snapshot.instances[0];instance.object_to_world(0,3)=.05f;
            instance.world_to_object=instance.object_to_world.inverse();snapshot.revisions.transforms++;
        }
        if(frame==13) {settings.realtime.internal_scale=.5f;settings.width=59;settings.height=43;}
        state.camera_cut=frame==16;
        raster.render_next_frame(snapshot,camera(frame*.002f),settings,state,a);
        ray.render_next_frame(snapshot,camera(frame*.002f),settings,state,b);
        RENDER_CHECK(difference(a,b)<1e-8);
        const auto da=raster.download_diagnostics(),db=ray.download_diagnostics();
        const auto p=da.size()/2;
        RENDER_CHECK(std::abs(da[p].motion_x-db[p].motion_x)<1e-5);
        RENDER_CHECK(std::abs(da[p].depth-db[p].depth)<1e-4);
        RENDER_CHECK(std::abs(da[p].history-db[p].history)<.01);
    }
    RENDER_CHECK(raster.statistics().realtime.history_resets==3);
    const auto bytes=provider->resident_bytes();
    auto changed=std::make_shared<Scene>(*snapshot.assets[0].local_scene);
    changed->triangles.clear();
    snapshot.assets[0].local_scene=changed;snapshot.assets[0].triangle_material_slots.clear();
    ++snapshot.assets[0].geometry_revision;++snapshot.revisions.geometry;
    state.camera_cut=false;
    raster.render_next_frame(snapshot,camera(),settings,state,a);
    RENDER_CHECK((a.pixel(29,21)-snapshot.environment).norm()<1e-5);
    RENDER_CHECK(provider->resident_bytes()<bytes);
    settings.realtime.raster_primary=false;raster.render_next_frame(snapshot,camera(),settings,state,a);
    RENDER_CHECK(!raster.statistics().realtime.raster_primary_active);
}

RENDER_TEST(test_hybrid_primary_fixes_shared_triangle_edge_coverage) {
    Context context;CudaRealtimeRenderer r(context.cuda);r.set_primary_visibility(make_opengl_primary_visibility(context.cuda));
    Scene scene;Material m;m.type=MaterialType::Emissive;m.emission=Color::Ones();scene.materials.push_back(m);quad(scene,-3,0);
    const auto snapshot=make_render_scene_snapshot(scene);
    RenderSettings settings;settings.width=settings.height=64;settings.path.cuda_device=context.cuda.device_id();
    settings.realtime.raster_primary=true;
    settings.realtime.taa=false;settings.realtime.denoise=false;
    Framebuffer f(1,1);InteractiveFrameState state;
    r.render_next_frame(snapshot,Camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1),settings,state,f);
    for(int y=0;y<64;++y)for(int x=0;x<64;++x) RENDER_CHECK((f.pixel(x,y)-Color::Ones()).norm()<1e-5);
}

RENDER_TEST(test_hybrid_interop_presentation_and_analytic_fallback) {
    Context context;CudaRealtimeRenderer renderer(context.cuda);
    auto visibility=make_opengl_primary_visibility(context.cuda);renderer.set_primary_visibility(visibility);
    CudaOpenGlInteropTexture output;RENDER_CHECK(output.initialize(context.cuda));
    Scene scene;Material m;m.type=MaterialType::Emissive;m.emission=Color(.2f,.4f,.6f);
    scene.materials.push_back(m);quad(scene,-3,0);auto snapshot=make_render_scene_snapshot(scene);
    RenderSettings settings;settings.width=67;settings.height=51;settings.path.cuda_device=context.cuda.device_id();
    settings.realtime.raster_primary=true;InteractiveFrameState state;
    for(int frame=0;frame<20;++frame) {
        CudaSurfaceHandle surface=0;const auto stream=renderer.stream_handle();
        RENDER_CHECK(output.begin_frame(settings.width,settings.height,stream,surface));
        renderer.render_next_frame_to_surface(snapshot,camera(),settings,state,surface);
        RENDER_CHECK(output.end_frame(stream));
    }
    std::vector<float> rgba(67*51*4);
    glGetTextureImage(output.texture(),0,GL_RGBA,GL_FLOAT,GLsizei(rgba.size()*sizeof(float)),rgba.data());
    const auto center=(25*67+33)*4;
    RENDER_CHECK(std::abs(rgba[center]-.2f)<1e-5 && std::abs(rgba[center+2]-.6f)<1e-5);
    RENDER_CHECK(renderer.statistics().framebuffer_downloads==0);
    RENDER_CHECK(renderer.statistics().realtime.raster_primary_active);

    // Raw analytic assets cannot be represented by the triangle visibility
    // target. Returning zero must recover full ray visibility, including sky.
    auto local=std::make_shared<Scene>(*snapshot.assets[0].local_scene);
    local->spheres.emplace_back(Vec3(0,0,-2),.4f,0);
    snapshot.assets[0].local_scene=local;snapshot.assets[0].sphere_material_slots={MaterialSlot::bound(0)};
    ++snapshot.assets[0].geometry_revision;++snapshot.revisions.geometry;
    Framebuffer frame(1,1);renderer.render_next_frame(snapshot,camera(),settings,state,frame);
    RENDER_CHECK(!renderer.statistics().realtime.raster_primary_active);
    RENDER_CHECK(!renderer.statistics().realtime.hardware_ray_tracing_active);
    RENDER_CHECK(!visibility->reason().empty());
    RENDER_CHECK((frame.pixel(33,25)-m.emission).norm()<1e-5);
    RENDER_CHECK(glGetError()==GL_NO_ERROR);
}

RENDER_TEST(test_hybrid_performance_distribution) {
    if(!std::getenv("RTRT_HYBRID_PERFORMANCE")) RENDER_SKIP("set RTRT_HYBRID_PERFORMANCE=1 for the GL/CUDA timing sweep");
    Context context;
#if RENDERER_HAS_CUDA
    std::vector<std::string> cases={"builtin","Computer Graphics Archive/CornellBox/CornellBox-Glossy.obj",
        "Computer Graphics Archive/bedroom/iscv2.obj"};
    if(const char* asset=std::getenv("RTRT_BENCH_ASSET")) cases={asset};
    nlohmann::json results=nlohmann::json::array();
    for(const auto& name:cases) {
        Scene scene;Camera view(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,960.0f/540);
        if(name=="builtin") scene=make_cornell_box_scene();
        else {
            auto loaded=load_scene_asset((std::filesystem::path(RENDERER_SOURCE_DIR)/name).string(),960,540);
            view=loaded.camera;scene=std::move(loaded.scene);
        }
        const auto triangles=scene.triangles.size();const auto snapshot=make_render_scene_snapshot(std::move(scene));
        for(bool raster:{false,true}) {
            CudaRealtimeRenderer r(context.cuda);r.set_primary_visibility(make_opengl_primary_visibility(context.cuda));
            RenderSettings settings;settings.width=960;settings.height=540;settings.path.cuda_device=context.cuda.device_id();
            settings.realtime.raster_primary=raster;
            InteractiveFrameState state;
            std::vector<double> wall,gpu,primary,raster_time;
            for(int frame=0;frame<348;++frame) {
                const auto start=std::chrono::steady_clock::now();
                r.render_next_frame_to_surface(snapshot,view,settings,state,0);
                RENDER_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(r.stream_handle()))==cudaSuccess);
                const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
                r.refresh_statistics();
                if(frame>=48) {
                    wall.push_back(elapsed);gpu.push_back(r.statistics().realtime.total_ms);
                    primary.push_back(r.statistics().realtime.gbuffer_ms);
                    raster_time.push_back(r.statistics().realtime.raster_primary_ms);
                }
            }
            auto distribution=[](std::vector<double> values) {
                std::sort(values.begin(),values.end());
                return nlohmann::json{{"p50",values[values.size()/2]},
                    {"p95",values[std::size_t(std::ceil(values.size()*.95))-1]},
                    {"p99",values[std::size_t(std::ceil(values.size()*.99))-1]}};
            };
            const auto& rt=r.statistics().realtime;
            nlohmann::json entry={{"scene",name},{"triangles",triangles},{"raster",raster},
                {"samples",wall.size()},{"submit_and_wait_ms",distribution(wall)},
                {"cuda_ms",distribution(gpu)},{"cuda_primary_ms",distribution(primary)},
                {"raster_ms",distribution(raster_time)},{"raster_bytes",rt.raster_primary_bytes}};
            results.push_back(entry);std::cout<<"hybrid timing "<<entry.dump()<<std::endl;
            RENDER_CHECK(rt.raster_primary_active==raster);
            RENDER_CHECK(r.statistics().framebuffer_downloads==0);
            RENDER_CHECK(rt.history_resets==1);
            if(const char* dir=std::getenv("RTRT_VALIDATION_DIR")) {
                std::filesystem::create_directories(dir);
                Framebuffer frame(1,1);r.download_current_frame(frame);Image image(960,540);
                DisplaySettings display;display.tone_mapper=ToneMapper::Aces;
                for(int y=0;y<540;++y)for(int x=0;x<960;++x)image.set_pixel(x,y,apply_display_transform(frame.pixel(x,y),display));
                const std::string stem=name=="builtin"?name:std::filesystem::path(name).stem().string();
                RENDER_CHECK(image.write_png((std::filesystem::path(dir)/(stem+(raster?"-raster.png":"-cuda.png"))).string()));
            }
            export_json("hybrid-performance.json",results);
        }
    }
#endif
}
