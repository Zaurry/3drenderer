#include "test_framework.h"
#include "render/realtime/cuda_realtime_renderer.h"
#include "render/realtime/realtime_settings_json.h"
#include "core/image.h"
#include "scene/scene_document.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#if RENDERER_HAS_CUDA
#include <cuda_runtime_api.h>
#endif

using namespace renderer;
namespace {
CudaDeviceContext device() {
    std::string reason;
    if (!cuda_path_backend_available(0, &reason)) RENDER_SKIP(reason);
    return CudaDeviceContext::create(0);
}
Camera front_camera(float x=0) { return Camera(Vec3(x,0,0),Vec3(x,0,-3),Vec3::UnitY(),45,1); }
RenderSettings small_settings(int size=32) {
    RenderSettings s; s.width=s.height=size; s.path.cuda_device=0; return s;
}
Scene plane_scene(bool emissive=false, bool slope=false) {
    Scene s; s.environment=Color::Zero();
    Material m; m.base_color=Color(.7f,.5f,.2f); m.roughness=1;
    if(emissive) {m.type=MaterialType::Emissive;m.emission=Color(.7f,.2f,.1f);}
    s.materials.push_back(m);
    const float left=slope?-4.0f:-3.0f,right=slope?-2.0f:-3.0f;
    s.triangles.emplace_back(Vec3(-3,-3,left),Vec3(3,-3,right),Vec3(3,3,right),0);
    s.triangles.emplace_back(Vec3(-3,-3,left),Vec3(3,3,right),Vec3(-3,3,left),0);
    return s;
}
void check_finite(const Framebuffer& f) {
    for(int y=0;y<f.height();++y) for(int x=0;x<f.width();++x) {
        RENDER_CHECK(f.pixel(x,y).allFinite());
        RENDER_CHECK(f.pixel(x,y).minCoeff()>=0);
    }
}
void translate(RenderSceneInstanceSnapshot& i, const Vec3& d) {
    i.object_to_world.topRightCorner<3,1>() += d;
    i.world_to_object=i.object_to_world.inverse();
    i.world_bounds=Bounds3(i.world_bounds.min+d,i.world_bounds.max+d);
}
void export_frame(const char* name,const Framebuffer& f) {
    const char* dir=std::getenv("RTRT_VALIDATION_DIR");
    if(!dir || !*dir) return;
    std::filesystem::create_directories(dir);
    std::ofstream hdr(std::filesystem::path(dir)/(std::string(name)+".pfm"),std::ios::binary);
    hdr << "PF\n" << f.width() << ' ' << f.height() << "\n-1.0\n";
    for(int y=f.height()-1;y>=0;--y)for(int x=0;x<f.width();++x) hdr.write(reinterpret_cast<const char*>(f.pixel(x,y).data()),3*sizeof(float));
    Image image(f.width(),f.height());
    DisplaySettings d; d.tone_mapper=ToneMapper::Aces;
    for(int y=0;y<f.height();++y) for(int x=0;x<f.width();++x)
        image.set_pixel(x,y,apply_display_transform(f.pixel(x,y),d));
    RENDER_CHECK(image.write_png((std::filesystem::path(dir)/name).string()));
}
double mse(const Framebuffer& f,const Image& ref) {
    double sum=0;
    for(int y=0;y<f.height();++y) for(int x=0;x<f.width();++x)
        sum+=(f.pixel(x,y)-ref.pixel(x,y)).squaredNorm();
    return sum/(3*f.width()*f.height());
}
void export_metrics(const char* name,const nlohmann::json& metrics) {
    if(const char* dir=std::getenv("RTRT_VALIDATION_DIR")) {
        std::filesystem::create_directories(dir);
        std::ofstream out(std::filesystem::path(dir)/name);
        out << metrics.dump(2);
    }
}
// Measure moving low-frequency blotches separately from individual pixel noise.
double block_difference(const Framebuffer& a,const Framebuffer& b) {
    double sum=0; int count=0;
    for(int y=8;y+8<a.height()-8;y+=8) for(int x=8;x+8<a.width()-8;x+=8) {
        Color delta=Color::Zero();
        for(int dy=0;dy<8;++dy) for(int dx=0;dx<8;++dx)
            delta+=a.pixel(x+dx,y+dy)-b.pixel(x+dx,y+dy);
        sum+=(delta/64).squaredNorm(); ++count;
    }
    return sum/std::max(1,3*count);
}
}

RENDER_TEST(test_realtime_settings_roundtrip_and_sanitization) {
    RealtimeRenderSettings s; s.internal_scale=.5f; s.diffuse_history=48; s.specular_iterations=0;
    s.transmission=false; s.debug_view=RealtimeDebugView::Motion; s.sharpening=.3f;
    s.raster_primary=false; s.low_discrepancy=false;s.hardware_ray_tracing=false;s.full_resolution_materials=false;s.shader_execution_reordering=false;
    s.split_dielectric=false;
    s.denoiser=RealtimeDenoiser::Optix;
    RENDER_CHECK(parse_realtime_settings(realtime_settings_json(s))==s);
    RENDER_CHECK(realtime_settings_json(s).at("denoiser")=="optix");
    RENDER_CHECK(parse_realtime_settings({}).denoiser==RealtimeDenoiser::Svgf);
    for(const auto& value:{nlohmann::json("future"),nlohmann::json(999),nlohmann::json(nullptr)})
        RENDER_CHECK(parse_realtime_settings({{"denoiser",value}}).denoiser==RealtimeDenoiser::Svgf);
    s.denoiser=static_cast<RealtimeDenoiser>(255);
    RENDER_CHECK(sanitize_realtime_settings(s).denoiser==RealtimeDenoiser::Svgf);
    s.internal_scale=std::numeric_limits<float>::quiet_NaN(); s.normal_threshold=std::numeric_limits<float>::infinity();
    s.max_bounces=-20; s.diffuse_iterations=900;
    auto safe=sanitize_realtime_settings(s);
    RENDER_CHECK(safe.internal_scale==1 && safe.normal_threshold==.85f);
    RENDER_CHECK(safe.max_bounces==1 && safe.diffuse_iterations==5);
    const auto parsed=parse_realtime_settings({{"max_bounces",1e99},{"taa","invalid"},{"internal_scale",nullptr}});
    RENDER_CHECK(parsed.max_bounces==64 && parsed.taa && parsed.internal_scale==1);
}

RENDER_TEST(test_realtime_constant_preservation_resize_and_resident_output) {
    CudaRealtimeRenderer r(device()); auto s=small_settings();
    Scene scene; scene.environment=Color(.3f,.6f,2); auto snapshot=make_render_scene_snapshot(scene);
    Framebuffer f(1,1); InteractiveFrameState state;
    for(int i=0;i<8;++i) r.render_next_frame_to_surface(snapshot,front_camera(),s,state,0);
    RENDER_CHECK(r.statistics().framebuffer_downloads==0);
    r.download_current_frame(f); check_finite(f);
    for(int y=0;y<32;++y) for(int x=0;x<32;++x) RENDER_CHECK((f.pixel(x,y)-scene.environment).norm()<1e-5f);
    auto generation=r.statistics().allocation_generation;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(r.statistics().allocation_generation==generation);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    s.realtime.debug_view=RealtimeDebugView::HistoryLength;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    s.realtime.debug_view=RealtimeDebugView::Final; s.realtime.internal_scale=.5f;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.width()==32 && r.statistics().internal_width==16);
    RENDER_CHECK(r.statistics().realtime.history_resets==2);
    RENDER_CHECK((f.pixel(16,16)-scene.environment).norm()<1e-5f);
    s.width=17;s.height=13;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    check_finite(f); RENDER_CHECK(f.width()==17 && f.height()==13);
}

RENDER_TEST(test_realtime_motion_jitter_slope_cut_and_object_history) {
    CudaRealtimeRenderer r(device()); auto s=small_settings();
    auto snapshot=make_render_scene_snapshot(plane_scene(true,true));
    Framebuffer f(1,1); InteractiveFrameState state;
    for(int i=0;i<10;++i) r.render_next_frame(snapshot,front_camera(),s,state,f);
    auto d=r.download_diagnostics(); const int p=16*32+16;
    RENDER_CHECK(std::abs(d[p].motion_x)<1e-6f && std::abs(d[p].motion_y)<1e-6f);
    RENDER_CHECK(d[p].history>8 && d[p].variance<1e-5f);
    RENDER_CHECK((f.pixel(16,16)-Color(.7f,.2f,.1f)).norm()<1e-4f);
    r.render_next_frame(snapshot,front_camera(.04f),s,state,f); d=r.download_diagnostics();
    RENDER_CHECK(d[p].motion_x>0 && d[p].history>8 && d[p].rejected==0);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    translate(snapshot.instances[0],Vec3(.04f,0,0)); snapshot.revisions.transforms++;
    r.render_next_frame(snapshot,front_camera(.04f),s,state,f); d=r.download_diagnostics();
    RENDER_CHECK(d[p].motion_x<0 && d[p].rejected==0);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    state.camera_cut=true;
    r.render_next_frame(snapshot,front_camera(),s,state,f); d=r.download_diagnostics();
    RENDER_CHECK(d[p].history==1 && d[p].variance==0 && d[p].rejected==1);
    RENDER_CHECK(r.statistics().realtime.history_resets==2);
}

RENDER_TEST(test_realtime_disocclusion_rejects_old_surface) {
    CudaRealtimeRenderer r(device()); auto s=small_settings();
    Scene scene; Material m; m.type=MaterialType::Emissive;m.emission=Color(2,0,0);scene.materials.push_back(m);
    scene.spheres.emplace_back(Vec3(0,0,-2),.5f,0);scene.environment=Color(0,0,1);
    auto snapshot=make_render_scene_snapshot(scene);Framebuffer f(1,1);InteractiveFrameState state;
    for(int i=0;i<8;++i) r.render_next_frame(snapshot,front_camera(),s,state,f);
    export_frame("disocclusion-before.png",f);
    translate(snapshot.instances[0],Vec3(1.5f,0,0));snapshot.revisions.transforms++;
    r.render_next_frame(snapshot,front_camera(),s,state,f);auto d=r.download_diagnostics();
    RENDER_CHECK(d[16*32+16].rejected==1 && d[16*32+16].history==1);
    RENDER_CHECK((f.pixel(16,16)-scene.environment).norm()<1e-4f);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    export_frame("disocclusion-after.png",f);
}

RENDER_TEST(test_realtime_punctual_and_area_shadow_switches) {
    auto context=device(); auto s=small_settings();
    s.realtime.denoise=false;s.realtime.taa=false;s.realtime.soft_shadows=false;s.realtime.max_bounces=1;s.realtime.samples_per_pixel=32;
    for(int kind=0;kind<4;++kind) {
        auto scene=plane_scene(); scene.spheres.emplace_back(Vec3(.5f,0,-2),.22f,0);
        if(kind==0) scene.point_lights.push_back(PointLight{Vec3(1,0,-1),Color::Constant(15)});
        if(kind==1) scene.directional_lights.push_back(DirectionalLight{Vec3(-1,0,-2),Color::Constant(3)});
        if(kind==2) {SpotLight l;l.position=Vec3(1,0,-1);l.direction=Vec3(-1,0,-2);l.intensity=Color::Constant(15);scene.spot_lights.push_back(l);}
        if(kind==3) {RectAreaLight l;l.position=Vec3(1,0,-1);l.axis_u=Vec3(.05f,0,0);l.axis_v=Vec3(0,.05f,0);l.radiance=Color::Constant(500);l.two_sided=true;scene.rect_area_lights.push_back(l);}
        CudaRealtimeRenderer r(context);Framebuffer shadow(1,1),lit(1,1);InteractiveFrameState state;
        auto snapshot=make_render_scene_snapshot(scene);
        r.render_next_frame(snapshot,front_camera(),s,state,shadow);
        for(auto& l:scene.point_lights) l.casts_shadows=false;
        for(auto& l:scene.directional_lights) l.casts_shadows=false;
        for(auto& l:scene.spot_lights) l.casts_shadows=false;
        for(auto& l:scene.rect_area_lights) l.casts_shadows=false;
        snapshot.point_lights=scene.point_lights;snapshot.directional_lights=scene.directional_lights;snapshot.spot_lights=scene.spot_lights;
        for(auto& instance:snapshot.instances) instance.emission_casts_shadows=false;
        snapshot.revisions.lighting++;
        r.render_next_frame(snapshot,front_camera(),s,state,lit);
        RENDER_CHECK(lit.pixel(16,16).sum()>.05f);
        RENDER_CHECK(lit.pixel(16,16).sum()>shadow.pixel(16,16).sum()+.04f);
    }
}

RENDER_TEST(test_realtime_light_change_reacts_without_full_reset) {
    CudaRealtimeRenderer r(device());auto s=small_settings();s.realtime.max_bounces=1;s.realtime.taa=false;
    auto scene=plane_scene();scene.directional_lights.push_back(DirectionalLight{Vec3(0,0,-1),Color::Ones()});
    auto snapshot=make_render_scene_snapshot(scene);Framebuffer f(1,1);InteractiveFrameState state;
    for(int i=0;i<16;++i) r.render_next_frame(snapshot,front_camera(),s,state,f);
    const float before=f.pixel(16,16).sum();
    snapshot.directional_lights[0].radiance*=4;snapshot.revisions.lighting++;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.pixel(16,16).sum()>3.5f*before);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    RENDER_CHECK(r.download_diagnostics()[16*32+16].reactive>=.5f);
}

RENDER_TEST(test_realtime_moving_shadow_responds_without_reset) {
    for(bool upscale:{false,true}) {
    auto context=device(); CudaRealtimeRenderer r(context); auto s=small_settings(64);
    if(upscale)s.realtime=realtime_1080p_quality_settings();
    s.realtime.max_bounces=1;s.realtime.soft_shadows=false;
    auto scene=plane_scene(); scene.spheres.emplace_back(Vec3(.5f,0,-2),.22f,0);
    scene.point_lights.push_back(PointLight{Vec3(1,0,-1),Color::Constant(15)});
    auto snapshot=make_render_scene_snapshot(scene);
    RENDER_CHECK(snapshot.instances.size()==2);
    Framebuffer f(1,1),reference(1,1);InteractiveFrameState state;
    for(int frame=0;frame<24;++frame) r.render_next_frame(snapshot,front_camera(),s,state,f);
    const auto reset_count=r.statistics().realtime.history_resets;
    auto reference_settings=s; reference_settings.path.samples_per_pixel=256; reference_settings.path.max_bounces=1;
    nlohmann::json errors=nlohmann::json::array();
    for(float dx:{1.5f,-1.5f}) {
        translate(snapshot.instances[1],Vec3(dx,0,0)); ++snapshot.revisions.transforms;
        const auto ground_truth=render_cuda_path(snapshot,front_camera(),reference_settings);
        reference.resize(64,64);
        for(int y=0;y<64;++y) for(int x=0;x<64;++x) reference.set_pixel(x,y,ground_truth.image.pixel(x,y));
        for(int frame=0;frame<4;++frame) r.render_next_frame(snapshot,front_camera(),s,state,f);
        double error=0,energy=0;
        for(int y=29;y<35;++y) for(int x=29;x<35;++x) {
            error+=(f.pixel(x,y)-reference.pixel(x,y)).squaredNorm();
            energy+=reference.pixel(x,y).squaredNorm();
        }
        const double normalized=error/std::max(energy,1.0);
        export_frame(dx>0?"shadow-reveal.png":"shadow-cover.png",f);
        export_frame(dx>0?"shadow-reveal-reference.png":"shadow-cover-reference.png",reference);
        errors.push_back(normalized);
        std::cout << "RTRT moving shadow: relative_patch_mse=" << normalized << '\n';
        RENDER_CHECK(normalized<.01);
        RENDER_CHECK(r.statistics().realtime.history_resets==reset_count);
        for(int frame=0;frame<20;++frame) r.render_next_frame(snapshot,front_camera(),s,state,f);
    }
    export_metrics(upscale?"moving-shadow-upscale.json":"moving-shadow.json",{{"four_frame_patch_errors",errors}});
    }
}

RENDER_TEST(test_realtime_offscreen_reflection_glass_and_alpha) {
    auto context=device();auto s=small_settings();
    s.realtime.denoise=false;s.realtime.taa=false;s.realtime.roulette_start=64;
    s.realtime.samples_per_pixel=64;s.realtime.direct_lighting=false;
    auto scene=plane_scene();scene.materials[0].type=MaterialType::Metal;
    scene.materials[0].base_color=Color::Ones();scene.materials[0].roughness=.02f;
    Material emitter;emitter.type=MaterialType::Emissive;emitter.emission=Color(1,.2f,.1f);scene.materials.push_back(emitter);
    // This emitter is behind the camera and can only be seen in reflection.
    scene.triangles.emplace_back(Vec3(-8,-8,1),Vec3(8,8,1),Vec3(8,-8,1),1);
    scene.triangles.emplace_back(Vec3(-8,-8,1),Vec3(-8,8,1),Vec3(8,8,1),1);
    auto snapshot=make_render_scene_snapshot(scene);CudaRealtimeRenderer r(context);Framebuffer f(1,1);InteractiveFrameState state;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.pixel(16,16).x()>.8f);export_frame("offscreen-mirror.png",f);
    s.realtime.reflections=false;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.pixel(16,16).sum()<1e-4f);
    s.realtime.reflections=true;
    scene=Scene{};scene.environment=Color(.3f,.6f,.9f);
    Material glass;glass.type=MaterialType::Dielectric;scene.materials.push_back(glass);
    scene.spheres.emplace_back(Vec3(0,0,-2),.5f,0);snapshot=make_render_scene_snapshot(scene);
    r.render_next_frame(snapshot,front_camera(),s,state,f);check_finite(f);
    RENDER_CHECK((f.pixel(16,16)-scene.environment).norm()<.04f);export_frame("glass.png",f);
    s.realtime.transmission=false;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.pixel(16,16).sum()<scene.environment.sum()*.3f);
    s.realtime.transmission=true;
    for(auto mode:{AlphaMode::Mask,AlphaMode::Blend}) {
        scene.materials[0].type=MaterialType::Diffuse;scene.materials[0].opacity=0;scene.materials[0].alpha_mode=mode;
        snapshot=make_render_scene_snapshot(scene);r.render_next_frame(snapshot,front_camera(),s,state,f);check_finite(f);
        RENDER_CHECK((f.pixel(16,16)-scene.environment).norm()<1e-4f);
    }
}

RENDER_TEST(test_realtime_rough_reflection_noise_does_not_reject_history) {
    CudaRealtimeRenderer r(device()); auto s=small_settings(64);
    s.realtime.direct_lighting=false; s.realtime.max_bounces=3;
    auto scene=plane_scene(); scene.environment=Color::Ones();
    scene.materials[0].type=MaterialType::Metal;
    scene.materials[0].base_color=Color::Ones(); scene.materials[0].roughness=.45f;
    Material light; light.type=MaterialType::Emissive; light.emission=Color::Ones();
    scene.materials.push_back(light);
    // Equal radiance at very different secondary distances. The BRDF sample
    // changes every frame, but there is no lighting or geometry change.
    scene.triangles.emplace_back(Vec3(0,-20,1),Vec3(20,20,1),Vec3(20,-20,1),1);
    scene.triangles.emplace_back(Vec3(0,-20,1),Vec3(0,20,1),Vec3(20,20,1),1);
    scene.triangles.emplace_back(Vec3(-20,-20,9),Vec3(0,20,9),Vec3(0,-20,9),1);
    scene.triangles.emplace_back(Vec3(-20,-20,9),Vec3(-20,20,9),Vec3(0,20,9),1);
    const auto snapshot=make_render_scene_snapshot(scene);
    InteractiveFrameState state; Framebuffer f(1,1),last(1,1);
    double reactive=0,boiling=0,settled=0; int count=0;
    for(int frame=0;frame<96;++frame) {
        const float x=frame<48?0:std::min(frame-48,16)*.008f;
        r.render_next_frame(snapshot,front_camera(x),s,state,f);
        if(frame>=32 && frame<48) {
            const auto d=r.download_diagnostics();
            for(int y=8;y<56;++y) for(int px=8;px<56;++px) {reactive+=d[y*64+px].reactive; ++count;}
            boiling+=block_difference(f,last)/16;
        }
        if(frame>=80) settled+=block_difference(f,last)/16;
        last=f;
    }
    reactive/=count;
    std::cout << "RTRT glossy stability: reactive=" << reactive << " block_flicker=" << boiling
        << " settled_block_flicker=" << settled << '\n';
    export_metrics("glossy-stability.json",{{"static_reactive_mean",reactive},
        {"static_block_flicker_mse",boiling},{"settled_block_flicker_mse",settled}});
    export_frame("rough-reflection-settled.png",f);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    RENDER_CHECK(reactive<.05);
    RENDER_CHECK(boiling<5e-6 && settled<5e-6);
}

RENDER_TEST(test_realtime_performance_distribution) {
    if(!std::getenv("RTRT_PERFORMANCE")) RENDER_SKIP("set RTRT_PERFORMANCE=1 for the GPU timing sweep");
    auto context=device();
#if RENDERER_HAS_CUDA
    const auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    nlohmann::json cases=nlohmann::json::array();
    for(float scale:{1.0f,.5f}) for(bool moving:{false,true}) {
        CudaRealtimeRenderer r(context); auto s=small_settings();
        s.width=960;s.height=540;s.realtime.internal_scale=scale;
        std::vector<double> gpu,wall,temporal,filter,reconstruction,lighting;
        InteractiveFrameState state;
        for(int frame=0;frame<120;++frame) {
            const float x=moving?.06f*std::sin(frame*.025f):0;
            Camera camera(Vec3(x,.15f,1.5f),Vec3(x,.15f,-2),Vec3::UnitY(),45,960.0f/540);
            const auto start=std::chrono::steady_clock::now();
            r.render_next_frame_to_surface(snapshot,camera,s,state,0);
            RENDER_CHECK(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(r.stream_handle()))==cudaSuccess);
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            r.refresh_statistics();
            if(frame>=24) {
                const auto& rt=r.statistics().realtime;
                gpu.push_back(rt.total_ms);wall.push_back(elapsed);temporal.push_back(rt.temporal_ms);
                filter.push_back(rt.filter_ms);reconstruction.push_back(rt.reconstruction_ms);lighting.push_back(rt.lighting_ms);
            }
        }
        auto distribution=[](std::vector<double> values) {
            std::sort(values.begin(),values.end());
            return nlohmann::json{{"p50",values[values.size()/2]},
                {"p95",values[std::size_t(std::ceil(values.size()*.95))-1]},
                {"p99",values[std::size_t(std::ceil(values.size()*.99))-1]}};
        };
        nlohmann::json entry={{"internal_scale",scale},{"moving",moving},{"samples",gpu.size()},
            {"gpu_ms",distribution(gpu)},{"cpu_submit_and_gpu_wait_ms",distribution(wall)},
            {"temporal_ms",distribution(temporal)},{"atrous_ms",distribution(filter)},
            {"reconstruction_ms",distribution(reconstruction)},{"lighting_ms",distribution(lighting)},
            {"framebuffer_bytes",r.statistics().realtime.framebuffer_bytes}};
        cases.push_back(entry);std::cout << "RTRT performance: " << entry.dump() << '\n';
        RENDER_CHECK(r.statistics().framebuffer_downloads==0);
        RENDER_CHECK(r.statistics().realtime.history_resets==1);
    }
    export_metrics("performance.json",cases);
#endif
}

RENDER_TEST(test_realtime_secondary_reflection_motion_responds_without_reset) {
    for(float roughness:{.45f,.02f}) {
    for(bool upscale:{false,true}) {
    CudaRealtimeRenderer renderer(device());auto settings=small_settings(64);
    if(upscale)settings.realtime=realtime_1080p_quality_settings();
    settings.realtime.direct_lighting=false;settings.realtime.max_bounces=3;
    auto scene=plane_scene();scene.environment=Color::Zero();
    scene.materials[0].type=MaterialType::Metal;
    scene.materials[0].base_color=Color::Ones();scene.materials[0].roughness=roughness;
    auto snapshot=make_render_scene_snapshot(scene);
    Scene light;Material material;material.type=MaterialType::Emissive;material.emission=Color::Ones();
    light.materials.push_back(material);
    light.triangles.emplace_back(Vec3(-12,-12,1),Vec3(12,12,1),Vec3(12,-12,1),0);
    light.triangles.emplace_back(Vec3(-12,-12,1),Vec3(-12,12,1),Vec3(12,12,1),0);
    auto emitter=make_render_scene_snapshot(light);
    snapshot.assets.push_back(emitter.assets[0]);snapshot.assets.back().asset_id=2;
    snapshot.instances.push_back(emitter.instances[0]);
    snapshot.instances.back().object_id=2;snapshot.instances.back().asset_index=1;
    Framebuffer frame(1,1);InteractiveFrameState state;
    auto patch=[](const Framebuffer& f) {
        double mean=0;for(int y=24;y<40;++y)for(int x=24;x<40;++x)mean+=f.pixel(x,y).mean();
        return mean/256;
    };
    for(int i=0;i<48;++i)renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);
    const double before=patch(frame);RENDER_CHECK(before>.7);
    translate(snapshot.instances.back(),Vec3(1000,0,0));++snapshot.revisions.transforms;
    // The mirror's primary surface and material remain unchanged. Only the
    // off-screen reflected object moves, so no global reset may hide lag.
    for(int i=0;i<8;++i)renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);
    const double after=patch(frame);
    std::cout<<"RTRT reflected motion: before="<<before<<" after_8="<<after<<'\n';
    const char* report=roughness<.06f?(upscale?"mirror-motion-upscale.json":"mirror-motion.json"):
        (upscale?"reflected-motion-upscale.json":"reflected-motion.json");
    export_metrics(report,{{"before",before},{"after_8_frames",after}});
    RENDER_CHECK(after<before*.1);
    RENDER_CHECK(renderer.statistics().realtime.history_resets==1);
    }
    }
}

RENDER_TEST(test_realtime_optix_denoiser_quality_against_offline_reference) {
    const auto context=device();auto s=small_settings(96);
    s.path.max_bounces=8;s.path.samples_per_pixel=512;
    const auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    // Keep the output TAA off so this verifies the neural denoiser itself.
    s.realtime.taa=false;s.realtime.temporal_upscale=false;
    s.realtime.denoiser=RealtimeDenoiser::Optix;
    CudaRealtimeRenderer probe(context);Framebuffer frame(1,1);InteractiveFrameState state;
    probe.render_next_frame(snapshot,camera,s,state,frame);
    if(!probe.statistics().realtime.optix_denoiser_active) {
        if(std::getenv("RTRT_REQUIRE_OPTIX_DENOISER"))throw TestFailure{probe.statistics().realtime.optix_denoiser_detail};
        RENDER_SKIP(probe.statistics().realtime.optix_denoiser_detail);
    }
    const auto reference=render_cuda_path(snapshot,camera,s).image;
    Framebuffer ref(96,96);double reference_energy=0;
    for(int y=0;y<96;++y)for(int x=0;x<96;++x) {
        ref.set_pixel(x,y,reference.pixel(x,y));reference_energy+=reference.pixel(x,y).sum();
    }
    export_frame("optix-reference-512spp.png",ref);
    nlohmann::json metrics=nlohmann::json::array();
    for(int variant=0;variant<4;++variant) {
        s.realtime.temporal=variant!=0;s.realtime.internal_scale=variant>=2?.5f:1.0f;
        s.realtime.full_resolution_materials=variant!=3;
        CudaRealtimeRenderer raw_renderer(context);auto raw_settings=s;raw_settings.realtime.denoise=false;
        Framebuffer raw(1,1);raw_renderer.render_next_frame(snapshot,camera,raw_settings,state,raw);
        // Compare to the same resolution/material reconstruction, not a native
        // input when evaluating a lower-resolution denoising configuration.
        export_frame(("optix-input-"+std::to_string(variant)+".png").c_str(),raw);
        CudaRealtimeRenderer r(context);Framebuffer previous(1,1);double flicker=0,error=0,raw_error=0;
        double surface_error=0,raw_surface_error=0;int surface_pixels=0;
        for(int i=0;i<16;++i) {
            r.render_next_frame(snapshot,camera,s,state,frame);check_finite(frame);
            if(i>0)raw_renderer.render_next_frame(snapshot,camera,raw_settings,state,raw);
            RENDER_CHECK(r.statistics().realtime.optix_denoiser_active);
            if(i>=8) {error+=mse(frame,reference)/8;raw_error+=mse(raw,reference)/8;}
            // Compare wall/floor shading separately from the bright emitter's
            // subpixel silhouette, which has unavoidable half-resolution error.
            if(i>=8)for(int y=24;y<86;++y)for(int x=10;x<86;++x) {
                surface_error+=(frame.pixel(x,y)-reference.pixel(x,y)).squaredNorm();
                raw_surface_error+=(raw.pixel(x,y)-reference.pixel(x,y)).squaredNorm();++surface_pixels;
            }
            if(i>=8)for(int y=0;y<96;++y)for(int x=0;x<96;++x)
                flicker+=(frame.pixel(x,y)-previous.pixel(x,y)).squaredNorm()/(8*3*96*96);
            previous=frame;
        }
        double energy=0;
        for(int y=0;y<96;++y)for(int x=0;x<96;++x)energy+=frame.pixel(x,y).sum();
        metrics.push_back({{"variant",variant},{"settings",realtime_settings_json(s.realtime)},
            {"raw_mse",raw_error},{"denoised_mse",error},{"energy_ratio",energy/reference_energy},
            {"surface_mse",surface_error/(3*surface_pixels)},{"raw_surface_mse",raw_surface_error/(3*surface_pixels)},
            {"flicker_mse",flicker},{"denoiser_bytes",r.statistics().realtime.optix_denoiser_bytes}});
        std::cout<<"OptiX denoiser variant="<<variant<<" raw_mse="<<raw_error<<" denoised_mse="<<error
            <<" energy="<<energy/reference_energy<<" flicker="<<flicker
            <<" surface_mse="<<surface_error/(3*surface_pixels)<<" raw_surface_mse="<<raw_surface_error/(3*surface_pixels)<<'\n';
        export_frame(("optix-cornell-"+std::to_string(variant)+".png").c_str(),frame);
        export_metrics("optix-quality.json",metrics);
        RENDER_CHECK(surface_error<raw_surface_error);
        RENDER_CHECK(error<raw_error*1.05);
        RENDER_CHECK(energy/reference_energy>.65 && energy/reference_energy<1.35);
    }
}

RENDER_TEST(test_realtime_svgf_quality_against_offline_reference) {
    auto context=device();auto s=small_settings(96);s.path.max_bounces=8;s.path.samples_per_pixel=1024;
    auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    const auto reference=render_cuda_path(snapshot,camera,s);
    Framebuffer ref(96,96);for(int y=0;y<96;++y)for(int x=0;x<96;++x)ref.set_pixel(x,y,reference.image.pixel(x,y));
    export_frame("cornell-reference-1024spp.png",ref);
    CudaRealtimeRenderer r(context);Framebuffer raw(1,1),filtered(1,1),native(1,1);InteractiveFrameState state;
    s.realtime.debug_view=RealtimeDebugView::Raw;
    r.render_next_frame(snapshot,camera,s,state,raw);check_finite(raw);export_frame("cornell-raw-1spp.png",raw);
    s.realtime.debug_view=RealtimeDebugView::Final;
    CudaRealtimeRenderer raw_renderer(context);auto raw_settings=s;raw_settings.realtime.debug_view=RealtimeDebugView::Raw;
    double flicker=0,raw_flicker=0;Framebuffer last=raw,last_raw=raw,current_raw(1,1);
    for(int i=0;i<48;++i) {
        r.render_next_frame(snapshot,camera,s,state,filtered);
        if(i>=40) for(int y=0;y<96;++y)for(int x=0;x<96;++x) flicker+=(filtered.pixel(x,y)-last.pixel(x,y)).squaredNorm();
        raw_renderer.render_next_frame(snapshot,camera,raw_settings,state,current_raw);
        if(i>=40) for(int y=0;y<96;++y)for(int x=0;x<96;++x) raw_flicker+=(current_raw.pixel(x,y)-last_raw.pixel(x,y)).squaredNorm();
        last_raw=current_raw;last=filtered;
    }
    check_finite(filtered);export_frame("cornell-svgf-taa.png",filtered);native=filtered;
    const double raw_error=mse(raw,reference.image),filtered_error=mse(filtered,reference.image);
    std::cout << "Native errors: raw=" << raw_error << " filtered=" << filtered_error << std::endl;
    s.realtime.internal_scale=.5f;
    for(int i=0;i<48;++i)r.render_next_frame(snapshot,camera,s,state,filtered);
    check_finite(filtered);export_frame("cornell-svgf-taau-50.png",filtered);
    const double upscale_error=mse(filtered,reference.image);
    std::cout << "RTRT quality: raw_mse=" << raw_error << " svgf_taa_mse=" << filtered_error
        << " raw_flicker_mse=" << raw_flicker/(8*96*96*3) << " taau_mse=" << upscale_error << " static_flicker_mse=" << flicker/(8*96*96*3) << '\n';
    if(const char* dir=std::getenv("RTRT_VALIDATION_DIR")) {
        std::ofstream out(std::filesystem::path(dir)/"metrics.json");
        out << nlohmann::json{{"reference_spp",1024},{"frames",48},{"raw_mse",raw_error},
            {"svgf_taa_mse",filtered_error},{"taau_50_mse",upscale_error},{"static_flicker_mse",flicker/(8*96*96*3)},{"raw_flicker_mse",raw_flicker/(8*96*96*3)}}.dump(2);
    }
    RENDER_CHECK(filtered_error<raw_error*.8);
    RENDER_CHECK(flicker<raw_flicker*.8);
}

RENDER_TEST(test_realtime_low_discrepancy_convergence) {
    auto context=device();auto settings=small_settings(64);
    settings.path.max_bounces=8;settings.path.samples_per_pixel=2048;
    const auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    const auto reference=render_cuda_path(snapshot,camera,settings);
    nlohmann::json results=nlohmann::json::array();
    double total_error[2]{};
    for(int seed=0;seed<4;++seed) for(int sobol=0;sobol<2;++sobol) {
        CudaRealtimeRenderer renderer(context);
        settings.path.sample_seed_offset=seed*101;
        settings.realtime.low_discrepancy=sobol!=0;
        settings.realtime.denoise=false;settings.realtime.debug_view=RealtimeDebugView::Raw;
        // Keep jitter for pixel coverage, but measure an arithmetic mean of raw
        // radiance, independently of denoising, clamping and TAA.
        Framebuffer frame(1,1),average(64,64);InteractiveFrameState state;
        for(int sample=0;sample<64;++sample) {
            renderer.render_next_frame(snapshot,camera,settings,state,frame);
            for(int y=0;y<64;++y)for(int x=0;x<64;++x)
                average.set_pixel(x,y,average.pixel(x,y)+frame.pixel(x,y)/64);
        }
        const double error=mse(average,reference.image);total_error[sobol]+=error;
        double energy=0,reference_energy=0;
        for(int y=0;y<64;++y)for(int x=0;x<64;++x) {
            energy+=average.pixel(x,y).sum();reference_energy+=reference.image.pixel(x,y).sum();
        }
        results.push_back({{"seed",seed},{"sobol",sobol!=0},{"raw_64spp_mse",error},
            {"energy_ratio",energy/reference_energy}});
        RENDER_CHECK(std::abs(energy/reference_energy-1)<.06);
    }
    export_metrics("sampling-convergence.json",results);
    std::cout<<"RTRT 64-sample raw convergence: PCG="<<total_error[0]/4<<" Sobol="<<total_error[1]/4<<'\n';
    RENDER_CHECK(total_error[1]<total_error[0]*.9);
}

RENDER_TEST(test_realtime_hardware_material_updates_and_instancing) {
    const auto context=device();CudaRealtimeRenderer hardware(context),software(context);
    // Jitter avoids the legacy software traversal's non-watertight shared
    // triangle diagonal. Watertight constant coverage is checked separately.
    auto settings=small_settings(37);settings.realtime.denoise=false;
    settings.realtime.debug_view=RealtimeDebugView::Raw;settings.realtime.max_bounces=3;
    Scene scene=plane_scene(true);scene.environment=Color(.2f,.4f,.1f);
    scene.materials[0].type=MaterialType::Pbr;scene.materials[0].two_sided=false;
    auto snapshot=make_render_scene_snapshot(scene);Framebuffer a(1,1),b(1,1);InteractiveFrameState state;
    for(int variant=0;variant<6;++variant) {
        if(variant==1) {
            auto& m=snapshot.instances[0].materials[0];m.alpha_mode=AlphaMode::Mask;m.opacity=0;
            ++snapshot.revisions.materials;
        }
        if(variant==2) {
            auto& m=snapshot.instances[0].materials[0];m.alpha_mode=AlphaMode::Opaque;m.opacity=1;
            ++snapshot.revisions.materials;
        }
        if(variant==3) {
            auto& instance=snapshot.instances[0];instance.object_to_world(0,0)=-1;
            instance.world_to_object=instance.object_to_world.inverse();
            instance.normal_to_world=instance.world_to_object.topLeftCorner<3,3>().transpose();
            ++snapshot.revisions.transforms;
        }
        if(variant==4) {
            auto instance=snapshot.instances[0];instance.object_id+=100;
            instance.materials[0].two_sided=true;translate(instance,Vec3(.1f,0,-.2f));
            snapshot.instances.push_back(instance);++snapshot.revisions.topology;++snapshot.revisions.geometry;
        }
        if(variant==5) {
            auto local=std::make_shared<Scene>(*snapshot.assets[0].local_scene);local->triangles.clear();
            snapshot.assets[0].local_scene=local;snapshot.assets[0].triangle_material_slots.clear();
            ++snapshot.assets[0].geometry_revision;++snapshot.revisions.geometry;
        }
        hardware.render_next_frame(snapshot,front_camera(),settings,state,a);
        if(variant==0 && !hardware.statistics().realtime.hardware_ray_tracing_active)
            RENDER_SKIP(hardware.statistics().realtime.hardware_ray_tracing_detail);
        RENDER_CHECK(hardware.statistics().realtime.hardware_ray_tracing_active);
        auto reference=settings;reference.realtime.hardware_ray_tracing=false;
        software.render_next_frame(snapshot,front_camera(),reference,state,b);
        check_finite(a);double error=0;
        for(int y=0;y<37;++y)for(int x=0;x<37;++x)error+=(a.pixel(x,y)-b.pixel(x,y)).squaredNorm();
        std::cout<<"hardware material variant="<<variant<<" mse="<<error/(37*37*3)<<'\n';
        RENDER_CHECK(error/(37*37*3)<1e-6);
    }
}

RENDER_TEST(test_realtime_full_resolution_texture_reconstruction) {
    const auto context=device();Scene scene;Material m;m.type=MaterialType::Pbr;
    m.base_color=Color::Ones();m.base_color_texture_id=0;m.roughness=1;scene.materials.push_back(m);
    scene.directional_lights.push_back(DirectionalLight{Vec3(0,0,-1),Color::Ones()});
    std::vector<Color> pixels;
    for(int y=0;y<128;++y)for(int x=0;x<128;++x)pixels.push_back((x/2+y/2)%2?Color(.8f,.4f,.1f):Color(.1f,.3f,.7f));
    scene.textures.emplace_back(128,128,std::move(pixels));
    TriangleVertex v[4];
    v[0].position=Vec3(-1.3f,-1.3f,-3);v[0].uv=Vec2(0,0);
    v[1].position=Vec3(1.3f,-1.3f,-3);v[1].uv=Vec2(1,0);
    v[2].position=Vec3(1.3f,1.3f,-3);v[2].uv=Vec2(1,1);
    v[3].position=Vec3(-1.3f,1.3f,-3);v[3].uv=Vec2(0,1);
    scene.triangles.emplace_back(v[0],v[1],v[2],0);scene.triangles.emplace_back(v[0],v[2],v[3],0);
    const auto snapshot=make_render_scene_snapshot(scene);auto settings=small_settings(64);
    settings.realtime.max_bounces=1;settings.realtime.taa=false;settings.realtime.temporal_upscale=false;
    Framebuffer reference(1,1),low(1,1),full(1,1);InteractiveFrameState state;
    CudaRealtimeRenderer native(context),old(context),detail(context);
    native.render_next_frame(snapshot,front_camera(),settings,state,reference);
    settings.realtime.internal_scale=.5f;settings.realtime.full_resolution_materials=false;
    old.render_next_frame(snapshot,front_camera(),settings,state,low);
    settings.realtime.full_resolution_materials=true;
    detail.render_next_frame(snapshot,front_camera(),settings,state,full);
    double old_error=0,new_error=0;
    for(int y=4;y<60;++y)for(int x=4;x<60;++x) {
        old_error+=(low.pixel(x,y)-reference.pixel(x,y)).squaredNorm();
        new_error+=(full.pixel(x,y)-reference.pixel(x,y)).squaredNorm();
    }
    std::cout<<"material texture reconstruction: bilinear="<<old_error/(56*56*3)<<" full="<<new_error/(56*56*3)<<'\n';
    RENDER_CHECK(old_error>1e-3 && new_error<old_error*.1);
}

RENDER_TEST(test_realtime_shader_reordering_preserves_paths) {
    const auto context=device();const auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    auto settings=small_settings(48);settings.realtime.debug_view=RealtimeDebugView::Raw;
    settings.realtime.samples_per_pixel=4;
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    CudaRealtimeRenderer ordinary(context),reordered(context);Framebuffer a(1,1),b(1,1);InteractiveFrameState state;
    for(int frame=0;frame<8;++frame) {
        settings.realtime.shader_execution_reordering=false;ordinary.render_next_frame(snapshot,camera,settings,state,a);
        if(!ordinary.statistics().realtime.hardware_ray_tracing_active)RENDER_SKIP(ordinary.statistics().realtime.hardware_ray_tracing_detail);
        settings.realtime.shader_execution_reordering=true;reordered.render_next_frame(snapshot,camera,settings,state,b);check_finite(b);
        double error=0;for(int y=0;y<48;++y)for(int x=0;x<48;++x)error+=(a.pixel(x,y)-b.pixel(x,y)).squaredNorm();
        RENDER_CHECK(error/(48*48*3)<1e-7);
    }
}

RENDER_TEST(test_realtime_san_miguel_quality) {
    if(!std::getenv("RTRT_SAN_MIGUEL_QUALITY"))RENDER_SKIP("set RTRT_SAN_MIGUEL_QUALITY=1 for complex-scene reference validation");
    const auto context=device();auto settings=small_settings();
    const int width=std::getenv("RTRT_QUALITY_WIDTH")?std::clamp(std::atoi(std::getenv("RTRT_QUALITY_WIDTH")),320,1920):320;
    const int height=width*9/16;settings.width=width;settings.height=height;
    auto document=SceneDocument::load("benchmarks/cases/san_miguel_first_scene.rscene",width,height);
    const auto snapshot=document.render_scene_snapshot();
    std::ifstream input("benchmarks/cases/san_miguel_first_scene.json");nlohmann::json preset;input>>preset;
    const auto& c=preset.at("camera");
    auto vector=[](const nlohmann::json& j){return Vec3(j[0].get<float>(),j[1].get<float>(),j[2].get<float>());};
    const Vec3 eye=vector(c.at("eye")),forward=vector(c.at("forward")),up=vector(c.at("up"));
    const Camera camera(eye,eye+forward,up,c.at("vertical_fov_degrees").get<float>(),float(width)/height);
    settings.path.max_bounces=8;settings.path.samples_per_pixel=1024;
    Image reference(width,height);
    if(const char* path=std::getenv("RTRT_REFERENCE_PFM")) {
        std::ifstream hdr(path,std::ios::binary);std::string magic;int w=0,h=0;float endian=0;
        hdr>>magic>>w>>h>>endian;hdr.get();RENDER_CHECK(magic=="PF" && w==width && h==height && endian==-1);
        for(int y=height-1;y>=0;--y)for(int x=0;x<width;++x) {
            Color pixel;hdr.read(reinterpret_cast<char*>(pixel.data()),3*sizeof(float));reference.set_pixel(x,y,pixel);
        }
        RENDER_CHECK(bool(hdr));
    } else reference=render_cuda_path(snapshot,camera,settings).image;
    Framebuffer ref(width,height);for(int y=0;y<height;++y)for(int x=0;x<width;++x)ref.set_pixel(x,y,reference.pixel(x,y));
    export_frame("san-miguel-reference.png",ref);
    nlohmann::json results=nlohmann::json::array();
    const int first_variant=std::getenv("RTRT_QUALITY_VARIANT")?std::clamp(std::atoi(std::getenv("RTRT_QUALITY_VARIANT")),0,7):0;
    const int last_variant=std::getenv("RTRT_QUALITY_VARIANT")?first_variant+1:8;
    for(int variant=first_variant;variant<last_variant;++variant) {
        const char* name=variant==0?"san-miguel-native.png":variant==1?"san-miguel-bilinear.png":variant==2?"san-miguel-material6.png":variant==3?"san-miguel-material8.png":variant==4?"san-miguel-quality60.png":variant==5?"san-miguel-long-taa.png":variant==6?"san-miguel-soft-normal.png":"san-miguel-final.png";
        settings.realtime=RealtimeRenderSettings{};
        settings.realtime.internal_scale=variant==0?1:(variant>=4?.45f:.5f);
        settings.realtime.full_resolution_materials=variant>=2;settings.realtime.max_bounces=variant==2?6:8;
        if(variant>=5){settings.realtime.taa_current_weight=.05f;settings.realtime.taa_clip_sigma=2.5f;}
        if(variant==6){settings.realtime.normal_power=16;settings.realtime.diffuse_iterations=3;}
        if(variant==7)settings.realtime=realtime_1080p_quality_settings();
        CudaRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        double flicker=0,blocks=0;
        for(int f=0;f<96;++f) {
            renderer.render_next_frame(snapshot,camera,settings,state,frame);check_finite(frame);
            if(f>=80) {
                blocks+=block_difference(frame,previous);
                for(int y=0;y<height;++y)for(int x=0;x<width;++x)flicker+=(frame.pixel(x,y)-previous.pixel(x,y)).squaredNorm()/(3*width*height);
            }
            previous=frame;
        }
        double display_error=0,energy=0,ref_energy=0;DisplaySettings display;display.tone_mapper=ToneMapper::Aces;
        for(int y=0;y<height;++y)for(int x=0;x<width;++x) {
            display_error+=(apply_display_transform(frame.pixel(x,y),display)-apply_display_transform(ref.pixel(x,y),display)).squaredNorm()/(3*width*height);
            energy+=frame.pixel(x,y).sum();ref_energy+=ref.pixel(x,y).sum();
        }
        export_frame(name,frame);
        results.push_back({{"name",name},{"settings",realtime_settings_json(settings.realtime)},{"hdr_mse",mse(frame,reference)},{"display_mse",display_error},
            {"energy_ratio",energy/ref_energy},{"static_flicker_mse",flicker/16},{"block_flicker_mse",blocks/16}});
        std::cout<<results.back().dump()<<'\n';
    }
    export_metrics("san-miguel-quality.json",{{"width",width},{"height",height},{"reference_spp",1024},{"results",results}});
}

RENDER_TEST(test_realtime_glass_split_preserves_fresnel_energy) {
    const auto context=device();auto scene=plane_scene();scene.environment=Color(.02f,.03f,.04f);
    scene.materials[0].type=MaterialType::Dielectric;scene.materials[0].ior=1.5f;
    Material light;light.type=MaterialType::Emissive;light.emission=Color(40,20,10);scene.materials.push_back(light);
    scene.triangles.emplace_back(Vec3(-8,-8,1),Vec3(8,8,1),Vec3(8,-8,1),1);
    scene.triangles.emplace_back(Vec3(-8,-8,1),Vec3(-8,8,1),Vec3(8,8,1),1);
    const auto snapshot=make_render_scene_snapshot(scene);auto settings=small_settings(32);
    settings.realtime.denoise=false;settings.realtime.taa=false;settings.realtime.temporal_upscale=false;
    settings.realtime.debug_view=RealtimeDebugView::Raw;settings.realtime.max_bounces=4;
    InteractiveFrameState state;double errors[2]{};
    for(int split=0;split<2;++split) {
        settings.realtime.split_dielectric=split!=0;CudaRealtimeRenderer renderer(context);Framebuffer frame(1,1);
        for(int f=0;f<16;++f) {
            renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);
            for(int y=4;y<28;++y)for(int x=4;x<28;++x) {
                const float dx=(2*(x+.5f)/32-1)*std::tan(22.5f*3.14159265358979323846f/180);
                const float dy=(2*(y+.5f)/32-1)*std::tan(22.5f*3.14159265358979323846f/180);
                const float cosine=1/std::sqrt(1+dx*dx+dy*dy);
                const float fresnel=.04f+.96f*std::pow(1-cosine,5);
                const Color expected=fresnel*light.emission+(1-fresnel)*scene.environment;
                errors[split]+=(frame.pixel(x,y)-expected).squaredNorm()/(16*24*24*3);
            }
        }
    }
    std::cout<<"glass Fresnel MSE: stochastic="<<errors[0]<<" split="<<errors[1]<<'\n';
    RENDER_CHECK(errors[0]>1 && errors[1]<1e-7);
    export_metrics("glass-fresnel.json",{{"stochastic_mse",errors[0]},{"split_mse",errors[1]}});

    // Emission is attached to the visible interface, so branch splitting must
    // count it once. Fully transparent alpha coverage must not multiply paths.
    scene.materials[0].emission=Color(1,2,3);
    CudaRealtimeRenderer emitting(context);Framebuffer lit(1,1);
    emitting.render_next_frame(make_render_scene_snapshot(scene),front_camera(),settings,state,lit);
    const Color expected=.04f*light.emission+.96f*scene.environment+scene.materials[0].emission;
    RENDER_CHECK((lit.pixel(16,16)-expected).norm()<1e-4f);
    scene.materials[0].alpha_mode=AlphaMode::Blend;scene.materials[0].opacity=0;
    CudaRealtimeRenderer transparent(context);
    transparent.render_next_frame(make_render_scene_snapshot(scene),front_camera(),settings,state,lit);
    RENDER_CHECK((lit.pixel(16,16)-scene.environment).norm()<1e-5f);
}

namespace {
Scene specular_firefly_scene(int material_case) {
    Scene scene;scene.environment=Color(.015f,.02f,.025f);
    Material ball;ball.type=material_case==2?MaterialType::Dielectric:MaterialType::Metal;
    ball.base_color=Color(.98f,.95f,.9f);ball.roughness=material_case==1?.18f:.02f;ball.ior=1.5f;
    scene.materials.push_back(ball);scene.spheres.emplace_back(Vec3(0,0,-3),.9f,0);
    std::vector<Color> pixels;
    for(int y=0;y<128;++y)for(int x=0;x<128;++x)
        pixels.push_back((x/8+y/8)%2?Color::Ones():Color(.002f,.003f,.004f));
    scene.textures.emplace_back(128,128,std::move(pixels));
    Material back;back.type=MaterialType::Emissive;back.emission=Color(12,9,6);back.emissive_texture_id=0;
    scene.materials.push_back(back);
    TriangleVertex v[4];
    v[0].position=Vec3(-6,-6,-6);v[0].uv=Vec2(0,0);
    v[1].position=Vec3(6,-6,-6);v[1].uv=Vec2(1,0);
    v[2].position=Vec3(6,6,-6);v[2].uv=Vec2(1,1);
    v[3].position=Vec3(-6,6,-6);v[3].uv=Vec2(0,1);
    scene.triangles.emplace_back(v[0],v[1],v[2],1);scene.triangles.emplace_back(v[0],v[2],v[3],1);
    Material strip;strip.type=MaterialType::Emissive;strip.emission=Color(80,60,40);scene.materials.push_back(strip);
    // A bright strip behind the camera is visible only through reflection.
    scene.triangles.emplace_back(Vec3(-4,.35f,1),Vec3(4,.55f,1),Vec3(4,.35f,1),2);
    scene.triangles.emplace_back(Vec3(-4,.35f,1),Vec3(-4,.55f,1),Vec3(4,.55f,1),2);
    return scene;
}
}

RENDER_TEST(test_realtime_native_optics_dense_and_empty_worklists) {
    const auto context=device();
    for(bool hardware:{true,false}) {
    CudaRealtimeRenderer native(context),upscaled(context);
    auto settings=small_settings(513);settings.height=257;
    settings.realtime.hardware_ray_tracing=hardware;
    settings.realtime.denoise=false;settings.realtime.taa=false;settings.realtime.temporal_upscale=false;
    settings.realtime.direct_lighting=false;settings.realtime.max_bounces=2;
    InteractiveFrameState state;Framebuffer expected(1,1),actual(1,1);
    // Odd dimensions cover partial warps. More than 65536 selected pixels
    // require multiple worklist strides; clearing between masks is essential.
    for(bool mirror:{true,false,true}) {
        auto scene=plane_scene(!mirror);scene.environment=Color(.4f,.3f,.2f);
        if(mirror) {
            scene.materials[0].type=MaterialType::Metal;scene.materials[0].roughness=.02f;
            scene.materials[0].base_color=Color(.98f,.95f,.9f);
        }
        const auto snapshot=make_render_scene_snapshot(scene);
        settings.realtime.internal_scale=1;
        native.render_next_frame(snapshot,front_camera(),settings,state,expected);
        settings.realtime.internal_scale=.5f;
        upscaled.render_next_frame(snapshot,front_camera(),settings,state,actual);
        double error=0;
        for(int y=0;y<settings.height;++y)for(int x=0;x<settings.width;++x)
            error+=(actual.pixel(x,y)-expected.pixel(x,y)).squaredNorm();
        RENDER_CHECK(error/(settings.width*settings.height)<1e-10);
    }
    }
}

RENDER_TEST(test_realtime_optix_denoiser_switch_resize_and_resident_output) {
    CudaRealtimeRenderer r(device());auto s=small_settings(32);
    s.realtime.hardware_ray_tracing=false;s.realtime.denoiser=RealtimeDenoiser::Optix;
    Scene scene;scene.environment=Color(.3f,.6f,2);auto snapshot=make_render_scene_snapshot(scene);
    InteractiveFrameState state;Framebuffer f(1,1);
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    if(!r.statistics().realtime.optix_denoiser_active) {
        RENDER_CHECK(!r.statistics().realtime.optix_denoiser_detail.empty());
        if(std::getenv("RTRT_REQUIRE_OPTIX_DENOISER"))
            throw TestFailure{r.statistics().realtime.optix_denoiser_detail};
        RENDER_SKIP(r.statistics().realtime.optix_denoiser_detail);
    }
    RENDER_CHECK(!r.statistics().realtime.hardware_ray_tracing_active);
    RENDER_CHECK(r.statistics().realtime.optix_denoiser_temporal);
    const auto generation=r.statistics().allocation_generation,bytes=r.statistics().realtime.optix_denoiser_bytes;
    RENDER_CHECK(bytes>0);
    const auto downloads=r.statistics().framebuffer_downloads;
    for(int i=0;i<8;++i)r.render_next_frame_to_surface(snapshot,front_camera(),s,state,0);
    RENDER_CHECK(r.statistics().framebuffer_downloads==downloads);
    r.download_current_frame(f);check_finite(f);
    RENDER_CHECK(r.statistics().allocation_generation==generation);
    RENDER_CHECK(r.statistics().realtime.optix_denoiser_bytes==bytes);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    RENDER_CHECK((f.pixel(16,16)-scene.environment).norm()<1e-5f);
    s.realtime.debug_view=RealtimeDebugView::Filtered;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    RENDER_CHECK((f.pixel(16,16)-scene.environment).norm()<1e-5f);
    s.realtime.debug_view=RealtimeDebugView::Final;
    s.realtime.denoise=false;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(!r.statistics().realtime.optix_denoiser_active && r.statistics().realtime.optix_denoiser_bytes==0);
    s.realtime.denoiser=RealtimeDenoiser::Svgf;s.realtime.denoise=true;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(!r.statistics().realtime.optix_denoiser_active && r.statistics().realtime.history_resets==3);
    s.realtime.denoiser=RealtimeDenoiser::Optix;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(r.statistics().realtime.optix_denoiser_active && r.statistics().realtime.history_resets==4);
    s.realtime.temporal=false;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(r.statistics().realtime.optix_denoiser_active && !r.statistics().realtime.optix_denoiser_temporal);
    s.width=37;s.height=23;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.width()==37 && f.height()==23);check_finite(f);
    for(bool native:{true,false}) {
        s.realtime.temporal=true;s.realtime.internal_scale=.5f;s.realtime.full_resolution_materials=native;
        r.render_next_frame(snapshot,front_camera(),s,state,f);
        RENDER_CHECK(r.statistics().realtime.optix_denoiser_active);check_finite(f);
        RENDER_CHECK((f.pixel(18,11)-scene.environment).norm()<1e-5f);
    }
    // Returning to the exact same dimensions must still start a new sequence.
    s.realtime.internal_scale=1;s.width=s.height=32;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(r.statistics().realtime.optix_denoiser_active);
    const auto resets=r.statistics().realtime.history_resets;
    state.camera_cut=true;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(r.statistics().realtime.history_resets==resets+1);
    RENDER_CHECK((f.pixel(16,16)-scene.environment).norm()<1e-5f);
}

RENDER_TEST(test_realtime_optix_denoiser_native_glass_and_mirror) {
    const auto context=device();InteractiveFrameState state;Framebuffer frame(1,1);
    for(bool glass:{false,true})for(float scale:{1.0f,.5f}) {
        auto scene=plane_scene();scene.environment=Color(.25f,.5f,1);
        auto& material=scene.materials[0];material.type=glass?MaterialType::Dielectric:MaterialType::Metal;
        material.base_color=Color::Ones();material.roughness=.02f;material.ior=1.5f;
        const auto snapshot=make_render_scene_snapshot(scene);auto settings=small_settings(37);
        settings.realtime.denoiser=RealtimeDenoiser::Optix;settings.realtime.internal_scale=scale;
        settings.realtime.direct_lighting=false;settings.realtime.max_bounces=4;
        CudaRealtimeRenderer r(context);
        for(int i=0;i<6;++i) {
            r.render_next_frame(snapshot,front_camera(),settings,state,frame);check_finite(frame);
            if(!r.statistics().realtime.optix_denoiser_active) {
                if(std::getenv("RTRT_REQUIRE_OPTIX_DENOISER"))throw TestFailure{r.statistics().realtime.optix_denoiser_detail};
                RENDER_SKIP(r.statistics().realtime.optix_denoiser_detail);
            }
        }
        // Both branches of glass and a white mirror see the same environment.
        // This also exercises the native optical mask after the low-res AOV pass.
        RENDER_CHECK((frame.pixel(18,18)-scene.environment).norm()<.08f);
        export_frame((std::string("optix-")+(glass?"glass-":"mirror-")+(scale<1?"half.png":"native.png")).c_str(),frame);
    }
}

RENDER_TEST(test_realtime_optix_denoiser_unavailable_uses_svgf) {
    const auto context=device();auto s=small_settings(32);
    s.realtime.hardware_ray_tracing=false;s.realtime.denoiser=RealtimeDenoiser::Optix;
    auto scene=plane_scene();scene.environment=Color(.3f,.2f,.1f);
    const auto snapshot=make_render_scene_snapshot(scene);InteractiveFrameState state;Framebuffer actual(1,1),expected(1,1);
    CudaRealtimeRenderer fallback(context),svgf(context);
    auto reference=s;reference.realtime.denoiser=RealtimeDenoiser::Svgf;
    for(int i=0;i<3;++i) {
        fallback.render_next_frame(snapshot,front_camera(),s,state,actual);
        if(fallback.statistics().realtime.optix_denoiser_active)
            RENDER_SKIP("OptiX available; run this fallback check with RENDERER_OPTIX=OFF");
        svgf.render_next_frame(snapshot,front_camera(),reference,state,expected);
        RENDER_CHECK(!fallback.statistics().realtime.optix_denoiser_detail.empty());
        check_finite(actual);
        for(int y=0;y<32;++y)for(int x=0;x<32;++x)
            RENDER_CHECK((actual.pixel(x,y)-expected.pixel(x,y)).squaredNorm()<1e-12f);
    }
    RENDER_CHECK(fallback.statistics().realtime.history_resets==1);
    s.realtime.denoiser=RealtimeDenoiser::Svgf;
    fallback.render_next_frame(snapshot,front_camera(),s,state,actual);
    RENDER_CHECK(fallback.statistics().realtime.optix_denoiser_detail.empty());
}

RENDER_TEST(test_realtime_optix_denoiser_scene_changes_and_disocclusion) {
    CudaRealtimeRenderer r(device());auto s=small_settings(64);
    s.realtime.denoiser=RealtimeDenoiser::Optix;s.realtime.hardware_ray_tracing=false;
    auto scene=plane_scene(true);scene.environment=Color(0,0,.2f);
    auto snapshot=make_render_scene_snapshot(scene);InteractiveFrameState state;Framebuffer f(1,1);
    for(int i=0;i<6;++i)r.render_next_frame(snapshot,front_camera(),s,state,f);
    if(!r.statistics().realtime.optix_denoiser_active) {
        if(std::getenv("RTRT_REQUIRE_OPTIX_DENOISER"))throw TestFailure{r.statistics().realtime.optix_denoiser_detail};
        RENDER_SKIP(r.statistics().realtime.optix_denoiser_detail);
    }
    // Camera translation in both axes uses the existing previous-to-current
    // instance mapping; a pure transform must not reset the neural sequence.
    const Camera moved(Vec3(.08f,.04f,0),Vec3(.08f,.04f,-3),Vec3::UnitY(),45,1);
    r.render_next_frame(snapshot,moved,s,state,f);
    auto d=r.download_diagnostics();RENDER_CHECK(d[32*64+32].motion_x>0 && d[32*64+32].motion_y<0);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);check_finite(f);
    translate(snapshot.instances[0],Vec3(12,0,0));++snapshot.revisions.transforms;
    r.render_next_frame(snapshot,moved,s,state,f);
    RENDER_CHECK(r.statistics().realtime.history_resets==1);
    RENDER_CHECK((f.pixel(32,32)-scene.environment).norm()<1e-5f);
    snapshot.instances[0].materials[0].emission=Color(0,2,0);++snapshot.revisions.materials;
    translate(snapshot.instances[0],Vec3(-12,0,0));++snapshot.revisions.transforms;
    r.render_next_frame(snapshot,moved,s,state,f);
    RENDER_CHECK(r.statistics().realtime.history_resets==2);check_finite(f);
    RENDER_CHECK(f.pixel(32,32).y()>1.5f && f.pixel(32,32).x()<.1f);
    r.reset(snapshot,s);r.render_next_frame(snapshot,moved,s,state,f);
    RENDER_CHECK(r.statistics().realtime.history_resets==3);
}

RENDER_TEST(test_realtime_thin_mirror_highlight_preserves_energy) {
    const auto context=device();auto scene=specular_firefly_scene(0);
    tessellate_spheres(scene,64,32);
    const auto snapshot=make_render_scene_snapshot(scene);auto settings=small_settings(128);
    settings.path.samples_per_pixel=512;settings.path.max_bounces=8;
    const auto reference=render_cuda_path(snapshot,front_camera(),settings).image;
    for(bool upscale:{false,true}) {
        settings.realtime=upscale?realtime_1080p_quality_settings():RealtimeRenderSettings{};
        CudaRealtimeRenderer renderer(context);Framebuffer frame(1,1);InteractiveFrameState state;
        double error=0,energy=0,reference_energy=0;int count=0;
        for(int f=0;f<64;++f) {
            renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);
            if(f<48)continue;
            for(int y=24;y<104;++y)for(int x=24;x<104;++x) {
                if((x-63.5f)*(x-63.5f)+(y-63.5f)*(y-63.5f)>40*40)continue;
                const Color c=frame.pixel(x,y),r=reference.pixel(x,y);
                energy+=c.sum();reference_energy+=r.sum();error+=(c-r).squaredNorm();++count;
            }
        }
        const double ratio=energy/reference_energy,mean_error=error/(3*count);
        std::cout<<"thin mirror "<<(upscale?"half":"native")<<": energy="<<ratio<<" MSE="<<mean_error<<'\n';
        export_metrics(upscale?"mirror-highlight-upscale.json":"mirror-highlight.json",
            {{"energy_ratio",ratio},{"hdr_mse",mean_error},{"hardware_rt",renderer.statistics().realtime.hardware_ray_tracing_active}});
        RENDER_CHECK(ratio>.8 && ratio<1.2);
        RENDER_CHECK(mean_error<3);
    }
}

RENDER_TEST(test_realtime_specular_firefly_stress) {
    if(!std::getenv("RTRT_FIREFLY_STRESS"))RENDER_SKIP("set RTRT_FIREFLY_STRESS=1 for mirror/gloss/glass HDR validation");
    const auto context=device();auto settings=small_settings(128);
    settings.path.samples_per_pixel=2048;settings.path.max_bounces=8;
    const Camera camera=front_camera();nlohmann::json results=nlohmann::json::array();
    for(int material_case=0;material_case<3;++material_case) {
        if(const char* selected=std::getenv("RTRT_FIREFLY_MATERIAL"))if(material_case!=std::atoi(selected))continue;
        const std::string label=material_case==0?"mirror":material_case==1?"glossy":"glass";
        Scene scene=specular_firefly_scene(material_case);
        if(std::getenv("RTRT_FIREFLY_MESH"))tessellate_spheres(scene,64,32);
        const auto snapshot=make_render_scene_snapshot(scene);
        const auto reference=render_cuda_path(snapshot,camera,settings).image;
        Framebuffer ref(128,128);for(int y=0;y<128;++y)for(int x=0;x<128;++x)ref.set_pixel(x,y,reference.pixel(x,y));
        export_frame(("firefly-"+label+"-reference.png").c_str(),ref);
        for(int profile=0;profile<2;++profile) {
            settings.realtime=profile?realtime_1080p_quality_settings():RealtimeRenderSettings{};
            if(std::getenv("RTRT_FIREFLY_RAW")) {
                settings.realtime.denoise=false;
                settings.realtime.debug_view=RealtimeDebugView::Raw;
            }
            if(std::getenv("RTRT_FIREFLY_NO_DENOISE"))settings.realtime.denoise=false;
            if(std::getenv("RTRT_FIREFLY_NO_TEMPORAL"))settings.realtime.temporal=false;
            if(const char* view=std::getenv("RTRT_FIREFLY_VIEW"))
                settings.realtime.debug_view=static_cast<RealtimeDebugView>(std::atoi(view));
            CudaRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
            double error=0,flicker=0,energy=0,reference_energy=0,excess=0;std::uint64_t spikes=0,pixels_measured=0;
            for(int f=0;f<96;++f) {
                renderer.render_next_frame(snapshot,camera,settings,state,frame);check_finite(frame);
                if(f>=64)for(int y=0;y<128;++y)for(int x=0;x<128;++x) {
                    if((x-63.5f)*(x-63.5f)+(y-63.5f)*(y-63.5f)>40*40)continue;
                    const Color c=frame.pixel(x,y),r=reference.pixel(x,y);
                    const float l=c.dot(Color(.2126f,.7152f,.0722f)),rl=r.dot(Color(.2126f,.7152f,.0722f));
                    const float tail=std::max(0.0f,l-(4*rl+2));
                    spikes+=tail>0;excess+=tail;error+=(c-r).squaredNorm();
                    flicker+=(c-previous.pixel(x,y)).squaredNorm();energy+=l;reference_energy+=rl;++pixels_measured;
                }
                previous=frame;
            }
            export_frame(("firefly-"+label+(profile?"-half.png":"-native.png")).c_str(),frame);
            results.push_back({{"material",label},{"profile",profile?"half":"native"},{"samples",pixels_measured},
                {"hardware_rt",renderer.statistics().realtime.hardware_ray_tracing_active},
                {"hdr_mse",error/(3*pixels_measured)},{"flicker_mse",flicker/(3*pixels_measured)},
                {"energy_ratio",energy/reference_energy},{"spike_count",spikes},{"excess_luminance",excess/pixels_measured}});
            std::cout<<results.back().dump()<<'\n';
        }
    }
    export_metrics("specular-firefly.json",{{"reference_spp",2048},{"results",results}});
}
