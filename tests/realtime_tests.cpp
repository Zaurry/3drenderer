#include "test_framework.h"
#include "render/optix/optix_realtime_renderer.h"
#include "render/realtime/realtime_settings_json.h"
#include "render/realtime/realtime_defaults.h"
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
    if (!optix_realtime_available(0, &reason)) {
        if(std::getenv("RTRT_REQUIRE_OPTIX"))throw TestFailure{reason};
        RENDER_SKIP(reason);
    }
    return CudaDeviceContext::create(0);
}
Camera front_camera(float x=0) { return Camera(Vec3(x,0,0),Vec3(x,0,-3),Vec3::UnitY(),45,1); }
RenderSettings small_settings(int size=32) {
    RenderSettings s; s.width=s.height=size; s.path.cuda_device=0;
    // Algorithm regressions intentionally exercise the one-SPP SVGF pipeline,
    // independent of the viewer's preferred quality/denoiser policy.
    s.realtime.samples_per_pixel=1;s.realtime.denoiser=RealtimeDenoiser::Svgf;
    s.realtime.diffuse_history=32;s.realtime.sharpening=0;
    return s;
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
void require_neural_denoiser(const OptixRealtimeRenderer& renderer) {
    if(renderer.statistics().realtime.optix_denoiser_active)return;
    const auto& reason=renderer.statistics().realtime.optix_denoiser_detail;
    if(std::getenv("RTRT_REQUIRE_OPTIX_DENOISER"))throw TestFailure{reason};
    RENDER_SKIP(reason);
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
    s.low_discrepancy=false;s.full_resolution_materials=false;s.shader_execution_reordering=false;
    s.split_dielectric=false;s.specular_antialiasing=false;s.regularize_indirect=false;
    s.denoiser=RealtimeDenoiser::Optix;
    RENDER_CHECK(parse_realtime_settings(realtime_settings_json(s))==s);
    RENDER_CHECK(realtime_settings_json(s).at("denoiser")=="optix");
    RENDER_CHECK(parse_realtime_settings({})==RealtimeRenderSettings{});
    for(const auto& value:{nlohmann::json("future"),nlohmann::json(999),nlohmann::json(nullptr)})
        RENDER_CHECK(parse_realtime_settings({{"denoiser",value}}).denoiser==RealtimeRenderSettings{}.denoiser);
    RENDER_CHECK(parse_realtime_settings({{"denoiser","svgf"}}).denoiser==RealtimeDenoiser::Svgf);
    s.denoiser=static_cast<RealtimeDenoiser>(255);
    RENDER_CHECK(sanitize_realtime_settings(s).denoiser==RealtimeRenderSettings{}.denoiser);
    s.internal_scale=std::numeric_limits<float>::quiet_NaN(); s.normal_threshold=std::numeric_limits<float>::infinity();
    s.max_bounces=-20; s.diffuse_iterations=900;
    auto safe=sanitize_realtime_settings(s);
    RENDER_CHECK(safe.internal_scale==1 && safe.normal_threshold==.85f);
    RENDER_CHECK(safe.max_bounces==1 && safe.diffuse_iterations==5);
    const auto parsed=parse_realtime_settings({{"max_bounces",1e99},{"taa","invalid"},{"internal_scale",nullptr}});
    RENDER_CHECK(parsed.max_bounces==64 && parsed.taa && parsed.internal_scale==1);
}

RENDER_TEST(test_realtime_section_resets_are_independent_and_persist) {
    const RealtimeRenderSettings defaults;
    auto s=defaults;
    s.samples_per_pixel=9;s.reflections=false;s.regularize_indirect=false;
    s.denoise=false;s.firefly_filter=false;s.denoiser=RealtimeDenoiser::Svgf;s.diffuse_history=7;
    s.internal_scale=.5f;s.taa=false;s.sharpening=.7f;
    s.debug_view=RealtimeDebugView::Normal;
    reset_realtime_settings(s,RealtimeSettingsSection::Denoising);
    RENDER_CHECK(s.denoise && s.firefly_filter && s.denoiser==defaults.denoiser);
    RENDER_CHECK(s.diffuse_history==defaults.diffuse_history);
    RENDER_CHECK(s.samples_per_pixel==9 && !s.reflections && !s.regularize_indirect);
    RENDER_CHECK(s.internal_scale==.5f && !s.taa && s.sharpening==.7f);
    RENDER_CHECK(s.debug_view==RealtimeDebugView::Normal);
    reset_realtime_settings(s,RealtimeSettingsSection::Lighting);
    RENDER_CHECK(s.samples_per_pixel==defaults.samples_per_pixel && s.reflections && s.regularize_indirect);
    RENDER_CHECK(s.internal_scale==.5f && s.debug_view==RealtimeDebugView::Normal);
    reset_realtime_settings(s,RealtimeSettingsSection::Reconstruction);
    RENDER_CHECK(s.internal_scale==defaults.internal_scale && s.taa && s.sharpening==defaults.sharpening);
    RENDER_CHECK(s.debug_view==RealtimeDebugView::Normal);
    reset_realtime_settings(s,RealtimeSettingsSection::Diagnostics);
    RENDER_CHECK(s==defaults);
    RENDER_CHECK(parse_realtime_settings(realtime_settings_json(s))==defaults);
    // Resetting a section again must be harmless.
    reset_realtime_settings(s,RealtimeSettingsSection::Denoising);
    RENDER_CHECK(s==defaults);
}

RENDER_TEST(test_realtime_constant_preservation_resize_and_resident_output) {
    OptixRealtimeRenderer r(device()); auto s=small_settings();
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
    OptixRealtimeRenderer r(device()); auto s=small_settings();
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

RENDER_TEST(test_realtime_spatial_filter_preserves_constant_at_partial_tiles) {
    const auto context=device();auto scene=plane_scene();scene.materials[0].specular_factor=0;
    // Also exercise the footprint filter's halo and inactive lanes at borders.
    scene.materials[0].normal_texture_id=0;
    scene.textures.emplace_back(1,1,std::vector<Color>{Color(.5f,.5f,1)});
    scene.directional_lights.push_back(DirectionalLight{Vec3(0,0,-1),Color::Constant(3)});
    const auto snapshot=make_render_scene_snapshot(scene);
    const Color expected=scene.materials[0].base_color*(3.0f/3.14159265358979323846f);
    // The light has zero angular radius, but soft_shadows remains enabled so
    // direct light goes through the spatial filter instead of its bypass.
    for(const auto size:{std::pair{1,1},std::pair{17,13},std::pair{33,9}})for(int passes:{1,2,5}) {
        auto settings=small_settings(size.first);settings.height=size.second;
        settings.realtime.taa=false;settings.realtime.temporal_upscale=false;
        settings.realtime.diffuse_iterations=passes;settings.realtime.specular_iterations=passes;
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1);InteractiveFrameState state;
        for(int sample=0;sample<3;++sample) {
            renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);check_finite(frame);
            for(int y=0;y<frame.height();++y)for(int x=0;x<frame.width();++x)
                RENDER_CHECK((frame.pixel(x,y)-expected).norm()<2e-5f);
        }
    }
}

RENDER_TEST(test_realtime_disocclusion_rejects_old_surface) {
    OptixRealtimeRenderer r(device()); auto s=small_settings();
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
        OptixRealtimeRenderer r(context);Framebuffer shadow(1,1),lit(1,1);InteractiveFrameState state;
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
    OptixRealtimeRenderer r(device());auto s=small_settings();s.realtime.max_bounces=1;s.realtime.taa=false;
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
    auto context=device(); OptixRealtimeRenderer r(context); auto s=small_settings(64);
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
    auto snapshot=make_render_scene_snapshot(scene);OptixRealtimeRenderer r(context);Framebuffer f(1,1);InteractiveFrameState state;
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.pixel(16,16).x()>.8f);export_frame("offscreen-mirror.png",f);
    s.realtime.reflections=false;r.render_next_frame(snapshot,front_camera(),s,state,f);
    RENDER_CHECK(f.pixel(16,16).sum()<1e-4f);
    s.realtime.reflections=true;
    scene=Scene{};scene.environment=Color(.3f,.6f,.9f);
    Material glass;glass.type=MaterialType::Dielectric;scene.materials.push_back(glass);
    scene.spheres.emplace_back(Vec3(0,0,-2),.5f,0);snapshot=make_render_scene_snapshot(scene);
    r.render_next_frame(snapshot,front_camera(),s,state,f);check_finite(f);
    export_frame("glass.png",f);
    // A lossless dielectric in a uniform environment must preserve radiance
    // across the complete silhouette, including grazing refraction paths.
    double glass_error=0;
    for(int y=0;y<f.height();++y)for(int x=0;x<f.width();++x)
        glass_error=std::max(glass_error,double((f.pixel(x,y)-scene.environment).norm()));
    std::cout<<"Uniform glass max error: "<<glass_error<<'\n';
    RENDER_CHECK(glass_error<.04);
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
    OptixRealtimeRenderer r(device()); auto s=small_settings(64);
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
        OptixRealtimeRenderer r(context); auto s=small_settings();
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
    for(auto denoiser:{RealtimeDenoiser::Svgf,RealtimeDenoiser::Optix}) {
    for(float roughness:{.45f,.02f}) {
    for(bool upscale:{false,true}) {
    OptixRealtimeRenderer renderer(device());auto settings=small_settings(64);
    if(upscale)settings.realtime=realtime_1080p_quality_settings();
    settings.realtime.denoiser=denoiser;
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
    if(denoiser==RealtimeDenoiser::Optix)require_neural_denoiser(renderer);
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
}

RENDER_TEST(test_realtime_optix_denoiser_quality_against_offline_reference) {
    const auto context=device();auto s=small_settings(96);
    s.path.max_bounces=8;s.path.samples_per_pixel=512;
    const auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    // Keep the output TAA off so this verifies the neural denoiser itself.
    s.realtime.taa=false;s.realtime.temporal_upscale=false;
    s.realtime.denoiser=RealtimeDenoiser::Optix;
    OptixRealtimeRenderer probe(context);Framebuffer frame(1,1);InteractiveFrameState state;
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
        OptixRealtimeRenderer raw_renderer(context);auto raw_settings=s;raw_settings.realtime.denoise=false;
        Framebuffer raw(1,1);raw_renderer.render_next_frame(snapshot,camera,raw_settings,state,raw);
        // Compare to the same resolution/material reconstruction, not a native
        // input when evaluating a lower-resolution denoising configuration.
        export_frame(("optix-input-"+std::to_string(variant)+".png").c_str(),raw);
        OptixRealtimeRenderer r(context);Framebuffer previous(1,1);double flicker=0,error=0,raw_error=0;
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
    OptixRealtimeRenderer r(context);Framebuffer raw(1,1),filtered(1,1),native(1,1);InteractiveFrameState state;
    s.realtime.debug_view=RealtimeDebugView::Raw;
    r.render_next_frame(snapshot,camera,s,state,raw);check_finite(raw);export_frame("cornell-raw-1spp.png",raw);
    s.realtime.debug_view=RealtimeDebugView::Final;
    OptixRealtimeRenderer raw_renderer(context);auto raw_settings=s;raw_settings.realtime.debug_view=RealtimeDebugView::Raw;
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
        OptixRealtimeRenderer renderer(context);
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

RENDER_TEST(test_realtime_terminal_vertex_preserves_direct_light_energy) {
    const auto context=device();auto settings=small_settings(32);
    settings.realtime.denoise=false;settings.realtime.taa=false;settings.realtime.temporal_upscale=false;
    settings.realtime.debug_view=RealtimeDebugView::Raw;settings.realtime.samples_per_pixel=32;
    settings.path.max_bounces=4;settings.path.samples_per_pixel=4096;
    nlohmann::json metrics=nlohmann::json::array();double largest_error=0;
    for(bool area:{false,true}) {
        auto scene=plane_scene();scene.materials[0].specular_factor=0;
        scene.environment=area?Color::Zero().eval():Color(.6f,.8f,1.2f);
        if(area) {
            Material light;light.type=MaterialType::Emissive;light.emission=Color(2,3,4);
            scene.materials.push_back(light);
            // The light is behind the camera and subtends a broad solid angle.
            scene.triangles.emplace_back(Vec3(-6,-6,1),Vec3(6,6,1),Vec3(6,-6,1),1);
            scene.triangles.emplace_back(Vec3(-6,-6,1),Vec3(-6,6,1),Vec3(6,6,1),1);
        }
        const auto snapshot=make_render_scene_snapshot(scene);
        const auto reference=render_cuda_path(snapshot,front_camera(),settings).image;
        double reference_energy=0;
        for(int y=0;y<32;++y)for(int x=0;x<32;++x)
            reference_energy+=area?reference.pixel(x,y).sum():scene.materials[0].base_color.cwiseProduct(scene.environment).sum();
        for(int depth:{1,4})for(int light_samples:{1,4}) {
            settings.realtime.max_bounces=depth;settings.realtime.light_samples=light_samples;
            OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),average(32,32);InteractiveFrameState state;
            for(int sample=0;sample<16;++sample) {
                renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);check_finite(frame);
                for(int y=0;y<32;++y)for(int x=0;x<32;++x)
                    average.set_pixel(x,y,average.pixel(x,y)+frame.pixel(x,y)/16);
            }
            double energy=0;
            for(int y=0;y<32;++y)for(int x=0;x<32;++x)energy+=average.pixel(x,y).sum();
            const double ratio=energy/reference_energy;largest_error=std::max(largest_error,std::abs(ratio-1));
            metrics.push_back({{"area_light",area},{"max_bounces",depth},{"light_samples",light_samples},{"energy_ratio",ratio}});
            export_frame((std::string("terminal-")+(area?"area-":"environment-")+std::to_string(depth)+"-"+std::to_string(light_samples)+".png").c_str(),average);
        }
    }
    export_metrics("terminal-energy.json",metrics);
    std::cout<<"Terminal direct-light energy: "<<metrics.dump()<<'\n';
    RENDER_CHECK(largest_error<.02);
}

RENDER_TEST(test_realtime_hardware_material_updates_and_instancing) {
    const auto context=device();OptixRealtimeRenderer hardware(context);
    // Exercise updates across the jitter sequence; exact shared-edge coverage
    // is checked independently by optix_interop_tests.
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
        // Compare incremental updates against a fresh acceleration build at
        // the same sample index, without retaining a second traversal backend.
        OptixRealtimeRenderer rebuilt(context);
        for(int frame=0;frame<=variant;++frame)
            rebuilt.render_next_frame(snapshot,front_camera(),settings,state,b);
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
    OptixRealtimeRenderer native(context),old(context),detail(context);
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

RENDER_TEST(test_realtime_subpixel_normal_detail_is_temporally_stable) {
    const auto context=device();Scene scene;Material material;
    material.type=MaterialType::Pbr;material.base_color=Color(.7f,.5f,.3f);
    material.normal_texture_id=0;material.roughness=1;material.specular_factor=0;
    scene.materials.push_back(material);
    scene.directional_lights.push_back(DirectionalLight{Vec3(-.7f,0,-1).normalized(),Color::Constant(3)});
    std::vector<Color> normals;
    for(int y=0;y<256;++y)for(int x=0;x<256;++x)
        normals.push_back(Color((x/2+y/3)%2?.9f:.1f,.5f,.8f));
    scene.textures.emplace_back(256,256,std::move(normals));
    TriangleVertex vertices[4];
    for(int i=0;i<4;++i) {
        const int x=i==1 || i==2,y=i>=2;
        vertices[i].position=Vec3(x?1.3f:-1.3f,y?1.3f:-1.3f,-3);
        vertices[i].uv=Vec2(float(x),float(y));
    }
    scene.triangles.emplace_back(vertices[0],vertices[1],vertices[2],0);
    scene.triangles.emplace_back(vertices[0],vertices[2],vertices[3],0);
    const auto snapshot=make_render_scene_snapshot(scene);
    nlohmann::json metrics=nlohmann::json::array();double worst=0;
    for(auto denoiser:{RealtimeDenoiser::Svgf,RealtimeDenoiser::Optix})for(float scale:{1.0f,.5f}) {
        OptixRealtimeRenderer renderer(context);auto settings=small_settings(96);
        settings.realtime.denoiser=denoiser;settings.realtime.internal_scale=scale;
        settings.realtime.max_bounces=1;settings.realtime.soft_shadows=false;
        Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        double flicker=0;const std::string label=std::string(denoiser==RealtimeDenoiser::Svgf?"svgf":"optix")+(scale==1?"-native":"-half");
        for(int sample=0;sample<96;++sample) {
            renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);check_finite(frame);
            if(denoiser==RealtimeDenoiser::Optix)require_neural_denoiser(renderer);
            if(sample>=64) {
                for(int y=8;y<88;++y)for(int x=8;x<88;++x)
                    flicker+=(frame.pixel(x,y)-previous.pixel(x,y)).squaredNorm()/(32*3*80*80);
                if(std::getenv("RTRT_STABILITY_SEQUENCE"))
                    export_frame(("normal-"+label+"-"+std::to_string(sample)+".png").c_str(),frame);
            }
            previous=frame;
        }
        const double rms=std::sqrt(flicker);worst=std::max(worst,rms);
        metrics.push_back({{"variant",label},{"static_flicker_rmse",rms}});
        std::cout<<"Subpixel normal stability "<<label<<" RMSE="<<rms<<'\n';
    }
    export_metrics("normal-stability.json",metrics);
    RENDER_CHECK(worst<.006);
}

RENDER_TEST(test_realtime_disabled_temporal_upscale_does_not_jitter) {
    const auto context=device();auto scene=plane_scene(true);
    // An off-center silhouette gives jitter an observable, deterministic edge.
    scene.triangles.clear();
    scene.triangles.emplace_back(Vec3(-.71f,-.8f,-3),Vec3(.63f,-.8f,-3),Vec3(.63f,.87f,-3),0);
    scene.triangles.emplace_back(Vec3(-.71f,-.8f,-3),Vec3(.63f,.87f,-3),Vec3(-.71f,.87f,-3),0);
    const auto snapshot=make_render_scene_snapshot(scene);double worst=0;
    for(auto denoiser:{RealtimeDenoiser::Svgf,RealtimeDenoiser::Optix})for(bool materials:{false,true}) {
        auto settings=small_settings(63);settings.realtime.internal_scale=.5f;
        settings.realtime.denoiser=denoiser;settings.realtime.temporal=false;
        settings.realtime.taa=true;settings.realtime.temporal_upscale=false;
        settings.realtime.full_resolution_materials=materials;settings.realtime.max_bounces=1;
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        for(int sample=0;sample<12;++sample) {
            renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);
            if(denoiser==RealtimeDenoiser::Optix)require_neural_denoiser(renderer);
            if(sample>0)for(int y=0;y<63;++y)for(int x=0;x<63;++x)
                worst=std::max(worst,double((frame.pixel(x,y)-previous.pixel(x,y)).norm()));
            previous=frame;
        }
    }
    std::cout<<"Disabled temporal upscaling: maximum pixel change="<<worst<<'\n';
    RENDER_CHECK(worst<1e-6);
}

RENDER_TEST(test_realtime_denoiser_static_edges_and_camera_stop) {
    const auto context=device();const auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    const auto camera=[](float x) {return Camera(Vec3(x,.15f,1.5f),Vec3(x,.15f,-2),Vec3::UnitY(),45,1);};
    nlohmann::json metrics=nlohmann::json::array();double worst_static=0,worst_stop=0;
    for(auto denoiser:{RealtimeDenoiser::Svgf,RealtimeDenoiser::Optix})for(float scale:{1.0f,.5f}) {
        auto settings=small_settings(96);settings.realtime=RealtimeRenderSettings{};
        settings.realtime.denoiser=denoiser;settings.realtime.internal_scale=scale;
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        const std::string label=std::string(denoiser==RealtimeDenoiser::Svgf?"svgf":"optix")+(scale==1?"-native":"-half");
        double flicker=0,stopped=0,blocks=0;DisplaySettings display;display.tone_mapper=ToneMapper::Aces;
        for(int sample=0;sample<192;++sample) {
            // Static convergence, continuous camera translation, then stopping
            // without a cut. This exposes stale-history rebound and edge wobble.
            const float x=sample<96?0:(sample<128?float(sample-95)*.002f:.064f);
            renderer.render_next_frame(snapshot,camera(x),settings,state,frame);check_finite(frame);
            if(denoiser==RealtimeDenoiser::Optix)require_neural_denoiser(renderer);
            if((sample>=64 && sample<96) || sample>=160) {
                double error=0;
                for(int y=0;y<96;++y)for(int x=0;x<96;++x)
                    error+=(apply_display_transform(frame.pixel(x,y),display)-apply_display_transform(previous.pixel(x,y),display)).squaredNorm()/(3*96*96);
                if(sample<96) {flicker+=error/32;blocks+=block_difference(frame,previous)/32;}
                else stopped+=error/32;
            }
            if(std::getenv("RTRT_STABILITY_SEQUENCE") && sample>=64)
                export_frame(("edges-"+label+"-"+std::to_string(sample)+".png").c_str(),frame);
            previous=frame;
        }
        worst_static=std::max(worst_static,std::sqrt(flicker));worst_stop=std::max(worst_stop,std::sqrt(stopped));
        metrics.push_back({{"variant",label},{"display_static_rmse",std::sqrt(flicker)},
            {"display_stopped_rmse",std::sqrt(stopped)},{"hdr_block_flicker_mse",blocks}});
        std::cout<<"Camera stability "<<metrics.back().dump()<<'\n';
        RENDER_CHECK(renderer.statistics().realtime.history_resets==1);
    }
    export_metrics("camera-stability.json",metrics);
    RENDER_CHECK(worst_static<.0025 && worst_stop<.0025);
}

RENDER_TEST(test_realtime_shader_reordering_preserves_paths) {
    const auto context=device();const auto snapshot=make_render_scene_snapshot(make_cornell_box_scene());
    auto settings=small_settings(48);settings.realtime.debug_view=RealtimeDebugView::Raw;
    settings.realtime.samples_per_pixel=4;
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    OptixRealtimeRenderer ordinary(context),reordered(context);Framebuffer a(1,1),b(1,1);InteractiveFrameState state;
    for(int frame=0;frame<8;++frame) {
        settings.realtime.shader_execution_reordering=false;ordinary.render_next_frame(snapshot,camera,settings,state,a);
        if(!ordinary.statistics().realtime.hardware_ray_tracing_active)RENDER_SKIP(ordinary.statistics().realtime.hardware_ray_tracing_detail);
        settings.realtime.shader_execution_reordering=true;reordered.render_next_frame(snapshot,camera,settings,state,b);check_finite(b);
        double error=0;for(int y=0;y<48;++y)for(int x=0;x<48;++x)error+=(a.pixel(x,y)-b.pixel(x,y)).squaredNorm();
        RENDER_CHECK(error/(48*48*3)<1e-7);
    }
}

RENDER_TEST(test_realtime_wet_normal_highlight_stability) {
    const auto context=device();Scene scene;Material material;material.type=MaterialType::Pbr;
    material.base_color=Color::Constant(.025f);material.roughness=.02f;material.normal_texture_id=0;
    scene.materials.push_back(material);
    scene.directional_lights.push_back(DirectionalLight{Vec3(0,0,-1),Color::Constant(80)});
    std::vector<Color> normals;
    for(int y=0;y<256;++y)for(int x=0;x<256;++x) {
        const Vec3 n=Vec3(.07f*std::sin(x*1.7f+y*.3f),.035f*std::cos(y*1.3f-x*.2f),1).normalized();
        normals.push_back((n+Vec3::Ones())*.5f);
    }
    scene.textures.emplace_back(256,256,std::move(normals));TriangleVertex vertices[4];
    for(int i=0;i<4;++i) {
        const int x=i==1 || i==2,y=i>=2;
        vertices[i].position=Vec3(x?1.3f:-1.3f,y?1.3f:-1.3f,-3);vertices[i].uv=Vec2(float(x),float(y));
    }
    scene.triangles.emplace_back(vertices[0],vertices[1],vertices[2],0);
    scene.triangles.emplace_back(vertices[0],vertices[2],vertices[3],0);
    const auto snapshot=make_render_scene_snapshot(scene);nlohmann::json results=nlohmann::json::array();double worst=0;
    if(std::getenv("RTRT_WET_REFERENCE")) {
        auto settings=small_settings(96);settings.path.samples_per_pixel=4096;settings.path.max_bounces=2;
        const auto reference=render_cuda_path(snapshot,front_camera(),settings).image;Framebuffer ref(96,96);double energy=0;
        for(int y=0;y<96;++y)for(int x=0;x<96;++x) {
            ref.set_pixel(x,y,reference.pixel(x,y));
            if(x>=8 && x<88 && y>=8 && y<88)energy+=reference.pixel(x,y).sum()/(3*80*80);
        }
        export_frame("wet-reference.png",ref);
        std::cout<<"Wet highlight reference mean radiance "<<energy<<'\n';
        export_metrics("wet-reference.json",{{"mean_radiance",energy},{"spp",4096}});
    }
    for(auto denoiser:{RealtimeDenoiser::Svgf,RealtimeDenoiser::Optix})for(bool moving:{false,true}) {
        auto settings=small_settings(96);settings.realtime=RealtimeRenderSettings{};
        settings.realtime.denoiser=denoiser;settings.realtime.max_bounces=1;
        if(std::getenv("RTRT_DISABLE_SPECULAR_AA"))settings.realtime.specular_antialiasing=false;
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        DisplaySettings display;display.tone_mapper=ToneMapper::Aces;double error=0,energy=0;
        const std::string label=std::string(denoiser==RealtimeDenoiser::Svgf?"svgf":"optix")+(moving?"-moving":"-static");
        for(int sample=0;sample<128;++sample) {
            const float shift=moving?sample*.001f:0;
            renderer.render_next_frame(snapshot,front_camera(shift),settings,state,frame);check_finite(frame);
            if(denoiser==RealtimeDenoiser::Optix)require_neural_denoiser(renderer);
            if(sample>=64)for(int y=8;y<88;++y)for(int x=8;x<88;++x) {
                error+=(apply_display_transform(frame.pixel(x,y),display)-apply_display_transform(previous.pixel(x,y),display)).squaredNorm()/(64*3*80*80);
                energy+=frame.pixel(x,y).sum()/(64*3*80*80);
            }
            if(std::getenv("RTRT_STABILITY_SEQUENCE") && sample>=96)
                export_frame(("wet-"+label+"-"+std::to_string(sample)+".png").c_str(),frame);
            previous=frame;
        }
        worst=std::max(worst,std::sqrt(error));
        results.push_back({{"variant",label},{"display_flicker_rmse",std::sqrt(error)},{"mean_radiance",energy}});
        std::cout<<"Wet highlight "<<results.back().dump()<<'\n';
    }
    export_metrics("wet-highlight.json",results);
    RENDER_CHECK(worst<.003);
}

RENDER_TEST(test_realtime_optix_continuous_motion_diffuse_history) {
    const auto context=device();auto scene=plane_scene();scene.environment=Color(.6f,.8f,1);
    auto& material=scene.materials[0];material.type=MaterialType::Pbr;
    material.roughness=.02f;material.specular_factor=0;
    const auto snapshot=make_render_scene_snapshot(scene);nlohmann::json results=nlohmann::json::array();
    for(float scale:{1.0f,.5f}) {
        auto settings=small_settings(96);settings.realtime.denoiser=RealtimeDenoiser::Optix;
        settings.realtime.internal_scale=scale;settings.realtime.max_bounces=2;
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        double error=0,reference_error=0;const Color expected=material.base_color.cwiseProduct(scene.environment);
        for(int sample=0;sample<128;++sample) {
            renderer.render_next_frame(snapshot,front_camera(sample*.004f),settings,state,frame);check_finite(frame);
            require_neural_denoiser(renderer);
            if(sample>=64)for(int y=16;y<80;++y)for(int x=16;x<80;++x) {
                error+=(frame.pixel(x,y)-previous.pixel(x,y)).squaredNorm()/(64*3*64*64);
                reference_error+=(frame.pixel(x,y)-expected).squaredNorm()/(64*3*64*64);
            }
            previous=frame;
        }
        results.push_back({{"internal_scale",scale},{"moving_rmse",std::sqrt(error)},
            {"reference_rmse",std::sqrt(reference_error)}});
        std::cout<<"Neural diffuse motion "<<results.back().dump()<<'\n';
        RENDER_CHECK(renderer.statistics().realtime.history_resets==1);
        RENDER_CHECK(std::sqrt(error)<.001 && std::sqrt(reference_error)<.004);
    }
    export_metrics("optix-diffuse-motion.json",results);
}

RENDER_TEST(test_realtime_cave_continuous_motion) {
    if(!std::getenv("RTRT_CAVE_SEQUENCE"))RENDER_SKIP("set RTRT_CAVE_SEQUENCE=1 for the local wet cave asset");
    const auto context=device();
    const int output_width=std::getenv("RTRT_CAVE_WIDTH")?std::clamp(std::atoi(std::getenv("RTRT_CAVE_WIDTH")),320,1920):320;
    const int output_height=output_width*9/16;
    auto settings=small_settings(output_width);settings.height=output_height;
    const char* scene_path=std::getenv("RTRT_CAVE_SCENE");
    const auto document=[&] {
        if(scene_path) {
            std::ifstream input(scene_path);nlohmann::json session;input>>session;
            return SceneDocument::from_session_snapshot(session,output_width,output_height);
        }
        return SceneDocument::load("Computer Graphics Archive/sceness/cave.rscene",output_width,output_height);
    }();
    const auto snapshot=document.render_scene_snapshot();
    const Vec3 eye(4.189f,1.031f,2.073f),forward=(Vec3(2.922f,.980f,1.550f)-eye).normalized();
    const Vec3 up=Vec3::UnitY(),right=forward.cross(up).normalized();
    nlohmann::json results=nlohmann::json::array();
    const float speed=std::getenv("RTRT_CAVE_SPEED")?std::stof(std::getenv("RTRT_CAVE_SPEED")):.004f;
    RealtimeRenderSettings configured;
    if(const char* path=std::getenv("RTRT_CAVE_CONFIG")) {
        std::ifstream input(path);nlohmann::json config;input>>config;
        configured=parse_realtime_settings(config);
    }
    for(auto denoiser:{RealtimeDenoiser::Svgf,RealtimeDenoiser::Optix})for(float scale:{1.0f,.5f}) {
        const int variant=(denoiser==RealtimeDenoiser::Optix?2:0)+(scale<1?1:0);
        if(const char* selected=std::getenv("RTRT_CAVE_VARIANT");selected && std::atoi(selected)!=variant)continue;
        settings.realtime=configured;settings.realtime.denoiser=denoiser;settings.realtime.internal_scale=scale;
        if(std::getenv("RTRT_DISABLE_SPECULAR_AA"))settings.realtime.specular_antialiasing=false;
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        DisplaySettings display;display.tone_mapper=ToneMapper::Aces;double errors[2]{},counts[2]{};
        const std::string label=std::string(denoiser==RealtimeDenoiser::Svgf?"svgf":"optix")+(scale==1?"-native":"-half");
        for(int sample=0;sample<192;++sample) {
            const Vec3 p=eye+right*(sample<96?0:float(sample-95)*speed);
            const Camera camera(p,p+forward,up,75,float(output_width)/output_height);
            renderer.render_next_frame(snapshot,camera,settings,state,frame);check_finite(frame);
            if(denoiser==RealtimeDenoiser::Optix)require_neural_denoiser(renderer);
            if(sample>=64) {
                const auto diagnostics=renderer.download_diagnostics();
                const int width=renderer.statistics().internal_width,height=renderer.statistics().internal_height;
                const int phase=sample>=96?1:0;
                for(int y=8;y<output_height-8;++y)for(int x=8;x<output_width-8;++x) {
                    const auto& d=diagnostics[std::min(height-1,y*height/output_height)*width+std::min(width-1,x*width/output_width)];
                    const float px=x+d.motion_x*output_width,py=y+d.motion_y*output_height;
                    const int bx=int(std::floor(px)),by=int(std::floor(py));
                    if(bx<0 || by<0 || bx+1>=output_width || by+1>=output_height || d.depth<=0)continue;
                    const float fx=px-bx,fy=py-by;
                    const Color old=(previous.pixel(bx,by)*(1-fx)+previous.pixel(bx+1,by)*fx)*(1-fy)+
                        (previous.pixel(bx,by+1)*(1-fx)+previous.pixel(bx+1,by+1)*fx)*fy;
                    errors[phase]+=(apply_display_transform(frame.pixel(x,y),display)-apply_display_transform(old,display)).squaredNorm();
                    counts[phase]+=3;
                }
            }
            if(std::getenv("RTRT_STABILITY_SEQUENCE") && ((sample>=80 && sample<96) || sample>=160))
                export_frame(("cave-"+label+"-"+std::to_string(sample)+".png").c_str(),frame);
            if(sample==95 || sample==191)export_frame(("cave-"+label+"-"+std::to_string(sample)+".png").c_str(),frame);
            if(std::getenv("RTRT_CAVE_REFERENCE") && variant==0 && (sample==95 || sample>=176)) {
                auto reference_settings=settings;reference_settings.path.samples_per_pixel=256;reference_settings.path.max_bounces=8;
                const auto reference=render_cuda_path(snapshot,camera,reference_settings).image;Framebuffer ref(output_width,output_height);
                for(int y=0;y<output_height;++y)for(int x=0;x<output_width;++x)ref.set_pixel(x,y,reference.pixel(x,y));
                export_frame(("cave-reference-"+std::to_string(sample)+".png").c_str(),ref);
            }
            previous=frame;
        }
        results.push_back({{"variant",label},{"static_rmse",std::sqrt(errors[0]/counts[0])},
            {"moving_reprojected_rmse",std::sqrt(errors[1]/counts[1])}});
        std::cout<<"Cave continuous motion "<<results.back().dump()<<'\n';
    }
    export_metrics("cave-motion.json",results);
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
        if(const char* denoiser=std::getenv("RTRT_QUALITY_DENOISER");denoiser && std::string(denoiser)=="optix")
            settings.realtime.denoiser=RealtimeDenoiser::Optix;
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
        double flicker=0,blocks=0;
        for(int f=0;f<96;++f) {
            renderer.render_next_frame(snapshot,camera,settings,state,frame);check_finite(frame);
            if(settings.realtime.denoiser==RealtimeDenoiser::Optix)
                RENDER_CHECK(renderer.statistics().realtime.optix_denoiser_active);
            if(f>=80) {
                blocks+=block_difference(frame,previous);
                for(int y=0;y<height;++y)for(int x=0;x<width;++x)flicker+=(frame.pixel(x,y)-previous.pixel(x,y)).squaredNorm()/(3*width*height);
                if(std::getenv("RTRT_STABILITY_SEQUENCE"))
                    export_frame((std::string(name)+"-"+std::to_string(f)+".png").c_str(),frame);
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
        settings.realtime.split_dielectric=split!=0;OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1);
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
    OptixRealtimeRenderer emitting(context);Framebuffer lit(1,1);
    emitting.render_next_frame(make_render_scene_snapshot(scene),front_camera(),settings,state,lit);
    const Color expected=.04f*light.emission+.96f*scene.environment+scene.materials[0].emission;
    RENDER_CHECK((lit.pixel(16,16)-expected).norm()<1e-4f);
    scene.materials[0].alpha_mode=AlphaMode::Blend;scene.materials[0].opacity=0;
    OptixRealtimeRenderer transparent(context);
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
    for(bool reorder:{true,false}) {
    OptixRealtimeRenderer native(context),upscaled(context);
    auto settings=small_settings(513);settings.height=257;
    settings.realtime.shader_execution_reordering=reorder;
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
    OptixRealtimeRenderer r(device());auto s=small_settings(32);
    // Exercise the actual preferred profile as well as the explicit one-SPP
    // algorithm fixtures above, including fallback, resize and scene cuts.
    s.realtime=RealtimeRenderSettings{};
    Scene scene;scene.environment=Color(.3f,.6f,2);auto snapshot=make_render_scene_snapshot(scene);
    InteractiveFrameState state;Framebuffer f(1,1);
    r.render_next_frame(snapshot,front_camera(),s,state,f);
    if(!r.statistics().realtime.optix_denoiser_active) {
        RENDER_CHECK(!r.statistics().realtime.optix_denoiser_detail.empty());
        if(std::getenv("RTRT_REQUIRE_OPTIX_DENOISER"))
            throw TestFailure{r.statistics().realtime.optix_denoiser_detail};
        RENDER_SKIP(r.statistics().realtime.optix_denoiser_detail);
    }
    RENDER_CHECK(r.statistics().realtime.hardware_ray_tracing_active);
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

RENDER_TEST(test_realtime_smooth_mirror_raw_furnace) {
    auto scene=plane_scene();scene.environment=Color(.25f,.5f,1);
    auto& material=scene.materials[0];material.type=MaterialType::Metal;
    material.base_color=Color::Ones();material.roughness=.02f;
    const auto snapshot=make_render_scene_snapshot(scene);auto settings=small_settings(48);
    settings.realtime.debug_view=RealtimeDebugView::Raw;settings.realtime.direct_lighting=false;
    settings.realtime.samples_per_pixel=32;settings.realtime.max_bounces=2;
    OptixRealtimeRenderer renderer(device());Framebuffer frame(1,1);InteractiveFrameState state;double mean=0;
    for(int i=0;i<8;++i) {
        renderer.render_next_frame(snapshot,front_camera(.3f),settings,state,frame);check_finite(frame);
        for(int y=8;y<40;++y)for(int x=8;x<40;++x)mean+=frame.pixel(x,y).z()/(8*32*32);
    }
    std::cout<<"Smooth mirror raw furnace energy="<<mean<<'\n';
    RENDER_CHECK(std::abs(mean-1)<.01);
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
        OptixRealtimeRenderer r(context);
        for(int i=0;i<6;++i) {
            r.render_next_frame(snapshot,front_camera(),settings,state,frame);check_finite(frame);
            if(!r.statistics().realtime.optix_denoiser_active) {
                if(std::getenv("RTRT_REQUIRE_OPTIX_DENOISER"))throw TestFailure{r.statistics().realtime.optix_denoiser_detail};
                RENDER_SKIP(r.statistics().realtime.optix_denoiser_detail);
            }
        }
        // Both branches of glass and a white mirror see the same environment.
        // This also exercises the native optical mask after the low-res AOV pass.
        std::cout<<"Optical furnace glass="<<glass<<" scale="<<scale<<" center="<<frame.pixel(18,18).transpose()<<'\n';
        export_frame((std::string("optix-")+(glass?"glass-":"mirror-")+(scale<1?"half.png":"native.png")).c_str(),frame);
        RENDER_CHECK((frame.pixel(18,18)-scene.environment).norm()<.08f);
    }
}

RENDER_TEST(test_realtime_availability_contract) {
    std::string reason;
    const bool available=optix_realtime_available(0,&reason);
    RENDER_CHECK(available==reason.empty());
    if(std::getenv("RTRT_REQUIRE_OPTIX"))RENDER_CHECK(available);
#if !RENDERER_HAS_OPTIX
    RENDER_CHECK(!available);
    bool threw=false;
    try {OptixRealtimeRenderer renderer{CudaDeviceContext{}};}
    catch(const std::runtime_error& e){threw=std::string(e.what()).find("OptiX")!=std::string::npos;}
    RENDER_CHECK(threw);
#endif
    // Removed selectors must not restore a software or raster tracing path.
    const auto old=parse_realtime_settings({{"hardware_ray_tracing",false},{"raster_primary",true}});
    const auto saved=realtime_settings_json(old);
    RENDER_CHECK(!saved.contains("hardware_ray_tracing") && !saved.contains("raster_primary"));
}

RENDER_TEST(test_realtime_optix_empty_first_frame) {
    OptixRealtimeRenderer r(device());
    Scene scene;scene.environment=Color(.12f,.4f,.8f);
    auto settings=small_settings(17);settings.realtime.denoise=false;
    settings.realtime.taa=false;settings.realtime.temporal_upscale=false;
    Framebuffer image(1,1);InteractiveFrameState state;
    auto snapshot=make_render_scene_snapshot(scene);
    for(float scale:{1.0f,.5f}) {
        settings.realtime.internal_scale=scale;
        r.render_next_frame(snapshot,front_camera(),settings,state,image);
        for(int y=0;y<17;++y)for(int x=0;x<17;++x)
            RENDER_CHECK((image.pixel(x,y)-scene.environment).norm()<1e-5f);
    }
    snapshot.environment_background_visible=false;++snapshot.revisions.environment;
    r.render_next_frame(snapshot,front_camera(),settings,state,image);
    for(int y=0;y<17;++y)for(int x=0;x<17;++x)RENDER_CHECK(image.pixel(x,y).norm()<1e-5f);
    RENDER_CHECK(r.statistics().realtime.hardware_ray_tracing_active);
    RENDER_CHECK(r.statistics().realtime.gas_builds==0 && r.statistics().realtime.ias_builds==0);
}

RENDER_TEST(test_realtime_optix_analytic_spheres_and_incremental_acceleration) {
    const auto context=device();OptixRealtimeRenderer r(context);
    Scene scene;Material sphere;sphere.type=MaterialType::Emissive;sphere.emission=Color(1,.2f,.1f);
    Material back=sphere;back.emission=Color(.1f,.3f,1);scene.materials={sphere,back};
    scene.triangles.emplace_back(Vec3(-5,-5,-4),Vec3(5,-5,-4),Vec3(5,5,-4),1);
    scene.triangles.emplace_back(Vec3(-5,-5,-4),Vec3(5,5,-4),Vec3(-5,5,-4),1);
    auto snapshot=make_render_scene_snapshot(scene);
    auto local=std::make_shared<Scene>(*snapshot.assets[0].local_scene);
    local->spheres.emplace_back(Vec3(0,0,-2),.5f,0);
    snapshot.assets[0].local_scene=local;snapshot.assets[0].sphere_material_slots={MaterialSlot::bound(0)};
    ++snapshot.assets[0].geometry_revision;++snapshot.revisions.geometry;
    auto settings=small_settings(33);settings.realtime.denoise=false;settings.realtime.taa=false;
    settings.realtime.temporal_upscale=false;settings.realtime.debug_view=RealtimeDebugView::Raw;
    Framebuffer image(1,1);InteractiveFrameState state;
    const auto render=[&]{r.render_next_frame(snapshot,front_camera(),settings,state,image);check_finite(image);};
    render();
    RENDER_CHECK((image.pixel(16,16)-sphere.emission).norm()<1e-5f);
    RENDER_CHECK(std::abs(r.download_diagnostics()[16*33+16].depth-1.5f)<1e-5f);
    RENDER_CHECK(r.statistics().realtime.hardware_ray_tracing_active && r.statistics().realtime.gas_builds==2);
    RENDER_CHECK(r.statistics().blas_build_count==0 && r.statistics().tlas_build_count==0);
    const auto builds=r.statistics().realtime.gas_builds;
    snapshot.instances[0].materials[0].emission=Color(.3f,.7f,.4f);++snapshot.revisions.materials;
    render();
    RENDER_CHECK((image.pixel(16,16)-Color(.3f,.7f,.4f)).norm()<1e-5f);
    RENDER_CHECK(r.statistics().realtime.gas_builds==builds);
    // Mirrored and nonuniformly scaled instances keep sphere parameterization
    // and share the triangle/sphere material tables of the logical instance.
    auto& instance=snapshot.instances[0];
    instance.object_to_world(0,0)=-1.4f;instance.object_to_world(1,1)=.75f;
    instance.world_to_object=instance.object_to_world.inverse();
    instance.normal_to_world=instance.world_to_object.topLeftCorner<3,3>().transpose();
    ++snapshot.revisions.transforms;render();
    RENDER_CHECK((image.pixel(16,16)-Color(.3f,.7f,.4f)).norm()<1e-5f);
    RENDER_CHECK(r.statistics().realtime.gas_builds==builds && r.statistics().realtime.ias_updates==1);
    // Only the sphere's acceptance class changes; the triangle GAS is reused.
    instance.materials[0].opacity=0;++snapshot.revisions.materials;render();
    RENDER_CHECK((image.pixel(16,16)-back.emission).norm()<1e-5f);
    RENDER_CHECK(r.statistics().realtime.gas_builds==builds+1);
    instance.materials[0].opacity=1;++snapshot.revisions.materials;render();
    RENDER_CHECK((image.pixel(16,16)-Color(.3f,.7f,.4f)).norm()<1e-5f);
    // Empty topology still produces the environment using the same OptiX path.
    snapshot.instances.clear();++snapshot.revisions.topology;++snapshot.revisions.geometry;render();
    RENDER_CHECK((image.pixel(16,16)-snapshot.environment).norm()<1e-5f);
    RENDER_CHECK(r.statistics().realtime.hardware_ray_tracing_active);
}

RENDER_TEST(test_realtime_optix_denoiser_scene_changes_and_disocclusion) {
    OptixRealtimeRenderer r(device());auto s=small_settings(64);
    s.realtime.denoiser=RealtimeDenoiser::Optix;
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
        if(!upscale)if(const char* path=std::getenv("RTRT_QUALITY_CONFIG")) {
            std::ifstream input(path);RENDER_CHECK(input.good());
            nlohmann::json config;input>>config;
            settings.realtime=parse_realtime_settings(config);
        }
        OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1);InteractiveFrameState state;
        double error=0,energy=0,reference_energy=0;int count=0;
        // Compare settled profiles after two history lengths, including the
        // quality profile's longer stationary accumulation.
        const int warmup=std::max(48,2*settings.realtime.diffuse_history);
        for(int f=0;f<warmup+16;++f) {
            renderer.render_next_frame(snapshot,front_camera(),settings,state,frame);
            if(f<warmup)continue;
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
            OptixRealtimeRenderer renderer(context);Framebuffer frame(1,1),previous(1,1);InteractiveFrameState state;
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
