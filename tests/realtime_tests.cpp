#include "test_framework.h"
#include "render/realtime/cuda_realtime_renderer.h"
#include "render/realtime/realtime_settings_json.h"
#include "core/image.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>

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
}

RENDER_TEST(test_realtime_settings_roundtrip_and_sanitization) {
    RealtimeRenderSettings s; s.internal_scale=.5f; s.diffuse_history=48; s.specular_iterations=0;
    s.transmission=false; s.debug_view=RealtimeDebugView::Motion; s.sharpening=.3f;
    RENDER_CHECK(parse_realtime_settings(realtime_settings_json(s))==s);
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
