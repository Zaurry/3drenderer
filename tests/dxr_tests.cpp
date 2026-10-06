#include "test_framework.h"
#include "render/dxr/dxr_renderer.h"
#include "core/image.h"
#include "scene/scene_document.h"
#include "scene/environment.h"
#include <cstdlib>
#include <filesystem>
#if RENDERER_HAS_DXR
#include "platform/d3d12/d3d12_context.h"
#include "render/dxr/dxr_omm.h"
#endif
using namespace renderer;
namespace {
#if RENDERER_HAS_DXR
std::shared_ptr<D3d12Context> device() {
    try{auto context=D3d12Context::create();
        if(std::getenv("DXR_VALIDATE"))RENDER_CHECK(context->capabilities().debug_layer_active);
        if(std::getenv("DXR_GPU_VALIDATION"))RENDER_CHECK(context->capabilities().gpu_validation_active);
        return context;}
    catch(const std::exception& error){if(std::getenv("DXR_STRICT"))throw;RENDER_SKIP(error.what());}
}
RenderSettings settings(int width=64,int height=64) {
    RenderSettings result;result.width=width;result.height=height;result.dxr.reconstruction=DxrReconstruction::Reference;
    result.dxr.restir_di=false;result.dxr.restir_pt=false;result.dxr.samples_per_pixel=16;result.dxr.opacity_micromaps=false;return result;
}
void check_finite(const Framebuffer& frame){for(int y=0;y<frame.height();++y)for(int x=0;x<frame.width();++x){RENDER_CHECK(frame.pixel(x,y).allFinite());RENDER_CHECK(frame.pixel(x,y).minCoeff()>=0);}}
void export_frame(const char* filename,const Framebuffer& frame) {
    const char* path=std::getenv("DXR_ARTIFACTS");if(!path)return;
    std::filesystem::create_directories(path);Image image(frame.width(),frame.height());DisplaySettings display;display.tone_mapper=ToneMapper::Aces;
    for(int y=0;y<frame.height();++y)for(int x=0;x<frame.width();++x)image.set_pixel(x,y,apply_display_transform(frame.pixel(x,y),display));
    RENDER_CHECK(image.write_png((std::filesystem::path(path)/filename).string()));
}
#endif
}
RENDER_TEST(dxr_reference_lambert_environment_energy) {
#if RENDERER_HAS_DXR
    auto context=device();DxrRenderer renderer(context);Scene source;source.environment=Color::Ones();
    Material material;material.base_color=Color(.6f,.4f,.2f);source.materials.push_back(material);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings();options.dxr.samples_per_pixel=64;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);Framebuffer image(1,1);
    renderer.render(scene,camera,options,{});renderer.readback(image);check_finite(image);
    Color mean=Color::Zero();for(int y=0;y<64;++y)for(int x=0;x<64;++x)mean+=image.pixel(x,y)/4096.f;
    std::cout<<"DXR Lambert environment mean="<<mean.transpose()<<'\n';RENDER_CHECK((mean-material.base_color).norm()<.012f);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_hdr_environment_reuse_preserves_quantized_alias_samples) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Ones();
    Material material;material.base_color=Color(.6f,.4f,.2f);source.materials.push_back(material);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    // More texels than one packed UV component can address. A bright source
    // exposes changes of the sampled direction between RIS and final shading.
    std::vector<Color> pixels(512*256,Color(.025f,.05f,.075f));
    for(int y=126;y<130;++y)for(int x=382;x<386;++x)pixels[y*512+x]=Color(800,250,60);
    source.environment_map=std::make_shared<EnvironmentMap>(512,256,std::move(pixels));
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(32,32);options.dxr.max_bounces=2;
    options.dxr.samples_per_pixel=64;options.dxr.shader_execution_reordering=false;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);DxrRenderer renderer(context);Framebuffer image(1,1);
    auto mean=[&]{Color sum=Color::Zero();for(int y=0;y<32;++y)for(int x=0;x<32;++x)sum+=image.pixel(x,y)/1024.f;return sum;};
    for(int frame=0;frame<16;++frame)renderer.render(scene,camera,options,{});renderer.readback(image);const Color expected=mean();
    RENDER_CHECK(expected.minCoeff()>0);
    options.dxr.samples_per_pixel=1;options.dxr.restir_di=true;
    for(bool pt:{false,true}) {
        options.dxr.restir_pt=pt;
        for(int frame=0;frame<256;++frame)renderer.render(scene,camera,options,{});renderer.readback(image);check_finite(image);
        const Color actual=mean();std::cout<<"HDR alias PT="<<pt<<" mean="<<actual.transpose()<<" reference="<<expected.transpose()<<'\n';
        RENDER_CHECK((actual-expected).norm()/expected.norm()<.025f);
    }
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_frame_resources_stabilize_and_reuse_after_resize) {
#if RENDERER_HAS_DXR
    auto context=device();DxrRenderer renderer(context);auto scene=make_render_scene_snapshot(make_cornell_box_scene());
    auto options=settings(64,64);options.dxr.samples_per_pixel=1;options.dxr.reconstruction=DxrReconstruction::NrdTaau;
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    auto render=[&]{renderer.render(scene,camera,options,{});context->submit();};
    for(int frame=0;frame<24;++frame)render();context->flush();
    const auto creations=context->resource_creations(),bytes=context->allocated_bytes();
    for(int frame=0;frame<80;++frame)render();context->flush();
    RENDER_CHECK(context->resource_creations()==creations);RENDER_CHECK(context->allocated_bytes()==bytes);
    for(int resize=0;resize<12;++resize){options.width=options.height=resize%2?64:96;render();context->flush();}
    const auto resized_creations=context->resource_creations(),resized_bytes=context->allocated_bytes();
    for(int resize=0;resize<12;++resize){options.width=options.height=resize%2?64:96;render();context->flush();}
    RENDER_CHECK(context->resource_reuses()>0);RENDER_CHECK(context->resource_creations()==resized_creations);
    RENDER_CHECK(context->allocated_bytes()==resized_bytes);RENDER_CHECK(context->pooled_bytes()<=128ull*1024*1024);
    Framebuffer image(1,1);renderer.readback(image);check_finite(image);context->check_validation();
    std::cout<<"DXR steady resource creations="<<creations<<" live bytes="<<bytes<<" reuse count="<<context->resource_reuses()<<'\n';
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_sidedness_negative_scale_and_blended_alpha) {
#if RENDERER_HAS_DXR
    auto context=device();DxrRenderer renderer(context);Scene source;source.environment=Color(.1f,.2f,.3f);
    Material material;material.type=MaterialType::Emissive;material.emission=Color(.8f,.5f,.2f);material.two_sided=false;source.materials.push_back(material);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(32,32);options.dxr.samples_per_pixel=1;Framebuffer image(1,1);
    const Camera front(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1),back(Vec3(0,0,-6),Vec3(0,0,-3),Vec3::UnitY(),45,1);
    InteractiveFrameState reset;reset.reset_requested=true;
    for(float scale:{1.f,-2.f}) {
        auto& instance=scene.instances[0];instance.object_to_world(0,0)=scale;instance.object_to_world(1,1)=1.5f;
        instance.world_to_object=instance.object_to_world.inverse();instance.normal_to_world=instance.world_to_object.topLeftCorner<3,3>().transpose();++scene.revisions.transforms;
        renderer.render(scene,front,options,reset);renderer.readback(image);RENDER_CHECK((image.pixel(16,16)-material.emission).norm()<1e-5f);
        renderer.render(scene,back,options,reset);renderer.readback(image);RENDER_CHECK((image.pixel(16,16)-scene.environment).norm()<1e-5f);
    }
    const auto builds=renderer.statistics().blas_builds;
    auto& edited=scene.instances[0].materials[0];edited.two_sided=true;++scene.revisions.materials;
    renderer.render(scene,back,options,reset);renderer.readback(image);RENDER_CHECK((image.pixel(16,16)-material.emission).norm()<1e-5f);
    RENDER_CHECK(renderer.statistics().blas_builds==builds);
    edited.alpha_mode=AlphaMode::Blend;edited.opacity=.25f;++scene.revisions.materials;options.dxr.samples_per_pixel=64;
    for(int frame=0;frame<64;++frame)renderer.render(scene,front,options,{});renderer.readback(image);
    Color mean=Color::Zero();for(int y=0;y<32;++y)for(int x=0;x<32;++x)mean+=image.pixel(x,y)/1024.f;
    const Color expected=.25f*material.emission+.75f*scene.environment;
    RENDER_CHECK((mean-expected).norm()<.002f);context->check_validation();std::cout<<"4096 SPP alpha mean="<<mean.transpose()<<'\n';
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_cornell_ser_and_instance_updates) {
#if RENDERER_HAS_DXR
    auto context=device();auto scene=make_render_scene_snapshot(make_cornell_box_scene());auto options=settings(128,128);
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    Framebuffer ordinary(1,1),reordered(1,1);
    options.dxr.shader_execution_reordering=false;
    {DxrRenderer renderer(context);for(int i=0;i<4;++i)renderer.render(scene,camera,options,{});renderer.readback(ordinary);}
    options.dxr.shader_execution_reordering=true;
    DxrRenderer renderer(context);for(int i=0;i<4;++i)renderer.render(scene,camera,options,{});renderer.readback(reordered);
    check_finite(ordinary);check_finite(reordered);double error=0,energy=0;
    for(int y=0;y<ordinary.height();++y)for(int x=0;x<ordinary.width();++x){error+=(ordinary.pixel(x,y)-reordered.pixel(x,y)).squaredNorm();energy+=ordinary.pixel(x,y).sum();}
    std::cout<<"DXR SER MSE="<<error/(128*128)<<" energy="<<energy/(128*128)<<'\n';
    RENDER_CHECK(energy>100);RENDER_CHECK(error/(128*128)<1e-7);export_frame("cornell-dxr-reference.png",ordinary);
    auto before=renderer.statistics();scene.instances.back().object_to_world(0,3)+=.1f;scene.instances.back().world_to_object=scene.instances.back().object_to_world.inverse();++scene.revisions.transforms;
    renderer.render(scene,camera,options,{});renderer.readback(reordered);
    auto after=renderer.statistics();RENDER_CHECK(before.blas_builds==after.blas_builds);RENDER_CHECK(after.tlas_updates==before.tlas_updates+1);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_texture_normal_uv_transform_and_bump_scale) {
#if RENDERER_HAS_DXR
    auto context=device();DxrRenderer renderer(context);Scene source;source.environment=Color::Ones();
    Material material;material.bump_texture_id=0;material.bump_scale=2;source.materials.push_back(material);
    std::vector<Color> ramp;for(int y=0;y<32;++y)for(int x=0;x<32;++x)ramp.emplace_back(0,float(x)/32,0);
    source.textures.emplace_back(32,32,std::move(ramp));
    source.textures.emplace_back(1,1,std::vector<Color>{Color(.75f,.5f,1)});
    const Vec3 positions[]={Vec3(-10,-10,-3),Vec3(10,-10,-3),Vec3(0,10,-3)};
    const Vec2 uvs[]={Vec2(0,0),Vec2(1,0),Vec2(.5f,1)};TriangleVertex v[3];
    for(int j=0;j<3;++j){v[j].position=positions[j];v[j].uv=uvs[j];v[j].uv1=Vec2(uvs[j].y(),uvs[j].x());v[j].has_uv1=true;}
    source.triangles.emplace_back(v[0],v[1],v[2],0);auto scene=make_render_scene_snapshot(std::move(source));
    auto options=settings(32,32);options.dxr.samples_per_pixel=1;options.dxr.debug_view=DxrDebugView::Normal;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);Framebuffer image(1,1);
    for(float scale:{1.f,-2.f}) {
        auto& instance=scene.instances[0];instance.object_to_world(0,0)=scale;instance.world_to_object=instance.object_to_world.inverse();
        instance.normal_to_world=instance.world_to_object.topLeftCorner<3,3>().transpose();++scene.revisions.transforms;
        auto& edited=instance.materials[0];edited.normal_texture_id=-1;++scene.revisions.materials;
        renderer.render(scene,camera,options,{});renderer.readback(image);
        Vec3 expected((scale>0?-1.f:1.f)*2*.7152f/32,0,1);expected.normalize();
        RENDER_CHECK((image.pixel(16,16)*2-Color::Ones()-expected).norm()<1e-4f);
        edited.normal_texture_id=1;edited.normal_texture_transform.texcoord=1;edited.normal_texture_transform.rotation=1.5707963267948966f;++scene.revisions.materials;
        renderer.render(scene,camera,options,{});renderer.readback(image);
        expected=Vec3(scale>0?-.5f:.5f,0,1).normalized();
        RENDER_CHECK((image.pixel(16,16)*2-Color::Ones()-expected).norm()<1e-4f);
    }
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_divergent_material_textures_and_samplers) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;std::vector<Color> expected;
    for(int y=0;y<8;++y)for(int x=0;x<8;++x) {
        const int index=y*8+x;const Color a(.05f+index*.01f,.1f,.2f),b(.1f,.2f,.05f+index*.01f);
        source.textures.emplace_back(2,1,std::vector<Color>{a,b});
        const bool repeat=index%2==0;
        source.textures.back().set_sampler(repeat?TextureWrap::Repeat:TextureWrap::ClampToEdge,TextureWrap::ClampToEdge,TextureFilter::Nearest,TextureFilter::Nearest);
        expected.push_back(repeat?a:b);
        Material material;material.type=MaterialType::Emissive;material.emission=Color::Ones();material.emissive_texture_id=index;
        source.materials.push_back(material);
        const float left=-2+x*.5f,bottom=-2+y*.5f;TriangleVertex vertices[4];
        vertices[0].position=Vec3(left,bottom,-2);vertices[1].position=Vec3(left+.5f,bottom,-2);
        vertices[2].position=Vec3(left+.5f,bottom+.5f,-2);vertices[3].position=Vec3(left,bottom+.5f,-2);
        for(auto& v:vertices)v.uv=Vec2(1.25f,.5f);
        source.triangles.emplace_back(vertices[0],vertices[1],vertices[2],index);source.triangles.emplace_back(vertices[0],vertices[2],vertices[3],index);
    }
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(64,64);options.dxr.samples_per_pixel=1;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-2),Vec3::UnitY(),90,1);Framebuffer image(1,1);DxrRenderer renderer(context);
    for(bool ser:{false,true}) {
        options.dxr.shader_execution_reordering=ser;renderer.render(scene,camera,options,{});renderer.readback(image);
        for(int y=0;y<64;++y)for(int x=0;x<64;++x)RENDER_CHECK((image.pixel(x,y)-expected[(7-y/8)*8+x/8]).norm()<1e-4f);
    }
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_relax_taau_history_and_resize) {
#if RENDERER_HAS_DXR
    auto context=device();auto scene=make_render_scene_snapshot(make_cornell_box_scene());auto options=settings(160,96);
    options.dxr.samples_per_pixel=1;options.dxr.reconstruction=DxrReconstruction::NrdTaau;options.dxr.internal_scale=2.f/3;
    DxrRenderer renderer(context);Framebuffer image(1,1);
    for(int frame=0;frame<40;++frame) {
        float x=frame<20?.05f*std::sin(frame*.05f):.05f*std::sin(19*.05f);
        const Camera camera(Vec3(x,.15f,1.5f),Vec3(x,.15f,-2),Vec3::UnitY(),45,160.f/96);
        InteractiveFrameState state;state.delta_seconds=1.f/60;renderer.render(scene,camera,options,state);context->submit();
    }
    renderer.readback(image);check_finite(image);export_frame("cornell-dxr-relax-taau.png",image);
    const auto stats=renderer.statistics();RENDER_CHECK(stats.nrd_active);RENDER_CHECK(stats.history_resets==1);RENDER_CHECK(stats.readbacks==1);
    RENDER_CHECK(stats.internal_width==107 && stats.internal_height==64);context->check_validation();
    options.width=96;options.height=64;
    renderer.render(scene,Camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1.5f),options,{});renderer.readback(image);
    RENDER_CHECK(image.width()==96 && image.height()==64);RENDER_CHECK(renderer.statistics().history_resets==2);check_finite(image);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_denoiser_environment_hit_distance_survives_empty_pt_reservoir) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Zero();Material material;source.materials.push_back(material);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(16,16);options.dxr.samples_per_pixel=1;
    options.dxr.internal_scale=1;options.dxr.reconstruction=DxrReconstruction::NrdTaau;options.dxr.debug_view=DxrDebugView::HitDistance;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);Framebuffer image(1,1);DxrRenderer renderer(context);
    for(bool reuse:{false,true}) {
        options.dxr.restir_di=reuse;options.dxr.restir_pt=reuse;
        // The black environment leaves the PT radiance reservoir empty.
        // Its original unoccluded BRDF ray must still provide a nonzero guide.
        for(int frame=0;frame<4;++frame){renderer.render(scene,camera,options,{});renderer.readback(image);
            for(int y=2;y<14;++y)for(int x=2;x<14;++x){RENDER_CHECK(image.pixel(x,y).x()>1);RENDER_CHECK(image.pixel(x,y).y()==0);}}
    }
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_taau_preserves_stationary_silhouette_coverage) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Zero();Material emitter;emitter.type=MaterialType::Emissive;emitter.emission=Color(.8f,.5f,.3f);source.materials.push_back(emitter);
    // This vertical silhouette crosses pixel 32 at 0.272 of its width.
    source.triangles.emplace_back(Vec3(.017f,-100,-2),Vec3(100,-100,-2),Vec3(100,100,-2),0);
    source.triangles.emplace_back(Vec3(.017f,-100,-2),Vec3(100,100,-2),Vec3(.017f,100,-2),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(64,64);options.dxr.samples_per_pixel=1;
    options.dxr.reconstruction=DxrReconstruction::NrdTaau;options.dxr.shader_execution_reordering=false;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-2),Vec3::UnitY(),90,1);DxrRenderer renderer(context);
    for(int frame=0;frame<64;++frame){renderer.render(scene,camera,options,{});context->submit();}
    Framebuffer previous(1,1),image(1,1);double temporal=0,energy=0;
    for(int frame=0;frame<64;++frame) {
        renderer.render(scene,camera,options,{});renderer.readback(image);check_finite(image);
        for(int y=0;y<64;++y)for(int x=0;x<64;++x){
            energy+=image.pixel(x,y).sum();
            if(frame)temporal+=(image.pixel(x,y)-previous.pixel(x,y)).squaredNorm();
        }
        previous=image;
    }
    const double expected=64.*64*(32-.272)*emitter.emission.sum();
    const double rmse=std::sqrt(temporal/(63.*64*64*3));
    std::cout<<"TAA silhouette energy ratio="<<energy/expected<<" temporal RMSE="<<rmse<<'\n';
    RENDER_CHECK(std::abs(energy/expected-1)<.01);RENDER_CHECK(rmse<.005);
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_taau_stationary_glossy_coverage_restarts_on_motion) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Zero();Material material;
    // A smooth PBR surface with deterministic emission isolates TAA coverage
    // from Monte Carlo noise and NRD's own temporal filter.
    material.type=MaterialType::Pbr;material.base_color=Color::Zero();material.specular_factor=0;
    material.roughness=0;material.emission=Color(.8f,.5f,.3f);source.materials.push_back(material);
    source.triangles.emplace_back(Vec3(.017f,-100,-2),Vec3(100,-100,-2),Vec3(100,100,-2),0);
    source.triangles.emplace_back(Vec3(.017f,-100,-2),Vec3(100,100,-2),Vec3(.017f,100,-2),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(64,64);options.dxr.samples_per_pixel=1;
    options.dxr.reconstruction=DxrReconstruction::NrdTaau;options.dxr.shader_execution_reordering=false;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-2),Vec3::UnitY(),90,1);DxrRenderer renderer(context);
    for(int frame=0;frame<64;++frame){renderer.render(scene,camera,options,{});context->submit();}
    Framebuffer previous(1,1),image(1,1);double temporal=0,energy=0;
    for(int frame=0;frame<64;++frame) {
        renderer.render(scene,camera,options,{});renderer.readback(image);check_finite(image);
        for(int y=0;y<64;++y)for(int x=0;x<64;++x){
            energy+=image.pixel(x,y).sum();
            if(frame)temporal+=(image.pixel(x,y)-previous.pixel(x,y)).squaredNorm();
        }
        previous=image;
    }
    const double expected=(32-.272)*64*64*material.emission.sum();
    const double rmse=std::sqrt(temporal/(63*64*64*3));
    std::cout<<"TAA stationary glossy energy="<<energy/expected<<" temporal RMSE="<<rmse<<'\n';
    RENDER_CHECK(std::abs(energy/expected-1)<.004);RENDER_CHECK(rmse<.002);
    // Moving the object exposes a formerly bright pixel. Static accumulation
    // must not retain that pixel, even with an unchanged camera.
    auto& instance=scene.instances[0];instance.object_to_world(0,3)=.5f;
    instance.world_to_object=instance.object_to_world.inverse();++scene.revisions.transforms;
    renderer.render(scene,camera,options,{});renderer.readback(image);
    RENDER_CHECK(image.pixel(36,32).maxCoeff()<.01f);
    // Camera motion back onto the surface also has to leave the static path.
    const Camera shifted(Vec3(.5f,0,0),Vec3(.5f,0,-2),Vec3::UnitY(),90,1);
    renderer.render(scene,shifted,options,{});renderer.readback(image);
    RENDER_CHECK((image.pixel(36,32)-material.emission).norm()<.01f);
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_taau_preserves_output_resolution_material_detail) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Zero();Material material;
    material.type=MaterialType::Emissive;material.emission=Color::Ones();material.emissive_texture_id=0;
    source.materials.push_back(material);std::vector<Color> pixels;
    PointLight movingLight;movingLight.stable_id=901;movingLight.position=Vec3(0,1,0);movingLight.intensity=Color::Ones();
    source.point_lights.push_back(movingLight);
    for(int y=0;y<64;++y)for(int x=0;x<64;++x)pixels.push_back((x/2)%2?Color(.8f,.2f,.1f):Color(.03f,.08f,.4f));
    source.textures.emplace_back(64,64,std::move(pixels));
    source.textures.back().set_sampler(TextureWrap::ClampToEdge,TextureWrap::ClampToEdge,TextureFilter::Nearest,TextureFilter::Nearest);
    TriangleVertex v[4];
    for(int y=0;y<2;++y)for(int x=0;x<2;++x){v[y*2+x].position=Vec3(-2+4*x,-2+4*y,-2);v[y*2+x].uv=Vec2(float(x),float(y));}
    source.triangles.emplace_back(v[0],v[1],v[3],0);source.triangles.emplace_back(v[0],v[3],v[2],0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(64,64);
    const Camera camera(Vec3::Zero(),Vec3(0,0,-2),Vec3::UnitY(),90,1);DxrRenderer renderer(context);
    options.dxr.samples_per_pixel=64;options.dxr.shader_execution_reordering=false;
    Framebuffer reference(1,1),image(1,1);
    renderer.render(scene,camera,options,{});renderer.readback(reference);
    options.dxr.reconstruction=DxrReconstruction::NrdTaau;options.dxr.samples_per_pixel=1;
    double errors[2]{};
    for(int detail=0;detail<2;++detail) {
        options.dxr.full_resolution_materials=detail!=0;
        // An animated light keeps the scene responsive. The emitter's own
        // radiance stays deterministic, isolating immediate material detail
        // from the separate stationary supersampling/convergence path.
        for(int frame=0;frame<128;++frame){scene.point_lights[0].position.x()=std::sin(frame*.1f);++scene.revisions.lighting;
            renderer.render(scene,camera,options,{});context->submit();}
        renderer.readback(image);check_finite(image);
        for(int y=4;y<60;++y)for(int x=4;x<60;++x)errors[detail]+=(image.pixel(x,y)-reference.pixel(x,y)).squaredNorm()/(56*56*3);
    }
    std::cout<<"TAA material MSE: low-resolution="<<errors[0]<<" output-resolution="<<errors[1]<<'\n';
    RENDER_CHECK(errors[0]>.01);RENDER_CHECK(errors[1]<.0001);
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_legacy_opacity_luminance_and_pbr_opaque) {
#if RENDERER_HAS_DXR
    auto context=device();DxrRenderer renderer(context);Scene source;source.environment=Color(.1f,.2f,.3f);
    Material material;material.type=MaterialType::Emissive;material.emission=Color(.8f,.6f,.4f);material.opacity_texture_id=0;
    source.materials.push_back(material);source.textures.emplace_back(1,1,std::vector<Color>{Color(0,1,0)});
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(16,16);options.dxr.samples_per_pixel=1;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);Framebuffer image(1,1);
    renderer.render(scene,camera,options,{});renderer.readback(image);RENDER_CHECK((image.pixel(8,8)-material.emission).norm()<1e-5f);
    scene.textures[0]=ImageTexture(1,1,std::vector<Color>{Color(1,0,0)});++scene.revisions.textures;
    renderer.render(scene,camera,options,{});renderer.readback(image);RENDER_CHECK((image.pixel(8,8)-scene.environment).norm()<1e-5f);
    auto& edited=scene.instances[0].materials[0];edited.type=MaterialType::Pbr;edited.base_color=Color::Zero();edited.specular_factor=0;++scene.revisions.materials;
    renderer.render(scene,camera,options,{});renderer.readback(image);RENDER_CHECK((image.pixel(8,8)-material.emission).norm()<1e-5f);
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_restir_di_energy_and_light_remapping) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Ones();Material m;m.base_color=Color(.6f,.4f,.2f);source.materials.push_back(m);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings();options.dxr.samples_per_pixel=1;options.dxr.restir_di=true;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);Framebuffer image(1,1);DxrRenderer renderer(context);
    for(int i=0;i<64;++i)renderer.render(scene,camera,options,{});renderer.readback(image);check_finite(image);
    Color mean=Color::Zero();for(int y=0;y<64;++y)for(int x=0;x<64;++x)mean+=image.pixel(x,y)/4096.f;
    std::cout<<"RTXDI DI Lambert mean="<<mean.transpose()<<'\n';RENDER_CHECK((mean-m.base_color).norm()<.02f);
    PointLight a;a.stable_id=11;a.position=Vec3(-1,1,0);a.intensity=Color(2,.5f,.2f);
    PointLight b;b.stable_id=12;b.position=Vec3(1,1,0);b.intensity=Color(.2f,.5f,2);
    scene.point_lights={a,b};++scene.revisions.lighting;options.dxr.reconstruction=DxrReconstruction::NrdTaau;
    for(int i=0;i<16;++i)renderer.render(scene,camera,options,{});
    auto before=renderer.statistics();std::swap(scene.point_lights[0],scene.point_lights[1]);++scene.revisions.lighting;
    for(int i=0;i<8;++i)renderer.render(scene,camera,options,{});
    scene.point_lights.erase(scene.point_lights.begin());++scene.revisions.lighting;
    for(int i=0;i<8;++i)renderer.render(scene,camera,options,{});
    renderer.readback(image);check_finite(image);RENDER_CHECK(renderer.statistics().restir_di_active);
    RENDER_CHECK(renderer.statistics().history_resets==before.history_resets);RENDER_CHECK(renderer.statistics().tlas_updates==before.tlas_updates);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_environment_rotation_sampling_matches_cpu_integral) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color(.8f,.5f,.3f);source.environment_intensity=.7f;
    Material material;material.base_color=Color(.6f,.4f,.2f);source.materials.push_back(material);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    std::vector<Color> texels;for(int y=0;y<4;++y)for(int x=0;x<8;++x)texels.emplace_back(.1f+x*.15f,.05f+y*.3f,.2f+(x%3)*.2f);
    source.environment_map=std::make_shared<EnvironmentMap>(8,4,std::move(texels));auto scene=make_render_scene_snapshot(std::move(source));
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),1,1);auto options=settings(64,64);options.dxr.samples_per_pixel=64;
    DxrRenderer renderer(context);Framebuffer image(1,1);
    for(float rotation:{0.f,90.f,180.f}) {
        scene.environment_rotation_degrees=rotation;++scene.revisions.environment;Color expected=Color::Zero();
        constexpr int cells=256;
        for(int y=0;y<cells;++y)for(int x=0;x<cells;++x){const float u=(x+.5f)/cells,phi=(y+.5f)/cells*6.28318530718f;
            const Vec3 direction(std::sqrt(u)*std::cos(phi),std::sqrt(u)*std::sin(phi),std::sqrt(1-u));
            expected+=environment_radiance(scene.environment,scene.environment_map,scene.environment_intensity,rotation,direction)/float(cells*cells);}
        expected=expected.cwiseProduct(material.base_color);
        for(bool reuse:{false,true}) {
            options.dxr.restir_di=reuse;renderer.reset(scene,options);
            // DI generates candidates once per pixel/frame, independent of the
            // reference path's 64 SPP. Accumulate more decorrelated reservoirs.
            for(int frame=0;frame<(reuse?128:16);++frame)renderer.render(scene,camera,options,{});renderer.readback(image);
            Color mean=Color::Zero();for(int y=0;y<image.height();++y)for(int x=0;x<image.width();++x)mean+=image.pixel(x,y)/float(image.width()*image.height());
            std::cout<<"Environment rotation="<<rotation<<" DI="<<reuse<<" mean="<<mean.transpose()<<" expected="<<expected.transpose()<<'\n';
            RENDER_CHECK((mean-expected).norm()/expected.norm()<.025f);
        }
    }
    scene.environment=Color::Zero();++scene.revisions.environment;
    renderer.render(scene,camera,options,{});renderer.readback(image);
    RENDER_CHECK(image.pixel(8,8).norm()==0);
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_visible_rectangle_has_one_sampling_identity) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Zero();Material diffuse;diffuse.base_color=Color(.7f,.5f,.3f);source.materials.push_back(diffuse);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto document=SceneDocument::from_scene(std::move(source),"Rectangle energy","rectangle-energy");
    const Color radiance(2.8f,5,7);const auto light=document.create_rect_area_light("Visible rectangle",Vec3::Zero(),Vec3(0,0,-1),radiance,2,2);
    auto scene=document.render_scene_snapshot();RENDER_CHECK(scene.rect_area_lights.size()==1);RENDER_CHECK(scene.rect_area_lights[0].stable_id==light);
    RENDER_CHECK(std::count_if(scene.instances.begin(),scene.instances.end(),[&](const auto& instance){return instance.emissive_light_id==light;})==1);
    double irradiance=0;constexpr int cells=256;
    for(int y=0;y<cells;++y)for(int x=0;x<cells;++x){const double u=-1+(x+.5)*2/cells,v=-1+(y+.5)*2/cells,r2=u*u+v*v+9;irradiance+=9/(r2*r2)*4/(cells*cells);}
    const Color expected=diffuse.base_color.cwiseProduct(radiance)*float(irradiance/3.141592653589793);
    auto options=settings(16,16);options.dxr.samples_per_pixel=64;options.dxr.max_bounces=2;
    const Camera camera(Vec3(0,0,-1.5f),Vec3(0,0,-3),Vec3::UnitY(),.25f,1);Framebuffer image(1,1);DxrRenderer renderer(context);
    for(bool reuse:{false,true}) {
        options.dxr.restir_di=reuse;options.dxr.restir_pt=reuse;
        for(int frame=0;frame<64;++frame)renderer.render(scene,camera,options,{});renderer.readback(image);
        Color mean=Color::Zero();for(int y=0;y<16;++y)for(int x=0;x<16;++x)mean+=image.pixel(x,y)/256;
        std::cout<<"Rectangle energy reuse="<<reuse<<" mean="<<mean.transpose()<<" expected="<<expected.transpose()<<'\n';
        RENDER_CHECK((mean-expected).norm()/expected.norm()<.015f);
    }
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_restir_pt_cornell_reference_energy) {
#if RENDERER_HAS_DXR
    auto context=device();auto scene=make_render_scene_snapshot(make_cornell_box_scene());auto options=settings(64,64);
    const Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1);
    Framebuffer reference(1,1),reused(1,1);DxrRenderer renderer(context);options.dxr.samples_per_pixel=64;
    for(int i=0;i<16;++i)renderer.render(scene,camera,options,{});renderer.readback(reference);
    options.dxr.samples_per_pixel=1;options.dxr.restir_di=true;options.dxr.restir_pt=true;
    for(int i=0;i<128;++i)renderer.render(scene,camera,options,{});renderer.readback(reused);check_finite(reused);
    double expected=0,actual=0,error=0;
    for(int y=0;y<64;++y)for(int x=0;x<64;++x){expected+=reference.pixel(x,y).sum();actual+=reused.pixel(x,y).sum();error+=(reference.pixel(x,y)-reused.pixel(x,y)).squaredNorm();}
    std::cout<<"RTXDI PT Cornell energy ratio="<<actual/expected<<" MSE="<<error/4096<<'\n';
    export_frame("cornell-dxr-1024spp.png",reference);export_frame("cornell-dxr-restir.png",reused);
    RENDER_CHECK(actual/expected>.94 && actual/expected<1.06);RENDER_CHECK(renderer.statistics().restir_pt_active);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_restir_di_preserves_glossy_direct_connections) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color(.6f,.8f,1);
    Material metal;metal.type=MaterialType::Metal;metal.roughness=.02f;metal.base_color=Color(.8f,.6f,.4f);
    source.materials.push_back(metal);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto document=SceneDocument::from_scene(std::move(source),"Glossy direct light");
    const Camera camera(Vec3(0,0,-1.5f),Vec3(0,0,-3),Vec3::UnitY(),.5f,1);
    auto options=settings(32,32);options.dxr.max_bounces=2;options.dxr.shader_execution_reordering=false;
    DxrRenderer renderer(context);Framebuffer image(1,1);
    auto mean=[&]{Color sum=Color::Zero();for(int y=0;y<image.height();++y)for(int x=0;x<image.width();++x)sum+=image.pixel(x,y);return (sum/float(image.width()*image.height())).eval();};
    for(bool rectangle:{false,true}) {
        if(rectangle){document.set_environment_intensity(0);document.create_rect_area_light("Emitter",Vec3::Zero(),Vec3(0,0,-1),Color(.6f,.8f,1),2,2);}
        const auto& scene=document.render_scene_snapshot();
        options.dxr.restir_di=options.dxr.restir_pt=false;options.dxr.samples_per_pixel=64;
        for(int frame=0;frame<16;++frame)renderer.render(scene,camera,options,{});renderer.readback(image);
        const Color expected=mean();RENDER_CHECK(expected.minCoeff()>.3f);
        for(bool pt:{false,true}) {
            options.dxr.restir_di=true;options.dxr.restir_pt=pt;options.dxr.samples_per_pixel=1;
            for(int frame=0;frame<128;++frame)renderer.render(scene,camera,options,{});renderer.readback(image);check_finite(image);
            const Color actual=mean();
            std::cout<<"Glossy direct rectangle="<<rectangle<<" PT="<<pt<<" mean="<<actual.transpose()<<" reference="<<expected.transpose()<<'\n';
            RENDER_CHECK((actual-expected).norm()/expected.norm()<.025f);
        }
    }
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_restir_di_temporal_horizon_energy) {
#if RENDERER_HAS_DXR
    auto context=device();Scene source;source.environment=Color::Zero();Material material;source.materials.push_back(material);
    // Neighbor normals pass the reuse threshold, but only half of the stripes
    // face this grazing light. A reservoir cannot represent the other's zero
    // support without the temporal bias-correction denominator.
    for(int stripe=0;stripe<256;++stripe) {
        const float x=-10+stripe*(20.f/256),next=x+20.f/256;TriangleVertex v[4];
        v[0].position=Vec3(x,-10,-3);v[1].position=Vec3(next,-10,-3);
        v[2].position=Vec3(next,10,-3);v[3].position=Vec3(x,10,-3);
        for(auto& vertex:v){vertex.normal=Vec3(stripe%2?.15f:-.15f,0,1).normalized();vertex.has_normal=true;}
        source.triangles.emplace_back(v[0],v[1],v[2],0);source.triangles.emplace_back(v[0],v[2],v[3],0);
    }
    DirectionalLight light;light.direction=Vec3(-1,0,-.06f).normalized();light.radiance=Color::Ones();light.angular_radius_radians=0;source.directional_lights.push_back(light);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(64,64);
    options.dxr.max_bounces=1;options.dxr.shader_execution_reordering=false;
    options.dxr.spatial_samples=0;options.dxr.history_length=32;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);Framebuffer image(1,1);DxrRenderer renderer(context);
    auto energy=[&]{double sum=0;for(int y=0;y<image.height();++y)for(int x=0;x<image.width();++x)sum+=image.pixel(x,y).sum();return sum;};
    options.dxr.samples_per_pixel=64;
    for(int frame=0;frame<16;++frame)renderer.render(scene,camera,options,{});renderer.readback(image);const double expected=energy();
    options.dxr.samples_per_pixel=1;options.dxr.restir_di=true;
    for(int frame=0;frame<256;++frame)renderer.render(scene,camera,options,{});renderer.readback(image);check_finite(image);
    const double ratio=energy()/expected;std::cout<<"DI temporal horizon energy ratio="<<ratio<<'\n';
    RENDER_CHECK(std::abs(ratio-1)<.025);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_psr_energy_and_reflected_object_motion) {
#if RENDERER_HAS_DXR
    auto context=device();Scene mirror_scene;mirror_scene.environment=Color::Zero();
    Material mirror;mirror.type=MaterialType::Metal;mirror.roughness=0;mirror.base_color=Color(.8f,.7f,.6f);
    mirror_scene.materials.push_back(mirror);mirror_scene.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    Scene target_scene;Material emissive;emissive.type=MaterialType::Emissive;emissive.emission=Color(.9f,.4f,.2f);
    target_scene.materials.push_back(emissive);target_scene.triangles.emplace_back(Vec3(-100,-100,2),Vec3(100,-100,2),Vec3(0,100,2),0);
    auto scene=make_render_scene_snapshot(std::move(mirror_scene));auto target=make_render_scene_snapshot(std::move(target_scene));
    scene.assets.push_back(target.assets.front());scene.instances.push_back(target.instances.front());
    scene.instances.back().asset_index=1;scene.instances.back().object_id=900;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);auto options=settings();options.dxr.samples_per_pixel=1;
    DxrRenderer renderer(context);Framebuffer baseline(1,1),replacement(1,1);
    options.dxr.specular_antialiasing=false;renderer.render(scene,camera,options,{});renderer.readback(baseline);
    options.dxr.specular_antialiasing=true;renderer.render(scene,camera,options,{});renderer.readback(replacement);
    const Color expected=mirror.base_color.cwiseProduct(emissive.emission);
    RENDER_CHECK((replacement.pixel(32,32)-expected).norm()<1e-5f);
    RENDER_CHECK((replacement.pixel(32,32)-baseline.pixel(32,32)).norm()<1e-5f);
    options.dxr.restir_di=true;options.dxr.restir_pt=true;options.dxr.debug_view=DxrDebugView::SpecularMotion;
    renderer.render(scene,camera,options,{});renderer.readback(replacement);
    RENDER_CHECK(std::abs(replacement.pixel(32,32).x()-.5f)<1e-5f);
    RENDER_CHECK(replacement.pixel(32,32).z()>.99f); // The primary surface was replaced.
    auto& moving=scene.instances.back();moving.object_to_world(0,3)=.1f;moving.world_to_object=moving.object_to_world.inverse();++scene.revisions.transforms;
    renderer.render(scene,camera,options,{});renderer.readback(replacement);check_finite(replacement);
    const float expected_motion=.5f-.1f*64/(8*camera.viewport_width()*32);
    std::cout<<"PSR reflected motion="<<replacement.pixel(32,32).x()<<" expected="<<expected_motion<<'\n';
    RENDER_CHECK(std::abs(replacement.pixel(32,32).x()-expected_motion)<.0002f);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_glass_reference_and_restir_fresnel_energy) {
#if RENDERER_HAS_DXR
    auto context=device();DxrRenderer renderer(context);Scene source;source.environment=Color::Ones();
    Material glass;glass.type=MaterialType::Dielectric;glass.base_color=Color(.8f,.6f,.4f);glass.ior=1.5f;source.materials.push_back(glass);
    source.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
    auto scene=make_render_scene_snapshot(std::move(source));auto options=settings(16,16);options.dxr.samples_per_pixel=64;
    const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),1,1);Framebuffer image(1,1);
    const Color expected=Color::Constant(.04f)+glass.base_color*.96f;
    for(bool reuse:{false,true}) {
        options.dxr.restir_di=options.dxr.restir_pt=reuse;
        for(int frame=0;frame<64;++frame)renderer.render(scene,camera,options,{});
        renderer.readback(image);check_finite(image);Color mean=Color::Zero();
        for(int y=0;y<16;++y)for(int x=0;x<16;++x)mean+=image.pixel(x,y)/256.f;
        std::cout<<"4096 SPP glass reuse="<<reuse<<" mean="<<mean.transpose()<<" expected="<<expected.transpose()<<'\n';
        RENDER_CHECK((mean-expected).norm()<.008f);
    }
    context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
RENDER_TEST(dxr_omm_alpha_equivalence_and_invalidation) {
#if RENDERER_HAS_DXR
    auto context=device();if(!context->capabilities().omm_supported)RENDER_SKIP("DXR 1.2 opacity micromaps unavailable");
    Scene source;source.environment=Color(.1f,.2f,.3f);Material mask;mask.type=MaterialType::Emissive;mask.emission=Color(.8f,.5f,.2f);
    mask.alpha_mode=AlphaMode::Mask;mask.base_color_texture_id=0;source.materials.push_back(mask);
    std::vector<Color> texels(4096,Color::Ones());std::vector<float> alpha(4096);
    for(int y=0;y<64;++y)for(int x=0;x<64;++x)alpha[y*64+x]=x<32?0.f:1.f;
    source.textures.emplace_back(64,64,texels,alpha,TextureUvOrigin::TopLeft);
    source.triangles.emplace_back(Vec3(-3,-3,-3),Vec3(3,-3,-3),Vec3(0,3,-3),0,Vec2(0,0),Vec2(1,0),Vec2(0,1));
    auto scene=make_render_scene_snapshot(std::move(source));const Camera camera(Vec3::Zero(),Vec3(0,0,-3),Vec3::UnitY(),45,1);
    const auto bake=bake_dxr_opacity(scene.assets[0],scene.instances[0],scene.textures);
    RENDER_CHECK(bake.counts[0]>0 && bake.counts[1]>0 && bake.counts[2]+bake.counts[3]>0);
    float area=0;for(unsigned index=0;index<256;++index){const auto t=dxr_microtriangle(index,4);for(const auto& v:t)RENDER_CHECK(v.minCoeff()>=0 && v.sum()<=1);const Vec2 a=t[1]-t[0],b=t[2]-t[0];area+=std::abs(a.x()*b.y()-a.y()*b.x())*.5f;}
    RENDER_CHECK(std::abs(area-.5f)<1e-6f);
    auto options=settings();options.dxr.samples_per_pixel=1;Framebuffer baseline(1,1),mapped(1,1);InteractiveFrameState state;state.reset_requested=true;
    {DxrRenderer renderer(context);for(int frame=0;frame<12;++frame)renderer.render(scene,camera,options,state);renderer.readback(baseline);}
    options.dxr.opacity_micromaps=true;DxrRenderer renderer(context);
    for(int frame=0;frame<12;++frame)renderer.render(scene,camera,options,state);renderer.readback(mapped);
    RENDER_CHECK(renderer.statistics().omm_active);double error=0;for(int y=0;y<64;++y)for(int x=0;x<64;++x)error+=(baseline.pixel(x,y)-mapped.pixel(x,y)).squaredNorm();
    std::cout<<"OMM alpha MSE="<<error/4096<<" builds="<<renderer.statistics().omm_builds<<'\n';RENDER_CHECK(error/4096<1e-8);context->check_validation();
    const auto previous_builds=renderer.statistics().omm_builds;
    auto& edited=scene.instances[0].materials[0];edited.base_color_texture_transform.offset.x()=.25f;++scene.revisions.materials;
    for(int frame=0;frame<8;++frame)renderer.render(scene,camera,options,state);renderer.readback(mapped);
    RENDER_CHECK(renderer.statistics().omm_builds==previous_builds+1);RENDER_CHECK(renderer.statistics().omm_active);check_finite(mapped);
    edited.type=MaterialType::Diffuse;edited.emission=Color::Zero();++scene.revisions.materials;options.dxr.restir_di=true;options.dxr.restir_pt=true;
    for(int frame=0;frame<4;++frame)renderer.render(scene,camera,options,state);renderer.readback(mapped);check_finite(mapped);
    RENDER_CHECK(renderer.statistics().omm_builds==previous_builds+1);context->check_validation();
#else
    if(std::getenv("DXR_STRICT"))RENDER_CHECK(false);RENDER_SKIP("DXR disabled in this build");
#endif
}
