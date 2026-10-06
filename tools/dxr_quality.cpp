#include "render/dxr/dxr_renderer.h"
#include "render/dxr/dxr_settings_json.h"
#include "platform/d3d12/d3d12_context.h"
#include "render/optix/optix_realtime_renderer.h"
#include "render/realtime/realtime_settings_json.h"
#include "scene/scene_document.h"
#include "core/image.h"
#include <nlohmann/json.hpp>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>

// Quality runs deliberately read back images. Use run_dxr_benchmarks.py for
// full-frame timing: its measured frames have no readbacks or reference work.
namespace {
using namespace renderer;
using Json=nlohmann::json;
struct Options {
    std::filesystem::path scene_file,camera_file,config,output;
    std::string scene="cornell",motion="static";
    int width=640,height=360,frames=64,warmup=64,reference_spp=1024,reference_stride=16;
    float motion_distance=.4f;
    bool rtrt=true;
};
Json read_json(const std::filesystem::path& path){std::ifstream input(path);if(!input)throw std::runtime_error("Cannot read "+path.string());return Json::parse(input);}
void write_json(const std::filesystem::path& path,const Json& value){std::ofstream output(path);output<<value.dump(2);if(!output)throw std::runtime_error("Cannot write "+path.string());}
Options parse(int argc,char** argv) {
    Options o;
    for(int i=1;i<argc;++i){const std::string key=argv[i];
        auto value=[&]()->std::string{if(i+1==argc)throw std::invalid_argument("Missing value: "+key);return argv[++i];};
        if(key=="--output")o.output=value();else if(key=="--scene-file")o.scene_file=value();
        else if(key=="--camera-preset")o.camera_file=value();else if(key=="--config")o.config=value();
        else if(key=="--scene")o.scene=value();else if(key=="--motion")o.motion=value();
        else if(key=="--width")o.width=std::stoi(value());else if(key=="--height")o.height=std::stoi(value());
        else if(key=="--frames")o.frames=std::stoi(value());else if(key=="--warmup")o.warmup=std::stoi(value());
        else if(key=="--reference-spp")o.reference_spp=std::stoi(value());else if(key=="--reference-stride")o.reference_stride=std::stoi(value());
        else if(key=="--motion-distance")o.motion_distance=std::stof(value());else if(key=="--no-rtrt")o.rtrt=false;
        else throw std::invalid_argument("Unknown option: "+key);
    }
    if(o.output.empty() || o.width<1 || o.height<1 || o.frames<2 || o.warmup<0 || o.reference_spp<64 || o.reference_spp%64 || o.reference_stride<1 || !std::isfinite(o.motion_distance))
        throw std::invalid_argument("Require --output, positive dimensions/counts and reference SPP divisible by 64");
    if(o.motion!="static" && o.motion!="camera-stop" && o.motion!="object" && o.motion!="light")throw std::invalid_argument("Motion must be static, camera-stop, object or light");
    if(std::filesystem::exists(o.output))throw std::invalid_argument("Output directory already exists");
    return o;
}
std::string frame_name(int frame){std::ostringstream text;text<<std::setfill('0')<<std::setw(4)<<frame;return text.str();}
void save_image(const std::filesystem::path& stem,const Framebuffer& frame) {
    Image png(frame.width(),frame.height());DisplaySettings display;display.tone_mapper=ToneMapper::Aces;
    std::ofstream linear(stem.string()+".pfm",std::ios::binary);linear<<"PF\n"<<frame.width()<<' '<<frame.height()<<"\n-1.0\n";
    for(int y=frame.height()-1;y>=0;--y)for(int x=0;x<frame.width();++x){
        const auto color=frame.pixel(x,y);if(!color.allFinite() || color.minCoeff()<0)throw std::runtime_error("Non-finite/negative radiance");
        linear.write(reinterpret_cast<const char*>(color.data()),3*sizeof(float));png.set_pixel(x,y,apply_display_transform(color,display));
    }
    if(!linear || !png.write_png(stem.string()+".png"))throw std::runtime_error("Failed to export quality image");
}
struct CameraPreset {
    Vec3 eye{0,.15f,1.5f},forward{0,0,-1},up{0,1,0};float fov=45;
    Camera at(const Options& o,int frame) const {
        Vec3 offset=Vec3::Zero();
        if(o.motion=="camera-stop" && frame>=0){float t=std::min(1.f,float(frame)/std::max(1,o.frames/2));offset.x()=o.motion_distance*std::sin(t*1.57079632679f);}
        return Camera(eye+offset,eye+offset+forward,up,fov,float(o.width)/o.height);
    }
};
CameraPreset camera_preset(const Options& o) {
    CameraPreset result;if(o.camera_file.empty())return result;const auto camera=read_json(o.camera_file).at("camera");
    auto vector=[&](const char* key){const auto& v=camera.at(key);return Vec3(v.at(0).get<float>(),v.at(1).get<float>(),v.at(2).get<float>());};
    result.eye=vector("eye");result.forward=vector("forward");result.up=vector("up");result.fov=camera.at("vertical_fov_degrees");return result;
}
RenderSceneSnapshot make_scene(const Options& o) {
    if(!o.scene_file.empty())return SceneDocument::load(o.scene_file,o.width,o.height).render_scene_snapshot();
    if(o.scene=="mirror") {
        Scene mirror;mirror.environment=Color(.02f,.025f,.03f);Material metal;metal.type=MaterialType::Metal;metal.roughness=0;metal.base_color=Color::Constant(.85f);
        mirror.materials.push_back(metal);mirror.triangles.emplace_back(Vec3(-100,-100,-3),Vec3(100,-100,-3),Vec3(0,100,-3),0);
        auto result=make_render_scene_snapshot(std::move(mirror));
        auto append=[&](float size,float z,Color emission){Scene source;Material material;material.type=MaterialType::Emissive;material.emission=emission;source.materials.push_back(material);
            source.triangles.emplace_back(Vec3(-size,-size,z),Vec3(size,-size,z),Vec3(size,size,z),0);
            source.triangles.emplace_back(Vec3(-size,-size,z),Vec3(size,size,z),Vec3(-size,size,z),0);
            auto object=make_render_scene_snapshot(std::move(source));auto instance=object.instances.front();instance.asset_index=int(result.assets.size());instance.object_id=900+instance.asset_index;
            result.assets.push_back(object.assets.front());result.instances.push_back(instance);};
        append(100,4,Color(.12f,.17f,.25f));append(.75f,2,Color(1,.08f,.03f));return result;
    }
    auto scene=make_cornell_box_scene();
    if(o.scene=="many-lights")for(unsigned j=0;j<256;++j){PointLight light;light.stable_id=10000+j;light.position=Vec3(-.85f+1.7f*(j%16)/15,.7f,-.8f+1.5f*(j/16)/15);light.intensity=Color(.025f,.02f,.015f);scene.point_lights.push_back(light);}
    else if(o.scene=="glass") {Material glass;glass.type=MaterialType::Dielectric;glass.ior=1.5f;glass.base_color=Color(.98f,1,.98f);scene.materials.push_back(glass);scene.spheres.emplace_back(Vec3(0,-.4f,-2),.55f,int(scene.materials.size()-1));}
    else if(o.scene!="cornell")throw std::invalid_argument("Scene must be cornell, many-lights, mirror or glass (or use --scene-file)");
    if((o.motion=="object" || o.motion=="light") && scene.spheres.empty())scene.spheres.emplace_back(Vec3(0,-.65f,-2),.3f,2);
    if(o.motion=="light" && scene.point_lights.empty()){PointLight light;light.stable_id=8100;light.position=Vec3(-.5f,.7f,-1.3f);light.intensity=Color(5,4,3);scene.point_lights.push_back(light);}
    return make_render_scene_snapshot(std::move(scene));
}
void update_scene(RenderSceneSnapshot& scene,const RenderSceneSnapshot& original,const Options& o,int frame) {
    if(frame<0 || (o.motion!="object" && o.motion!="light"))return;
    // Deterministic translation exposes background during the first half, then
    // remains still so convergence and residual trails can be measured.
    const float t=std::min(1.f,float(frame)/std::max(1,o.frames/2));const float offset=o.motion_distance*std::sin(t*1.57079632679f);
    if(o.motion=="object") {
        auto& instance=scene.instances.back();const auto previous=instance.object_to_world;
        instance.object_to_world=original.instances.back().object_to_world;instance.object_to_world(0,3)+=offset;
        instance.world_bounds=original.instances.back().world_bounds;instance.world_bounds.min.x()+=offset;instance.world_bounds.max.x()+=offset;
        if(instance.object_to_world!=previous){instance.world_to_object=instance.object_to_world.inverse();instance.normal_to_world=instance.world_to_object.topLeftCorner<3,3>().transpose();++scene.revisions.transforms;}
    } else if(!scene.point_lights.empty()) {
        const Vec3 position=original.point_lights.front().position+Vec3(offset,0,0);
        if(!scene.point_lights.front().position.isApprox(position)){scene.point_lights.front().position=position;++scene.revisions.lighting;}
    } else if(!scene.rect_area_lights.empty()) {
        auto& light=scene.rect_area_lights.front();const Vec3 position=original.rect_area_lights.front().position+Vec3(offset,0,0);
        if(!light.position.isApprox(position)){
            light.position=position;++scene.revisions.lighting;
            for(std::size_t j=0;j<scene.instances.size();++j)if(light.stable_id && scene.instances[j].emissive_light_id==light.stable_id){
                auto& instance=scene.instances[j];instance.object_to_world=original.instances[j].object_to_world;instance.object_to_world(0,3)+=offset;
                instance.world_to_object=instance.object_to_world.inverse();instance.normal_to_world=instance.world_to_object.topLeftCorner<3,3>().transpose();++scene.revisions.transforms;
                instance.world_bounds=original.instances[j].world_bounds;instance.world_bounds.min.x()+=offset;instance.world_bounds.max.x()+=offset;
            }
        }
    } else throw std::runtime_error("Light motion requires a point or rectangle light");
}
Json error_metrics(const Framebuffer& image,const Framebuffer& reference) {
    double squared=0,absolute=0,energy=0,actual=0,display_squared=0;DisplaySettings display;display.tone_mapper=ToneMapper::Aces;
    for(int y=0;y<image.height();++y)for(int x=0;x<image.width();++x){const Color a=image.pixel(x,y),b=reference.pixel(x,y);
        squared+=(a-b).squaredNorm();absolute+=(a-b).cwiseAbs().sum();energy+=b.sum();actual+=a.sum();
        display_squared+=(apply_display_transform(a,display)-apply_display_transform(b,display)).squaredNorm();}
    const double count=3.*image.width()*image.height(),mse=display_squared/count;
    return {{"linear_rmse",std::sqrt(squared/count)},{"relative_l1",absolute/std::max(energy,1e-12)},
        {"energy_ratio",actual/std::max(energy,1e-12)},{"display_psnr_db",mse>0?-10*std::log10(mse):120.}};
}
double residual_flicker(const Framebuffer& a,const Framebuffer& b,const Framebuffer& ra,const Framebuffer& rb) {
    double error=0;
    for(int y=0;y<a.height();++y)for(int x=0;x<a.width();++x){
        auto log_color=[](const Color& c){return (c.array()+1).log().matrix().eval();};
        error+=(log_color(a.pixel(x,y))-log_color(b.pixel(x,y))-log_color(ra.pixel(x,y))+log_color(rb.pixel(x,y))).squaredNorm();}
    return std::sqrt(error/(3.*a.width()*a.height()));
}
int run(const Options& o) {
    auto context=D3d12Context::create(false); // Strict: failures are never OpenGL fallback.
    const int reference_spp=o.scene=="glass"?std::max(4096,o.reference_spp):o.reference_spp;
    const auto original=make_scene(o);if(original.instances.empty())throw std::runtime_error("Empty scene");
    const auto camera=camera_preset(o);RenderSettings settings;settings.width=o.width;settings.height=o.height;
    if(!o.config.empty())settings.dxr=parse_dxr_settings(read_json(o.config));
    settings.realtime.internal_scale=settings.dxr.internal_scale;settings.realtime.samples_per_pixel=settings.dxr.samples_per_pixel;settings.realtime.max_bounces=settings.dxr.max_bounces;
    std::filesystem::create_directories(o.output/"reference");std::filesystem::create_directories(o.output/"dxr");
    const int stationary_start=o.motion=="static"?0:o.frames/2;
    Json report={{"schema_version",2},{"purpose","quality validation with explicit readback; not a performance benchmark"},
        {"adapter",context->capabilities().adapter},{"driver",context->capabilities().driver_version},
        {"scene_file",o.scene_file.generic_string()},{"scene",o.scene_file.empty()?o.scene:o.scene_file.stem().string()},{"motion",o.motion},{"motion_distance",o.motion_distance},
        {"width",o.width},{"height",o.height},{"frames",o.frames},{"warmup",o.warmup},{"reference_spp",reference_spp},
        {"reference_stride",o.reference_stride},{"stationary_start_frame",stationary_start},{"settings",dxr_settings_json(settings.dxr)},
        {"dlss_quality_coverage","Headless NRD/TAAU; validate DLSS through the DXGI viewer"}};
    report["camera"]={{"eye",{camera.eye.x(),camera.eye.y(),camera.eye.z()}},{"forward",{camera.forward.x(),camera.forward.y(),camera.forward.z()}},
        {"up",{camera.up.x(),camera.up.y(),camera.up.z()}},{"vertical_fov_degrees",camera.fov}};
    report["rtrt_settings"]=realtime_settings_json(settings.realtime);
    report["debug_layer_active"]=context->capabilities().debug_layer_active;report["gpu_validation_active"]=context->capabilities().gpu_validation_active;
    std::set<int> comparisons{0,o.frames/2,o.frames-1};for(int frame=0;frame<o.frames;frame+=o.reference_stride)comparisons.insert(frame);
    std::map<int,Framebuffer> references;
    {
        DxrRenderer renderer(context);auto scene=original;auto reference=settings;reference.dxr.reconstruction=DxrReconstruction::Reference;
        reference.dxr.debug_view=DxrDebugView::Final;
        reference.dxr.samples_per_pixel=64;reference.dxr.restir_di=reference.dxr.restir_pt=false;reference.dxr.specular_antialiasing=false;
        auto previous_camera=camera.at(o,-1);auto previous_revisions=scene.revisions;int previous_reference=-1;
        report["reference_frames"]=Json::array();
        for(int frame:comparisons){update_scene(scene,original,o,frame);const auto view=camera.at(o,frame);
            const bool unchanged=previous_reference>=0 && scene.revisions==previous_revisions &&
                view.eye().isApprox(previous_camera.eye(),0) && view.forward().isApprox(previous_camera.forward(),0) &&
                view.up().isApprox(previous_camera.up(),0) && view.viewport_width()==previous_camera.viewport_width() &&
                view.viewport_height()==previous_camera.viewport_height();
            Framebuffer image(1,1);
            if(unchanged)image=references.at(previous_reference);
            else {InteractiveFrameState state;state.reset_requested=true;
                for(int sample=0;sample<reference_spp;sample+=64){renderer.render(scene,view,reference,state);context->submit();state.reset_requested=false;}
                renderer.readback(image);
            }
            save_image(o.output/"reference"/frame_name(frame),image);references.emplace(frame,std::move(image));
            report["reference_frames"].push_back({{"frame",frame},{"spp",reference_spp},{"reused_from",unchanged?previous_reference:-1}});
            previous_camera=view;previous_revisions=scene.revisions;previous_reference=frame;
            std::cout<<"reference frame="<<frame<<" spp="<<reference_spp<<" identical_state_reused="<<unchanged<<'\n';
        }
    }
    auto capture=[&](const char* label,auto render) {
        std::filesystem::create_directories(o.output/label);auto scene=original;Json measurements=Json::array();
        Framebuffer previous(1,1),last_frame(1,1);int previous_reference=-1;double stationary_flicker=0;unsigned static_pairs=0;
        for(int frame=-o.warmup;frame<o.frames;++frame){update_scene(scene,original,o,frame);Framebuffer image(1,1);InteractiveFrameState state;state.delta_seconds=1.f/60;
            render(scene,camera.at(o,frame),state,image,frame>=0);
            if(frame<0)continue;
            save_image(o.output/label/frame_name(frame),image);
            if(comparisons.contains(frame)){
                auto metrics=error_metrics(image,references.at(frame));metrics["frame"]=frame;
                if(previous_reference>=0)metrics["temporal_residual_rmse"]=residual_flicker(image,previous,references.at(frame),references.at(previous_reference));
                measurements.push_back(metrics);previous_reference=frame;previous=image;
                std::cout<<label<<" frame="<<frame<<" relative_l1="<<metrics["relative_l1"]<<'\n';
            }
            // Both frames of a pair must be stationary. Static captures use
            // every consecutive pair; animated trajectories stop halfway.
            if(frame>stationary_start){stationary_flicker+=residual_flicker(image,last_frame,references.at(stationary_start),references.at(stationary_start));++static_pairs;}
            last_frame=std::move(image);
        }
        report[label]={{"status","measured"},{"comparisons",measurements},{"stationary_pairs",static_pairs},
            {"stationary_temporal_log_rmse",static_pairs?Json(stationary_flicker/static_pairs):Json(nullptr)}};
        write_json(o.output/"report.json",report);
    };
    {DxrRenderer renderer(context);capture("dxr",[&](const auto& scene,const auto& view,const auto& state,Framebuffer& image,bool readback){renderer.render(scene,view,settings,state);context->submit();if(readback)renderer.readback(image);});
        const auto stats=renderer.statistics();report["dxr"]["reconstruction"]=stats.reconstruction;report["dxr"]["history_resets"]=stats.history_resets;report["dxr"]["internal_size"]={stats.internal_width,stats.internal_height};}
    context->check_validation();
    std::string reason;const bool rtrt_available=optix_realtime_available(0,&reason);
    if(o.rtrt && rtrt_available){OptixRealtimeRenderer renderer(CudaDeviceContext::create(0));
        capture("rtrt",[&](const auto& scene,const auto& view,const auto& state,Framebuffer& image,bool){renderer.render_next_frame(scene,view,settings,state,image);});
        const auto stats=renderer.statistics();report["rtrt"]["internal_size"]={stats.internal_width,stats.internal_height};
        report["rtrt"]["samples_per_pixel"]=settings.realtime.samples_per_pixel;report["rtrt"]["max_bounces"]=settings.realtime.max_bounces;
    } else report["rtrt"]={{"status","skipped"},{"reason",o.rtrt?reason:"Explicit --no-rtrt"}};
    report["completed"]=!o.rtrt || rtrt_available;write_json(o.output/"report.json",report);
    return report["completed"].get<bool>()?0:2;
}
}
int main(int argc,char** argv){std::cout<<std::unitbuf;try{return run(parse(argc,argv));}catch(const std::exception& error){std::cerr<<"DXR quality validation failed: "<<error.what()<<'\n';return 1;}}
