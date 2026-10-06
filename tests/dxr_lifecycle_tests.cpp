#include "test_framework.h"
#include "platform/sdl/sdl_display_backend.h"
#include "platform/d3d12/d3d12_context.h"
#include "render/interactive/viewer_render_backend.h"
#include "render/optix/optix_realtime_renderer.h"
#include "scene/scene_document.h"
#include <SDL3/SDL.h>
#include <imgui.h>
#include <nlohmann/json.hpp>
#include <cstdlib>

using namespace renderer;
namespace {
void require_device() {
    const auto capabilities=query_dxr_capabilities();
    if(!capabilities.available){if(std::getenv("DXR_STRICT"))throw TestFailure{capabilities.reason};RENDER_SKIP(capabilities.reason);}
}
SDL_Window* main_window() {
    int count=0;SDL_Window** windows=SDL_GetWindows(&count);SDL_Window* result=nullptr;
    for(int i=0;i<count;++i)if(std::string(SDL_GetWindowTitle(windows[i]))=="DXR lifecycle validation"){result=windows[i];break;}
    SDL_free(windows);RENDER_CHECK(result);return result;
}
}
RENDER_TEST(dxr_viewer_switch_resize_minimize_preserves_document_and_layout) {
    require_device();
    auto document=SceneDocument::from_scene(make_cornell_box_scene(),"Lifecycle Cornell","cornell_box");
    auto transaction=document.begin_edit();
    const auto edit=document.create_group("Unsaved edit");transaction.commit();
    RENDER_CHECK(document.dirty() && document.can_undo());
    const auto saved=document.session_snapshot();const auto source_id=document.render_scene_snapshot().source_id;
    SdlDisplayBackend display;RENDER_CHECK(display.initialize(320,240,"DXR lifecycle validation",true,false));
    ImGui::GetIO().IniFilename=nullptr;
    RenderSettings settings;settings.width=80;settings.height=64;settings.opengl.ddgi.enabled=false;
    settings.dxr.reconstruction=DxrReconstruction::NrdTaau;settings.dxr.samples_per_pixel=1;
    Camera camera(Vec3(0,.15f,1.5f),Vec3(0,.15f,-2),Vec3::UnitY(),45,1.25f);
    const auto root=std::filesystem::path(RENDERER_SOURCE_DIR);
    std::unique_ptr<ViewerRenderBackend> backend;
    std::vector<InteractiveRenderMode> modes{InteractiveRenderMode::Dxr,InteractiveRenderMode::OpenGl,InteractiveRenderMode::Dxr};
    if(optix_realtime_available(0))modes.insert(modes.end(),{InteractiveRenderMode::Rtrt,InteractiveRenderMode::Dxr});
    std::weak_ptr<D3d12Context> old_device;
    for(unsigned cycle=0;cycle<2;++cycle)for(auto mode:modes) {
        std::cout<<"Lifecycle cycle="<<cycle<<" mode="<<int(mode)<<'\n';
        backend.reset();
        const bool dxr=mode==InteractiveRenderMode::Dxr;
        if(!dxr)old_device=display.dxr_context();
        display.switch_presentation(dxr);
        if(!dxr)RENDER_CHECK(old_device.expired());
        backend=make_viewer_render_backend(mode,root/"shaders/opengl/raster.vert",root/"shaders/opengl/raster.frag",false,display.dxr_context());
        backend->reset(document.render_scene_snapshot(),settings);
        for(int frame=0;frame<8;++frame) {
            SDL_Window* window=main_window();
            if(frame==2){RENDER_CHECK(SDL_SetWindowSize(window,384,256));settings.width=96;settings.height=64;}
            if(frame==4){RENDER_CHECK(SDL_MinimizeWindow(window));}
            if(frame==5){RENDER_CHECK(SDL_RestoreWindow(window));}
            (void)display.poll_input();display.begin_ui_frame();
            ImGui::SetNextWindowPos(ImVec2(12,18),ImGuiCond_Once);ImGui::SetNextWindowSize(ImVec2(220,130),ImGuiCond_Once);
            ImGui::Begin("Preserved DXR layout");ImGui::TextUnformatted("DXR / OpenGL / RTRT lifecycle");
            const auto position=ImGui::GetWindowPos();RENDER_CHECK(std::abs(position.x-12)<1 && std::abs(position.y-18)<1);ImGui::End();
            const auto& output=backend->render(document.render_scene_snapshot(),camera,settings,{});display.present(output,{});display.wait_for_frame();
        }
        RENDER_CHECK(document.find(edit));RENDER_CHECK(document.dirty() && document.can_undo());
        RENDER_CHECK(document.session_snapshot()==saved);RENDER_CHECK(document.render_scene_snapshot().source_id==source_id);
        if(dxr){const auto stats=std::get<DxrStatistics>(backend->statistics());RENDER_CHECK(stats.active && stats.readbacks==0);display.dxr_context()->check_validation();}
    }
    backend.reset();document.undo();RENDER_CHECK(document.find(edit)==nullptr);
}
RENDER_TEST(dxr_removed_device_is_reported) {
    require_device();auto context=D3d12Context::create(false);
    context->begin();context->submit();context->flush();context->device()->RemoveDevice();
    bool reported=false;try{context->begin();context->submit();context->flush();}catch(const std::exception& error){reported=std::string(error.what()).find("DXR")!=std::string::npos;}
    RENDER_CHECK(reported);
}
