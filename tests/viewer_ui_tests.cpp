#include "test_framework.h"
#include "interactive/realtime_panel.h"

namespace {
struct UiContext {
    ImGuiContext* context=ImGui::CreateContext();
    UiContext() {
        auto& io=ImGui::GetIO();io.IniFilename=nullptr;io.LogFilename=nullptr;
        io.DisplaySize=ImVec2(600,1600);io.DeltaTime=1.0f/60;
        unsigned char* pixels=nullptr;int width=0,height=0;
        io.Fonts->GetTexDataAsRGBA32(&pixels,&width,&height);
        io.Fonts->SetTexID(ImTextureID(1));
    }
    ~UiContext() {ImGui::DestroyContext(context);}
    template<class Draw> void frame(Draw draw,ImVec2 mouse=ImVec2(-100,-100),bool pressed=false) {
        auto& io=ImGui::GetIO();io.AddMousePosEvent(mouse.x,mouse.y);io.AddMouseButtonEvent(0,pressed);
        ImGui::NewFrame();
        ImGui::SetNextWindowPos(ImVec2(0,0));ImGui::SetNextWindowSize(ImVec2(420,1500));
        ImGui::Begin("Settings test",nullptr,ImGuiWindowFlags_NoSavedSettings|ImGuiWindowFlags_NoResize|ImGuiWindowFlags_NoMove);
        draw();ImGui::End();ImGui::Render();
    }
};
}

RENDER_TEST(test_settings_reset_works_when_collapsed_without_toggling_header) {
    UiContext ui;int resets=0;bool open=true;ImVec2 target;
    const auto draw=[&] {
        ImGui::SetNextItemOpen(false,ImGuiCond_Once);
        open=renderer::settings_header("Closed section",[&]{++resets;});
        const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();
        target=ImVec2((a.x+b.x)*.5f,(a.y+b.y)*.5f);
    };
    ui.frame(draw);ui.frame(draw,target);ui.frame(draw,target,true);ui.frame(draw,target,false);
    RENDER_CHECK(resets==1 && !open);
}

RENDER_TEST(test_settings_reset_remains_enabled_and_does_not_reset_siblings) {
    UiContext ui;renderer::RealtimeRenderSettings settings;
    settings.denoise=false;settings.firefly_filter=false;settings.samples_per_pixel=9;
    int lighting_resets=0,denoising_resets=0;ImVec2 target;
    const auto draw=[&] {
        renderer::settings_header("Lighting",[&]{++lighting_resets;renderer::reset_realtime_settings(settings,renderer::RealtimeSettingsSection::Lighting);});
        const bool open=renderer::settings_header("Denoising",[&]{++denoising_resets;renderer::reset_realtime_settings(settings,renderer::RealtimeSettingsSection::Denoising);});
        const auto a=ImGui::GetItemRectMin(),b=ImGui::GetItemRectMax();target=ImVec2((a.x+b.x)*.5f,(a.y+b.y)*.5f);
        if(open) {
            ImGui::BeginDisabled(!settings.denoise);
            ImGui::Checkbox("Firefly suppression",&settings.firefly_filter);
            ImGui::EndDisabled();
        }
    };
    ui.frame(draw);ui.frame(draw,target);ui.frame(draw,target,true);ui.frame(draw,target,false);
    RENDER_CHECK(denoising_resets==1 && lighting_resets==0);
    RENDER_CHECK(settings.denoise && settings.firefly_filter && settings.samples_per_pixel==9);
}

RENDER_TEST(test_realtime_panel_draws_both_denoisers_and_disabled_controls) {
    UiContext ui;renderer::RealtimeRenderSettings settings;
    for(auto denoiser:{renderer::RealtimeDenoiser::Svgf,renderer::RealtimeDenoiser::Optix})for(bool enabled:{false,true}) {
        settings.denoiser=denoiser;settings.denoise=enabled;
        const auto before=settings;
        ui.frame([&]{renderer::draw_realtime_panel(settings);});
        RENDER_CHECK(settings==before);
    }
}
