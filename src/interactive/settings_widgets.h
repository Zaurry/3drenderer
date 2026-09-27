#pragma once

#include <imgui.h>

namespace renderer {

// Keep Reset outside disabled controls and usable while the section is closed.
template<class Reset>
bool settings_header(const char* label, Reset reset,
                     const char* tooltip="Restore this section's default settings.") {
    const float right=ImGui::GetCursorPosX()+ImGui::GetContentRegionAvail().x;
    const float button_width=ImGui::CalcTextSize("Reset").x+2*ImGui::GetStyle().FramePadding.x;
    const bool open=ImGui::CollapsingHeader(label,
        ImGuiTreeNodeFlags_DefaultOpen|ImGuiTreeNodeFlags_AllowOverlap);
    ImGui::SameLine(right-button_width);
    ImGui::PushID(label);
    if(ImGui::SmallButton("Reset"))reset();
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",tooltip);
    ImGui::PopID();
    return open;
}

// Subsections have their own reset so tuning one effect need not reset its siblings.
template<class Reset>
void settings_subsection(const char* label, Reset reset,
                         const char* tooltip="Restore this section's default settings.") {
    ImGui::SeparatorText(label);
    ImGui::PushID(label);
    if(ImGui::SmallButton("Reset"))reset();
    if(ImGui::IsItemHovered())ImGui::SetTooltip("%s",tooltip);
    ImGui::PopID();
}

} // namespace renderer
