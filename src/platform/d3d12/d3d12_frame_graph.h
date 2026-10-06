#pragma once
#include "platform/d3d12/d3d12_context.h"
#include <initializer_list>
#include <set>
#include <string_view>
#include <unordered_map>

namespace renderer {
// Ordered single-queue frame graph. A pass declares every shared image/buffer
// it touches. Dependencies and UAV hazards follow the last producer/consumer;
// state transitions and fence retention are handled centrally. SDK passes may
// manage their private scratch resources internally.
class D3d12FrameGraph {
public:
    enum Access { ShaderRead, UnorderedRead, UnorderedWrite };
    struct Use {D3d12ResourcePtr resource;Access access;};
    struct Pass {std::string_view name;std::vector<unsigned> dependencies;};
    explicit D3d12FrameGraph(D3d12Context& context):context_(context){}
    template<class Record>
    void run(std::string_view name,std::initializer_list<Use> uses,Record&& record) {
        std::set<unsigned> dependencies;bool uav_hazard=false;
        const auto index=unsigned(passes_.size());
        for(const auto& use:uses)if(use.resource) {
            const auto target=use.access==ShaderRead?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            auto previous=last_.find(use.resource.get());
            // History may have been written by the preceding submitted frame.
            if(previous==last_.end() && use.resource->state==D3D12_RESOURCE_STATE_UNORDERED_ACCESS && target==D3D12_RESOURCE_STATE_UNORDERED_ACCESS)uav_hazard=true;
            if(previous!=last_.end() && (previous->second.write || use.access==UnorderedWrite)) {
                dependencies.insert(previous->second.pass);
                uav_hazard|=use.resource->state==D3D12_RESOURCE_STATE_UNORDERED_ACCESS && target==D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            }
        }
        if(uav_hazard)context_.uav_barrier({});
        for(const auto& use:uses)if(use.resource) {
            context_.transition(use.resource,use.access==ShaderRead?D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE:D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
            last_[use.resource.get()]={index,use.access==UnorderedWrite};
        }
        passes_.push_back({name,{dependencies.begin(),dependencies.end()}});
        record();
    }
    const std::vector<Pass>& passes() const {return passes_;}
private:
    struct LastUse {unsigned pass;bool write;};
    D3d12Context& context_;
    std::unordered_map<const D3d12Resource*,LastUse> last_;
    std::vector<Pass> passes_;
};
}
