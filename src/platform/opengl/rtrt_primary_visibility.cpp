#include "platform/opengl/rtrt_primary_visibility.h"
#include "platform/opengl/gl_shader_program.h"

#if RENDERER_HAS_CUDA
#include <cuda_gl_interop.h>
#include <cuda_runtime_api.h>
#include <array>
#include <climits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace renderer {
namespace {
constexpr const char* vertex_source=R"GLSL(#version 450 core
layout(location=0) in vec3 position;
uniform mat4 object_to_world;
uniform vec3 eye, camera_right, camera_up, camera_forward;
uniform vec2 viewport, jitter_ndc;
out vec2 barycentric;
void main() {
    vec3 p=(object_to_world*vec4(position,1)).xyz-eye;
    float z=dot(p,camera_forward);
    // z=0 retains the entire positive camera half-space, without introducing
    // near/far visibility gaps relative to the ray tracer. Depth is written below.
    gl_Position=vec4(2*dot(p,camera_right)/viewport.x+jitter_ndc.x*z,
                     2*dot(p,camera_up)/viewport.y+jitter_ndc.y*z,0,z);
    int corner=gl_VertexID%3;
    barycentric=vec2(corner==1?1:0,corner==2?1:0);
}
)GLSL";
constexpr const char* fragment_source=R"GLSL(#version 450 core
in vec2 barycentric;
uniform uint instance_id;
layout(location=0) out uvec4 visibility;
void main() {
    // Monotonic reversed depth in (0,1), using the interpolated reciprocal w.
    gl_FragDepth=gl_FragCoord.w/(1+gl_FragCoord.w);
    visibility=uvec4(instance_id,uint(gl_PrimitiveID)+1u,floatBitsToUint(barycentric));
}
)GLSL";

// The viewer and offscreen tests may leave arbitrary GL state behind.
struct RasterState {
    GLint framebuffer=0,program=0,vao=0,viewport[4]{},depth_func=0,polygon[2]{},origin=0,clip_depth=0;
    GLboolean depth_mask=0,color_mask[4]{};
    GLdouble depth_range[2]{};
    static constexpr std::array<GLenum,9> caps={GL_DEPTH_TEST,GL_BLEND,GL_CULL_FACE,
        GL_SCISSOR_TEST,GL_STENCIL_TEST,GL_RASTERIZER_DISCARD,GL_POLYGON_OFFSET_FILL,
        GL_DEPTH_CLAMP,GL_COLOR_LOGIC_OP};
    std::array<GLboolean,caps.size()> enabled{};
    RasterState() {
        glGetIntegerv(GL_DRAW_FRAMEBUFFER_BINDING,&framebuffer);glGetIntegerv(GL_CURRENT_PROGRAM,&program);
        glGetIntegerv(GL_VERTEX_ARRAY_BINDING,&vao);glGetIntegerv(GL_VIEWPORT,viewport);
        glGetIntegerv(GL_DEPTH_FUNC,&depth_func);glGetIntegerv(GL_POLYGON_MODE,polygon);
        glGetIntegerv(GL_CLIP_ORIGIN,&origin);glGetIntegerv(GL_CLIP_DEPTH_MODE,&clip_depth);
        glGetBooleanv(GL_DEPTH_WRITEMASK,&depth_mask);glGetBooleanv(GL_COLOR_WRITEMASK,color_mask);
        glGetDoublev(GL_DEPTH_RANGE,depth_range);
        for(std::size_t i=0;i<caps.size();++i) {enabled[i]=glIsEnabled(caps[i]);glDisable(caps[i]);}
        glEnable(GL_DEPTH_TEST);glDepthFunc(GL_GREATER);glDepthMask(GL_TRUE);glDepthRange(0,1);
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE);glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
        glClipControl(GL_LOWER_LEFT,GL_NEGATIVE_ONE_TO_ONE);
    }
    ~RasterState() {
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER,GLuint(framebuffer));glUseProgram(GLuint(program));
        glBindVertexArray(GLuint(vao));glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
        glDepthFunc(GLenum(depth_func));glDepthMask(depth_mask);glDepthRange(depth_range[0],depth_range[1]);
        glColorMask(color_mask[0],color_mask[1],color_mask[2],color_mask[3]);
        glPolygonMode(GL_FRONT_AND_BACK,GLenum(polygon[0]));glClipControl(GLenum(origin),GLenum(clip_depth));
        for(std::size_t i=0;i<caps.size();++i) {if(enabled[i])glEnable(caps[i]);else glDisable(caps[i]);}
    }
};

class OpenGlPrimaryVisibility final : public RealtimePrimaryVisibility {
public:
    explicit OpenGlPrimaryVisibility(CudaDeviceContext context):context_(context) {}
    ~OpenGlPrimaryVisibility() override {
        context_.activate();
        release_target();release_geometry();
        if(query_) glDeleteQueries(1,&query_);
    }
    CudaSurfaceHandle begin_frame(const RenderSceneSnapshot& snapshot,const Camera& camera,
        int width,int height,float jitter_x,float jitter_y,CudaStreamHandle handle) override {
        if(failed_) return 0;
        context_.activate();
        if(!GLAD_GL_VERSION_4_5) return fail("Primary rasterization needs OpenGL 4.5");
        for(const auto& asset:snapshot.assets) {
            if(asset.local_scene && !asset.local_scene->spheres.empty()) {
                reason_="Analytic primitives use CUDA primary traversal";return 0;
            }
        }
        if(!program_ && !program_.load_sources(vertex_source,fragment_source,reason_)) return fail(reason_);
        if(!sync_geometry(snapshot)) return 0;
        if(!resize(width,height)) return 0;
        if(query_pending_) {
            GLint ready=0;glGetQueryObjectiv(query_,GL_QUERY_RESULT_AVAILABLE,&ready);
            if(ready) {
                GLuint64 ns=0;glGetQueryObjectui64v(query_,GL_QUERY_RESULT,&ns);
                gpu_ms_=float(double(ns)*1e-6);query_pending_=false;
            }
        }
        {
            RasterState restore;
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER,framebuffer_);glViewport(0,0,width,height);
            const bool measure=!query_pending_;
            if(measure) glBeginQuery(GL_TIME_ELAPSED,query_);
            const GLuint clear[4]={0,0,0,0};const GLfloat clear_depth=0;
            glClearBufferuiv(GL_COLOR,0,clear);glClearBufferfv(GL_DEPTH,0,&clear_depth);
            glUseProgram(program_.id());
            auto vec=[&](const char* name,const Vec3& value) {
                glUniform3fv(glGetUniformLocation(program_.id(),name),1,value.data());
            };
            vec("eye",camera.eye());vec("camera_right",camera.right());
            vec("camera_up",camera.up());vec("camera_forward",camera.forward());
            glUniform2f(glGetUniformLocation(program_.id(),"viewport"),camera.viewport_width(),camera.viewport_height());
            glUniform2f(glGetUniformLocation(program_.id(),"jitter_ndc"),-2*jitter_x/width,2*jitter_y/height);
            const GLint transform=glGetUniformLocation(program_.id(),"object_to_world");
            const GLint instance_id=glGetUniformLocation(program_.id(),"instance_id");
            for(std::size_t i=0;i<snapshot.instances.size();++i) {
                const auto& instance=snapshot.instances[i];
                if(instance.asset_index<0 || std::size_t(instance.asset_index)>=assets_.size()) continue;
                const auto& asset=assets_[std::size_t(instance.asset_index)];
                if(asset.vertices==0) continue;
                glUniformMatrix4fv(transform,1,GL_FALSE,instance.object_to_world.data());
                glUniform1ui(instance_id,GLuint(i+1));glBindVertexArray(asset.vao);
                glDrawArrays(GL_TRIANGLES,0,asset.vertices);
            }
            if(measure) {glEndQuery(GL_TIME_ELAPSED);query_pending_=true;}
        }
        const auto stream=reinterpret_cast<cudaStream_t>(handle);
        cudaError_t error=cudaGraphicsMapResources(1,&resource_,stream);
        if(error!=cudaSuccess) return fail(cudaGetErrorString(error));
        mapped_=true;mapped_stream_=stream;
        cudaArray_t array=nullptr;
        error=cudaGraphicsSubResourceGetMappedArray(&array,resource_,0,0);
        if(error==cudaSuccess) {
            cudaResourceDesc desc{};desc.resType=cudaResourceTypeArray;desc.res.array.array=array;
            error=cudaCreateSurfaceObject(&surface_,&desc);
        }
        if(error!=cudaSuccess) {release_target();return fail(cudaGetErrorString(error));}
        reason_.clear();return CudaSurfaceHandle(surface_);
    }
    void end_frame(CudaStreamHandle handle) override {
        if(!mapped_) return;
        const auto destroy=cudaDestroySurfaceObject(surface_);surface_=0;
        const auto unmap=cudaGraphicsUnmapResources(1,&resource_,reinterpret_cast<cudaStream_t>(handle));
        if(unmap==cudaSuccess) mapped_=false;
        if(destroy!=cudaSuccess || unmap!=cudaSuccess) {
            const auto error=destroy!=cudaSuccess?destroy:unmap;
            fail(cudaGetErrorString(error));
            throw std::runtime_error("RTRT primary visibility unmap failed: "+reason_);
        }
    }
    float gpu_milliseconds() const override {return gpu_ms_;}
    std::uint64_t resident_bytes() const override {
        return std::uint64_t(width_)*height_*20+geometry_bytes_;
    }
    const std::string& reason() const override {return reason_;}
private:
    struct Asset {
        GLuint vao=0,vbo=0;GLsizei vertices=0;
        std::uint64_t id=0,revision=0;
        std::shared_ptr<const Scene> source;
    };
    CudaSurfaceHandle fail(std::string message) {reason_=std::move(message);failed_=true;return 0;}
    bool sync_geometry(const RenderSceneSnapshot& snapshot) {
        bool changed=source_!=snapshot.source_id || geometry_revision_!=snapshot.revisions.geometry || assets_.size()!=snapshot.assets.size();
        if(!changed) for(std::size_t i=0;i<assets_.size();++i) {
            const auto& a=snapshot.assets[i];
            if(assets_[i].id!=a.asset_id || assets_[i].revision!=a.geometry_revision || assets_[i].source!=a.local_scene) {changed=true;break;}
        }
        if(!changed) return true;
        release_geometry();assets_.resize(snapshot.assets.size());
        for(std::size_t i=0;i<assets_.size();++i) {
            auto& out=assets_[i];const auto& asset=snapshot.assets[i];
            out.id=asset.asset_id;out.revision=asset.geometry_revision;out.source=asset.local_scene;
            if(!asset.local_scene) continue;
            const auto& triangles=asset.local_scene->triangles;
            if(triangles.size()>std::size_t(INT_MAX/3)) {fail("Primary raster mesh exceeds GL draw limits");return false;}
            if(triangles.empty()) continue;
            std::vector<float> positions;positions.reserve(triangles.size()*9);
            for(const auto& triangle:triangles) for(int v=0;v<3;++v) {
                const Vec3& p=triangle.vertex(v).position;
                positions.insert(positions.end(),{p.x(),p.y(),p.z()});
            }
            out.vertices=GLsizei(triangles.size()*3);
            glCreateBuffers(1,&out.vbo);glNamedBufferData(out.vbo,GLsizeiptr(positions.size()*sizeof(float)),positions.data(),GL_STATIC_DRAW);
            glCreateVertexArrays(1,&out.vao);glVertexArrayVertexBuffer(out.vao,0,out.vbo,0,3*sizeof(float));
            glEnableVertexArrayAttrib(out.vao,0);glVertexArrayAttribFormat(out.vao,0,3,GL_FLOAT,GL_FALSE,0);glVertexArrayAttribBinding(out.vao,0,0);
            geometry_bytes_+=positions.size()*sizeof(float);
        }
        source_=snapshot.source_id;geometry_revision_=snapshot.revisions.geometry;return true;
    }
    bool resize(int width,int height) {
        if(width_==width && height_==height && resource_) return true;
        release_target();
        glCreateTextures(GL_TEXTURE_2D,1,&texture_);glTextureStorage2D(texture_,1,GL_RGBA32UI,width,height);
        glCreateRenderbuffers(1,&depth_);glNamedRenderbufferStorage(depth_,GL_DEPTH_COMPONENT32F,width,height);
        glCreateFramebuffers(1,&framebuffer_);glNamedFramebufferTexture(framebuffer_,GL_COLOR_ATTACHMENT0,texture_,0);
        glNamedFramebufferRenderbuffer(framebuffer_,GL_DEPTH_ATTACHMENT,GL_RENDERBUFFER,depth_);
        glNamedFramebufferDrawBuffer(framebuffer_,GL_COLOR_ATTACHMENT0);
        if(glCheckNamedFramebufferStatus(framebuffer_,GL_FRAMEBUFFER)!=GL_FRAMEBUFFER_COMPLETE) {
            fail("Primary visibility framebuffer is incomplete");return false;
        }
        const auto error=cudaGraphicsGLRegisterImage(&resource_,texture_,GL_TEXTURE_2D,
            cudaGraphicsRegisterFlagsReadOnly|cudaGraphicsRegisterFlagsSurfaceLoadStore);
        if(error!=cudaSuccess) {fail(cudaGetErrorString(error));return false;}
        width_=width;height_=height;
        if(!query_) glGenQueries(1,&query_);
        return true;
    }
    void release_target() noexcept {
        if(mapped_) cudaStreamSynchronize(mapped_stream_);
        if(surface_) {cudaDestroySurfaceObject(surface_);surface_=0;}
        if(mapped_) {cudaGraphicsUnmapResources(1,&resource_,mapped_stream_);mapped_=false;}
        if(resource_) {cudaGraphicsUnregisterResource(resource_);resource_=nullptr;}
        if(framebuffer_) glDeleteFramebuffers(1,&framebuffer_);
        if(depth_) glDeleteRenderbuffers(1,&depth_);
        if(texture_) glDeleteTextures(1,&texture_);
        framebuffer_=depth_=texture_=0;width_=height_=0;
    }
    void release_geometry() noexcept {
        for(const auto& a:assets_) {if(a.vao)glDeleteVertexArrays(1,&a.vao);if(a.vbo)glDeleteBuffers(1,&a.vbo);}
        assets_.clear();geometry_bytes_=0;
    }
    CudaDeviceContext context_;
    GlShaderProgram program_;
    std::vector<Asset> assets_;
    GLuint framebuffer_=0,texture_=0,depth_=0,query_=0;
    cudaGraphicsResource_t resource_=nullptr;
    cudaSurfaceObject_t surface_=0;
    cudaStream_t mapped_stream_=nullptr;
    std::uint64_t source_=0,geometry_revision_=0,geometry_bytes_=0;
    int width_=0,height_=0;
    bool mapped_=false,failed_=false,query_pending_=false;
    float gpu_ms_=0;
    std::string reason_;
};
} // namespace
std::shared_ptr<RealtimePrimaryVisibility> make_opengl_primary_visibility(CudaDeviceContext context) {
    return std::make_shared<OpenGlPrimaryVisibility>(context);
}
} // namespace renderer
#else
namespace renderer {
std::shared_ptr<RealtimePrimaryVisibility> make_opengl_primary_visibility(CudaDeviceContext) {return {};}
}
#endif
