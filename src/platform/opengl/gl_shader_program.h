#pragma once

#include <glad/gl.h>

#include <filesystem>
#include <string>

namespace renderer {

class GlShaderProgram {
public:
    GlShaderProgram() = default;
    ~GlShaderProgram();

    GlShaderProgram(const GlShaderProgram&) = delete;
    GlShaderProgram& operator=(const GlShaderProgram&) = delete;
    GlShaderProgram(GlShaderProgram&& other) noexcept;
    GlShaderProgram& operator=(GlShaderProgram&& other) noexcept;

    bool load(
        const std::filesystem::path& vertex_path,
        const std::filesystem::path& fragment_path,
        std::string& error);
    bool load_sources(
        const std::string& vertex_source,
        const std::string& fragment_source,
        std::string& error);

    GLuint id() const;
    explicit operator bool() const;
    void reset();

private:
    GLuint program_ = 0;
};

std::string read_text_file(const std::filesystem::path& path);

}  // namespace renderer
