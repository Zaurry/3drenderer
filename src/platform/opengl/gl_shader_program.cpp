#include "platform/opengl/gl_shader_program.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace renderer {

namespace {

GLuint compile_shader(GLenum type, const std::string& source, std::string& error) {
    const GLuint shader = glCreateShader(type);
    const char* source_pointer = source.c_str();
    glShaderSource(shader, 1, &source_pointer, nullptr);
    glCompileShader(shader);

    GLint compiled = GL_FALSE;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_TRUE) {
        return shader;
    }

    GLint log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> log(static_cast<std::size_t>(std::max(log_length, 1)));
    glGetShaderInfoLog(shader, log_length, nullptr, log.data());
    error = type == GL_VERTEX_SHADER ? "Vertex shader compilation failed:\n" :
        "Fragment shader compilation failed:\n";
    error += log.data();
    glDeleteShader(shader);
    return 0;
}

GLuint link_program(GLuint vertex_shader, GLuint fragment_shader, std::string& error) {
    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint linked = GL_FALSE;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (linked == GL_TRUE) {
        return program;
    }

    GLint log_length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &log_length);
    std::vector<char> log(static_cast<std::size_t>(std::max(log_length, 1)));
    glGetProgramInfoLog(program, log_length, nullptr, log.data());
    error = "Shader program link failed:\n";
    error += log.data();
    glDeleteProgram(program);
    return 0;
}

}  // namespace

GlShaderProgram::~GlShaderProgram() {
    reset();
}

GlShaderProgram::GlShaderProgram(GlShaderProgram&& other) noexcept
    : program_(std::exchange(other.program_, 0)) {}

GlShaderProgram& GlShaderProgram::operator=(GlShaderProgram&& other) noexcept {
    if (this != &other) {
        reset();
        program_ = std::exchange(other.program_, 0);
    }
    return *this;
}

bool GlShaderProgram::load(
    const std::filesystem::path& vertex_path,
    const std::filesystem::path& fragment_path,
    std::string& error) {
    try {
        return load_sources(read_text_file(vertex_path), read_text_file(fragment_path), error);
    } catch (const std::exception& exception) {
        error = exception.what();
        return false;
    }
}

bool GlShaderProgram::load_sources(
    const std::string& vertex_source,
    const std::string& fragment_source,
    std::string& error) {
    error.clear();
    const GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, vertex_source, error);
    if (vertex_shader == 0) {
        return false;
    }
    const GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, fragment_source, error);
    if (fragment_shader == 0) {
        glDeleteShader(vertex_shader);
        return false;
    }

    const GLuint replacement = link_program(vertex_shader, fragment_shader, error);
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    if (replacement == 0) {
        return false;
    }

    reset();
    program_ = replacement;
    return true;
}

GLuint GlShaderProgram::id() const {
    return program_;
}

GlShaderProgram::operator bool() const {
    return program_ != 0;
}

void GlShaderProgram::reset() {
    if (program_ != 0) {
        glDeleteProgram(program_);
        program_ = 0;
    }
}

std::string read_text_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Failed to open shader file: " + path.string());
    }
    std::ostringstream contents;
    contents << input.rdbuf();
    if (!input.good() && !input.eof()) {
        throw std::runtime_error("Failed to read shader file: " + path.string());
    }
    return contents.str();
}

}  // namespace renderer
