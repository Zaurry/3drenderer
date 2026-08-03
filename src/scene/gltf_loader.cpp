#include "scene/gltf_loader.h"

#include "scene/material.h"
#include "scene/texture.h"

#include <fastgltf/core.hpp>
#include <fastgltf/tools.hpp>
#include <fastgltf/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace renderer {

namespace {

struct TextureCacheKey {
    std::size_t texture_index = 0;
    TextureEncoding encoding = TextureEncoding::Srgb;

    bool operator==(const TextureCacheKey&) const = default;
};

struct TextureCacheKeyHash {
    std::size_t operator()(const TextureCacheKey& key) const {
        return key.texture_index * 1315423911U + static_cast<std::size_t>(key.encoding);
    }
};

struct ImageCacheKey {
    std::size_t image_index = 0;
    TextureEncoding encoding = TextureEncoding::Srgb;

    bool operator==(const ImageCacheKey&) const = default;
};

struct ImageCacheKeyHash {
    std::size_t operator()(const ImageCacheKey& key) const {
        return key.image_index * 2654435761U + static_cast<std::size_t>(key.encoding);
    }
};

Camera default_camera(const Bounds3& bounds, int width, int height) {
    const Vec3 center = (bounds.min + bounds.max) * 0.5f;
    const float radius = std::max(0.5f, (bounds.max - bounds.min).norm() * 0.5f);
    const float aspect = static_cast<float>(std::max(width, 1)) /
        static_cast<float>(std::max(height, 1));
    return Camera(
        center + Vec3(0.0f, radius * 0.15f, radius * 2.4f),
        center,
        Vec3(0.0f, 1.0f, 0.0f),
        45.0f,
        aspect);
}

Mat4 convert_matrix(const fastgltf::math::fmat4x4& source) {
    Mat4 result;
    for (int column = 0; column < 4; ++column) {
        for (int row = 0; row < 4; ++row) {
            result(row, column) = source[static_cast<std::size_t>(column)][static_cast<std::size_t>(row)];
        }
    }
    return result;
}

TextureTransform convert_transform(const fastgltf::TextureInfo& source) {
    TextureTransform result;
    result.texcoord = static_cast<int>(source.texCoordIndex);
    if (source.transform) {
        result.offset = Vec2(
            source.transform->uvOffset.x(),
            source.transform->uvOffset.y());
        result.scale = Vec2(
            source.transform->uvScale.x(),
            source.transform->uvScale.y());
        result.rotation = source.transform->rotation;
        if (source.transform->texCoordIndex) {
            result.texcoord = static_cast<int>(*source.transform->texCoordIndex);
        }
    }
    if (result.texcoord < 0 || result.texcoord > 1) {
        throw std::runtime_error("glTF material references TEXCOORD_2 or higher; only TEXCOORD_0/1 are supported");
    }
    return result;
}

std::vector<unsigned char> read_file_bytes(
    const std::filesystem::path& path,
    std::size_t offset) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("Failed to open glTF image: " + path.string());
    }
    const std::streamoff size = stream.tellg();
    if (size < 0 || static_cast<std::uint64_t>(size) < offset) {
        throw std::runtime_error("Invalid glTF image byte offset: " + path.string());
    }
    const std::size_t byte_count = static_cast<std::size_t>(size) - offset;
    std::vector<unsigned char> bytes(byte_count);
    stream.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (byte_count > 0U) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(byte_count));
    }
    if (!stream) {
        throw std::runtime_error("Failed to read glTF image: " + path.string());
    }
    return bytes;
}

ImageTexture decode_image(
    const fastgltf::Asset& asset,
    const fastgltf::Image& image,
    const std::filesystem::path& base,
    TextureEncoding encoding,
    const std::string& label) {
    return std::visit(fastgltf::visitor{
        [&](const fastgltf::sources::URI& source) -> ImageTexture {
            if (!source.uri.isLocalPath()) {
                throw std::runtime_error("Remote glTF image URIs are not supported: " + label);
            }
            std::filesystem::path path(std::string(source.uri.path().begin(), source.uri.path().end()));
            if (path.is_relative()) {
                path = base / path;
            }
            const std::vector<unsigned char> bytes = read_file_bytes(path, source.fileByteOffset);
            return ImageTexture::load_from_memory(
                bytes.data(),
                bytes.size(),
                encoding,
                path.string(),
                TextureUvOrigin::TopLeft);
        },
        [&](const fastgltf::sources::Array& source) -> ImageTexture {
            return ImageTexture::load_from_memory(
                reinterpret_cast<const unsigned char*>(source.bytes.data()),
                source.bytes.size_bytes(),
                encoding,
                label,
                TextureUvOrigin::TopLeft);
        },
        [&](const fastgltf::sources::Vector& source) -> ImageTexture {
            return ImageTexture::load_from_memory(
                reinterpret_cast<const unsigned char*>(source.bytes.data()),
                source.bytes.size(),
                encoding,
                label,
                TextureUvOrigin::TopLeft);
        },
        [&](const fastgltf::sources::ByteView& source) -> ImageTexture {
            return ImageTexture::load_from_memory(
                reinterpret_cast<const unsigned char*>(source.bytes.data()),
                source.bytes.size_bytes(),
                encoding,
                label,
                TextureUvOrigin::TopLeft);
        },
        [&](const fastgltf::sources::BufferView& source) -> ImageTexture {
            const auto bytes = fastgltf::DefaultBufferDataAdapter{}(asset, source.bufferViewIndex);
            return ImageTexture::load_from_memory(
                reinterpret_cast<const unsigned char*>(bytes.data()),
                bytes.size_bytes(),
                encoding,
                label,
                TextureUvOrigin::TopLeft);
        },
        [&](const auto&) -> ImageTexture {
            throw std::runtime_error("Unsupported glTF image data source: " + label);
        }}, image.data);
}

TextureWrap convert_wrap(fastgltf::Wrap wrap) {
    switch (wrap) {
        case fastgltf::Wrap::ClampToEdge:
            return TextureWrap::ClampToEdge;
        case fastgltf::Wrap::MirroredRepeat:
            return TextureWrap::MirroredRepeat;
        case fastgltf::Wrap::Repeat:
            return TextureWrap::Repeat;
    }
    return TextureWrap::Repeat;
}

TextureFilter convert_filter(fastgltf::Filter filter) {
    switch (filter) {
        case fastgltf::Filter::Nearest:
            return TextureFilter::Nearest;
        case fastgltf::Filter::Linear:
            return TextureFilter::Linear;
        case fastgltf::Filter::NearestMipMapNearest:
            return TextureFilter::NearestMipmapNearest;
        case fastgltf::Filter::LinearMipMapNearest:
            return TextureFilter::LinearMipmapNearest;
        case fastgltf::Filter::NearestMipMapLinear:
            return TextureFilter::NearestMipmapLinear;
        case fastgltf::Filter::LinearMipMapLinear:
            return TextureFilter::LinearMipmapLinear;
    }
    return TextureFilter::Linear;
}

int texture_id_for(
    const fastgltf::Asset& asset,
    std::size_t texture_index,
    TextureEncoding encoding,
    const std::filesystem::path& base,
    std::vector<ImageTexture>& textures,
    std::unordered_map<TextureCacheKey, int, TextureCacheKeyHash>& cache,
    std::unordered_map<ImageCacheKey, ImageTexture, ImageCacheKeyHash>& image_cache) {
    if (texture_index >= asset.textures.size()) {
        throw std::runtime_error("glTF texture index is out of range");
    }
    const TextureCacheKey key{texture_index, encoding};
    if (const auto found = cache.find(key); found != cache.end()) {
        return found->second;
    }
    const fastgltf::Texture& texture = asset.textures[texture_index];
    if (!texture.imageIndex || *texture.imageIndex >= asset.images.size()) {
        throw std::runtime_error("glTF texture does not reference a PNG/JPEG image");
    }
    const int id = static_cast<int>(textures.size());
    const std::string label = asset.images[*texture.imageIndex].name.empty()
        ? "glTF image " + std::to_string(*texture.imageIndex)
        : std::string(asset.images[*texture.imageIndex].name);
    const ImageCacheKey image_key{*texture.imageIndex, encoding};
    auto decoded_image = image_cache.find(image_key);
    if (decoded_image == image_cache.end()) {
        decoded_image = image_cache.emplace(
            image_key,
            decode_image(
                asset,
                asset.images[*texture.imageIndex],
                base,
                encoding,
                label)).first;
    }
    ImageTexture decoded = decoded_image->second;
    TextureWrap wrap_s = TextureWrap::Repeat;
    TextureWrap wrap_t = TextureWrap::Repeat;
    TextureFilter min_filter = TextureFilter::LinearMipmapLinear;
    TextureFilter mag_filter = TextureFilter::Linear;
    if (texture.samplerIndex) {
        if (*texture.samplerIndex >= asset.samplers.size()) {
            throw std::runtime_error("glTF sampler index is out of range");
        }
        const fastgltf::Sampler& sampler = asset.samplers[*texture.samplerIndex];
        wrap_s = convert_wrap(sampler.wrapS);
        wrap_t = convert_wrap(sampler.wrapT);
        if (sampler.minFilter) {
            min_filter = convert_filter(*sampler.minFilter);
        }
        if (sampler.magFilter) {
            mag_filter = convert_filter(*sampler.magFilter);
        }
    }
    decoded.set_sampler(wrap_s, wrap_t, min_filter, mag_filter);
    textures.push_back(std::move(decoded));
    cache.emplace(key, id);
    return id;
}

template <typename TextureInfoType>
void apply_texture_info(
    const fastgltf::Asset& asset,
    const std::optional<TextureInfoType>& info,
    TextureEncoding encoding,
    const std::filesystem::path& base,
    std::vector<ImageTexture>& textures,
    std::unordered_map<TextureCacheKey, int, TextureCacheKeyHash>& cache,
    std::unordered_map<ImageCacheKey, ImageTexture, ImageCacheKeyHash>& image_cache,
    int& texture_id,
    TextureTransform& transform) {
    if (!info) {
        return;
    }
    texture_id = texture_id_for(
        asset,
        info->textureIndex,
        encoding,
        base,
        textures,
        cache,
        image_cache);
    transform = convert_transform(*info);
}

std::vector<Material> convert_materials(
    const fastgltf::Asset& asset,
    const std::filesystem::path& base,
    std::vector<ImageTexture>& textures,
    std::vector<std::string>& names) {
    std::vector<Material> result;
    result.reserve(asset.materials.size() + 1U);
    std::unordered_map<TextureCacheKey, int, TextureCacheKeyHash> cache;
    std::unordered_map<ImageCacheKey, ImageTexture, ImageCacheKeyHash> image_cache;
    for (const fastgltf::Material& source : asset.materials) {
        Material material;
        material.type = MaterialType::Pbr;
        material.ior = std::max(1.0f, static_cast<float>(source.ior));
        material.emission = Color(
            source.emissiveFactor.x(),
            source.emissiveFactor.y(),
            source.emissiveFactor.z()) * static_cast<float>(source.emissiveStrength);
        material.alpha_cutoff = source.alphaCutoff;
        material.two_sided = source.doubleSided;
        switch (source.alphaMode) {
            case fastgltf::AlphaMode::Opaque:
                material.alpha_mode = AlphaMode::Opaque;
                break;
            case fastgltf::AlphaMode::Mask:
                material.alpha_mode = AlphaMode::Mask;
                break;
            case fastgltf::AlphaMode::Blend:
                material.alpha_mode = AlphaMode::Blend;
                break;
        }
#if FASTGLTF_ENABLE_DEPRECATED_EXT
        if (source.specularGlossiness) {
            material.pbr_workflow = PbrWorkflow::SpecularGlossiness;
            material.base_color = Color(
                source.specularGlossiness->diffuseFactor.x(),
                source.specularGlossiness->diffuseFactor.y(),
                source.specularGlossiness->diffuseFactor.z());
            material.opacity = source.specularGlossiness->diffuseFactor.w();
            material.specular_color = Color(
                source.specularGlossiness->specularFactor.x(),
                source.specularGlossiness->specularFactor.y(),
                source.specularGlossiness->specularFactor.z());
            material.glossiness = std::clamp(
                static_cast<float>(source.specularGlossiness->glossinessFactor),
                0.0f,
                1.0f);
            material.roughness = 1.0f - material.glossiness;
            apply_texture_info(
                asset,
                source.specularGlossiness->diffuseTexture,
                TextureEncoding::Srgb,
                base,
                textures,
                cache,
                image_cache,
                material.base_color_texture_id,
                material.base_color_texture_transform);
            apply_texture_info(
                asset,
                source.specularGlossiness->specularGlossinessTexture,
                TextureEncoding::Srgb,
                base,
                textures,
                cache,
                image_cache,
                material.specular_glossiness_texture_id,
                material.specular_glossiness_texture_transform);
        } else
#endif
        {
            material.base_color = Color(
                source.pbrData.baseColorFactor.x(),
                source.pbrData.baseColorFactor.y(),
                source.pbrData.baseColorFactor.z());
            material.opacity = source.pbrData.baseColorFactor.w();
            material.metallic = std::clamp(
                static_cast<float>(source.pbrData.metallicFactor),
                0.0f,
                1.0f);
            material.roughness = std::clamp(
                static_cast<float>(source.pbrData.roughnessFactor),
                0.0f,
                1.0f);
            apply_texture_info(
                asset,
                source.pbrData.baseColorTexture,
                TextureEncoding::Srgb,
                base,
                textures,
                cache,
                image_cache,
                material.base_color_texture_id,
                material.base_color_texture_transform);
            apply_texture_info(
                asset,
                source.pbrData.metallicRoughnessTexture,
                TextureEncoding::Linear,
                base,
                textures,
                cache,
                image_cache,
                material.metallic_roughness_texture_id,
                material.metallic_roughness_texture_transform);
            if (source.specular) {
                material.specular_factor = std::clamp(
                    static_cast<float>(source.specular->specularFactor),
                    0.0f,
                    1.0f);
                material.specular_color = Color(
                    source.specular->specularColorFactor.x(),
                    source.specular->specularColorFactor.y(),
                    source.specular->specularColorFactor.z()).cwiseMax(Color::Zero());
                apply_texture_info(
                    asset,
                    source.specular->specularTexture,
                    TextureEncoding::Linear,
                    base,
                    textures,
                    cache,
                    image_cache,
                    material.specular_texture_id,
                    material.specular_texture_transform);
                apply_texture_info(
                    asset,
                    source.specular->specularColorTexture,
                    TextureEncoding::Srgb,
                    base,
                    textures,
                    cache,
                    image_cache,
                    material.specular_color_texture_id,
                    material.specular_color_texture_transform);
            }
        }
        apply_texture_info(
            asset,
            source.normalTexture,
            TextureEncoding::Linear,
            base,
            textures,
            cache,
            image_cache,
            material.normal_texture_id,
            material.normal_texture_transform);
        if (source.normalTexture) {
            material.normal_scale = source.normalTexture->scale;
        }
        apply_texture_info(
            asset,
            source.occlusionTexture,
            TextureEncoding::Linear,
            base,
            textures,
            cache,
            image_cache,
            material.occlusion_texture_id,
            material.occlusion_texture_transform);
        if (source.occlusionTexture) {
            material.occlusion_strength = source.occlusionTexture->strength;
        }
        apply_texture_info(
            asset,
            source.emissiveTexture,
            TextureEncoding::Srgb,
            base,
            textures,
            cache,
            image_cache,
            material.emissive_texture_id,
            material.emissive_texture_transform);
        result.push_back(material);
        names.push_back(source.name.empty()
            ? "Material " + std::to_string(names.size() + 1U)
            : std::string(source.name));
    }
    Material fallback;
    fallback.type = MaterialType::Pbr;
    fallback.base_color = Color(0.8f, 0.8f, 0.8f);
    fallback.roughness = 1.0f;
    result.push_back(fallback);
    names.push_back("<default>");
    return result;
}

template <typename FastVector>
Vec3 vec3(const FastVector& value) {
    return Vec3(value.x(), value.y(), value.z());
}

const fastgltf::Accessor& checked_accessor(
    const fastgltf::Asset& asset,
    std::size_t index,
    const char* semantic) {
    if (index >= asset.accessors.size()) {
        throw std::runtime_error(std::string("glTF ") + semantic + " accessor index is out of range");
    }
    return asset.accessors[index];
}

const fastgltf::Accessor& checked_vertex_accessor(
    const fastgltf::Asset& asset,
    std::size_t index,
    std::size_t vertex_count,
    const char* semantic) {
    const fastgltf::Accessor& accessor = checked_accessor(asset, index, semantic);
    if (accessor.count != vertex_count) {
        throw std::runtime_error(std::string("glTF ") + semantic +
            " accessor count does not match POSITION");
    }
    return accessor;
}

void append_primitive(
    const fastgltf::Asset& asset,
    const fastgltf::Primitive& primitive,
    int fallback_material,
    Scene& scene,
    Bounds3& bounds,
    std::vector<std::string>& warnings) {
    if (!primitive.targets.empty()) {
        throw std::runtime_error("glTF morph targets are not supported");
    }
    if (primitive.type == fastgltf::PrimitiveType::Points ||
        primitive.type == fastgltf::PrimitiveType::Lines ||
        primitive.type == fastgltf::PrimitiveType::LineLoop ||
        primitive.type == fastgltf::PrimitiveType::LineStrip) {
        warnings.push_back("Skipped a non-triangle glTF primitive.");
        return;
    }
    const auto position_attribute = primitive.findAttribute("POSITION");
    if (position_attribute == primitive.attributes.end()) {
        throw std::runtime_error("glTF primitive is missing POSITION");
    }
    const fastgltf::Accessor& position_accessor = checked_accessor(
        asset,
        position_attribute->accessorIndex,
        "POSITION");
    if (position_accessor.type != fastgltf::AccessorType::Vec3) {
        throw std::runtime_error("glTF POSITION accessor must use VEC3");
    }
    std::vector<TriangleVertex> vertices(position_accessor.count);
    fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
        asset,
        position_accessor,
        [&](const fastgltf::math::fvec3& value, std::size_t index) {
            vertices[index].position = vec3(value);
        });

    if (const auto attribute = primitive.findAttribute("NORMAL"); attribute != primitive.attributes.end()) {
        const fastgltf::Accessor& accessor = checked_vertex_accessor(
            asset,
            attribute->accessorIndex,
            vertices.size(),
            "NORMAL");
        if (accessor.type != fastgltf::AccessorType::Vec3) {
            throw std::runtime_error("glTF NORMAL accessor must use VEC3");
        }
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
            asset,
            accessor,
            [&](const fastgltf::math::fvec3& value, std::size_t index) {
                const Vec3 normal = vec3(value);
                vertices[index].normal = usable_direction(normal) ? normal.normalized() : Vec3::Zero();
                vertices[index].has_normal = usable_direction(normal);
            });
    }
    if (const auto attribute = primitive.findAttribute("TEXCOORD_0"); attribute != primitive.attributes.end()) {
        const fastgltf::Accessor& accessor = checked_vertex_accessor(
            asset,
            attribute->accessorIndex,
            vertices.size(),
            "TEXCOORD_0");
        if (accessor.type != fastgltf::AccessorType::Vec2) {
            throw std::runtime_error("glTF TEXCOORD_0 accessor must use VEC2");
        }
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
            asset,
            accessor,
            [&](const fastgltf::math::fvec2& value, std::size_t index) {
                vertices[index].uv = Vec2(value.x(), value.y());
            });
    }
    if (const auto attribute = primitive.findAttribute("TEXCOORD_1"); attribute != primitive.attributes.end()) {
        const fastgltf::Accessor& accessor = checked_vertex_accessor(
            asset,
            attribute->accessorIndex,
            vertices.size(),
            "TEXCOORD_1");
        if (accessor.type != fastgltf::AccessorType::Vec2) {
            throw std::runtime_error("glTF TEXCOORD_1 accessor must use VEC2");
        }
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec2>(
            asset,
            accessor,
            [&](const fastgltf::math::fvec2& value, std::size_t index) {
                vertices[index].uv1 = Vec2(value.x(), value.y());
                vertices[index].has_uv1 = true;
            });
    }
    if (const auto attribute = primitive.findAttribute("TANGENT"); attribute != primitive.attributes.end()) {
        const fastgltf::Accessor& accessor = checked_vertex_accessor(
            asset,
            attribute->accessorIndex,
            vertices.size(),
            "TANGENT");
        if (accessor.type != fastgltf::AccessorType::Vec4) {
            throw std::runtime_error("glTF TANGENT accessor must use VEC4");
        }
        fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
            asset,
            accessor,
            [&](const fastgltf::math::fvec4& value, std::size_t index) {
                vertices[index].tangent = Vec4(value.x(), value.y(), value.z(), value.w());
                vertices[index].has_tangent = true;
            });
    }
    if (const auto attribute = primitive.findAttribute("COLOR_0"); attribute != primitive.attributes.end()) {
        const fastgltf::Accessor& accessor = checked_vertex_accessor(
            asset,
            attribute->accessorIndex,
            vertices.size(),
            "COLOR_0");
        if (accessor.type == fastgltf::AccessorType::Vec3) {
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec3>(
                asset,
                accessor,
                [&](const fastgltf::math::fvec3& value, std::size_t index) {
                    vertices[index].color = vec3(value);
                    vertices[index].alpha = 1.0f;
                    vertices[index].has_color = true;
                });
        } else if (accessor.type == fastgltf::AccessorType::Vec4) {
            fastgltf::iterateAccessorWithIndex<fastgltf::math::fvec4>(
                asset,
                accessor,
                [&](const fastgltf::math::fvec4& value, std::size_t index) {
                    vertices[index].color = Color(value.x(), value.y(), value.z());
                    vertices[index].alpha = value.w();
                    vertices[index].has_color = true;
                });
        } else {
            throw std::runtime_error("glTF COLOR_0 accessor must use VEC3 or VEC4");
        }
    }

    std::vector<std::uint32_t> indices;
    if (primitive.indicesAccessor) {
        const fastgltf::Accessor& accessor = checked_accessor(
            asset,
            *primitive.indicesAccessor,
            "indices");
        if (accessor.type != fastgltf::AccessorType::Scalar) {
            throw std::runtime_error("glTF indices accessor must use SCALAR");
        }
        indices.resize(accessor.count);
        fastgltf::copyFromAccessor<std::uint32_t>(asset, accessor, indices.data());
    } else {
        indices.resize(vertices.size());
        for (std::size_t index = 0; index < indices.size(); ++index) {
            indices[index] = static_cast<std::uint32_t>(index);
        }
    }
    if (primitive.materialIndex && *primitive.materialIndex >= scene.materials.size() - 1U) {
        throw std::runtime_error("glTF primitive material index is out of range");
    }
    const int material_id = primitive.materialIndex
        ? static_cast<int>(*primitive.materialIndex)
        : fallback_material;
    const auto append_triangle = [&](std::uint32_t i0, std::uint32_t i1, std::uint32_t i2) {
        if (i0 >= vertices.size() || i1 >= vertices.size() || i2 >= vertices.size()) {
            throw std::runtime_error("glTF primitive index is out of range");
        }
        scene.triangles.emplace_back(vertices[i0], vertices[i1], vertices[i2], material_id);
        bounds.expand(vertices[i0].position);
        bounds.expand(vertices[i1].position);
        bounds.expand(vertices[i2].position);
    };
    if (primitive.type == fastgltf::PrimitiveType::Triangles) {
        if (indices.size() % 3U != 0U) {
            throw std::runtime_error("glTF triangle index count is not divisible by three");
        }
        for (std::size_t index = 0; index < indices.size(); index += 3U) {
            append_triangle(indices[index], indices[index + 1U], indices[index + 2U]);
        }
    } else if (primitive.type == fastgltf::PrimitiveType::TriangleStrip) {
        for (std::size_t index = 2; index < indices.size(); ++index) {
            if ((index & 1U) == 0U) {
                append_triangle(indices[index - 2U], indices[index - 1U], indices[index]);
            } else {
                append_triangle(indices[index - 1U], indices[index - 2U], indices[index]);
            }
        }
    } else if (primitive.type == fastgltf::PrimitiveType::TriangleFan) {
        for (std::size_t index = 2; index < indices.size(); ++index) {
            append_triangle(indices[0], indices[index - 1U], indices[index]);
        }
    }
}

}  // namespace

LoadedGltfScene load_gltf_scene(
    const std::filesystem::path& path,
    int width,
    int height) {
    constexpr fastgltf::Extensions supported_extensions =
        fastgltf::Extensions::KHR_texture_transform |
        fastgltf::Extensions::KHR_lights_punctual |
        fastgltf::Extensions::KHR_mesh_quantization |
        fastgltf::Extensions::KHR_materials_pbrSpecularGlossiness |
        fastgltf::Extensions::KHR_materials_specular |
        fastgltf::Extensions::KHR_materials_emissive_strength |
        fastgltf::Extensions::KHR_materials_ior |
        fastgltf::Extensions::KHR_materials_transmission;
    fastgltf::Parser parser(supported_extensions);
    auto file = fastgltf::MappedGltfFile::FromPath(path);
    if (!file) {
        throw std::runtime_error(
            "Failed to open glTF asset '" + path.string() + "': " +
            std::string(fastgltf::getErrorMessage(file.error())));
    }
    constexpr fastgltf::Options options =
        fastgltf::Options::LoadExternalBuffers |
        fastgltf::Options::LoadExternalImages |
        fastgltf::Options::GenerateMeshIndices |
        fastgltf::Options::DecomposeNodeMatrices;
    auto parsed = parser.loadGltf(file.get(), path.parent_path(), options);
    if (!parsed) {
        throw std::runtime_error(
            "Failed to parse glTF asset '" + path.string() + "': " +
            std::string(fastgltf::getErrorMessage(parsed.error())));
    }
    fastgltf::Asset asset = std::move(parsed.get());
    const auto has_extension = [](const auto& extensions, const char* name) {
        return std::any_of(
            extensions.begin(),
            extensions.end(),
            [name](const auto& extension) { return extension == name; });
    };
    if (has_extension(asset.extensionsRequired, "KHR_materials_transmission")) {
        throw std::runtime_error(
            "glTF required extension 'KHR_materials_transmission' is not supported: "
            "transmission BSDF rendering is not implemented");
    }
    if (!asset.animations.empty() || !asset.skins.empty()) {
        throw std::runtime_error("glTF animations and skins are not supported in the static scene importer");
    }
    if (asset.scenes.empty()) {
        throw std::runtime_error("glTF asset contains no scene");
    }

    LoadedGltfScene loaded;
    if (has_extension(asset.extensionsUsed, "KHR_materials_transmission")) {
        loaded.warnings.push_back(
            "KHR_materials_transmission is optional but not rendered; using the core material fallback.");
    }
    std::vector<ImageTexture> textures;
    std::vector<std::string> material_names;
    const std::vector<Material> materials = convert_materials(
        asset,
        path.parent_path(),
        textures,
        material_names);
    loaded.meshes.reserve(asset.meshes.size());
    for (std::size_t mesh_index = 0; mesh_index < asset.meshes.size(); ++mesh_index) {
        LoadedScene mesh{
            Scene(),
            Camera(Vec3(0.0f, 0.0f, 1.0f), Vec3::Zero(), Vec3(0.0f, 1.0f, 0.0f), 45.0f, 1.0f),
            Bounds3()};
        mesh.scene.materials = materials;
        mesh.scene.textures = textures;
        mesh.material_names = material_names;
        for (const fastgltf::Primitive& primitive : asset.meshes[mesh_index].primitives) {
            append_primitive(
                asset,
                primitive,
                static_cast<int>(materials.size() - 1U),
                mesh.scene,
                mesh.bounds,
                mesh.warnings);
        }
        if (mesh.scene.triangles.empty()) {
            mesh.warnings.push_back("glTF mesh contains no supported triangle primitives.");
            mesh.bounds = Bounds3(
                Vec3(-0.5f, -0.5f, -0.5f),
                Vec3(0.5f, 0.5f, 0.5f));
        }
        mesh.scene.environment = Color(0.02f, 0.025f, 0.03f);
        mesh.camera = default_camera(mesh.bounds, width, height);
        loaded.meshes.push_back(std::move(mesh));
    }

    loaded.cameras.reserve(asset.cameras.size());
    for (const fastgltf::Camera& camera : asset.cameras) {
        GltfCameraAsset result;
        result.name = camera.name.empty() ? "Camera" : std::string(camera.name);
        std::visit(fastgltf::visitor{
            [&](const fastgltf::Camera::Perspective& source) {
                result.vertical_fov_radians = source.yfov;
                result.aspect_ratio = source.aspectRatio.value_or(0.0f);
                result.near_plane = source.znear;
                result.far_plane = source.zfar.value_or(100000.0f);
            },
            [&](const fastgltf::Camera::Orthographic& source) {
                result.orthographic = true;
                result.x_magnification = source.xmag;
                result.y_magnification = source.ymag;
                result.near_plane = source.znear;
                result.far_plane = source.zfar;
            }}, camera.camera);
        loaded.cameras.push_back(result);
    }

    const std::size_t scene_index = asset.defaultScene.value_or(0U);
    if (scene_index >= asset.scenes.size()) {
        throw std::runtime_error("glTF default scene index is out of range");
    }
    std::unordered_set<std::size_t> visiting;
    std::unordered_set<std::size_t> visited;
    std::function<void(std::size_t, int)> visit_node = [&](std::size_t node_index, int parent) {
        if (node_index >= asset.nodes.size()) {
            throw std::runtime_error("glTF node index is out of range");
        }
        if (visiting.contains(node_index)) {
            throw std::runtime_error("glTF node hierarchy contains a cycle");
        }
        if (!visited.insert(node_index).second) {
            throw std::runtime_error("glTF node hierarchy references the same node more than once");
        }
        visiting.insert(node_index);
        const fastgltf::Node& source = asset.nodes[node_index];
        if (source.skinIndex) {
            throw std::runtime_error("glTF skinned nodes are not supported");
        }
        GltfNodeAsset node;
        node.name = source.name.empty() ? "Node " + std::to_string(node_index) : std::string(source.name);
        node.mesh_index = source.meshIndex ? static_cast<int>(*source.meshIndex) : -1;
        if (node.mesh_index >= 0 && static_cast<std::size_t>(node.mesh_index) >= asset.meshes.size()) {
            throw std::runtime_error("glTF node mesh index is out of range");
        }
        node.parent_index = parent;
        node.local_transform = convert_matrix(fastgltf::getTransformMatrix(source));
        node.camera_index = source.cameraIndex ? static_cast<int>(*source.cameraIndex) : -1;
        if (node.camera_index >= 0 &&
            static_cast<std::size_t>(node.camera_index) >= asset.cameras.size()) {
            throw std::runtime_error("glTF node camera index is out of range");
        }
        if (source.lightIndex) {
            if (*source.lightIndex >= asset.lights.size()) {
                throw std::runtime_error("glTF punctual light index is out of range");
            }
            const fastgltf::Light& light = asset.lights[*source.lightIndex];
            node.light_color = Color(light.color.x(), light.color.y(), light.color.z());
            node.light_intensity = light.intensity;
            node.light_range = light.range.value_or(0.0f);
            switch (light.type) {
                case fastgltf::LightType::Directional:
                    node.light_type = GltfNodeAsset::LightType::Directional;
                    break;
                case fastgltf::LightType::Point:
                    node.light_type = GltfNodeAsset::LightType::Point;
                    break;
                case fastgltf::LightType::Spot:
                    node.light_type = GltfNodeAsset::LightType::Spot;
                    node.spot_inner_cone_radians = light.innerConeAngle.value_or(0.0f);
                    node.spot_outer_cone_radians = light.outerConeAngle.value_or(0.7853981634f);
                    break;
            }
        }
        const int output_index = static_cast<int>(loaded.nodes.size());
        loaded.nodes.push_back(std::move(node));
        for (const std::size_t child : source.children) {
            visit_node(child, output_index);
        }
        visiting.erase(node_index);
    };
    for (const std::size_t root : asset.scenes[scene_index].nodeIndices) {
        visit_node(root, -1);
    }
    if (loaded.nodes.empty()) {
        throw std::runtime_error("glTF default scene contains no nodes");
    }
    return loaded;
}

}  // namespace renderer
