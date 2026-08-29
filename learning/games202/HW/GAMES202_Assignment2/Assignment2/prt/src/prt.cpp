#include <nori/integrator.h>
#include <nori/scene.h>
#include <nori/ray.h>
#include <filesystem/resolver.h>
#include <sh/spherical_harmonics.h>
#include <sh/default_image.h>
#include <Eigen/Core>
#include <algorithm>
#include <fstream>
#include <random>
#include <stb_image.h>

NORI_NAMESPACE_BEGIN

namespace ProjEnv
{
    std::vector<std::unique_ptr<float[]>>
    LoadCubemapImages(const std::string &cubemapDir, int &width, int &height,
                      int &channel)
    {
        std::vector<std::string> cubemapNames{"negx.jpg", "posx.jpg", "posy.jpg",
                                              "negy.jpg", "posz.jpg", "negz.jpg"};
        std::vector<std::unique_ptr<float[]>> images(6);
        for (int i = 0; i < 6; i++)
        {
            std::string filename = cubemapDir + "/" + cubemapNames[i];
            int w, h, c;
            float *image = stbi_loadf(filename.c_str(), &w, &h, &c, 3);
            if (!image)
            {
                std::cout << "Failed to load image: " << filename << std::endl;
                exit(-1);
            }
            if (i == 0)
            {
                width = w;
                height = h;
                channel = c;
            }
            else if (w != width || h != height || c != channel)
            {
                std::cout << "Dismatch resolution for 6 images in cubemap" << std::endl;
                exit(-1);
            }
            images[i] = std::unique_ptr<float[]>(image);
            int index = (0 * 128 + 0) * channel;
            // std::cout << images[i][index + 0] << "\t" << images[i][index + 1] << "\t"
            //           << images[i][index + 2] << std::endl;
        }
        return images;
    }

    const Eigen::Vector3f cubemapFaceDirections[6][3] = {
        {{0, 0, 1}, {0, -1, 0}, {-1, 0, 0}},  // negx
        {{0, 0, 1}, {0, -1, 0}, {1, 0, 0}},   // posx
        {{1, 0, 0}, {0, 0, -1}, {0, -1, 0}},  // negy
        {{1, 0, 0}, {0, 0, 1}, {0, 1, 0}},    // posy
        {{-1, 0, 0}, {0, -1, 0}, {0, 0, -1}}, // negz
        {{1, 0, 0}, {0, -1, 0}, {0, 0, 1}},   // posz
    };

    float CalcPreArea(const float &x, const float &y)
    {
        return std::atan2(x * y, std::sqrt(x * x + y * y + 1.0));
    }

    float CalcArea(const float &u_, const float &v_, const int &width,
                   const int &height)
    {
        // transform from [0..res - 1] to [- (1 - 1 / res) .. (1 - 1 / res)]
        // ( 0.5 is for texel center addressing)
        float u = (2.0 * (u_ + 0.5) / width) - 1.0;
        float v = (2.0 * (v_ + 0.5) / height) - 1.0;

        // shift from a demi texel, mean 1.0 / size  with u and v in [-1..1]
        float invResolutionW = 1.0 / width;
        float invResolutionH = 1.0 / height;

        // u and v are the -1..1 texture coordinate on the current face.
        // get projected area for this texel
        float x0 = u - invResolutionW;
        float y0 = v - invResolutionH;
        float x1 = u + invResolutionW;
        float y1 = v + invResolutionH;
        float angle = CalcPreArea(x0, y0) - CalcPreArea(x0, y1) -
                      CalcPreArea(x1, y0) + CalcPreArea(x1, y1);

        return angle;
    }

    // template <typename T> T ProjectSH() {}

    template <size_t SHOrder>
    std::vector<Eigen::Array3f> PrecomputeCubemapSH(const std::vector<std::unique_ptr<float[]>> &images,
                                                    const int &width, const int &height,
                                                    const int &channel)
    {
        std::vector<Eigen::Vector3f> cubemapDirs;
        cubemapDirs.reserve(6 * width * height);
        for (int i = 0; i < 6; i++)
        {
            Eigen::Vector3f faceDirX = cubemapFaceDirections[i][0];
            Eigen::Vector3f faceDirY = cubemapFaceDirections[i][1];
            Eigen::Vector3f faceDirZ = cubemapFaceDirections[i][2];
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float u = 2 * ((x + 0.5) / width) - 1;
                    float v = 2 * ((y + 0.5) / height) - 1;
                    Eigen::Vector3f dir = (faceDirX * u + faceDirY * v + faceDirZ).normalized();
                    cubemapDirs.push_back(dir);
                }
            }
        }
        constexpr int SHNum = (SHOrder + 1) * (SHOrder + 1);
        std::vector<Eigen::Array3f> SHCoeffiecents(SHNum);
        for (int i = 0; i < SHNum; i++)
            SHCoeffiecents[i] = Eigen::Array3f(0);
        std::vector<float> texelWeights(width * height);
        float sumWeight = 0;
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                const int texelIndex = y * width + x;
                const float weight = CalcArea(static_cast<float>(x), static_cast<float>(y), width, height);
                texelWeights[texelIndex] = weight;
                sumWeight += 6.0f * weight;
            }
        }
        for (int i = 0; i < 6; i++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Eigen::Vector3f dir = cubemapDirs[i * width * height + y * width + x];
                    int index = (y * width + x) * channel;
                    Eigen::Array3f Le(images[i][index + 0], images[i][index + 1],
                                      images[i][index + 2]);
                    const float weight = texelWeights[y * width + x];
                    const Eigen::Vector3d dirD = dir.cast<double>();
                    for (int l = 0; l <= static_cast<int>(SHOrder); ++l)
                    {
                        for (int m = -l; m <= l; ++m)
                        {
                            const int coeffIndex = sh::GetIndex(l, m);
                            const float basis = static_cast<float>(sh::EvalSH(l, m, dirD));
                            SHCoeffiecents[coeffIndex] += Le * (basis * weight);
                        }
                    }
                }
            }
        }
        const float normalization = static_cast<float>(4.0 * M_PI) / sumWeight;
        for (Eigen::Array3f &coefficient : SHCoeffiecents)
            coefficient *= normalization;
        return SHCoeffiecents;
    }
}

class PRTIntegrator : public Integrator
{
public:
    static constexpr int SHOrder = 2;
    static constexpr int SHCoeffLength = (SHOrder + 1) * (SHOrder + 1);

    enum class Type
    {
        Unshadowed = 0,
        Shadowed = 1,
        Interreflection = 2
    };

    PRTIntegrator(const PropertyList &props)
    {
        /* No parameters this time */
        m_SampleCount = props.getInteger("PRTSampleCount", 100);
        if (m_SampleCount <= 0)
            throw NoriException("PRTSampleCount must be greater than zero.");

        m_CubemapPath = props.getString("cubemap");
        auto type = props.getString("type", "unshadowed");
        if (type == "unshadowed")
        {
            m_Type = Type::Unshadowed;
        }
        else if (type == "shadowed")
        {
            m_Type = Type::Shadowed;
        }
        else if (type == "interreflection")
        {
            m_Type = Type::Interreflection;
            m_Bounce = props.getInteger("bounce", 1);
            if (m_Bounce < 0)
                throw NoriException("The interreflection bounce count cannot be negative.");
        }
        else
        {
            throw NoriException("Unsupported type: %s.", type);
        }
    }

    virtual void preprocess(const Scene *scene) override
    {
        if (scene->getMeshes().empty())
            throw NoriException("PRT preprocessing requires at least one mesh.");

        // The assignment's output format contains transport for the first mesh only.
        const auto mesh = scene->getMeshes()[0];
        if (mesh->getVertexNormals().cols() != mesh->getVertexPositions().cols())
            throw NoriException("PRT preprocessing requires one normal per mesh vertex.");
        // Projection environment
        auto cubePath = getFileResolver()->resolve(m_CubemapPath);
        auto lightPath = cubePath / "light.txt";
        auto transPath = cubePath / "transport.txt";
        std::ofstream lightFout(lightPath.str());
        std::ofstream fout(transPath.str());
        int width, height, channel;
        std::vector<std::unique_ptr<float[]>> images =
            ProjEnv::LoadCubemapImages(cubePath.str(), width, height, channel);
        auto envCoeffs = ProjEnv::PrecomputeCubemapSH<SHOrder>(images, width, height, channel);
        m_LightCoeffs.resize(3, SHCoeffLength);
        for (int i = 0; i < envCoeffs.size(); i++)
        {
            lightFout << (envCoeffs)[i].x() << " " << (envCoeffs)[i].y() << " " << (envCoeffs)[i].z() << std::endl;
            m_LightCoeffs.col(i) = (envCoeffs)[i];
        }
        std::cout << "Computed light sh coeffs from: " << cubePath.str() << " to: " << lightPath.str() << std::endl;
        // Projection transport
        m_TransportSHCoeffs.resize(SHCoeffLength, mesh->getVertexCount());
        fout << mesh->getVertexCount() << std::endl;
        for (int i = 0; i < mesh->getVertexCount(); i++)
        {
            const Point3f &v = mesh->getVertexPositions().col(i);
            // Normalize again so imported/scaled normals cannot distort the cosine term.
            const Normal3f n = mesh->getVertexNormals().col(i).normalized();
            auto shFunc = [&](double phi, double theta) -> double {
                // ProjectFunction samples (phi, theta) on the unit sphere.
                // Convert the sample to a world-space incident direction.
                const Eigen::Vector3d d = sh::ToVector(phi, theta);
                const Vector3f wi = d.cast<float>();

                // Lambertian transport receives light only from the normal-facing hemisphere.
                const double cosine = std::max(0.0, static_cast<double>(n.dot(wi)));
                if (cosine <= 0.0)
                    return 0.0;

                if (m_Type == Type::Unshadowed)
                {
                    // Unshadowed PRT ignores visibility; its transfer is max(n dot wi, 0).
                    return cosine;
                }
                else
                {
                    // Shadowed and interreflection modes both start with direct visibility.
                    // Ray3f uses Epsilon as mint, which avoids the surface at the ray origin.
                    const Ray3f visibilityRay(v, wi);
                    const bool blocked = scene->rayIntersect(visibilityRay);
                    return blocked ? 0.0 : cosine;
                }
            };
            auto shCoeff = sh::ProjectFunction(SHOrder, shFunc, m_SampleCount);
            for (int j = 0; j < shCoeff->size(); j++)
            {
                m_TransportSHCoeffs.col(i).coeffRef(j) = (*shCoeff)[j];
            }
        }
        if (m_Type == Type::Interreflection)
        {
            /*
             * Iterate indirect bounces:
             *   1. previousBounce stores transport arriving in exactly the prior bounce.
             *   2. Sample directions and retain upper-hemisphere rays that hit this mesh.
             *   3. Barycentrically interpolate prior-bounce SH coefficients at the hit.
             *   4. Weight by the source vertex cosine and integrate over directions.
             *
             * Propagating only previousBounce prevents older paths from being counted twice.
             */
            const int sampleSide = static_cast<int>(std::floor(std::sqrt(m_SampleCount)));
            const int actualSampleCount = sampleSide * sampleSide;
            const float sphereSampleWeight =
                static_cast<float>(4.0 * M_PI / actualSampleCount);
            const float lambertFactor = 1.0f / static_cast<float>(M_PI);

            Eigen::MatrixXf previousBounce = m_TransportSHCoeffs;
            for (int bounce = 0; bounce < m_Bounce; ++bounce)
            {
                Eigen::MatrixXf nextBounce =
                    Eigen::MatrixXf::Zero(SHCoeffLength, mesh->getVertexCount());

                for (int vertexIndex = 0; vertexIndex < mesh->getVertexCount(); ++vertexIndex)
                {
                    const Point3f &v = mesh->getVertexPositions().col(vertexIndex);
                    const Normal3f n = mesh->getVertexNormals().col(vertexIndex).normalized();

                    // A stable per-vertex/per-bounce seed makes precomputation reproducible.
                    std::seed_seq seed{0x505254, bounce, vertexIndex};
                    std::mt19937 generator(seed);
                    std::uniform_real_distribution<double> uniform(0.0, 1.0);

                    for (int t = 0; t < sampleSide; ++t)
                    {
                        for (int p = 0; p < sampleSide; ++p)
                        {
                            // Use the same stratified uniform-sphere sampling as ProjectFunction.
                            const double alpha = (t + uniform(generator)) / sampleSide;
                            const double beta = (p + uniform(generator)) / sampleSide;
                            const double phi = 2.0 * M_PI * beta;
                            const double theta = std::acos(2.0 * alpha - 1.0);
                            const Vector3f wi = sh::ToVector(phi, theta).cast<float>();

                            const float cosine = std::max(0.0f, n.dot(wi));
                            if (cosine <= 0.0f)
                                continue;

                            Intersection hit;
                            if (!scene->rayIntersect(Ray3f(v, wi), hit) || hit.mesh != mesh)
                                continue;

                            // Intersection::tri_index stores the hit triangle's vertex indices.
                            const Eigen::Index idx0 = static_cast<Eigen::Index>(hit.tri_index.x());
                            const Eigen::Index idx1 = static_cast<Eigen::Index>(hit.tri_index.y());
                            const Eigen::Index idx2 = static_cast<Eigen::Index>(hit.tri_index.z());
                            if (idx0 < 0 || idx1 < 0 || idx2 < 0 ||
                                idx0 >= previousBounce.cols() ||
                                idx1 >= previousBounce.cols() ||
                                idx2 >= previousBounce.cols())
                                continue;

                            // Interpolate prior-bounce transport at the exact hit point.
                            const Eigen::Matrix<float, SHCoeffLength, 1> hitCoeffs =
                                hit.bary.x() * previousBounce.col(idx0) +
                                hit.bary.y() * previousBounce.col(idx1) +
                                hit.bary.z() * previousBounce.col(idx2);
                            nextBounce.col(vertexIndex) += cosine * hitCoeffs;
                        }
                    }
                }

                // Uniform-sphere MC weight is 4*pi/N; 1/pi is a unit-albedo Lambert BRDF.
                nextBounce *= sphereSampleWeight * lambertFactor;
                m_TransportSHCoeffs += nextBounce;
                previousBounce = std::move(nextBounce);
            }
        }

        // Save in face format
        for (int f = 0; f < mesh->getTriangleCount(); f++)
        {
            const MatrixXu &F = mesh->getIndices();
            uint32_t idx0 = F(0, f), idx1 = F(1, f), idx2 = F(2, f);
            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx0).coeff(j) << " ";
            }
            fout << std::endl;
            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx1).coeff(j) << " ";
            }
            fout << std::endl;
            for (int j = 0; j < SHCoeffLength; j++)
            {
                fout << m_TransportSHCoeffs.col(idx2).coeff(j) << " ";
            }
            fout << std::endl;
        }
        std::cout << "Computed SH coeffs"
                  << " to: " << transPath.str() << std::endl;
    }

    Color3f Li(const Scene *scene, Sampler *sampler, const Ray3f &ray) const
    {
        Intersection its;
        if (!scene->rayIntersect(ray, its))
            return Color3f(0.0f);

        const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> sh0 = m_TransportSHCoeffs.col(its.tri_index.x()),
                                                                sh1 = m_TransportSHCoeffs.col(its.tri_index.y()),
                                                                sh2 = m_TransportSHCoeffs.col(its.tri_index.z());
        const Eigen::Matrix<Vector3f::Scalar, SHCoeffLength, 1> rL = m_LightCoeffs.row(0), gL = m_LightCoeffs.row(1), bL = m_LightCoeffs.row(2);

        Color3f c0 = Color3f(rL.dot(sh0), gL.dot(sh0), bL.dot(sh0)),
                c1 = Color3f(rL.dot(sh1), gL.dot(sh1), bL.dot(sh1)),
                c2 = Color3f(rL.dot(sh2), gL.dot(sh2), bL.dot(sh2));

        const Vector3f &bary = its.bary;
        Color3f c = bary.x() * c0 + bary.y() * c1 + bary.z() * c2;
        // Interpolate the three shaded vertex colors at the camera-ray hit point.
        return c;
    }

    std::string toString() const
    {
        return "PRTIntegrator[]";
    }

private:
    Type m_Type;
    int m_Bounce = 1;
    int m_SampleCount = 100;
    std::string m_CubemapPath;
    Eigen::MatrixXf m_TransportSHCoeffs;
    Eigen::MatrixXf m_LightCoeffs;
};

NORI_REGISTER_CLASS(PRTIntegrator, "prt");
NORI_NAMESPACE_END
