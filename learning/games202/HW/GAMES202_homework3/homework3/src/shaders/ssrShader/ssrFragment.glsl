#ifdef GL_ES
precision highp float;
#endif

uniform vec3 uLightDir;
uniform vec3 uCameraPos;
uniform vec3 uLightRadiance;
uniform sampler2D uGDiffuse;
uniform sampler2D uGDepth;
uniform sampler2D uGNormalWorld;
uniform sampler2D uGShadow;
uniform sampler2D uGPosWorld;
uniform int uRenderMode;
uniform int uSampleCount;
uniform int uRayMarchSteps;

varying mat4 vWorldToScreen;
varying highp vec4 vPosWorld;

#define M_PI 3.1415926535897932384626433832795
#define TWO_PI 6.283185307
#define INV_PI 0.31830988618
#define INV_TWO_PI 0.15915494309

const int MAX_RAY_MARCH_STEPS = 128;
const int RAY_REFINEMENT_STEPS = 6;
const float RAY_MIN_DISTANCE = 0.05;
const float RAY_MAX_DISTANCE = 12.0;
const float RAY_MIN_STEP = 0.04;
const float RAY_MAX_STEP = 0.16;
const float RAY_ORIGIN_BIAS = 0.02;
const float RAY_MIN_THICKNESS = 0.03;

float Rand1(inout float p) {
  p = fract(p * .1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}

vec2 Rand2(inout float p) {
  return vec2(Rand1(p), Rand1(p));
}

float InitRand(vec2 uv) {
	vec3 p3  = fract(vec3(uv.xyx) * .1031);
  p3 += dot(p3, p3.yzx + 33.33);
  return fract((p3.x + p3.y) * p3.z);
}

vec3 SampleHemisphereUniform(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = uv.x;
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(1.0 - z*z);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = INV_TWO_PI;
  return dir;
}

vec3 SampleHemisphereCos(inout float s, out float pdf) {
  vec2 uv = Rand2(s);
  float z = sqrt(1.0 - uv.x);
  float phi = uv.y * TWO_PI;
  float sinTheta = sqrt(uv.x);
  vec3 dir = vec3(sinTheta * cos(phi), sinTheta * sin(phi), z);
  pdf = z * INV_PI;
  return dir;
}

void LocalBasis(vec3 n, out vec3 b1, out vec3 b2) {
  float sign_ = sign(n.z);
  if (n.z == 0.0) {
    sign_ = 1.0;
  }
  float a = -1.0 / (sign_ + n.z);
  float b = n.x * n.y * a;
  b1 = vec3(1.0 + sign_ * n.x * n.x * a, sign_ * b, -sign_ * n.x);
  b2 = vec3(b, sign_ + n.y * n.y * a, -n.y);
}

vec4 Project(vec4 a) {
  return a / a.w;
}

float GetDepth(vec3 posWorld) {
  float depth = (vWorldToScreen * vec4(posWorld, 1.0)).w;
  return depth;
}

/*
 * Transform point from world space to screen space([0, 1] x [0, 1])
 *
 */
vec2 GetScreenCoordinate(vec3 posWorld) {
  vec2 uv = Project(vWorldToScreen * vec4(posWorld, 1.0)).xy * 0.5 + 0.5;
  return uv;
}

float GetGBufferDepth(vec2 uv) {
  float depth = texture2D(uGDepth, uv).x;
  if (depth < 1e-2) {
    depth = 1000.0;
  }
  return depth;
}

vec3 GetGBufferNormalWorld(vec2 uv) {
  vec3 normal = texture2D(uGNormalWorld, uv).xyz;
  return normal;
}

vec3 GetGBufferPosWorld(vec2 uv) {
  vec3 posWorld = texture2D(uGPosWorld, uv).xyz;
  return posWorld;
}

float GetGBufferuShadow(vec2 uv) {
  float visibility = texture2D(uGShadow, uv).x;
  return visibility;
}

vec3 GetGBufferDiffuse(vec2 uv) {
  vec3 diffuse = texture2D(uGDiffuse, uv).xyz;
  diffuse = pow(diffuse, vec3(2.2));
  return diffuse;
}

/*
 * Evaluate diffuse bsdf value.
 *
 * wi, wo are all in world space.
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDiffuse(vec3 wi, vec3 wo, vec2 uv) {
  vec3 n = normalize(GetGBufferNormalWorld(uv));
  if (dot(n, wi) <= 0.0 || dot(n, wo) <= 0.0) {
    return vec3(0.0);
  }
  return GetGBufferDiffuse(uv) * INV_PI;
}

/*
 * Evaluate directional light with shadow map
 * uv is in screen space, [0, 1] x [0, 1].
 *
 */
vec3 EvalDirectionalLight(vec2 uv) {
  return uLightRadiance * clamp(GetGBufferuShadow(uv), 0.0, 1.0);
}

bool InsideScreen(vec2 uv) {
  return all(greaterThanEqual(uv, vec2(0.0))) &&
         all(lessThanEqual(uv, vec2(1.0)));
}

bool RayMarch(vec3 ori, vec3 dir, out vec3 hitPos) {
  dir = normalize(dir);
  float travel = RAY_MIN_DISTANCE;
  vec3 previousPos = ori + dir * travel;
  vec4 previousClip = vWorldToScreen * vec4(previousPos, 1.0);
  if (previousClip.w <= 0.0) {
    return false;
  }

  vec2 previousUv = Project(previousClip).xy * 0.5 + 0.5;
  if (!InsideScreen(previousUv)) {
    return false;
  }

  float previousSceneDepth = GetGBufferDepth(previousUv);
  bool previousValid = previousSceneDepth < 999.0;
  float previousDelta = previousClip.w - previousSceneDepth;

  for (int i = 0; i < MAX_RAY_MARCH_STEPS; ++i) {
    if (i >= uRayMarchSteps) {
      break;
    }
    float stepLength = mix(
        RAY_MIN_STEP,
        RAY_MAX_STEP,
        clamp(travel / RAY_MAX_DISTANCE, 0.0, 1.0));
    travel += stepLength;
    if (travel > RAY_MAX_DISTANCE) {
      break;
    }

    vec3 rayPos = ori + dir * travel;
    vec4 rayClip = vWorldToScreen * vec4(rayPos, 1.0);
    if (rayClip.w <= 0.0) {
      return false;
    }

    vec2 uv = Project(rayClip).xy * 0.5 + 0.5;
    if (!InsideScreen(uv)) {
      return false;
    }

    float sceneDepth = GetGBufferDepth(uv);
    bool currentValid = sceneDepth < 999.0;
    float depthDelta = rayClip.w - sceneDepth;

    // A hit must cross the visible surface from its camera-facing side.
    // Merely accepting a small absolute depth difference produces many
    // false hits around silhouettes and at coarse step sizes.
    if (previousValid && currentValid &&
        previousDelta < 0.0 && depthDelta >= 0.0) {
      vec3 nearPos = previousPos;
      vec3 farPos = rayPos;
      vec2 refinedUv = uv;
      float refinedDelta = depthDelta;

      for (int refine = 0; refine < RAY_REFINEMENT_STEPS; ++refine) {
        vec3 midpoint = (nearPos + farPos) * 0.5;
        vec4 midpointClip = vWorldToScreen * vec4(midpoint, 1.0);
        if (midpointClip.w <= 0.0) {
          farPos = midpoint;
          continue;
        }

        vec2 midpointUv = Project(midpointClip).xy * 0.5 + 0.5;
        if (!InsideScreen(midpointUv)) {
          nearPos = midpoint;
          continue;
        }

        float midpointSceneDepth = GetGBufferDepth(midpointUv);
        if (midpointSceneDepth >= 999.0) {
          nearPos = midpoint;
          continue;
        }

        float midpointDelta = midpointClip.w - midpointSceneDepth;
        if (midpointDelta >= 0.0) {
          farPos = midpoint;
          refinedUv = midpointUv;
          refinedDelta = midpointDelta;
        } else {
          nearPos = midpoint;
        }
      }

      float thickness = max(
          RAY_MIN_THICKNESS,
          GetGBufferDepth(refinedUv) * 0.005);
      vec3 hitNormal = normalize(GetGBufferNormalWorld(refinedUv));
      if (refinedDelta <= thickness && dot(hitNormal, dir) < -1e-3) {
        hitPos = GetGBufferPosWorld(refinedUv);
        return true;
      }
    }

    previousPos = rayPos;
    previousValid = currentValid;
    previousDelta = depthDelta;
  }

  return false;
}

#define MAX_SAMPLE_NUM 4

void main() {
  float s = InitRand(gl_FragCoord.xy);

  vec2 uv = GetScreenCoordinate(vPosWorld.xyz);
  vec3 position = GetGBufferPosWorld(uv);
  vec3 normal = normalize(GetGBufferNormalWorld(uv));
  vec3 wo = normalize(uCameraPos - position);
  vec3 lightDir = normalize(uLightDir);

  vec3 direct = EvalDiffuse(lightDir, wo, uv) *
      EvalDirectionalLight(uv) * max(dot(normal, lightDir), 0.0);

  if (uRenderMode == 3) {
    vec3 reflectionDir = reflect(-wo, normal);
    vec3 reflectionHit;
    vec3 reflectedAlbedo = vec3(0.0);
    if (RayMarch(
        position + normal * RAY_ORIGIN_BIAS,
        reflectionDir,
        reflectionHit)) {
      reflectedAlbedo = GetGBufferDiffuse(
          GetScreenCoordinate(reflectionHit));
    }
    vec3 debugColor = pow(
        clamp(reflectedAlbedo, vec3(0.0), vec3(1.0)),
        vec3(1.0 / 2.2));
    gl_FragColor = vec4(debugColor, 1.0);
    return;
  }

  vec3 indirect = vec3(0.0);
  if (uRenderMode != 1) {
    vec3 tangent;
    vec3 bitangent;
    LocalBasis(normal, tangent, bitangent);

    for (int i = 0; i < MAX_SAMPLE_NUM; ++i) {
      if (i >= uSampleCount) {
        break;
      }
      float pdf;
      vec3 localDir = SampleHemisphereCos(s, pdf);
      vec3 sampleDir = normalize(
          localDir.x * tangent +
          localDir.y * bitangent +
          localDir.z * normal);

      vec3 hitPos;
      if (RayMarch(
          position + normal * RAY_ORIGIN_BIAS,
          sampleDir,
          hitPos)) {
        vec2 hitUv = GetScreenCoordinate(hitPos);
        vec3 hitNormal = normalize(GetGBufferNormalWorld(hitUv));
        vec3 hitDirect = EvalDiffuse(lightDir, -sampleDir, hitUv) *
            EvalDirectionalLight(hitUv) *
            max(dot(hitNormal, lightDir), 0.0);
        vec3 bsdf = EvalDiffuse(sampleDir, wo, uv);
        float cosine = max(dot(normal, sampleDir), 0.0);
        indirect += bsdf * hitDirect * cosine / max(pdf, 1e-4);
      }
    }
    indirect /= max(float(uSampleCount), 1.0);
  }

  vec3 L = direct + indirect;
  if (uRenderMode == 1) {
    L = direct;
  } else if (uRenderMode == 2) {
    L = indirect;
  }
  vec3 color = pow(clamp(L, vec3(0.0), vec3(1.0)), vec3(1.0 / 2.2));
  gl_FragColor = vec4(color, 1.0);
}
