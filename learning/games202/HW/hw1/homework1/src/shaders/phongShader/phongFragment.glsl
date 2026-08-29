#ifdef GL_ES
precision highp float;
#endif

// Blinn-Phong 光照所需参数。
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uCameraPos;
uniform int uLightCount;

uniform vec3 uLightPos0;
uniform vec3 uLightIntensity0;
uniform vec3 uLightPos1;
uniform vec3 uLightIntensity1;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow Map / PCF / PCSS 所需参数。
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586
#define GOLDEN_ANGLE 2.399963229728653

uniform sampler2D uShadowMap0;
uniform sampler2D uShadowMap1;
uniform float uShadowMapSize;
uniform float uLightNearPlane0;
uniform float uLightFarPlane0;
uniform float uLightFrustumWidth0;
uniform float uLightRadius0;
uniform float uLightNearPlane1;
uniform float uLightFarPlane1;
uniform float uLightFrustumWidth1;
uniform float uLightRadius1;
uniform float uReceiveShadow;
uniform int uShadowMode;

varying highp vec4 vPositionFromLight0;
varying highp vec4 vPositionFromLight1;

highp float rand_2to1(vec2 uv) {
  const highp float a = 12.9898;
  const highp float b = 78.233;
  const highp float c = 43758.5453;
  highp float dt = dot(uv.xy, vec2(a, b));
  highp float sn = mod(dt, PI);
  return fract(sin(sn) * c);
}

/*
 * 与 shadowFragment.glsl::pack() 配套，将 RGBA8 四通道恢复成一个
 * [0, 1] 深度值。
 */
float unpack(vec4 rgbaDepth) {
  const vec4 bitShift = vec4(
      1.0,
      1.0 / 256.0,
      1.0 / (256.0 * 256.0),
      1.0 / (256.0 * 256.0 * 256.0)
  );
  return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];

/*
 * 使用黄金角螺旋生成 20 个圆盘样本。
 *
 * 原框架的角步长恰好是 π，样本只落在两条射线上。这里保留已验证兼容
 * WebGL1 的动态数组写法，只把角步长改为黄金角，避免短周期和条纹。
 */
void poissonDiskSamples(const in vec2 randomSeed) {
  float inverseSampleCount = 1.0 / float(NUM_SAMPLES);
  float angle = rand_2to1(randomSeed) * PI2;
  float radius = inverseSampleCount;

  for (int i = 0; i < NUM_SAMPLES; ++i) {
    poissonDisk[i] =
        vec2(cos(angle), sin(angle)) * pow(radius, 0.75);
    radius += inverseSampleCount;
    angle += GOLDEN_ANGLE;
  }
}

/*
 * 将 light clip space 坐标转换为纹理坐标。
 * 透视除法后坐标位于 NDC [-1, 1]，阴影纹理查询则使用 [0, 1]。
 */
vec3 getShadowCoord(vec4 coords) {
  vec3 shadowCoord = coords.xyz / coords.w;
  return shadowCoord * 0.5 + 0.5;
}

// 阴影贴图只覆盖单位立方体；范围外的片元不应采样纹理边缘值。
bool outsideShadowMap(vec3 shadowCoord) {
  return shadowCoord.x < 0.0 || shadowCoord.x > 1.0 ||
         shadowCoord.y < 0.0 || shadowCoord.y > 1.0 ||
         shadowCoord.z < 0.0 || shadowCoord.z > 1.0;
}

/*
 * 深度偏移用于抑制 shadow acne。
 * 表面越接近平行于入射光，插值和有限精度带来的误差越明显，因此采用
 * 与法线夹角相关的 bias，同时设置一个较小下限避免正面受光处自遮挡。
 */
float getShadowBias(vec3 lightPosition) {
  vec3 normal = normalize(vNormal);
  vec3 lightDir = normalize(lightPosition - vFragPos);
  float slope = 1.0 - max(dot(normal, lightDir), 0.0);
  return max(0.0005, 0.0025 * slope);
}

/*
 * 正交投影下，阴影贴图中的深度与光源空间距离呈线性关系。
 * PCSS 的相似三角形公式要求距离具有一致的物理单位，所以先把 [0, 1]
 * 深度还原到 [near, far] 的光源空间距离。
 */
float shadowDepthToLightDistance(
    float depth,
    float nearPlane,
    float farPlane
) {
  return mix(nearPlane, farPlane, depth);
}

/*
 * 在接收点周围搜索比接收点更靠近光源的样本，并返回平均遮挡深度。
 * 返回 -1.0 表示搜索区域内没有遮挡物。
 */
float findBlocker(
    sampler2D shadowMap,
    vec2 uv,
    float zReceiver,
    float nearPlane,
    float farPlane,
    float frustumWidth,
    float lightRadius,
    vec3 lightPosition
) {
  poissonDiskSamples(uv);

  float receiverDistance = shadowDepthToLightDistance(
      zReceiver,
      nearPlane,
      farPlane
  );
  float lightRadiusUV = lightRadius / frustumWidth;

  /*
   * 从光源到接收面的投影锥体随距离扩张。搜索范围使用世界空间距离推导，
   * 最后换算成 UV 半径，从而与阴影贴图分辨率解耦。
   */
  float searchRadius = lightRadiusUV *
      max(receiverDistance - nearPlane, 0.0) /
      max(receiverDistance, EPS);

  float blockerDepthSum = 0.0;
  float blockerCount = 0.0;
  float bias = getShadowBias(lightPosition);

  for (int i = 0; i < BLOCKER_SEARCH_NUM_SAMPLES; ++i) {
    vec2 sampleUV = uv + poissonDisk[i] * searchRadius;

    // 超出光源视锥的样本表示没有遮挡，不能让纹理边缘参与平均。
    if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
        sampleUV.y < 0.0 || sampleUV.y > 1.0) {
      continue;
    }

    float sampleDepth = unpack(texture2D(shadowMap, sampleUV));
    if (sampleDepth < zReceiver - bias) {
      blockerDepthSum += sampleDepth;
      blockerCount += 1.0;
    }
  }

  // 合法深度始终位于 [0, 1]，因此 -1 可安全用作哨兵。
  if (blockerCount < 0.5) {
    return -1.0;
  }
  return blockerDepthSum / blockerCount;
}

/*
 * Percentage Closer Filtering：
 * 先逐样本完成深度比较，再对二值可见性求平均。不能先平均深度再比较，
 * 否则阴影边缘的含义会发生变化。
 */
float PCF(
    sampler2D shadowMap,
    vec4 coords,
    float filterRadius,
    vec3 lightPosition
) {
  if (coords.w <= 0.0) {
    return 1.0;
  }

  vec3 shadowCoord = getShadowCoord(coords);
  if (outsideShadowMap(shadowCoord)) {
    return 1.0;
  }

  poissonDiskSamples(shadowCoord.xy);

  float visibility = 0.0;
  float validSampleCount = 0.0;
  float bias = getShadowBias(lightPosition);

  for (int i = 0; i < PCF_NUM_SAMPLES; ++i) {
    vec2 sampleUV =
        shadowCoord.xy + poissonDisk[i] * filterRadius;

    if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
        sampleUV.y < 0.0 || sampleUV.y > 1.0) {
      continue;
    }

    float closestDepth = unpack(texture2D(shadowMap, sampleUV));
    visibility +=
        shadowCoord.z - bias <= closestDepth ? 1.0 : 0.0;
    validSampleCount += 1.0;
  }

  return validSampleCount > 0.5
      ? visibility / validSampleCount
      : 1.0;
}

float PCSS(
    sampler2D shadowMap,
    vec4 coords,
    float nearPlane,
    float farPlane,
    float frustumWidth,
    float lightRadius,
    vec3 lightPosition
) {
  if (coords.w <= 0.0) {
    return 1.0;
  }

  vec3 shadowCoord = getShadowCoord(coords);
  if (outsideShadowMap(shadowCoord)) {
    return 1.0;
  }

  // STEP 1：搜索遮挡物并计算其平均深度。
  float averageBlockerDepth = findBlocker(
      shadowMap,
      shadowCoord.xy,
      shadowCoord.z,
      nearPlane,
      farPlane,
      frustumWidth,
      lightRadius,
      lightPosition
  );
  if (averageBlockerDepth < 0.0) {
    return 1.0;
  }

  /*
   * STEP 2：利用相似三角形估计半影半径。
   * penumbra / lightRadius = (receiver - blocker) / blocker
   */
  float receiverDistance =
      shadowDepthToLightDistance(
          shadowCoord.z,
          nearPlane,
          farPlane
      );
  float blockerDistance =
      shadowDepthToLightDistance(
          averageBlockerDepth,
          nearPlane,
          farPlane
      );
  float lightRadiusUV = lightRadius / frustumWidth;
  float penumbraRadius = lightRadiusUV *
      max(receiverDistance - blockerDistance, 0.0) /
      max(blockerDistance, nearPlane + EPS);

  /*
   * STEP 3：以估计出的半影半径执行 PCF。至少取一个 texel，避免非常近
   * 的接触阴影因半径趋近于零而受浮点噪声影响；上限仅防止异常深度让
   * 采样跨越大半张贴图。
   */
  float texelSize = 1.0 / uShadowMapSize;

  /*
   * 极少量错误 blocker 不应把过滤核放大到四分之一张贴图。上限取光源
   * 投影半径的两倍：仍允许远处阴影明显变软，但不会吞掉大片地面。
   */
  float maxPenumbraRadius = 1.5 * lightRadiusUV;
  penumbraRadius =
      clamp(penumbraRadius, texelSize, maxPenumbraRadius);
  return PCF(
      shadowMap,
      coords,
      penumbraRadius,
      lightPosition
  );
}

// 最基础的单样本 Two-Pass Shadow Map 深度比较。
float useShadowMap(
    sampler2D shadowMap,
    vec4 shadowCoord,
    vec3 lightPosition
) {
  if (shadowCoord.w <= 0.0) {
    return 1.0;
  }

  vec3 projectedCoord = getShadowCoord(shadowCoord);
  if (outsideShadowMap(projectedCoord)) {
    return 1.0;
  }

  float closestDepth =
      unpack(texture2D(shadowMap, projectedCoord.xy));
  return projectedCoord.z - getShadowBias(lightPosition) <= closestDepth
      ? 1.0
      : 0.0;
}

/*
 * 单独计算一盏灯的直接光。每盏灯只乘自己的 visibility，因此暖光被遮挡
 * 时冷光仍能照亮表面，这是多光源 Shadow Map 与单一总可见度的本质区别。
 */
vec3 evaluateLight(
    vec3 lightPosition,
    vec3 lightIntensity,
    float visibility,
    vec3 color,
    vec3 normal,
    vec3 viewDir
) {
  /*
   * 原代码使用 normalize(uLightPos) 计算方向，却又使用
   * distance(uLightPos, vFragPos) 做点光源衰减，两套模型互相矛盾。
   * 这里统一使用“片元到光源”的向量，并给平方距离设置下限。
   */
  vec3 lightVector = lightPosition - vFragPos;
  float lightDistanceSquared =
      max(dot(lightVector, lightVector), EPS);
  vec3 lightDir =
      lightVector * inversesqrt(lightDistanceSquared);

  float diff = max(dot(lightDir, normal), 0.0);
  vec3 lightAttenuation =
      lightIntensity / lightDistanceSquared;
  vec3 diffuse = diff * lightAttenuation * color;

  /*
   * 当 lightDir 与 viewDir 几乎相反时，两者之和接近零。直接 normalize
   * 会产生未定义值并污染整个 radiance，表现为模型大片变黑。
   */
  vec3 halfVector = lightDir + viewDir;
  float halfLengthSquared = dot(halfVector, halfVector);
  float spec = 0.0;
  if (diff > 0.0 && halfLengthSquared > EPS) {
    vec3 halfDir =
        halfVector * inversesqrt(halfLengthSquared);
    spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  }
  vec3 specular = uKs * lightAttenuation * spec;
  return visibility * (diffuse + specular);
}

vec3 blinnPhong(float visibility0, float visibility1) {
  vec3 color = pow(
      texture2D(uSampler, vTextureCoord).rgb,
      vec3(2.2)
  );

  float normalLengthSquared = dot(vNormal, vNormal);
  vec3 normal = normalLengthSquared > EPS
      ? vNormal * inversesqrt(normalLengthSquared)
      : vec3(0.0, 1.0, 0.0);

  vec3 viewVector = uCameraPos - vFragPos;
  float viewDistanceSquared =
      max(dot(viewVector, viewVector), EPS);
  vec3 viewDir =
      viewVector * inversesqrt(viewDistanceSquared);

  // 环境项只计算一次，直接光则逐光源累加。
  vec3 radiance = 0.08 * color;
  radiance += evaluateLight(
      uLightPos0,
      uLightIntensity0,
      visibility0,
      color,
      normal,
      viewDir
  );
  if (uLightCount > 1) {
    radiance += evaluateLight(
        uLightPos1,
        uLightIntensity1,
        visibility1,
        color,
        normal,
        viewDir
    );
  }

  return pow(radiance, vec3(1.0 / 2.2));
}

float calculateVisibility(
    sampler2D shadowMap,
    vec4 positionFromLight,
    float nearPlane,
    float farPlane,
    float frustumWidth,
    float lightRadius,
    vec3 lightPosition
) {
  if (uShadowMode == 0) {
    return useShadowMap(
        shadowMap,
        positionFromLight,
        lightPosition
    );
  }
  if (uShadowMode == 1) {
    return PCF(
        shadowMap,
        positionFromLight,
        3.0 / uShadowMapSize,
        lightPosition
    );
  }
  if (uShadowMode == 2) {
    return PCSS(
        shadowMap,
        positionFromLight,
        nearPlane,
        farPlane,
        frustumWidth,
        lightRadius,
        lightPosition
    );
  }
  return 1.0;
}

void main(void) {
  float visibility0 = 1.0;
  float visibility1 = 1.0;

  /*
   * 不接收阴影的投射物仍参与两盏灯的 Blinn-Phong 光照，只跳过 Shadow
   * Map 查询。这样不会产生模型自遮挡 acne，也不会丢失多光源效果。
   */
  if (uReceiveShadow > 0.5 && uShadowMode != 3) {
    visibility0 = calculateVisibility(
        uShadowMap0,
        vPositionFromLight0,
        uLightNearPlane0,
        uLightFarPlane0,
        uLightFrustumWidth0,
        uLightRadius0,
        uLightPos0
    );
    if (uLightCount > 1) {
      visibility1 = calculateVisibility(
          uShadowMap1,
          vPositionFromLight1,
          uLightNearPlane1,
          uLightFarPlane1,
          uLightFrustumWidth1,
          uLightRadius1,
          uLightPos1
      );
    }
  }

  vec3 phongColor = blinnPhong(visibility0, visibility1);
  gl_FragColor = vec4(phongColor, 1.0);
}
