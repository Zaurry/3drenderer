attribute vec3 aVertexPosition;
attribute vec3 aNormalPosition;
attribute vec2 aTextureCoord;

uniform mat4 uModelMatrix;
uniform mat4 uViewMatrix;
uniform mat4 uProjectionMatrix;
uniform mat4 uLightMVP0;
uniform mat4 uLightMVP1;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;
varying highp vec4 vPositionFromLight0;
varying highp vec4 vPositionFromLight1;

void main(void) {

  vFragPos = (uModelMatrix * vec4(aVertexPosition, 1.0)).xyz;
  // 当前模型只使用均匀缩放，mat3(model) 足以同步动态旋转后的法线。
  vNormal = mat3(uModelMatrix) * aNormalPosition;

  gl_Position = uProjectionMatrix * uViewMatrix * uModelMatrix *
                vec4(aVertexPosition, 1.0);

  vTextureCoord = aTextureCoord;
  vPositionFromLight0 =
      uLightMVP0 * vec4(aVertexPosition, 1.0);
  vPositionFromLight1 =
      uLightMVP1 * vec4(aVertexPosition, 1.0);
}
