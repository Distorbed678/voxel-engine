#version 460
#include "particles/particles_common.glsl"

layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vKind;
layout(location = 2) in float vFade;
layout(location = 3) flat in float vSeed;
layout(location = 0) out vec4 outColor;

void main() {
    FrameData f = pc.frame.d;
    float r = length(vUV * 2.0 - 1.0);
    float a = vKind == PK_RAIN ? 1.0 - abs(vUV.x * 2.0 - 1.0) : 1.0 - smoothstep(0.45, 1.0, r);
    vec3 skyAmb = f.skyZenith.rgb * 1.2 + f.skyHorizon.rgb * 0.3;
    vec3 lit = skyAmb * 0.6 + f.sunColor.rgb * f.sunDir.w * 0.35;
    vec3 c;
    float alpha;
    switch (vKind) {
        case PK_SNOW: c = vec3(0.95, 0.97, 1.0) * lit; alpha = 0.9; break;
        case PK_RAIN: c = vec3(0.75, 0.8, 0.9) * lit; alpha = 0.55; break;
        case PK_ASH: c = vec3(0.35, 0.33, 0.32) * lit; alpha = 0.85; break;
        case PK_EMBER: c = vec3(4.0, 1.4, 0.3); alpha = 1.0; break;
        case PK_FIREFLY: {
            float blink = 0.5 + 0.5 * sin(f.time * 2.3 + vSeed * 1.7);
            c = vec3(1.6, 2.4, 0.6) * (0.3 + 2.2 * blink * blink);
            alpha = 1.0;
            break;
        }
        default: c = vec3(1.0, 0.95, 0.7) * lit; alpha = 0.7; break;   // pollen
    }
    alpha *= a * vFade;
    if (alpha < 0.01) discard;
    outColor = vec4(c, alpha);
}
