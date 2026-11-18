#version 460
#include "particles/particles_common.glsl"

layout(location = 0) in vec2 vUV;
layout(location = 1) flat in uint vKind;
layout(location = 2) in float vFade;
layout(location = 3) flat in float vSeed;
layout(location = 4) in vec4 vLight;
layout(location = 0) out vec4 outColor;

void main() {
    FrameData f = pc.frame.d;
    ParticleType T = pc.types.t[vKind - 1u];
    uint shape = uint(T.look.z);
    float r = length(vUV * 2.0 - 1.0);
    float a;
    switch (shape) {
        case SHAPE_STREAK: a = 1.0 - abs(vUV.x * 2.0 - 1.0); break;
        case SHAPE_SQUARE: a = 1.0; break;
        case SHAPE_SPARK: a = exp(-r * r * 5.0); break;
        default: a = 1.0 - smoothstep(0.45, 1.0, r); break;
    }
    vec3 base = mix(T.color.rgb, T.color2.rgb, fract(vSeed * 0.618034));
    // lit by the sky (where it reaches), the sun and the light volume; glowing types light
    // themselves, blinking ones pulse
    vec3 skyAmb = (f.skyZenith.rgb * 1.2 + f.skyHorizon.rgb * 0.3) * f.lightParams.x * vLight.a;
    vec3 lit = skyAmb * 0.6 + f.sunColor.rgb * f.sunDir.w * 0.35 * vLight.a + vLight.rgb;
    float glow = T.color2.a;
    float b = 0.5 + 0.5 * sin(f.time * 2.3 + vSeed * 1.7);
    float pulse = mix(1.0, 0.3 + 2.2 * b * b, T.look.y);
    vec3 c = base * (lit * max(1.0 - glow, 0.0) + glow * pulse);
    if (shape == SHAPE_SPARK) c *= 1.0 + 2.0 * exp(-r * r * 30.0);
    float alpha = T.color.a * a * vFade;
    if (alpha < 0.01) discard;
    outColor = vec4(c, alpha);
}
