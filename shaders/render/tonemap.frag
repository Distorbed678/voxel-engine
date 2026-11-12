#version 460
#include "common.glsl"

layout(push_constant, scalar) uniform PC {
    FrameRef frame;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

vec3 aces(vec3 x) {
    const float a = 2.51, b = 0.03, c = 2.43, d = 0.59, e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

vec3 linear_to_srgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(0.0031308, c));
}

void main() {
    FrameData f = pc.frame.d;
    vec3 hdr = texture(gTex2D[nonuniformEXT(f.hdrTex)], vUV).rgb;
    vec3 c = aces(hdr * f.exposure.x);
    // subtle vignette
    vec2 q = vUV - 0.5;
    c *= 1.0 - dot(q, q) * 0.35;
    c = linear_to_srgb(c);

    if ((f.flags & FLAG_CROSSHAIR) != 0u) {
        vec2 px = vUV * f.resolution - f.resolution * 0.5;
        vec2 a = abs(px);
        bool cross = (a.x < 1.0 && a.y < 9.0) || (a.y < 1.0 && a.x < 9.0);
        if (cross) c = vec3(1.0) - c;
    }
    outColor = vec4(c, 1.0);
}
