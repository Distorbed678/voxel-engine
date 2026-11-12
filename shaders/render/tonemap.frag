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
    if ((f.flags & FLAG_TAA) != 0u) {
        // TAA softens the image: sharpen against the 4 neighbours, clamped to their range
        // so edges don't ring
        ivec2 p = ivec2(gl_FragCoord.xy);
        ivec2 hi = ivec2(f.resolution) - 1;
        vec3 n = texelFetch(gTex2D[nonuniformEXT(f.hdrTex)], clamp(p + ivec2(0, -1), ivec2(0), hi), 0).rgb;
        vec3 s = texelFetch(gTex2D[nonuniformEXT(f.hdrTex)], clamp(p + ivec2(0, 1), ivec2(0), hi), 0).rgb;
        vec3 e = texelFetch(gTex2D[nonuniformEXT(f.hdrTex)], clamp(p + ivec2(1, 0), ivec2(0), hi), 0).rgb;
        vec3 w = texelFetch(gTex2D[nonuniformEXT(f.hdrTex)], clamp(p + ivec2(-1, 0), ivec2(0), hi), 0).rgb;
        vec3 mn = min(hdr, min(min(n, s), min(e, w)));
        vec3 mx = max(hdr, max(max(n, s), max(e, w)));
        hdr = clamp(hdr + (4.0 * hdr - (n + s + e + w)) * 0.3, mn, mx);
    }
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
