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
    if (f.exposure.z > 0.0) {
        hdr += texture(gTex2D[nonuniformEXT(f.bloomTex)], vUV).rgb * f.exposure.z;
    }
    // per-biome colour grading: tint, saturation around luminance, contrast around mid-grey
    hdr *= f.grade.rgb;
    float lum = dot(hdr, vec3(0.2126, 0.7152, 0.0722));
    hdr = max(mix(vec3(lum), hdr, f.grade.a), vec3(0.0));
    vec3 e = hdr * f.exposure.x;
    e = 0.18 * pow(e / 0.18 + 1e-6, vec3(f.exposure.w));
    vec3 c = aces(e);
    if (f.debugMode == DBG_OVERDRAW) {
        // fragments shaded per pixel: 1 blue, 2 cyan, 4 green, 8 yellow, 16+ red/white
        uint n = imageLoad(gUImage2D[f.overdrawImg], ivec2(gl_FragCoord.xy)).r;
        float t = clamp(log2(float(max(n, 1u))) / 4.0, 0.0, 1.0);
        vec3 heat = n == 0u ? vec3(0.0) : (t < 0.25 ? mix(vec3(0.1, 0.2, 0.9), vec3(0.1, 0.8, 0.9), t * 4.0)
                  : t < 0.5 ? mix(vec3(0.1, 0.8, 0.9), vec3(0.2, 0.9, 0.2), t * 4.0 - 1.0)
                  : t < 0.75 ? mix(vec3(0.2, 0.9, 0.2), vec3(1.0, 0.9, 0.1), t * 4.0 - 2.0)
                  : mix(vec3(1.0, 0.9, 0.1), vec3(1.0, 0.15, 0.1), t * 4.0 - 3.0));
        if (n > 24u) heat = vec3(1.0);
        c = heat;
    }
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
