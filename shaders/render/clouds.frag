#version 460
// Volumetric cloud layer (ray marched), composited over the scene with premultiplied alpha.
// The ray is clamped by the scene depth so terrain correctly occludes clouds and vice versa.
#include "common.glsl"
#include "noise.glsl"
#include "sky.glsl"

layout(push_constant, scalar) uniform PC {
    FrameRef frame;
    float coverage;
    float base;
    float thickness;
    float density;
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

float cloud_density(vec3 wp, float t) {
    vec3 q = wp * 0.0045 + vec3(t * 0.004, 0.0, t * 0.0015);
    float n = value3(q, 91u) * 0.55 + value3(q * 2.7, 92u) * 0.3 + value3(q * 7.1, 93u) * 0.15;
    n = n * 0.5 + 0.5;
    float h = clamp((wp.y - pc.base) / pc.thickness, 0.0, 1.0);
    float shape = smoothstep(0.0, 0.25, h) * (1.0 - smoothstep(0.55, 1.0, h));
    return max(n * shape - (1.0 - pc.coverage), 0.0) * pc.density;
}

void main() {
    FrameData f = pc.frame.d;
    vec4 ndc = vec4(vUV * 2.0 - 1.0, 1e-4, 1.0);
    vec4 wpRel = f.invViewProj * ndc;
    vec3 dir = normalize(wpRel.xyz / wpRel.w);
    vec3 camW = vec3(f.camBlock.xyz) + f.camFrac.xyz;   // fine for clouds (low frequency)

    // scene distance along the ray (reversed infinite Z: viewZ = near / depth)
    float depth = texelFetch(gTex2D[nonuniformEXT(f.depthTex)], ivec2(gl_FragCoord.xy), 0).r;
    float maxT = 1e9;
    if (depth > 0.0) {
        float viewZ = f.proj[3][2] / depth;
        maxT = viewZ / max(dot(dir, f.camForward.xyz), 1e-3);
    }

    float y0 = pc.base, y1 = pc.base + pc.thickness;
    float t0, t1;
    if (abs(dir.y) < 1e-4) {
        if (camW.y < y0 || camW.y > y1) discard;
        t0 = 0.0; t1 = 3000.0;
    } else {
        float ta = (y0 - camW.y) / dir.y;
        float tb = (y1 - camW.y) / dir.y;
        t0 = max(min(ta, tb), 0.0);
        t1 = max(ta, tb);
    }
    t1 = min(t1, min(maxT, 6000.0));
    if (t1 <= t0) discard;

    const int STEPS = 40;
    float dt = (t1 - t0) / float(STEPS);
    float jitter = hash_to_unit(hash_u32(uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + f.frameIndex * 26699u));
    float t = t0 + dt * jitter;
    vec3 L = f.sunDir.y > -0.05 ? f.sunDir.xyz : -f.sunDir.xyz;
    vec3 sunCol = f.sunColor.rgb * f.sunDir.w;
    vec3 ambient = mix(f.skyHorizon.rgb, f.skyZenith.rgb, 0.5) * 1.1;
    float trans = 1.0;
    vec3 scatter = vec3(0.0);
    float mu = dot(dir, L);
    float phase = mix(0.8, 3.0, pow(max(mu, 0.0), 8.0));   // forward scattering glow
    for (int i = 0; i < STEPS; ++i) {
        vec3 wp = camW + dir * t;
        float d = cloud_density(wp, f.time);
        if (d > 0.002) {
            // light march toward the sun
            float ld = 0.0;
            for (int j = 1; j <= 3; ++j) ld += cloud_density(wp + L * float(j) * 18.0, f.time);
            float lightT = exp(-ld * 18.0 * 1.2);
            float powder = 1.0 - exp(-d * dt * 2.0);
            vec3 c = sunCol * lightT * phase * (0.6 + 0.4 * powder) + ambient * 0.9;
            float a = exp(-d * dt);
            scatter += trans * (1.0 - a) * c;
            trans *= a;
            if (trans < 0.02) break;
        }
        t += dt;
    }
    // fade with distance into the sky haze
    float fade = 1.0 - smoothstep(2500.0, 6000.0, t0);
    float alpha = (1.0 - trans) * fade;
    outColor = vec4(scatter * fade, alpha);
}
