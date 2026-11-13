#version 460
// The cloud sea: a permanent deck of cloud between the surface and the sky layer (World Output:
// base, top, density). Its top and its underside are lumpy height fields.
//   above it   a sea of cloud tops, lit by the sun, fading into the horizon haze
//   inside     thick fog
//   below      an overcast underside that fades in over the last kilometre below the base, so
//              from the ground the sky looks normal (and the sky layer stays hidden: it isn't
//              streamed while the camera is below the deck)
// Composited over the scene with premultiplied alpha; terrain in front of it hides it.
#include "common.glsl"
#include "noise.glsl"
#include "sky.glsl"

layout(push_constant, scalar) uniform PC {
    FrameRef frame;
    float base;
    float top;
    float density;
    float amp;      // height of the lumps on either face
} pc;

layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

// Lump height in [0, 1] (1 = the deck's outer face): rounded mounds with creases between them,
// drifting slowly.
float lumps(vec2 xz, float t, uint seed) {
    vec2 q = xz + vec2(t * 1.6, t * 0.7);
    float b = billow2(q / 900.0, seed, 4, 2.3, 0.5);
    return clamp(1.0 - abs(b), 0.0, 1.0);
}
float top_at(vec2 xz, float t) { return pc.top - pc.amp * (1.0 - lumps(xz, t, 71u)); }
float bot_at(vec2 xz, float t) { return pc.base + pc.amp * (1.0 - lumps(xz, t, 72u)); }

const int STEPS = 28;

void main() {
    FrameData f = pc.frame.d;
    vec4 ndc = vec4(vUV * 2.0 - 1.0, 1e-4, 1.0);
    vec4 wpRel = f.invViewProj * ndc;
    vec3 dir = normalize(wpRel.xyz / wpRel.w);
    vec3 cam = vec3(f.camBlock.xyz) + f.camFrac.xyz;
    float time = f.time;

    // scene distance along the ray (reversed infinite Z: viewZ = near / depth)
    float depth = texelFetch(gTex2D[nonuniformEXT(f.depthTex)], ivec2(gl_FragCoord.xy), 0).r;
    float maxT = 1e9;
    if (depth > 0.0) {
        float viewZ = f.proj[3][2] / depth;
        maxT = viewZ / max(dot(dir, f.camForward.xyz), 1e-3);
    }
    float jitter = hash_to_unit(hash_u32(uint(gl_FragCoord.x) * 1973u + uint(gl_FragCoord.y) * 9277u + f.frameIndex * 26699u));

    vec3 L = f.sunDir.y > -0.05 ? f.sunDir.xyz : -f.sunDir.xyz;
    vec3 sunCol = f.sunColor.rgb * f.sunDir.w;
    vec3 ambient = mix(f.skyHorizon.rgb, f.skyZenith.rgb, 0.6);
    vec3 horizon = sky_color(f, normalize(vec3(dir.x, 0.0, dir.z)));
    float topC = top_at(cam.xz, time), botC = bot_at(cam.xz, time);

    if (cam.y >= topC && dir.y < -1e-4) {
        // ---- above: march down to the lumpy top face ----
        float t0 = max((pc.top - cam.y) / dir.y, 0.0);
        float t1 = (pc.top - pc.amp - cam.y) / dir.y;
        float dt = (t1 - t0) / float(STEPS);
        float tPrev = t0, t = t0 + dt * jitter;
        bool hit = false;
        for (int i = 0; i < STEPS; ++i) {
            vec3 p = cam + dir * t;
            if (p.y <= top_at(p.xz, time)) { hit = true; break; }
            tPrev = t;
            t += dt;
        }
        if (!hit) t = t1;  // the deck's lowest point: below that it's solid
        // refine between the last point above and the first one inside
        float a = tPrev, b = t;
        for (int i = 0; i < 5 && hit; ++i) {
            float m = 0.5 * (a + b);
            vec3 p = cam + dir * m;
            if (p.y <= top_at(p.xz, time)) b = m; else a = m;
        }
        t = b;
        if (t > maxT) discard;
        vec3 p = cam + dir * t;
        const float e = 6.0;
        float h = top_at(p.xz, time);
        vec3 n = normalize(vec3(h - top_at(p.xz + vec2(e, 0.0), time), e, h - top_at(p.xz + vec2(0.0, e), time)));
        float lum = clamp((h - (pc.top - pc.amp)) / max(pc.amp, 1.0), 0.0, 1.0);  // creases are darker
        float diff = clamp((dot(n, L) + 0.35) / 1.35, 0.0, 1.0);
        float rim = pow(1.0 - clamp(dot(n, -dir), 0.0, 1.0), 4.0) * clamp(dot(dir, L) + 0.3, 0.0, 1.0);
        vec3 col = sunCol * (diff * 0.85 + rim * 0.6) + ambient * mix(0.55, 1.05, lum);
        col *= 0.95;
        // far away the sea melts into the horizon
        float haze = 1.0 - exp(-t / 14000.0);
        col = mix(col, horizon, clamp(haze * 1.15, 0.0, 1.0));
        outColor = vec4(col, 1.0) * pc.density;
        return;
    }

    if (cam.y <= botC) {
        // ---- below: the underside, faded in near the base ----
        float vis = smoothstep(pc.base - 1200.0, pc.base - 150.0, cam.y) * pc.density;
        if (vis <= 0.0 || dir.y < 1e-4) discard;
        float t0 = max((pc.base - cam.y) / dir.y, 0.0);
        float t1 = (pc.base + pc.amp - cam.y) / dir.y;
        float dt = (t1 - t0) / float(STEPS);
        float tPrev = t0, t = t0 + dt * jitter;
        bool hit = false;
        for (int i = 0; i < STEPS; ++i) {
            vec3 p = cam + dir * t;
            if (p.y >= bot_at(p.xz, time)) { hit = true; break; }
            tPrev = t;
            t += dt;
        }
        if (!hit) t = t1;
        float a = tPrev, b = t;
        for (int i = 0; i < 5 && hit; ++i) {
            float m = 0.5 * (a + b);
            vec3 p = cam + dir * m;
            if (p.y >= bot_at(p.xz, time)) b = m; else a = m;
        }
        t = b;
        if (t > maxT) discard;
        vec3 p = cam + dir * t;
        float lum = clamp((bot_at(p.xz, time) - pc.base) / max(pc.amp, 1.0), 0.0, 1.0);
        // grey underside: sky light from below, a little sun through the deck
        vec3 col = ambient * mix(0.75, 0.45, lum) + sunCol * 0.06;
        float haze = 1.0 - exp(-t / 9000.0);
        col = mix(col, horizon, haze);
        outColor = vec4(col, 1.0) * vis * (1.0 - 0.5 * haze);
        return;
    }

    if (cam.y > botC && cam.y < topC) {
        // ---- inside: fog, thinning towards the faces ----
        float lo = botC, hi = topC;
        float tExit = abs(dir.y) < 1e-4 ? 1e9 : ((dir.y > 0.0 ? hi : lo) - cam.y) / dir.y;
        float len = min(min(tExit, maxT), 3000.0);
        float a = 1.0 - exp(-len / 40.0);
        float k = clamp((cam.y - pc.base) / max(pc.top - pc.base, 1.0), 0.0, 1.0);
        vec3 col = ambient * mix(0.6, 1.0, k) + sunCol * mix(0.05, 0.45, k);
        outColor = vec4(col, 1.0) * a * pc.density;
        return;
    }
    discard;
}
