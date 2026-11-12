#version 460
#include "render/terrain_common.glsl"
#include "sky.glsl"
#include "light/light_sample.glsl"

layout(location = 0) in vec3 vRel;
layout(location = 1) in vec3 vLocal;
layout(location = 2) perprimitiveEXT flat in uvec2 pQuad;
layout(location = 3) perprimitiveEXT flat in uint pMeta;

#ifndef SHADOW
layout(location = 0) out vec4 outColor;
#endif

#ifndef SHADOW
float shadow_lookup(FrameData f, vec3 rel, vec3 N, vec3 L) {
    if ((f.flags & FLAG_SHADOWS) == 0u) return 1.0;
    float z = dot(rel, f.camForward.xyz);
    if (z > f.shadowSplits.w) return 1.0;
    uint c = z < f.shadowSplits.x ? 0u : (z < f.shadowSplits.y ? 1u : (z < f.shadowSplits.z ? 2u : 3u));
    float texel = f.shadowTexel[c];
    vec3 p = rel + N * texel * 1.5 + L * texel * 0.5;
    vec4 lp = f.shadowVP[c] * vec4(p, 1.0);
    vec2 uv = lp.xy * 0.5 + 0.5;
    float ref = lp.z + 0.0004;
    float sum = 0.0;
    vec2 ts = vec2(1.0 / float(f.shadowRes));
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x)
        sum += texture(gShadowArray[nonuniformEXT(f.shadowTex)], vec4(uv + vec2(x, y) * ts, float(c), ref));
    float s = sum / 9.0;
    // fade out at the far end of the last cascade
    float fade = smoothstep(f.shadowSplits.w * 0.85, f.shadowSplits.w, z);
    return mix(s, 1.0, fade);
}
#endif

// texture-space axes (T = +u, B = +v i.e. image-down) for each face direction
void faceFrame(uint dir, out vec3 T, out vec3 B) {
    switch (dir) {
        case 0u: T = vec3(0, 0, -1); B = vec3(0, -1, 0); break;  // +X
        case 1u: T = vec3(0, 0, 1);  B = vec3(0, -1, 0); break;  // -X
        case 2u: T = vec3(1, 0, 0);  B = vec3(0, 0, 1);  break;  // +Y
        case 3u: T = vec3(1, 0, 0);  B = vec3(0, 0, -1); break;  // -Y
        case 4u: T = vec3(1, 0, 0);  B = vec3(0, -1, 0); break;  // +Z
        default: T = vec3(-1, 0, 0); B = vec3(0, -1, 0); break;  // -Z
    }
}

void main() {
    FrameData f = pc.frame.d;
#ifndef SHADOW
    COUNT_OVERDRAW(f);
#endif
    uint dir = (pQuad.x >> 25) & 7u;
    uint block = pQuad.y & 0xFFFFu;
    uint ao = (pQuad.y >> 16) & 0xFFu;
    uint lod = pQuad.x >> 28;
    uint biome = (pQuad.y >> 24) & 127u;
    BiomeGpu bio = pc.biomes.b[biome];
    uint axis = dir >> 1;
    vec3 N = vec3(dir_normal(dir));
    vec3 T, B;
    faceFrame(dir, T, B);

    // texture coordinates in block units (repeat sampler => tiles per block)
    vec2 uv = vec2(dot(vLocal, T), dot(vLocal, B));
    // rotated logs: the bark grain (texture v) follows the log's axis on its side faces
    uint logAxis = BLOCK_STATE(block) & 3u;
    if (logAxis != 0u && (pc.blocks.b[BLOCK_ID(block)].flags & 15u) != KIND_MODEL) {
        uint la = logAxis == 1u ? 0u : 2u;
        if (axis != la && abs(B[la]) < 0.5) uv = uv.yx;
    }
    uint layer = face_layer(pc.blocks, block, dir, f.time);
    vec4 albedo = texture(gTex2DArray[nonuniformEXT(f.blockTexAlbedo)], vec3(uv, float(layer)));
#ifdef TRANSPARENT
    if (albedo.a < 0.02) discard;
#else
    if (albedo.a < 0.5) discard;
#endif
#ifdef SHADOW
    return;
#else
    vec4 nmap = texture(gTex2DArray[nonuniformEXT(f.blockTexNormal)], vec3(uv, float(layer)));

    // Hytale-style biome tinting: recolour the green/blue parts of tintable blocks
    uint tint = (pc.blocks.b[BLOCK_ID(block)].flags >> 4) & 15u;
    if (tint != 0u) {
        vec3 tc = tint == 1u ? bio.grass.rgb : (tint == 2u ? bio.foliage.rgb : bio.water.rgb);
        tc = pow(tc, vec3(2.2));
        float lum = dot(albedo.rgb, vec3(0.2126, 0.7152, 0.0722));
        float tl = max(dot(tc, vec3(0.2126, 0.7152, 0.0722)), 1e-3);
        vec3 tinted = lum / tl * tc;
        float mask = tint == 3u ? 1.0 : smoothstep(0.0, 0.04, albedo.g - max(albedo.r, albedo.b));
        albedo.rgb = mix(albedo.rgb, tinted, mask * 0.85);
    }

    // per-cell AO, bilinear from the 4 packed corners (u-,v-), (u+,v-), (u+,v+), (u-,v+)
    float scale = float(1u << lod);
    vec3 cellPos = vLocal / scale;
    vec2 cuv = vec2(cellPos[(axis + 1u) % 3u], cellPos[(axis + 2u) % 3u]);
    vec2 fc = fract(cuv);
    vec4 aoc = vec4(float(ao & 3u), float((ao >> 2) & 3u), float((ao >> 4) & 3u), float((ao >> 6) & 3u)) / 3.0;
    float aoV = mix(mix(aoc.x, aoc.y, fc.x), mix(aoc.w, aoc.z, fc.x), fc.y);
    aoV = mix(0.25, 1.0, aoV);
    aoV = aoV * aoV;

    vec3 tn = nmap.xyz * 2.0 - 1.0;
    vec3 Nw = normalize(T * tn.x + B * tn.y + N * max(tn.z, 0.2));

    // the moon takes over as the light source at night
    vec3 sunDir = f.sunDir.y > -0.05 ? f.sunDir.xyz : -f.sunDir.xyz;
    float ndl = max(dot(Nw, sunDir), 0.0) * smoothstep(-0.05, 0.05, dot(N, sunDir) + 0.05);
    vec3 sun = f.sunColor.rgb * f.sunDir.w;
    vec3 skyAmb = mix(f.groundColor.rgb, f.skyZenith.rgb * 1.2 + f.skyHorizon.rgb * 0.3, Nw.y * 0.5 + 0.5);
    float shadow = ndl > 0.0 ? shadow_lookup(f, vRel, N, sunDir) : 1.0;
    shadow = mix(1.0, shadow, f.camForward.w);
    // light volume: sky light dims sun + ambient in enclosed spaces, block light adds colour
    vec4 Lv = sample_light(f, vRel, N * 0.5);
    float skyVis = Lv.a * Lv.a;
    float sunVis = smoothstep(0.3, 0.8, Lv.a);
    vec3 blockL = Lv.rgb * Lv.rgb * BLOCK_LIGHT_GAIN;
    vec3 lit = albedo.rgb * (sun * ndl * shadow * sunVis + skyAmb * aoV * 0.55 * skyVis + (blockL + CAVE_AMBIENT) * aoV);
    // emissive blocks
    uint emission = (pc.blocks.b[BLOCK_ID(block)].flags >> 8) & 0xFFu;
    // self-glow: squared albedo keeps saturated colours (lava stays orange instead of white)
    lit += albedo.rgb * albedo.rgb * float(emission) / 255.0 * EMISSIVE_GAIN;

    vec3 viewDir = normalize(vRel);
#ifdef TRANSPARENT
    if (tint == 3u) {
        // ---- water: waves, depth absorption, fresnel sky reflection, sun glint ----
        vec3 wp = vRel + f.camFrac.xyz + vec3(f.camBlock.xyz & 4095);
        vec3 Nn = N;
        if (dir == 2u) {
            float t = f.time;
            vec2 g = vec2(0.0);
            g += vec2(cos(wp.x * 0.9 + t * 1.3), cos(wp.z * 0.7 + t * 1.1)) * 0.06;
            g += vec2(cos(wp.x * 2.3 - wp.z * 1.7 + t * 2.1), cos(wp.z * 2.9 + wp.x * 0.6 - t * 1.7)) * 0.03;
            g += vec2(cos((wp.x + wp.z) * 5.1 + t * 3.3), cos((wp.x - wp.z) * 4.7 - t * 2.9)) * 0.012;
            Nn = normalize(vec3(-g.x, 1.0, -g.y));
        }
        float depthTex = texelFetch(gTex2D[nonuniformEXT(f.depthTex)], ivec2(gl_FragCoord.xy), 0).r;
        float sceneZ = depthTex > 0.0 ? f.proj[3][2] / depthTex : 1e6;
        float waterZ = f.proj[3][2] / gl_FragCoord.z;
        float thick = max(sceneZ - waterZ, 0.0);
        vec3 deep = pow(bio.water.rgb, vec3(2.2)) * 0.35;
        vec3 shallow = albedo.rgb;
        float absorb = 1.0 - exp(-thick * 0.18);
        vec3 body = mix(shallow, deep, absorb) * (sun * max(dot(Nn, sunDir), 0.0) * 0.6 * sunVis + skyAmb * 0.8 * skyVis + blockL * 0.6);
        float cosT = max(dot(Nn, -viewDir), 0.0);
        float fresnel = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);
        vec3 refl = sky_color(f, reflect(viewDir, Nn));
        float spec = pow(max(dot(reflect(-sunDir, Nn), -viewDir), 0.0), 220.0) * shadow * sunVis;
        vec3 wc = mix(body, refl, fresnel) + sun * spec * 4.0;
        float a = clamp(mix(0.45, 0.95, absorb) + fresnel * 0.5, 0.0, 1.0);
        vec3 fogged = apply_fog(f, wc, vRel, viewDir);
        outColor = vec4(fogged, a);
        return;
    }
#endif
    vec3 color = apply_fog(f, lit, vRel, viewDir);

    switch (f.debugMode) {
        case DBG_NORMALS: color = Nw * 0.5 + 0.5; break;
        case DBG_LOD: color = hsv2rgb(vec3(float(lod) / 6.0, 0.7, 0.9)) * (0.6 + 0.4 * aoV); break;
        case DBG_CHUNKS: {
            ivec3 cp = ivec3(floor((vec3(f.camBlock.xyz) + f.camFrac.xyz + vRel - N * 0.01) / (32.0 * scale)));
            color = hash_color(uint(cp.x) * 73856093u ^ uint(cp.y) * 19349663u ^ uint(cp.z) * 83492791u ^ lod) * (0.6 + 0.4 * aoV);
            break;
        }
        case DBG_MESHLETS: color = hash_color(pMeta) * (0.5 + 0.5 * aoV); break;
        case DBG_AO: color = vec3(aoV); break;
        case DBG_ALBEDO: color = albedo.rgb; break;
        case DBG_ZONE: color = pow(bio.zoneColor.rgb, vec3(2.2)) * (0.35 + 0.65 * (ndl * 0.7 + 0.3)) * aoV; break;
        case DBG_BIOME: color = pow(bio.mapColor.rgb, vec3(2.2)) * (0.35 + 0.65 * (ndl * 0.7 + 0.3)) * aoV; break;
        case DBG_LIGHTING: color = vec3(0.7) * (sun * ndl * shadow * sunVis + skyAmb * aoV * 0.55 * skyVis + (blockL + CAVE_AMBIENT) * aoV); break;
        default: break;
    }
#ifdef TRANSPARENT
    outColor = vec4(color, albedo.a);
    return;
#else
    outColor = vec4(color, 1.0);
#endif

    // GPU picking: closest fragment under the pick pixel wins (reversed-Z => larger is closer)
    uint pp = f.pickPixel;
    if (uint(gl_FragCoord.x) == (pp & 0xFFFFu) && uint(gl_FragCoord.y) == (pp >> 16)) {
        uint payload = dir | (block << 3) | (lod << 19) | (1u << 31);
        uint64_t packed = (uint64_t(floatBitsToUint(gl_FragCoord.z)) << 32) | uint64_t(payload);
        atomicMax(pc.pick.v[0], packed);
    }
#endif
}
