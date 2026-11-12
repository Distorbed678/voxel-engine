#version 460
#include "render/terrain_common.glsl"
#include "sky.glsl"

layout(location = 0) in vec3 vRel;
layout(location = 1) in vec2 vUV;
layout(location = 2) perprimitiveEXT flat in uvec4 pData;

layout(location = 0) out vec4 outColor;

vec3 unpack_normal(uint u) {
    return vec3(float(u & 1023u), float((u >> 10) & 1023u), float((u >> 20) & 1023u)) / 1023.0 * 2.0 - 1.0;
}

void main() {
    FrameData f = pc.frame.d;
    uint block = pData.x & 0xFFFFu;
    uint biome = (pData.x >> 16) & 127u;
    uint texDir = pData.y & 255u;
    bool twoSided = (pData.y & 256u) != 0u;
    vec3 N = normalize(unpack_normal(pData.z));
    if (twoSided && !gl_FrontFacing) N = -N;
    uint layer = face_layer(pc.blocks, block, texDir);
    vec4 albedo = texture(gTex2DArray[nonuniformEXT(f.blockTexAlbedo)], vec3(vUV, float(layer)));
    if (albedo.a < 0.5) discard;

    BiomeGpu bio = pc.biomes.b[biome];
    uint tint = (pc.blocks.b[block].flags >> 4) & 15u;
    if (tint != 0u) {
        vec3 tc = pow(tint == 1u ? bio.grass.rgb : bio.foliage.rgb, vec3(2.2));
        float lum = dot(albedo.rgb, vec3(0.2126, 0.7152, 0.0722));
        float tl = max(dot(tc, vec3(0.2126, 0.7152, 0.0722)), 1e-3);
        float mask = smoothstep(0.0, 0.04, albedo.g - max(albedo.r, albedo.b));
        albedo.rgb = mix(albedo.rgb, lum / tl * tc, mask * 0.85);
    }

    vec3 sunDir = f.sunDir.y > -0.05 ? f.sunDir.xyz : -f.sunDir.xyz;
    vec3 sun = f.sunColor.rgb * f.sunDir.w;
    // plants: soft, mostly-up normal and a darker base (fake AO)
    vec3 Ns = twoSided ? normalize(mix(N, vec3(0, 1, 0), 0.6)) : N;
    float ndl = twoSided ? (0.35 + 0.65 * max(dot(Ns, sunDir), 0.0)) : max(dot(Ns, sunDir), 0.0);
    float ao = twoSided ? mix(0.45, 1.0, clamp(1.0 - vUV.y, 0.0, 1.0)) : 0.9;
    vec3 skyAmb = mix(f.groundColor.rgb, f.skyZenith.rgb * 1.2 + f.skyHorizon.rgb * 0.3, Ns.y * 0.5 + 0.5);
    vec3 lit = albedo.rgb * (sun * ndl + skyAmb * ao * 0.55);
    uint emission = (pc.blocks.b[block].flags >> 8) & 0xFFu;
    lit += albedo.rgb * float(emission) / 255.0 * 4.0;
    vec3 color = apply_fog(f, lit, vRel, normalize(vRel));
    if (f.debugMode == DBG_ALBEDO) color = albedo.rgb;
    if (f.debugMode == DBG_NORMALS) color = N * 0.5 + 0.5;
    if (f.debugMode == DBG_BIOME) color = pow(bio.mapColor.rgb, vec3(2.2)) * ao;
    if (f.debugMode == DBG_ZONE) color = pow(bio.zoneColor.rgb, vec3(2.2)) * ao;
    outColor = vec4(color, 1.0);

    uint pp = f.pickPixel;
    if (uint(gl_FragCoord.x) == (pp & 0xFFFFu) && uint(gl_FragCoord.y) == (pp >> 16)) {
        uint payload = 2u | (block << 3) | (pData.w << 19) | (1u << 30) | (1u << 31);
        uint64_t packed = (uint64_t(floatBitsToUint(gl_FragCoord.z)) << 32) | uint64_t(payload);
        atomicMax(pc.pick.v[0], packed);
    }
}
