#ifndef COMMON_GLSL
#define COMMON_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_buffer_reference2 : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_atomic_int64 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_shader_image_load_formatted : require

// ---- bindless images (set 0) ----------------------------------------------------------
layout(set = 0, binding = 0) uniform sampler2D gTex2D[];
layout(set = 0, binding = 0) uniform sampler2DArray gTex2DArray[];
layout(set = 0, binding = 0) uniform sampler2DArrayShadow gShadowArray[];
layout(set = 0, binding = 0) uniform sampler3D gTex3D[];
layout(set = 0, binding = 1) uniform image2D gImage2D[];

// ---- per-frame constants ----------------------------------------------------------------
struct FrameData {
    mat4 viewProj;        // camera-relative (translation removed)
    mat4 invViewProj;
    mat4 view;
    mat4 proj;
    mat4 prevViewProj;
    vec4 frustum[6];      // camera-relative planes, xyz normal (inward), w distance
    ivec4 camBlock;       // integer camera block position
    vec4 camFrac;         // fractional camera position inside the block
    vec4 sunDir;          // xyz direction TO the sun, w = intensity
    vec4 sunColor;
    vec4 skyZenith;
    vec4 skyHorizon;
    vec4 groundColor;
    vec4 fogColor;        // rgb, a = density
    vec4 fogParams;       // x = height falloff, y = start distance, z = sea level, w = max fog
    vec2 resolution;
    float time;
    float dt;
    uint frameIndex;
    uint debugMode;
    uint flags;
    uint pickPixel;       // x | (y << 16)
    uint blockTexAlbedo;  // bindless indices
    uint blockTexNormal;
    uint hdrTex;
    uint depthTex;
    vec4 exposure;        // x = exposure, y = time of day [0,1), z = bloom strength (0 = off)
    mat4 shadowVP[4];     // camera-relative -> cascade clip
    vec4 shadowPlanes[24];
    vec4 shadowSplits;    // far distance (view depth) of each cascade
    vec4 camForward;      // xyz camera forward, w = shadow strength
    uint shadowTex;
    uint shadowRes;
    uint lightTex;        // bindless sampled index of the light volume
    uint bloomTex;        // bindless sampled index of the bloom result (strength in exposure.z)
    ivec4 lightOrigin;    // light volume min corner relative to camBlock, w = LIGHT_ENABLED
    ivec4 coarseOrigin;   // coarse light volume min corner relative to camBlock, w = LIGHT_ENABLED
    uint coarseTex;       // bindless sampled index of the coarse light volume
    uint pad1;
    uint pad2;
    uint pad3;
};

layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer FrameRef {
    FrameData d;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) buffer U32Buf { uint v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer U64Buf { uint64_t v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer UVec2Buf { uvec2 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer F32Buf { float v[]; };

// Debug modes (keep in sync with Rust `DebugView`)
#define DBG_NONE 0u
#define DBG_NORMALS 1u
#define DBG_LOD 2u
#define DBG_CHUNKS 3u
#define DBG_MESHLETS 4u
#define DBG_AO 5u
#define DBG_ALBEDO 6u
#define DBG_OVERDRAW 7u
#define DBG_ZONE 8u
#define DBG_BIOME 9u
#define DBG_LIGHTING 10u

#define FLAG_WIREFRAME 1u
#define FLAG_SHADOWS 2u
#define FLAG_FOG 4u
#define FLAG_CROSSHAIR 8u
#define FLAG_TAA 16u

vec3 hsv2rgb(vec3 c) {
    vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
    return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

uint hash_u32(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

vec3 hash_color(uint x) {
    uint h = hash_u32(x);
    return hsv2rgb(vec3(float(h & 1023u) / 1023.0, 0.55, 0.9));
}

#endif
