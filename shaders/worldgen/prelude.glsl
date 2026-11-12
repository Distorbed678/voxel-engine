// Included before the generated world-generation code.
#include "common.glsl"
#include "noise.glsl"
#include "voxel.glsl"
// the core block ids (B_STONE, B_WATER, ...) come with the generated code

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer EditBuf { uvec2 e[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer ParamBuf { float p[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) readonly buffer PrefabTable { uvec4 t[]; };

layout(push_constant, scalar) uniform GenPC {
    GenJobBuf jobs;
    VoxelBuf voxels;
    JobResultBuf results;
    EditBuf edits;
    ParamBuf params;     // live-tweakable node parameters (no recompile needed)
    ColumnBuf columns;   // per job, per column: biome id (written by the terrain pass)
    PrefabTable prefabTable;
    ColumnBuf prefabData;
    uint numJobs;
    uint seed;
    uint editCount;
    uint pad;
    BlockTable blocks;   // kinds and rotations (BlockInfo.flags), so block edits never recompile
    uint64_t propPrims;  // props pass: scratch for building prop instances (props_lib.glsl)
    uint64_t propSegs;
} pc;

// Globals set by the main template before calling into generated code.
uint gSeed;
uint gLod;      // LOD level of the chunk being generated (0 = full detail)
float gScale;   // cell size in blocks (1 << gLod)

// Node parameters are read through this so dragging a slider never recompiles the shader.
float P(uint i) { return pc.params.p[i]; }

uint block_kind(uint b) { return pc.blocks.b[b & 0xFFFu].flags & 15u; }
// a voxel value turned `steps` quarter turns around Y, +X towards +Z (BlockRegistry::rotated_y)
uint rotate_value(uint v, uint steps) {
    uint id = v & 0xFFFu, st = v >> 12;
    uint rot = (pc.blocks.b[id].flags >> 25) & 3u;
    if (st == 0u) return v;
    if (rot == 1u && (steps & 1u) == 1u && st <= 2u) st = 3u - st;  // column axis X <-> Z
    else if (rot == 2u) st = ((st - 1u + steps) & 3u) + 1u;        // stairs facing
    return id | (st << 12);
}

// ---- helpers available to generated code -------------------------------------------------
float curve_eval(uint base, uint count, float x) {
    // Piecewise curve stored in params: count points of (x, y), sorted by x, smooth segments.
    if (count == 0u) return x;
    float x0 = P(base), y0 = P(base + 1u);
    if (x <= x0) return y0;
    for (uint i = 1u; i < count; ++i) {
        float x1 = P(base + i * 2u), y1 = P(base + i * 2u + 1u);
        if (x <= x1) {
            float t = (x - x0) / max(x1 - x0, 1e-6);
            t = t * t * (3.0 - 2.0 * t);
            return mix(y0, y1, t);
        }
        x0 = x1; y0 = y1;
    }
    return y0;
}

float terrace(float x, float stepv, float smoothness) {
    float s = max(stepv, 1e-3);
    float t = x / s;
    float fl = floor(t);
    float fr = t - fl;
    float k = clamp(smoothness, 0.0, 1.0) * 0.5;
    fr = smoothstep(0.5 - k, 0.5 + k + 1e-4, fr);
    return (fl + fr) * s;
}

// Nearest two jittered cells (F1/F2) for zone & biome layout.
void voronoi2(vec2 q, float jitter, uint seed, out ivec2 c1, out ivec2 c2, out float d1, out float d2) {
    ivec2 base = ivec2(floor(q));
    d1 = 1e9; d2 = 1e9; c1 = base; c2 = base;
    for (int y = -1; y <= 1; ++y)
    for (int x = -1; x <= 1; ++x) {
        ivec2 c = base + ivec2(x, y);
        vec2 fp = vec2(c) + 0.5 + (rand2v(c, seed) - 0.5) * jitter;
        float d = distance(q, fp);
        if (d < d1) { d2 = d1; c2 = c1; d1 = d; c1 = c; }
        else if (d < d2) { d2 = d; c2 = c; }
    }
}

float yband(float y, float y0, float y1) {
    return smoothstep(y0, y0 + 10.0, y) * (1.0 - smoothstep(y1 - 10.0, y1, y));
}

// Positive inside a tunnel (carve), 0..1
float cave_spaghetti(vec3 p, float size, float thick, float y0, float y1, uint seed) {
    float band = yband(p.y, y0, y1);
    if (band <= 0.0) return 0.0;
    vec3 q = p / vec3(size, size * 0.55, size);
    float a = simplex3(q, seed);
    float b = simplex3(q + vec3(31.7, 11.3, 57.1), seed + 1u);
    float t = thick * band;
    return (t * t - (a * a + b * b)) / max(t * t, 1e-6);
}

float cave_cheese(vec3 p, float size, float thr, float y0, float y1, uint seed) {
    float band = yband(p.y, y0, y1);
    if (band <= 0.0) return 0.0;
    float n = fbm3(p / vec3(size, size * 0.5, size), seed, 2, 2.0, 0.5);
    return (n - thr) * band * 8.0;
}

float cave_ravine(vec3 p, float size, float width, float yTop, float depth, uint seed) {
    float y0 = yTop - depth;
    if (p.y < y0 || p.y > yTop + 40.0) return 0.0;
    float n = abs(simplex2(p.xz / size, seed) + 0.2 * simplex2(p.xz / (size * 0.2), seed + 1u));
    float t = clamp((p.y - y0) / depth, 0.0, 1.0);
    float w = width * mix(0.15, 1.0, t) * (0.65 + 0.35 * simplex2(p.xz / 90.0, seed + 2u));
    return (w - n) / max(w, 1e-4);
}

uint strata_pick(vec3 p, float bh, float wob, uint seed, uint b1, uint b2, uint b3, uint b4, uint b5, uint b6) {
    float y = p.y + simplex2(p.xz / 160.0, seed) * wob;
    int band = int(floor(y / max(bh, 0.5)));
    uint h = hash_u32(uint(band) * 2654435761u ^ seed) % 9u;
    uint arr[9] = uint[9](b1, b2, b3, b1, b4, b5, b1, b6, b2);
    return arr[h];
}

// Meandering river network: 1 in the channel, falling to 0 across the banks. The channel follows
// the zero line of a warped low-frequency noise; distance to it is |n| / |grad n| (in blocks).
float river_mask(vec2 xz, float spacing, float width, float bank, float warp, uint seed) {
    vec2 q = xz + vec2(fbm2(xz / (spacing * 0.7), seed + 1u, 3, 2.0, 0.5), fbm2(xz / (spacing * 0.7), seed + 2u, 3, 2.0, 0.5)) * warp;
    float eps = spacing * 0.01;
    float n = fbm2(q / spacing, seed, 2, 2.0, 0.45);
    float nx = fbm2((q + vec2(eps, 0.0)) / spacing, seed, 2, 2.0, 0.45);
    float nz = fbm2((q + vec2(0.0, eps)) / spacing, seed, 2, 2.0, 0.45);
    float g = length(vec2(nx - n, nz - n)) / eps;
    float dist = abs(n) / max(g, 1e-6);
    return 1.0 - smoothstep(width * 0.5, width * 0.5 + max(bank, 1e-3), dist);
}

#define NO_CAVE_BIOME 0xFFFFu

uint ground_cover(vec3 p, uint ground, uint on, float chance, vec4 w, uvec4 b, uint seed) {
    if (ground != on) return 0u;
    ivec3 c = ivec3(floor(p));
    if (rand3(c, seed) >= chance) return 0u;
    float tot = max(w.x + w.y + w.z + w.w, 1e-4);
    float s = rand3(c, seed ^ 0x68bc21ebu) * tot;
    if (s < w.x) return b.x;
    s -= w.x;
    if (s < w.y) return b.y;
    s -= w.y;
    if (s < w.z) return b.z;
    return b.w;
}

// ---- props ------------------------------------------------------------------------------
#define PROP_GRAPH 1u
#define PROP_PREFAB 7u

struct PropLayer {
    uint biome;
    uint kind;       // PROP_GRAPH or PROP_PREFAB
    uint graph;      // prop graph index (PROP_GRAPH)
    uint variant;    // prefab index (PROP_PREFAB)
    uint b0; uint b1; uint b2;
    float a; float b; float c;
    float R; float H; float below;   // the box searched around an anchor
    float spacing;
    float chance;
    float jitter;
    uint surface;
    bool anySurface;
    bool buried;     // inside the ground (geodes) instead of on open surfaces
    bool far;        // kept on coarse LODs
    bool replaceAll; // overwrites terrain, not only air
    float yMin; float yMax;
    uint seed;
    bool hasMask;
};

PropLayer prop_default() {
    PropLayer L;
    L.biome = 0xFFFFu; L.kind = 0u; L.graph = 0u; L.variant = 0u; L.b0 = 0u; L.b1 = 0u; L.b2 = 0u;
    L.a = 1.0; L.b = 1.0; L.c = 1.0; L.R = 1.0; L.H = 1.0; L.below = 0.0;
    L.spacing = 16.0; L.chance = 0.0; L.jitter = 0.8;
    L.surface = 0u; L.anySurface = false; L.buried = false; L.far = false; L.replaceAll = false;
    L.yMin = -1e9; L.yMax = 1e9; L.seed = 0u; L.hasMask = false;
    return L;
}

// prop graph library (props pass only: the build functions use its shared arrays)
#ifdef PROPS_PASS
#include "worldgen/props_lib.glsl"
#endif
