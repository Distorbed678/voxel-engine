// Props pass: one workgroup per chunk job. Runs after the terrain pass for the same batch.
//   phase 1: every thread evaluates candidate anchors (jittered grid per prop layer) - biome,
//            chance, mask, surface scan and surface block tests - and appends valid anchors.
//   phase 2: all threads rasterise each anchor's shape into the chunk's voxel slot.
// Props are deterministic per anchor, so a tree crossing a chunk border is drawn identically
// (piece by piece) by every chunk it touches.
#include "worldgen/props.glsl"

layout(local_size_x = 256) in;

#define MAX_ANCHORS 384
shared ivec4 sAnchors[MAX_ANCHORS];
shared uint sCount;

void main() {
    uint jobIdx = gl_WorkGroupID.x;
    GenJob job = pc.jobs.j[jobIdx];
    gSeed = pc.seed;
    gLod = job.lod;
    gScale = float(1 << job.lod);
    wg_init();
    int scale = 1 << job.lod;
    int half_ = scale >> 1;
    ivec3 rmin = job.origin - ivec3(scale);
    ivec3 rmax = job.origin + ivec3((CS + 1) * scale);
    uint lid = gl_LocalInvocationIndex;
    if (lid == 0u) sCount = 0u;
    barrier();

    if (NUM_PROP_LAYERS > 0u) {
        for (uint layer = 0u; layer < NUM_PROP_LAYERS; ++layer) {
            PropLayer L = prop_default();
            prop_layer(layer, L);
            if (L.chance <= 0.0) continue;
            float R, H, below;
            prop_bounds(L, R, H, below);
            // small props disappear on coarse LODs
            if (gLod > 0u && max(R, H) < 2.0 * gScale) continue;
            if (gLod > 2u && L.kind != PROP_SPIKE && L.kind != PROP_TREE) continue;
            bool geode = L.kind == PROP_GEODE;
            if (geode && gLod > 0u) continue;
            int sp = max(int(L.spacing), 2);
            int Ri = int(ceil(R));
            ivec3 c0 = ivec3(floor(vec3(rmin - Ri) / float(sp)));
            ivec3 c1 = ivec3(floor(vec3(rmax + Ri) / float(sp)));
            if (!geode) { c0.y = 0; c1.y = 0; }
            uvec3 nn = uvec3(c1 - c0 + 1);
            uint total = nn.x * nn.y * nn.z;
            for (uint i = lid; i < total; i += 256u) {
                ivec3 cell = c0 + ivec3(i % nn.x, (i / nn.x) % nn.y, i / (nn.x * nn.y));
                float r = geode ? rand3(cell, L.seed) : rand2(cell.xz, L.seed);
                if (r >= L.chance) continue;
                vec3 j = (rand3v(cell, L.seed ^ 0x9E37u) - 0.5) * L.jitter;
                ivec3 a = ivec3(floor((vec3(cell) + 0.5 + j) * float(sp)));
                if (a.x + Ri < rmin.x || a.x - Ri >= rmax.x || a.z + Ri < rmin.z || a.z - Ri >= rmax.z) continue;
                if (geode && (float(a.y) < L.yMin || float(a.y) > L.yMax || a.y + Ri < rmin.y || a.y - Ri >= rmax.y)) continue;
                WgColumn col;
                wg_column(vec2(a.xz) + 0.5, col);
                if (col.biome[0] != L.biome || (!geode && col.bw[0] < 0.55)) continue;
                if (L.hasMask) {
                    float m = prop_mask(layer, vec3(float(a.x) + 0.5, 64.0, float(a.z) + 0.5), col.h[0]);
                    if (r >= L.chance * clamp(m, 0.0, 1.0)) continue;
                }
                if (!geode) {
                    // surface scan: first solid cell with air above, top-down
                    int yTop = rmax.y + int(below) + scale;
                    int yBot = rmin.y - int(H) - scale;
                    bool prevAir = false;
                    int surf = -2147483647;
                    for (int y = yTop + scale; y >= yBot; y -= scale) {
                        float d = wg_density(vec3(float(a.x) + 0.5, float(y) + 0.5, float(a.z) + 0.5), col);
                        if (d > 0.0 && prevAir && y <= yTop) { surf = y; break; }
                        prevAir = d <= 0.0;
                    }
                    if (surf == -2147483647) continue;
                    if (float(surf) < L.yMin || float(surf) > L.yMax) continue;
                    if (!L.anySurface) {
                        if (float(surf) < gSeaLevel) continue;
                        uint m = wg_material(vec3(float(a.x) + 0.5, float(surf) + 0.5, float(a.z) + 0.5), 0.0, float(scale), true, col);
                        if (m != L.surface) continue;
                    } else if (float(surf) < gSeaLevel - 1.0 && L.kind != PROP_SPIKE) {
                        continue;
                    }
                    a.y = surf + scale;
                }
                uint idx = atomicAdd(sCount, 1u);
                if (idx < MAX_ANCHORS) sAnchors[idx] = ivec4(a, int(layer));
            }
        }
    }
    barrier();

    uint count = min(sCount, uint(MAX_ANCHORS));
    uint written = 0u;
    uint removed = 0u;
    uint base = jobIdx * SLOT_U16;
    for (uint ai = 0u; ai < count; ++ai) {
        ivec4 an = sAnchors[ai];
        ivec3 anchor = an.xyz;
        PropLayer L = prop_default();
        prop_layer(uint(an.w), L);
        float R, H, below;
        prop_bounds(L, R, H, below);
        uint seed = hash3i(anchor, L.seed);
        int Ri = int(ceil(R));
        ivec3 bmin = max(anchor - ivec3(Ri, int(ceil(below)), Ri), rmin);
        ivec3 bmax = min(anchor + ivec3(Ri + 1, int(ceil(H)) + 1, Ri + 1), rmax);
        if (any(greaterThanEqual(bmin, bmax))) continue;
        ivec3 cmin = (bmin - rmin) / scale;
        ivec3 cmax = min((bmax - rmin + scale - 1) / scale, ivec3(CA));
        ivec3 cn = cmax - cmin;
        uint n = uint(cn.x * cn.y * cn.z);
        for (uint t = lid; t < n; t += 256u) {
            ivec3 c = cmin + ivec3(int(t) % cn.x, (int(t) / cn.x) % cn.y, int(t) / (cn.x * cn.y));
            ivec3 q = rmin + c * scale + half_;
            vec3 rel = vec3(q - anchor);
            uint b = prop_shape(L, rel, seed);
            uint vi = base + uint((c.y * CA + c.z) * CA + c.x);
            if (L.kind == PROP_GEODE) {
                if (b == 0xFFFFu) continue;
                if (b == 0u && uint(pc.voxels.v[vi]) != 0u) removed += 1u;
                pc.voxels.v[vi] = uint16_t(b);
                continue;
            }
            if (b == 0u) continue;
            uint cur = uint(pc.voxels.v[vi]);
            uint ck = block_kind(cur);
            bool replace = cur == 0u || ck == KIND_MODEL || (ck == KIND_CUTOUT && block_kind(b) == KIND_OPAQUE);
            if (replace) {
                pc.voxels.v[vi] = uint16_t(b);
                written += 1u;
            }
        }
    }
    uint s = subgroupAdd(written);
    uint rm = subgroupAdd(removed);
    if (subgroupElect()) {
        if (s > 0u) atomicAdd(pc.results.r[jobIdx].nonAir, s);
        if (rm > 0u) atomicAdd(pc.results.r[jobIdx].opaque, 0u - rm);
    }
}
