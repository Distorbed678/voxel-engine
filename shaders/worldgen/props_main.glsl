// Props pass: one workgroup per chunk job. Runs after the terrain pass for the same batch.
//   phase 1: every thread evaluates candidate anchors (jittered grid per prop layer) - biome,
//            chance, mask, surface scan and surface block tests - and appends valid anchors.
//   phase 2: 16 anchors at a time, one thread each builds the instance's primitives (its Prop
//            graph, props_lib.glsl) into the job's scratch; then anchor by anchor, in a fixed
//            order, all threads rasterise the cells an instance covers into the chunk's slot.
// Props are deterministic per anchor, so a tree crossing a chunk border is drawn identically
// (piece by piece) by every chunk it touches.
// Preview jobs (GENJOB_PREVIEW, the prop editor) have one anchor at the origin: the job's prop
// graph (pad1) with the job's seed (editOffset), on nothing.
#include "worldgen/props.glsl"

layout(local_size_x = 256) in;

#define MAX_ANCHORS 384
#define PROP_SKY 8  // cells of open air a surface needs above it to grow props
shared ivec4 sAnchors[MAX_ANCHORS];
shared ivec4 sSorted[MAX_ANCHORS];
shared uint sCount;
// instances built in parallel, BUILD_BATCH at a time: primitive count and bounds (cells,
// relative to the anchor) of each
#define BUILD_BATCH 16u
shared uint sBuiltN[BUILD_BATCH];
shared ivec3 sBuiltLo[BUILD_BATCH];
shared ivec3 sBuiltHi[BUILD_BATCH];

// the layer of an anchor: a biome's prop layer, or (w < 0) the preview of a prop graph
void anchor_layer(int w, out PropLayer L) {
    L = prop_default();
    if (w >= 0) {
        prop_layer(uint(w), L);
    } else {
        L.kind = PROP_GRAPH;
        L.graph = uint(-1 - w);
        L.R = 64.0; L.H = 128.0; L.below = 64.0;
        L.replaceAll = true;
        L.far = true;
    }
}

// a fixed order of anchors (prop layer, then position), the same in every chunk
bool anchor_before(ivec4 a, ivec4 b) {
    if (a.w != b.w) return a.w < b.w;
    if (a.y != b.y) return a.y < b.y;
    if (a.z != b.z) return a.z < b.z;
    return a.x < b.x;
}

void main() {
    // this layer's jobs (prelude.glsl): it draws into the cells it owns
    uint jobIdx = pc.jobList.j[gl_WorkGroupID.x] & ~JOBLIST_COLUMNS;
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
    bool preview = (job.flags & GENJOB_PREVIEW) != 0u;
    if (lid == 0u) {
        sCount = 0u;
        if (preview) {
            sAnchors[0] = ivec4(0, 0, 0, -1 - int(job.pad1));
            sCount = 1u;
        }
    }
    barrier();

    if (NUM_PROP_LAYERS > 0u && !preview) {
        for (uint layer = 0u; layer < NUM_PROP_LAYERS; ++layer) {
            PropLayer L = prop_default();
            prop_layer(layer, L);
            if (L.chance <= 0.0) continue;
            float R, H, below;
            prop_bounds(L, R, H, below);
            // small props disappear on coarse LODs, the far ones stay
            if (gLod > 0u && max(R, H) < 2.0 * gScale) continue;
            if (gLod > 2u && !L.far) continue;
            bool geode = L.buried;
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
                if (geode) {
                    uint idx = atomicAdd(sCount, 1u);
                    if (idx < MAX_ANCHORS) sAnchors[idx] = ivec4(a, int(layer));
                    continue;
                }
                // Surface scan, top-down over every height this prop could reach the chunk from.
                // Every open surface counts (a solid cell under PROP_SKY cells of air that no cave
                // carved), not just the highest one in range, so a prop is anchored the same way
                // in every chunk it touches. Cave floors and spots under low overhangs don't.
                // Surfaces the layer would reject anyway (height limits, under water) aren't scanned.
                float lo = max(float(rmin.y - int(H) - scale), L.yMin);
                if (!L.anySurface) lo = max(lo, gSeaLevel);
                else if (!L.far) lo = max(lo, gSeaLevel - 1.0);
                float hi = min(float(rmax.y + int(below) + scale), L.yMax);
                if (lo > hi) continue;
                int yBot = int(ceil(lo));
                int yTop = int(floor(hi)) & ~(scale - 1);  // on the LOD's cell grid
                int airRun = 0;
                for (int y = yTop + PROP_SKY * scale; y >= yBot; y -= scale) {
                    bool carved;
                    float d = wg_density_c(vec3(float(a.x) + 0.5, float(y) + 0.5, float(a.z) + 0.5), col, carved);
                    if (d <= 0.0) {
                        airRun = carved ? 0 : airRun + 1;
                        continue;
                    }
                    bool open = airRun >= PROP_SKY;
                    airRun = 0;
                    int surf = y;
                    if (!open || surf > yTop) continue;
                    if (!L.anySurface) {
                        if (float(surf) < gSeaLevel) continue;
                        uint m = wg_material(vec3(float(a.x) + 0.5, float(surf) + 0.5, float(a.z) + 0.5), 0.0, float(scale), true, col);
                        if (m != L.surface) continue;
                    } else if (float(surf) < gSeaLevel - 1.0 && !L.far) {
                        continue;
                    }
                    uint idx = atomicAdd(sCount, 1u);
                    if (idx < MAX_ANCHORS) sAnchors[idx] = ivec4(a.x, surf + scale, a.z, int(layer));
                }
            }
        }
    }
    barrier();

    // Overlapping props resolve by order (first writer wins between leaves, prefabs overwrite),
    // so draw the anchors in a fixed order, one at a time: the append order above depends on
    // thread timing, and every chunk a prop touches must draw the same blocks.
    uint count = min(sCount, uint(MAX_ANCHORS));
    for (uint i = lid; i < count; i += 256u) {
        ivec4 a = sAnchors[i];
        uint rank = 0u;
        for (uint j = 0u; j < count; ++j) {
            ivec4 b = sAnchors[j];
            if (anchor_before(b, a) || (b == a && j < i)) ++rank;
        }
        sSorted[rank] = a;
    }
    uint written = 0u;
    uint removed = 0u;
    uint base = jobIdx * SLOT_U16;
    // Anchors in batches: each thread of the first BUILD_BATCH builds one instance into its own
    // region of the job's scratch; then, one at a time in the fixed order, an instance is copied to
    // shared memory and all threads draw the cells it covers.
    for (uint b0 = 0u; b0 < count; b0 += BUILD_BATCH) {
        uint nb = min(uint(BUILD_BATCH), count - b0);
        barrier();
        if (lid < nb) {
            ivec4 an = sSorted[b0 + lid];
            PropLayer L;
            anchor_layer(an.w, L);
            sBuiltN[lid] = 0u;
            if (L.kind == PROP_GRAPH) {
                uint slot = jobIdx * BUILD_BATCH + lid;
                gPrimBase = slot * MAX_PRIMS;
                gSegBase = slot * MAX_SEGS;
                gXf = xf_identity();
                gInst = preview ? job.editOffset : hash3i(an.xyz, L.seed);
                gAnchorP = vec3(an.xyz) + vec3(0.5, 0.0, 0.5);
                gPrimN = 0u;
                gSegN = 0u;
                gGroupN = 0u;
                prop_build(L.graph);
                vec3 lo = vec3(1e9), hi = vec3(-1e9);
                for (uint i = 0u; i < gPrimN; ++i) {
                    Prim P = PRIM(i);
                    vec4 b = prim_bound(P, P.m.x & 15u);
                    float r = b.w + P.n.x + 1.0;
                    lo = min(lo, b.xyz - r);
                    hi = max(hi, b.xyz + r);
                }
                sBuiltN[lid] = gPrimN;
                sBuiltLo[lid] = ivec3(floor(lo));
                sBuiltHi[lid] = ivec3(ceil(hi)) + 1;
            }
        }
        for (uint k = 0u; k < nb; ++k) {
            barrier();
            ivec4 an = sSorted[b0 + k];
            ivec3 anchor = an.xyz;
            PropLayer L;
            anchor_layer(an.w, L);
            float R, H, below;
            prop_bounds(L, R, H, below);
            uint seed = preview ? job.editOffset : hash3i(anchor, L.seed);
            int Ri = int(ceil(R));
            ivec3 bmin = max(anchor - ivec3(Ri, int(ceil(below)), Ri), rmin);
            ivec3 bmax = min(anchor + ivec3(Ri + 1, int(ceil(H)) + 1, Ri + 1), rmax);
            bool graph = L.kind == PROP_GRAPH;
            uint primCount = 0u;
            if (graph) {
                primCount = sBuiltN[k];
                uint src = (jobIdx * BUILD_BATCH + k) * MAX_PRIMS;
                for (uint i = lid; i < primCount; i += 256u) sPrim[i] = PrimBuf(pc.propPrims).p[src + i];
                barrier();
                bmin = max(bmin, anchor + sBuiltLo[k]);
                bmax = min(bmax, anchor + sBuiltHi[k]);
            }
            if (any(greaterThanEqual(bmin, bmax))) continue;
            ivec3 cmin = (bmin - rmin) / scale;
            ivec3 cmax = min((bmax - rmin + scale - 1) / scale, ivec3(CA));
            ivec3 cn = cmax - cmin;
            uint n = uint(cn.x * cn.y * cn.z);
            // prop space: the anchor column's centre at x = z = 0, the ground top at y = 0
            vec3 origin = vec3(anchor) + vec3(0.5, 0.0, 0.5);
            for (uint t = lid; t < n; t += 256u) {
                ivec3 c = cmin + ivec3(int(t) % cn.x, (int(t) / cn.x) % cn.y, int(t) / (cn.x * cn.y));
                ivec3 q = rmin + c * scale + half_;
                if (!preview && !wg_owns(float(q.y) + 0.5)) continue;
                uint vi = base + uint((c.y * CA + c.z) * CA + c.x);
                uint cur = uint(pc.voxels.v[vi]);
                if (!graph) {
                    uint b = shape_prefab(L, vec3(q - anchor), seed);
                    if (b != 0u) {
                        pc.voxels.v[vi] = uint16_t(b);
                        written += 1u;
                    }
                    continue;
                }
                vec3 cell = vec3(rmin + c * scale) + 0.5 * gScale;
                uint b = prop_eval(cell - origin, primCount);
                if (b == PROP_NOTHING) continue;
                uint ck = block_kind(cur);
                bool replace = L.replaceAll || cur == 0u || ck == KIND_MODEL || (ck == KIND_CUTOUT && block_kind(b) == KIND_OPAQUE);
                if (!replace || b == cur) continue;
                if (b == 0u) {
                    removed += 1u;
                } else {
                    written += 1u;
                }
                pc.voxels.v[vi] = uint16_t(b);
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
