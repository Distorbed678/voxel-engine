// Terrain pass main. Expects the generated code to define (see crates/worldgen/src/compile.rs):
//   struct WgColumn; void wg_init(); void wg_column(vec2 xz, out WgColumn c);
//   float wg_density_c(vec3 p, in WgColumn c, out bool carved);  (> 0 = solid; carved = cave air)
//   uint  wg_material(vec3 p, float depth, float airAbove, bool skyOpen, in WgColumn c);
//   uint  wg_fluid(vec3 p, bool skyOpen, in WgColumn c);
//   uint  wg_decorate(vec3 p, uint ground, in WgColumn c);
//   uint  wg_main_biome(in WgColumn c);
//   uint  wg_cave_biome(vec3 p, in WgColumn c);                    (NO_CAVE_BIOME if the zone has none)
//   uint  wg_cave_material(uint b, vec3 p, float depth, float airAbove);
//   uint  wg_cave_decorate(uint b, vec3 p, uint ground);
//
// One invocation = one (x, z) column of 34 cells (chunk + apron), generated top-down so we know
// how deep below the surface each solid cell is (material layering) and whether the cell can
// see the sky (oceans vs. dry caves). Cells removed by the zone's cave carver are cave air: the
// solid cell below one is a cave floor, the solid cell above one a cave ceiling, and both take
// the zone's cave biome material (floors also get its ground cover).

#ifndef WG_LOOKAHEAD
#define WG_LOOKAHEAD 8
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    uint jobIdx = gl_WorkGroupID.z;
    GenJob job = pc.jobs.j[jobIdx];
    gSeed = pc.seed;
    gLod = job.lod;
    gScale = float(1 << job.lod);
    wg_init();
    ivec2 lc = ivec2(gl_GlobalInvocationID.xy);
    bool valid = lc.x < CA && lc.y < CA;

    uint nonAir = 0u;
    uint opaque = 0u;

    if (valid) {
        int scale = 1 << job.lod;
        int half_ = scale >> 1;
        vec2 xz = vec2(job.origin.xz + (lc - 1) * scale + half_) + 0.5;
        WgColumn col;
        wg_column(xz, col);
        pc.columns.c[jobIdx * uint(CA * CA) + uint(lc.y * CA + lc.x)] = uint16_t(wg_main_biome(col));

        uint base = jobIdx * SLOT_U16;
        float depth = 0.0;
        float airAbove = 0.0;
        bool skyOpen = true;
        uint prevBlock = 0u;
        int prevLy = -1;
        bool prevCarved = false;
        bool prevSolid = false;
        // single loop (one wg_density call site keeps the generated shader small):
        // ly >= CA is look-ahead above the chunk (only tracks depth / sky visibility)
        for (int ly = CA - 1 + WG_LOOKAHEAD; ly >= 0; --ly) {
            vec3 p = vec3(xz.x, float(job.origin.y + (ly - 1) * scale + half_) + 0.5, xz.y);
            bool carved;
            float d = wg_density_c(p, col, carved);
            bool solid = d > 0.0;
            if (ly < CA) {
                uint b = solid ? wg_material(p, depth, airAbove, skyOpen, col) : wg_fluid(p, skyOpen, col);
                bool decorated = false;
                if (solid && prevCarved) {
                    // cave floor
                    uint cb = wg_cave_biome(p, col);
                    if (cb != NO_CAVE_BIOME) {
                        b = wg_cave_material(cb, p, 0.0, airAbove);
                        if (prevLy >= 0 && prevBlock == 0u && job.lod == 0u) {
                            uint deco = wg_cave_decorate(cb, p + vec3(0.0, float(scale), 0.0), b);
                            if (deco != 0u) {
                                pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(deco);
                                nonAir += 1u;
                            }
                        }
                        decorated = true;
                    }
                }
                if (!solid && carved && prevSolid && prevLy >= 0 && prevBlock != 0u) {
                    // cave ceiling: re-material the solid cell above
                    vec3 q = p + vec3(0.0, float(scale), 0.0);
                    uint cb = wg_cave_biome(q, col);
                    if (cb != NO_CAVE_BIOME) {
                        pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(wg_cave_material(cb, q, max(depth, 1.0), 0.0));
                    }
                }
                if (!decorated && solid && depth == 0.0 && prevLy >= 0 && prevBlock == 0u && job.lod == 0u) {
                    uint deco = wg_decorate(p + vec3(0.0, 1.0, 0.0), b, col);
                    if (deco != 0u) {
                        pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(deco);
                        nonAir += 1u;
                    }
                }
                pc.voxels.v[base + uint(ly * CA * CA + lc.y * CA + lc.x)] = uint16_t(b);
                if (b != 0u) nonAir += 1u;
                if (solid && b != 0u) opaque += 1u;
                prevBlock = b;
                prevLy = ly;
            }
            prevCarved = carved && !solid;
            prevSolid = solid;
            if (solid) { depth += float(scale); airAbove = 0.0; skyOpen = false; }
            else { depth = 0.0; airAbove += float(scale); }
        }
    }

    uint sNonAir = subgroupAdd(nonAir);
    uint sOpaque = subgroupAdd(opaque);
    if (subgroupElect()) {
        atomicAdd(pc.results.r[jobIdx].nonAir, sNonAir);
        atomicAdd(pc.results.r[jobIdx].opaque, sOpaque);
    }
}
