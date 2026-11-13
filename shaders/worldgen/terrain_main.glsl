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
//   float wg_depth_max(in WgColumn c);                            (deepest depth threshold used)
//
// One invocation = one (x, z) column of 34 cells (chunk + apron), generated top-down so we know
// how deep below the surface each solid cell is (material layering) and whether the cell can
// see the sky (oceans vs. dry caves; see the windows below). Cells removed by the zone's cave carver are cave air: the
// solid cell below one is a cave floor, the solid cell above one a cave ceiling, and both take
// the zone's cave biome material (floors also get its ground cover).

#ifndef WG_LOOKAHEAD
#define WG_LOOKAHEAD 8
#endif

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

void main() {
    // this layer's jobs (prelude.glsl): it writes the cells it owns, and maybe the column biomes
    uint entry = pc.jobList.j[gl_WorkGroupID.z];
    uint jobIdx = entry & ~JOBLIST_COLUMNS;
    bool writeColumns = (entry & JOBLIST_COLUMNS) != 0u;
    GenJob job = pc.jobs.j[jobIdx];
    gSeed = pc.seed;
    gLod = job.lod;
    gScale = float(1 << job.lod);
    wg_init();
    ivec2 lc = ivec2(gl_GlobalInvocationID.xy);
    bool valid = lc.x < CA && lc.y < CA;

    uint nonAir = 0u;
    uint opaque = 0u;

    if ((job.flags & GENJOB_PREVIEW) != 0u) {
        // a prop preview stands on nothing: the props pass draws it into empty space
        if (valid) {
            uint base = jobIdx * SLOT_U16;
            for (int ly = 0; ly < CA; ++ly) pc.voxels.v[base + uint(ly * CA * CA + lc.y * CA + lc.x)] = uint16_t(0u);
            pc.columns.c[jobIdx * uint(CA * CA) + uint(lc.y * CA + lc.x)] = uint16_t(0u);
        }
        valid = false;
    }

    if (valid) {
        int scale = 1 << job.lod;
        int half_ = scale >> 1;
        vec2 xz = vec2(job.origin.xz + (lc - 1) * scale + half_) + 0.5;
        WgColumn col;
        wg_column(xz, col);
        if (writeColumns) pc.columns.c[jobIdx * uint(CA * CA) + uint(lc.y * CA + lc.x)] = uint16_t(wg_main_biome(col));

        uint base = jobIdx * SLOT_U16;
        // A cell must come out the same whichever chunk generates it (chunks also generate their
        // neighbours' apron cells), so what a cell sees above it is measured over fixed windows:
        //   depth     the run of solid cells directly above, capped at the deepest material
        //             threshold of the column's biomes (wg_depth_max), so layering stays exact;
        //   skyOpen   no solid cell up to WG_LOOKAHEAD cells above the cell, or above sea level
        //             for a cell below it (flooded caves, ocean floors);
        //   airAbove  air directly above (materials only test > 0).
        // The look-ahead (ly >= CA) covers these windows for cells above sea level. When they
        // reach further, the loop probes upwards after the top look-ahead cell: below a solid one
        // to measure deep layering, below an air one (under sea level) to look for a roof. ly = -1
        // is the cell below the slot, evaluated only for what it does to the bottom apron cell
        // (cave ceiling material, ground cover on it).
        // Single loop: one wg_density call site keeps the generated shader small.
        // Only the cells this layer owns are written. The loop spans them (one cell of margin
        // either way; the writes test ownership exactly) with the look-ahead above and the cell
        // below: every window is measured relative to its cell, so where the loop starts doesn't
        // change a result.
        float fs = float(scale);
        float cellY0 = float(job.origin.y - scale + half_) + 0.5;  // centre of cell ly = 0
        int lyLo = int(clamp(ceil((pc.ownMin - cellY0) / fs) - 1.0, 0.0, float(CA)));
        int lyHi = int(clamp(ceil((pc.ownMax - cellY0) / fs), -1.0, float(CA - 1)));
        int top = lyHi + WG_LOOKAHEAD;
        float skyReach = float(WG_LOOKAHEAD) * fs;
        float seaReach = gSeaLevel + skyReach;
        int needCells = min(int(ceil(wg_depth_max(col) / fs)), WG_LOOKAHEAD + 32);
        float depthCap = float(needCells) * fs;
        bool depthWanted = needCells > WG_LOOKAHEAD;
        bool skyWanted = cellY0 - fs < gSeaLevel && cellY0 + float(top) * fs < seaReach;
        bool depthOn = false;
        bool skyOn = false;
        int probe = 0;
        float depth = 0.0;
        float airAbove = 0.0;
        float solidY = 1e30;  // centre of the lowest solid cell seen so far
        uint prevBlock = 0u;
        int prevLy = -1;
        bool prevCarved = false;
        bool prevSolid = false;
        for (int ly = top; ly >= lyLo - 1 && lyHi >= lyLo;) {
            vec3 p = vec3(xz.x, cellY0 + float(probe > 0 ? probe : ly) * fs, xz.y);
            bool carved;
            float d = wg_density_c(p, col, carved);
            bool solid = d > 0.0;
            if (probe > 0) {
                int k = probe - top;
                if (depthOn) {
                    if (solid) depth = min(depth + fs, depthCap);
                    depthOn = solid && k < needCells - WG_LOOKAHEAD;
                }
                if (skyOn) {
                    if (solid || k >= 512) solidY = p.y;
                    skyOn = !solid && k < 512 && p.y + fs <= seaReach;
                }
                if (depthOn || skyOn) {
                    ++probe;
                } else {
                    probe = 0;
                    --ly;
                }
                continue;
            }
            if (ly < CA) {
                bool below = ly < 0;
                bool skyOpen = solidY > max(p.y, gSeaLevel) + skyReach;
                uint b = solid ? wg_material(p, depth, airAbove, skyOpen, col) : wg_fluid(p, skyOpen, col);
                bool decorated = false;
                if (solid && prevCarved) {
                    // cave floor
                    uint cb = wg_cave_biome(p, col);
                    if (cb != NO_CAVE_BIOME) {
                        b = wg_cave_material(cb, p, 0.0, airAbove);
                        if (prevLy >= 0 && prevBlock == 0u && job.lod == 0u && wg_owns(p.y + fs)) {
                            uint deco = wg_cave_decorate(cb, p + vec3(0.0, float(scale), 0.0), b);
                            if (deco != 0u) {
                                pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(deco);
                                nonAir += 1u;
                            }
                        }
                        decorated = true;
                    }
                }
                if (!solid && carved && prevSolid && prevLy >= 0 && prevBlock != 0u && wg_owns(p.y + fs)) {
                    // cave ceiling: re-material the solid cell above
                    vec3 q = p + vec3(0.0, float(scale), 0.0);
                    uint cb = wg_cave_biome(q, col);
                    if (cb != NO_CAVE_BIOME) {
                        pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(wg_cave_material(cb, q, max(depth, 1.0), 0.0));
                    }
                }
                if (!decorated && solid && depth == 0.0 && prevLy >= 0 && prevBlock == 0u && job.lod == 0u && wg_owns(p.y + fs)) {
                    uint deco = wg_decorate(p + vec3(0.0, 1.0, 0.0), b, col);
                    if (deco != 0u) {
                        pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(deco);
                        nonAir += 1u;
                    }
                }
                if (!below && wg_owns(p.y)) {
                    pc.voxels.v[base + uint(ly * CA * CA + lc.y * CA + lc.x)] = uint16_t(b);
                    if (b != 0u) nonAir += 1u;
                    if (solid && b != 0u) opaque += 1u;
                }
                prevBlock = b;
                prevLy = ly;
            }
            prevCarved = carved && !solid;
            prevSolid = solid;
            if (solid) { depth = min(depth + fs, depthCap); airAbove = 0.0; solidY = p.y; }
            else { depth = 0.0; airAbove += fs; }
            if (ly == top) {
                depthOn = solid && depthWanted;
                skyOn = !solid && skyWanted;
                if (depthOn || skyOn) {
                    probe = top + 1;
                    continue;
                }
            }
            --ly;
        }
    }

    uint sNonAir = subgroupAdd(nonAir);
    uint sOpaque = subgroupAdd(opaque);
    if (subgroupElect()) {
        atomicAdd(pc.results.r[jobIdx].nonAir, sNonAir);
        atomicAdd(pc.results.r[jobIdx].opaque, sOpaque);
    }
}
