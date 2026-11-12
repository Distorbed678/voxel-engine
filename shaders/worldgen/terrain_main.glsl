// Terrain pass main. Expects the generated code to define:
//   struct WgColumn;                                        (2D, per column)
//   void  wg_column(vec2 xz, out WgColumn c);
//   float wg_density(vec3 p, in WgColumn c);               (> 0 = solid)
//   uint  wg_material(vec3 p, float depth, float ceilDist, in WgColumn c);
//   uint  wg_fluid(vec3 p, in WgColumn c);                  (block for air cells, 0 = air)
//   uint  wg_decorate(vec3 p, uint ground, in WgColumn c);  (block placed above ground, 0 = none)
//
// One invocation = one (x, z) column of 34 cells (chunk + apron). Cells are generated top-down
// so we know how deep below the surface each solid cell is (for material layering).

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
    ivec2 lc = ivec2(gl_GlobalInvocationID.xy);
    bool valid = lc.x < CA && lc.y < CA;

    uint nonAir = 0u;
    uint opaque = 0u;

    if (valid) {
        int scale = 1 << job.lod;
        int half_ = scale >> 1;
        // world position of this cell's sample (block centre of the cell's central block)
        vec2 xz = vec2(job.origin.xz + (lc - 1) * scale + half_) + 0.5;
        WgColumn col;
        wg_column(xz, col);

        uint base = jobIdx * SLOT_U16;
        float depth = 0.0;      // solid cells directly above (in blocks)
        float airAbove = 0.0;   // air cells directly above (in blocks), for ceilings
        // look-ahead above the chunk so surface depth is continuous across chunk borders
        for (int ly = CA - 1 + WG_LOOKAHEAD; ly >= CA; --ly) {
            vec3 p = vec3(xz.x, float(job.origin.y + (ly - 1) * scale + half_) + 0.5, xz.y);
            float d = wg_density(p, col);
            if (d > 0.0) { depth += float(scale); airAbove = 0.0; }
            else { depth = 0.0; airAbove += float(scale); }
        }
        uint prevBlock = 0u;   // block of the cell above (in this column)
        int prevLy = -1;
        for (int ly = CA - 1; ly >= 0; --ly) {
            vec3 p = vec3(xz.x, float(job.origin.y + (ly - 1) * scale + half_) + 0.5, xz.y);
            float d = wg_density(p, col);
            uint b;
            if (d > 0.0) {
                b = wg_material(p, depth, airAbove, col);
                // surface decoration on top of ground (grass tufts, flowers, ...)
                if (depth == 0.0 && prevLy >= 0 && prevBlock == 0u) {
                    uint deco = wg_decorate(p + vec3(0.0, float(scale), 0.0), b, col);
                    if (deco != 0u) {
                        pc.voxels.v[base + uint(prevLy * CA * CA + lc.y * CA + lc.x)] = uint16_t(deco);
                        nonAir += 1u;
                    }
                }
                depth += float(scale);
                airAbove = 0.0;
            } else {
                b = wg_fluid(p, col);
                depth = 0.0;
                airAbove += float(scale);
            }
            pc.voxels.v[base + uint(ly * CA * CA + lc.y * CA + lc.x)] = uint16_t(b);
            if (b != 0u) nonAir += 1u;
            if (d > 0.0 && b != 0u) opaque += 1u;
            prevBlock = b;
            prevLy = ly;
        }
    }

    // Reduce occupancy statistics (used to skip meshing of empty / fully solid chunks).
    uint sNonAir = subgroupAdd(nonAir);
    uint sOpaque = subgroupAdd(opaque);
    if (subgroupElect()) {
        atomicAdd(pc.results.r[jobIdx].nonAir, sNonAir);
        atomicAdd(pc.results.r[jobIdx].opaque, sOpaque);
    }
}
