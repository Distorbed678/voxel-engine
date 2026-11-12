//! Bootstrap terrain used until the node graph compiler produces code (and as a fallback when a
//! graph fails to compile). Implements the generated-code contract of
//! shaders/worldgen/terrain_main.glsl.

pub const FALLBACK_TERRAIN: &str = r#"
struct WgColumn {
    float height;
    float mountain;
};

void wg_column(vec2 xz, out WgColumn c) {
    float continental = fbm2(xz / 1400.0, gSeed + 1u, 4, 2.0, 0.5);
    float hills = fbm2(xz / 240.0, gSeed + 2u, 5, 2.0, 0.5);
    float ridges = ridged2(xz / 700.0, gSeed + 3u, 5, 2.0, 0.5);
    float mountainMask = smoothstep(0.05, 0.45, continental);
    c.height = 66.0 + continental * 36.0 + hills * 14.0 + mountainMask * (ridges * 0.5 + 0.5) * 110.0;
    c.mountain = mountainMask;
}

float wg_density(vec3 p, in WgColumn c) {
    float d = c.height - p.y;
    d += fbm3(p / vec3(52.0, 36.0, 52.0), gSeed + 4u, 3, 2.0, 0.5) * 9.0 * (0.35 + c.mountain);
    // spaghetti tunnels: intersection of two noise iso-surfaces
    float a = simplex3(p / vec3(80.0, 48.0, 80.0), gSeed + 5u);
    float b = simplex3(p / vec3(80.0, 48.0, 80.0), gSeed + 6u);
    if (gLod >= 2u) return d; // caves are invisible from afar: skip them on coarse LODs
    if (a * a + b * b < 0.010 && p.y < c.height - 2.0) d = min(d, -1.0);
    // cheese caverns
    float cheese = simplex3(p / vec3(110.0, 60.0, 110.0), gSeed + 7u);
    if (cheese > 0.6 && p.y < c.height - 14.0) d = min(d, -1.0);
    return d;
}

uint wg_material(vec3 p, float depth, float airAbove, in WgColumn c) {
    bool underground = p.y < c.height - 7.0;
    if (underground) {
        if (p.y < -48.0) return B_DEEPSLATE;
        float ore = simplex3(p / 6.0, gSeed + 20u);
        if (ore > 0.82) return B_COAL_ORE;
        if (ore < -0.86 && p.y < 30.0) return B_IRON_ORE;
        if (depth < 1.0 && airAbove > 0.0) return B_GRAVEL;
        return B_STONE;
    }
    float slopeish = c.mountain;
    if (depth < 1.0) {
        if (p.y < 64.5) return B_SAND;
        if (p.y > 178.0) return B_SNOW;
        if (p.y > 150.0 && slopeish > 0.6) return B_STONE;
        return B_GRASS;
    }
    if (depth < 4.0) {
        if (p.y < 64.5) return B_SAND;
        if (p.y > 150.0) return B_STONE;
        return B_DIRT;
    }
    return B_STONE;
}

uint wg_fluid(vec3 p, in WgColumn c) {
    if (p.y < 62.0 && p.y > c.height - 3.0) return B_WATER;
    return 0u;
}

uint wg_decorate(vec3 p, uint ground, in WgColumn c) {
    return 0u;
}
"#;
