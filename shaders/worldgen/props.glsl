// Procedural prop shapes. `r` is the cell sample position relative to the anchor (the first air
// block above the ground for surface props, the centre for geodes). Returns a block or 0.
// gScale (cell size) is used to keep thin features (trunks) visible on coarse LODs.

float prop_rand(uint seed, uint k) { return hash_to_unit(hash_u32(seed * 0x9E3779B9u + k * 0x85EBCA6Bu)); }

void prop_bounds(in PropLayer L, out float R, out float H, out float below) {
    switch (L.kind) {
        case PROP_TREE: R = L.c + 3.0 + (L.variant == 3u ? 2.0 : 0.0) + (L.variant == 4u ? 3.0 : 0.0); H = L.b + L.c + 3.0; below = 1.0; break;
        case PROP_MUSHROOM: R = L.c + 1.0; H = L.b + 2.0; below = 0.0; break;
        case PROP_BOULDER: R = L.b + 1.0; H = L.b * 1.6 + 1.0; below = L.b; break;
        case PROP_COLUMN: R = max(L.c, 2.0) + 1.0; H = L.b + 1.0; below = 1.0; break;
        case PROP_SPIKE: R = L.c + 1.0; H = L.b + 1.0; below = 3.0; break;
        case PROP_GEODE: R = L.c + 1.0; H = L.c + 1.0; below = L.c + 1.0; break;
        default: R = 0.0; H = 0.0; below = 0.0; break;
    }
}

bool trunk_at(vec3 r, vec2 off, float halfw) {
    vec2 d = abs(r.xz - off);
    return max(d.x, d.y) <= max(halfw, gScale * 0.5 - 0.01);
}

float leaf_noise(vec3 r, uint seed) {
    return hash_to_unit(hash_u32(uint(int(r.x) + 64) * 73856093u ^ uint(int(r.y) + 64) * 19349663u ^ uint(int(r.z) + 64) * 83492791u ^ seed));
}

uint shape_tree(in PropLayer L, vec3 r, uint seed) {
    float h = floor(mix(L.a, L.b + 0.999, prop_rand(seed, 1u)));
    float R = L.c * mix(0.85, 1.15, prop_rand(seed, 2u));
    float ln = leaf_noise(r, seed);
    uint trunk = L.b0, leaves = L.b1;
    switch (L.variant) {
    case 0u: { // round (oak)
        if (r.y >= 0.0 && r.y < h && trunk_at(r, vec2(0), 0.0)) return trunk;
        vec3 c = vec3(0.0, h - 1.0, 0.0);
        float d = length((r - c) * vec3(1.0, 1.35, 1.0));
        if (d < R + 0.4 - ln * 0.9 && r.y >= h - R) return leaves;
        break;
    }
    case 1u: { // birch: tall, narrow canopy
        if (r.y >= 0.0 && r.y < h && trunk_at(r, vec2(0), 0.0)) return trunk;
        vec3 c = vec3(0.0, h - 1.5, 0.0);
        float d = length((r - c) * vec3(1.0, 0.65, 1.0));
        if (d < R + 0.3 - ln * 0.8 && r.y >= h - R * 2.0) return leaves;
        break;
    }
    case 2u: { // spruce: stacked cones
        if (r.y >= 0.0 && r.y < h && trunk_at(r, vec2(0), 0.0)) return trunk;
        if (r.y >= 2.0 && r.y <= h + 1.0) {
            float t = (r.y - 2.0) / max(h - 1.0, 1.0);
            float rad = (1.0 - t) * R + 0.6;
            if (mod(r.y, 2.0) < 1.0) rad *= 0.7;
            if (length(r.xz) <= rad - ln * 0.4) return leaves;
        }
        break;
    }
    case 3u: { // palm: curved trunk, drooping fronds
        float dir = prop_rand(seed, 3u) * 6.2831;
        vec2 bend = vec2(cos(dir), sin(dir));
        float t = clamp(r.y / h, 0.0, 1.0);
        vec2 off = bend * t * t * 2.5;
        if (r.y >= 0.0 && r.y < h && trunk_at(r, off, 0.0)) return trunk;
        vec2 top = bend * 2.5;
        vec2 d = r.xz - top;
        float dist = length(d);
        float droop = floor(dist * dist / (R * 1.6));
        bool arm = abs(d.x) < 0.6 || abs(d.y) < 0.6 || abs(abs(d.x) - abs(d.y)) < 0.8;
        if (arm && dist <= R && abs(r.y - (h - droop)) < 0.6) return leaves;
        if (dist < 1.2 && abs(r.y - h) < 0.6) return leaves;
        break;
    }
    case 4u: { // acacia: leaning trunk, flat canopy
        float dir = prop_rand(seed, 3u) * 6.2831;
        vec2 bend = vec2(cos(dir), sin(dir));
        float split = floor(h * 0.5);
        vec2 off = r.y > split ? bend * (r.y - split) : vec2(0);
        if (r.y >= 0.0 && r.y < h && trunk_at(r, off, 0.0)) return trunk;
        vec2 top = bend * (h - split);
        vec2 d = r.xz - top;
        if (abs(r.y - h) < 0.6 && length(d) <= R + 0.5 - ln) return leaves;
        if (abs(r.y - (h + 1.0)) < 0.6 && length(d) <= R * 0.55) return leaves;
        break;
    }
    case 5u: { // dead tree with branches
        if (r.y >= 0.0 && r.y < h && trunk_at(r, vec2(0), 0.0)) return trunk;
        for (uint i = 0u; i < 3u; ++i) {
            float by = floor(h * (0.45 + 0.18 * float(i)));
            float a = prop_rand(seed, 10u + i) * 6.2831;
            vec2 bd = vec2(cos(a), sin(a));
            float along = dot(r.xz, bd);
            if (along > 0.5 && along < R + 0.5 && length(r.xz - bd * along) < 0.6 && abs(r.y - (by + floor(along * 0.6))) < 0.6) return trunk;
        }
        break;
    }
    case 6u: { // bush
        vec3 c = vec3(0.0, 0.0, 0.0);
        float d = length((r - c) * vec3(1.0, 1.4, 1.0));
        if (r.y >= 0.0 && d < R * 0.75 - ln * 0.5) return leaves;
        break;
    }
    default: { // willow: round canopy with hanging strands
        if (r.y >= 0.0 && r.y < h && trunk_at(r, vec2(0), 0.0)) return trunk;
        vec3 c = vec3(0.0, h, 0.0);
        float d = length((r - c) * vec3(1.0, 1.6, 1.0));
        if (d < R + 0.3 - ln * 0.7) return leaves;
        float rr = length(r.xz);
        if (rr > R - 1.2 && rr < R + 0.3 && r.y > h - 4.0 - ln * 3.0 && r.y < h && leaf_noise(vec3(r.x, 0.0, r.z), seed) > 0.45) return leaves;
        break;
    }
    }
    return 0u;
}

uint shape_mushroom(in PropLayer L, vec3 r, uint seed) {
    float h = floor(mix(L.a, L.b + 0.999, prop_rand(seed, 1u)));
    float R = L.c * mix(0.85, 1.15, prop_rand(seed, 2u));
    if (r.y >= 0.0 && r.y < h && trunk_at(r, vec2(0), 0.5)) return L.b0;
    float rr = length(r.xz);
    if (abs(r.y - h) < 0.6 && rr <= R) return L.b1;
    if (abs(r.y - (h + 1.0)) < 0.6 && rr <= R - 1.5) return L.b1;
    if (abs(r.y - (h - 1.0)) < 0.6 && rr <= R + 0.5 && rr > R - 1.0) return L.b1;
    return 0u;
}

uint shape_boulder(in PropLayer L, vec3 r, uint seed) {
    float R = mix(L.a, L.b, prop_rand(seed, 1u));
    vec3 c = vec3(0.0, R * 0.25 - 1.0, 0.0);
    float n = leaf_noise(floor(r * 0.5), seed);
    float d = length((r - c) * vec3(1.0, 1.25, 1.0));
    if (d < R + n * 0.8 - 0.4) return leaf_noise(r, seed ^ 7u) > 0.6 ? L.b1 : L.b0;
    return 0u;
}

uint shape_column(in PropLayer L, vec3 r, uint seed) {
    float h = floor(mix(L.a, L.b + 0.999, prop_rand(seed, 1u)));
    if (L.c > 0.75) {
        // thick hexagon-ish pillar
        if (r.y >= -1.0 && r.y < h && length(r.xz) <= L.c) return L.b0;
        return 0u;
    }
    if (r.y >= 0.0 && r.y < h && trunk_at(r, vec2(0), 0.0)) return L.b0;
    if (L.variant == 1u && h >= 3.0) {
        for (uint i = 0u; i < 2u; ++i) {
            if (prop_rand(seed, 20u + i) < 0.45) continue;
            float ay = floor(mix(1.0, h - 2.0, prop_rand(seed, 30u + i)));
            vec2 d = (i == 0u) ? vec2(1, 0) : vec2(-1, 0);
            if (prop_rand(seed, 40u) > 0.5) d = d.yx;
            // horizontal stub then vertical arm
            if (abs(r.y - ay) < 0.6 && length(r.xz - d) < 0.6) return L.b0;
            if (r.y >= ay && r.y < ay + 2.0 + floor(prop_rand(seed, 50u + i) * 2.0) && length(r.xz - d * 2.0) < 0.6) return L.b0;
            if (abs(r.y - ay) < 0.6 && length(r.xz - d * 2.0) < 0.6) return L.b0;
        }
    }
    return 0u;
}

uint shape_spike(in PropLayer L, vec3 r, uint seed) {
    float h = mix(L.a, L.b, prop_rand(seed, 1u));
    if (r.y < -3.0 || r.y >= h) return 0u;
    float t = clamp(r.y / h, 0.0, 1.0);
    float rad = L.c * pow(1.0 - t, 1.3) + 0.35;
    if (length(r.xz) <= rad) return L.b0;
    return 0u;
}

uint shape_geode(in PropLayer L, vec3 r, uint seed) {
    float R = L.c * mix(0.8, 1.1, prop_rand(seed, 1u));
    float d = length(r) + (leaf_noise(floor(r * 0.5), seed) - 0.5) * 0.8;
    if (d > R) return 0xFFFFu;           // outside: leave untouched
    if (d > R - 1.0) return L.b0;        // outer shell
    if (d > R - 2.0) return L.b1;        // middle shell
    if (d > R - 3.0) return L.b2;        // crystal lining
    return 0u;                           // hollow
}

uint prop_shape(in PropLayer L, vec3 r, uint seed) {
    switch (L.kind) {
        case PROP_TREE: return shape_tree(L, r, seed);
        case PROP_MUSHROOM: return shape_mushroom(L, r, seed);
        case PROP_BOULDER: return shape_boulder(L, r, seed);
        case PROP_COLUMN: return shape_column(L, r, seed);
        case PROP_SPIKE: return shape_spike(L, r, seed);
        case PROP_GEODE: return shape_geode(L, r, seed);
        default: return 0u;
    }
}
