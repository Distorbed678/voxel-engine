// Prefab props (the other props are Prop graphs: props_lib.glsl). `r` is the cell position
// relative to the anchor (the first air block above the ground). Returns a block or 0.

uint shape_prefab(in PropLayer L, vec3 r, uint seed) {
    uvec4 e = pc.prefabTable.t[L.variant];
    ivec3 size = ivec3(e.yzw);
    ivec3 q = ivec3(floor(r)) + ivec3(0, int(L.b1), 0);
    uint rot = L.b0 != 0u ? (seed % 4u) : 0u;
    // rotate around the prefab's horizontal centre
    ivec2 c = ivec2(q.x, q.z);
    for (uint i = 0u; i < rot; ++i) c = ivec2(-c.y, c.x);
    ivec3 l = ivec3(c.x + size.x / 2, q.y, c.y + size.z / 2);
    if (any(lessThan(l, ivec3(0))) || any(greaterThanEqual(l, size))) return 0u;
    // local directions map to world ones by `rot` turns the other way
    return rotate_value(uint(pc.prefabData.c[e.x + uint((l.y * size.z + l.z) * size.x + l.x)]), (4u - rot) & 3u);
}

// the box searched around an anchor: radius, height above, depth below
void prop_bounds(in PropLayer L, out float R, out float H, out float below) {
    if (L.kind == PROP_PREFAB) {
        R = ceil(max(L.a, L.c) * 0.75) + 1.0;
        H = L.b + 1.0;
        below = float(L.b1);
    } else {
        R = L.R;
        H = L.H;
        below = L.below;
    }
}
