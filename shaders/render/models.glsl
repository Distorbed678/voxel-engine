#ifndef MODELS_GLSL
#define MODELS_GLSL
// Small non-cube block models (plants, slabs, stairs, lanterns, ...). Each model is at most
// MODEL_MAX_QUADS quads in block-local [0,1]^3 space. Keep ids in sync with blocks.rs `models`.

#define MODEL_MAX_QUADS 10u
#define MODEL_CROSS 1u
#define MODEL_SLAB 2u
#define MODEL_STAIRS 3u
#define MODEL_LANTERN 4u
#define MODEL_PEBBLES 5u
#define MODEL_MUSHROOM 6u
#define MODEL_TALL_CROSS 7u

struct ModelQuad {
    vec3 o;        // origin corner
    vec3 U;        // edge along texture u
    vec3 V;        // edge along texture -v (up)
    vec4 uv;       // u0, v0, u1, v1 (v0 = bottom)
    uint texDir;   // which block face texture to use
    bool twoSided;
    bool wind;
};

// Axis-aligned box face i (0..5 = +X,-X,+Y,-Y,+Z,-Z) of box [a, b]
ModelQuad box_face(vec3 a, vec3 b, uint i) {
    ModelQuad q;
    q.twoSided = false;
    q.wind = false;
    q.texDir = i;
    vec3 s = b - a;
    switch (i) {
        case 0u: q.o = vec3(b.x, a.y, b.z); q.U = vec3(0, 0, -s.z); q.V = vec3(0, s.y, 0); q.uv = vec4(1.0 - b.z, a.y, 1.0 - a.z, b.y); break;
        case 1u: q.o = vec3(a.x, a.y, a.z); q.U = vec3(0, 0, s.z); q.V = vec3(0, s.y, 0); q.uv = vec4(a.z, a.y, b.z, b.y); break;
        case 2u: q.o = vec3(a.x, b.y, b.z); q.U = vec3(s.x, 0, 0); q.V = vec3(0, 0, -s.z); q.uv = vec4(a.x, 1.0 - b.z, b.x, 1.0 - a.z); break;
        case 3u: q.o = vec3(a.x, a.y, a.z); q.U = vec3(s.x, 0, 0); q.V = vec3(0, 0, s.z); q.uv = vec4(a.x, a.z, b.x, b.z); break;
        case 4u: q.o = vec3(a.x, a.y, b.z); q.U = vec3(s.x, 0, 0); q.V = vec3(0, s.y, 0); q.uv = vec4(a.x, a.y, b.x, b.y); break;
        default: q.o = vec3(b.x, a.y, a.z); q.U = vec3(-s.x, 0, 0); q.V = vec3(0, s.y, 0); q.uv = vec4(1.0 - b.x, a.y, 1.0 - a.x, b.y); break;
    }
    return q;
}

// rotate a block-local point / vector around the block's vertical axis (rot * 90 degrees)
vec3 rot_p(vec3 p, uint rot) {
    vec3 c = p - vec3(0.5, 0.0, 0.5);
    for (uint i = 0u; i < rot; ++i) c = vec3(-c.z, c.y, c.x);
    return c + vec3(0.5, 0.0, 0.5);
}
vec3 rot_v(vec3 v, uint rot) {
    for (uint i = 0u; i < rot; ++i) v = vec3(-v.z, v.y, v.x);
    return v;
}

// Returns false for unused slots.
bool model_quad(uint model, uint i, uint seed, uint nmask, uint state, out ModelQuad q) {
    q.twoSided = false;
    q.wind = false;
    switch (model) {
    case MODEL_CROSS:
    case MODEL_TALL_CROSS: {
        if (i >= 2u) return false;
        float h = model == MODEL_TALL_CROSS ? 1.25 : mix(0.75, 1.0, float(seed & 255u) / 255.0);
        float w = model == MODEL_TALL_CROSS ? 1.1 : 0.9;
        vec2 j = (vec2(float((seed >> 8) & 255u), float((seed >> 16) & 255u)) / 255.0 - 0.5) * 0.25;
        vec3 c = vec3(0.5 + j.x, 0.0, 0.5 + j.y);
        vec3 d = i == 0u ? vec3(0.7071, 0, 0.7071) : vec3(0.7071, 0, -0.7071);
        q.o = c - d * 0.5 * w;
        q.U = d * w;
        q.V = vec3(0, h, 0);
        q.uv = vec4(0, 0, 1, 1);
        q.texDir = 4u;
        q.twoSided = true;
        q.wind = true;
        return true;
    }
    case MODEL_SLAB:
        if (i >= 6u) return false;
        q = box_face(vec3(0), vec3(1, 0.5, 1), i);
        return true;
    case MODEL_STAIRS: {
        // orientation: the tall back faces an opaque horizontal neighbour (auto-connect)
        uint rot = 0u;
        if ((nmask & (1u << 4u)) != 0u) rot = 0u;       // +Z neighbour
        else if ((nmask & (1u << 1u)) != 0u) rot = 1u;  // -X
        else if ((nmask & (1u << 5u)) != 0u) rot = 2u;  // -Z
        else if ((nmask & (1u << 0u)) != 0u) rot = 3u;  // +X
        if (state != 0u) rot = (state - 1u) & 3u;       // placed with a fixed facing
        if (i < 6u) q = box_face(vec3(0), vec3(1, 0.5, 1), i);
        else if (i < 10u) {
            uint faces[4] = uint[4](2u, 5u, 0u, 1u); // top, front(-Z), +X, -X of the back step
            q = box_face(vec3(0, 0.5, 0.5), vec3(1, 1, 1), faces[i - 6u]);
        } else return false;
        q.o = rot_p(q.o, rot);
        q.U = rot_v(q.U, rot);
        q.V = rot_v(q.V, rot);
        return true;
    }
    case MODEL_LANTERN:
        if (i < 6u) { q = box_face(vec3(0.3, 0.0, 0.3), vec3(0.7, 0.55, 0.7), i); return true; }
        if (i == 6u) { q = box_face(vec3(0.38, 0.55, 0.38), vec3(0.62, 0.66, 0.62), 2u); return true; }
        return false;
    case MODEL_PEBBLES: {
        if (i >= 10u) return false;
        uint b = i / 5u;
        vec2 c = b == 0u ? vec2(0.3, 0.35) : vec2(0.68, 0.62);
        float s = b == 0u ? 0.16 : 0.12;
        float hgt = b == 0u ? 0.14 : 0.1;
        uint faces[5] = uint[5](2u, 0u, 1u, 4u, 5u);
        q = box_face(vec3(c.x - s, 0.0, c.y - s), vec3(c.x + s, hgt, c.y + s), faces[i % 5u]);
        return true;
    }
    case MODEL_MUSHROOM: {
        if (i < 4u) {
            uint faces[4] = uint[4](0u, 1u, 4u, 5u);
            q = box_face(vec3(0.42, 0.0, 0.42), vec3(0.58, 0.45, 0.58), faces[i]);
            q.texDir = 3u; // stem uses the bottom texture (cap colour is on top/sides)
            return true;
        }
        if (i < 10u) {
            q = box_face(vec3(0.22, 0.42, 0.22), vec3(0.78, 0.68, 0.78), i - 4u);
            return true;
        }
        return false;
    }
    default:
        if (i >= 6u) return false;
        q = box_face(vec3(0.1), vec3(0.9), i);
        return true;
    }
}

#endif
