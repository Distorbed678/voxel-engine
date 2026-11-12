// Prop graphs: building and evaluating the primitives of one placed instance.
// Included by the prelude in the props pass (PROPS_PASS), before the generated build functions.
//
// A Prop graph compiles to `propN_build()`, run by one thread per instance. It appends primitives
// (sphere, ellipsoid, tapered capsule, box, cylinder/cone) to its list in prop space: the anchor
// column is at x = z = 0 and y = 0 is the top of the ground, so cell centres sit at integer x/z
// and half-integer y. Tree nodes first grow a skeleton of segments and emit primitives
// from it. Then every thread tests its cells against the list (`prop_eval`): the list is built
// once per instance, never per voxel.
//
// Each primitive has a block (and a second, speckled block), a priority (where shapes overlap the
// higher one wins, the later one on ties) and an op: Add places its block, Carve places air,
// Remove takes back what was placed before it, Keep takes back what lies outside it. Smooth
// unions melt consecutive primitives of one group together. Noise displaces the surface; holes
// leave out a share of the cells.

#define PRIM_SPHERE 0u
#define PRIM_ELLIPSOID 1u
#define PRIM_CAPSULE 2u
#define PRIM_BOX 3u
#define PRIM_CYLINDER 4u

#define OP_ADD 0u
#define OP_CARVE 1u
#define OP_REMOVE 2u
#define OP_KEEP 3u

#define MAX_PRIMS 192u
#define MAX_SEGS 96u
// prop_eval result: leave the cell as it is
#define PROP_NOTHING 0xFFFFu

#define SEG_TIP 16u
#define SEG_CHAIN 32u

struct Prim {
    vec4 a;   // centre / start (prop space); w = radius (at the start)
    vec4 b;   // end (capsule, cylinder) / radii (ellipsoid) / half size (box); w = radius at the end
    vec4 q;   // rotation (ellipsoid, box) as a quaternion
    vec4 n;   // x: noise amount, y: noise frequency, z: share of holes, w: share of block 2
    uvec4 m;  // x: kind | op << 4 | priority << 8 | group << 16; y: block | block2 << 16; z: seed; w: smoothness (float bits)
};

struct Seg {
    vec4 a;      // start, w = radius
    vec4 b;      // end, w = radius
    uint flags;  // level (bits 0..3) | SEG_TIP | SEG_CHAIN (first segment of a chain)
};

// Instances are built in parallel (one thread each) into the job's scratch in device memory
// (`pc.propPrims` / `pc.propSegs`, at gPrimBase / gSegBase); the instance being drawn is copied to
// `sPrim` for the per-cell evaluation.
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer PrimBuf { Prim p[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer SegBuf { Seg s[]; };
shared Prim sPrim[MAX_PRIMS];
#define PRIM(i) PrimBuf(pc.propPrims).p[gPrimBase + (i)]
#define SEG(i) SegBuf(pc.propSegs).s[gSegBase + (i)]

// ---- build state (one thread builds one instance) -------------------------------------------

struct Xf {
    mat3 m;   // rotation * scale
    vec3 t;
    float s;  // scale (radii)
};

Xf gXf;
uint gInst;      // instance seed (per anchor; each Repeat copy gets its own)
vec3 gAnchorP;   // world position of the anchor: density nodes in a prop graph see it as p
uint gPrimN;
uint gSegN;
uint gGroupN;
uint gPrimBase;  // this instance's region of the scratch
uint gSegBase;

mat3 rot_y(float a) { float c = cos(a), s = sin(a); return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c); }
mat3 rot_x(float a) { float c = cos(a), s = sin(a); return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c); }
mat3 rot_z(float a) { float c = cos(a), s = sin(a); return mat3(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0); }

Xf xf_identity() { return Xf(mat3(1.0), vec3(0.0), 1.0); }
// parent * child
Xf xf_mul(Xf a, Xf b) { return Xf(a.m * b.m, a.m * b.t + a.t, a.s * b.s); }
// offset, yaw / pitch / roll in degrees, uniform scale
Xf xf_make(vec3 t, vec3 ypr, float s) {
    vec3 r = radians(ypr);
    return Xf(rot_y(r.x) * rot_x(r.y) * rot_z(r.z) * s, t, s);
}
Xf xf_mirror(vec3 axes) { return Xf(mat3(vec3(axes.x, 0.0, 0.0), vec3(0.0, axes.y, 0.0), vec3(0.0, 0.0, axes.z)), vec3(0.0), 1.0); }
vec3 xf_point(vec3 p) { return gXf.m * p + gXf.t; }

// per-instance random number in [0, 1) for key k
float pr(uint k) { return hash_to_unit(hash_u32(gInst * 0x9E3779B9u + k * 0x85EBCA6Bu + 0x632BE5ABu)); }

vec4 quat_from_mat(mat3 m) {
    // the rotation part of m (scale and mirroring removed: shapes using it are symmetric)
    vec3 c0 = normalize(m[0]), c1 = normalize(m[1]);
    vec3 c2 = cross(c0, c1);
    c1 = cross(c2, c0);
    float tr = c0.x + c1.y + c2.z;
    vec4 q;
    if (tr > 0.0) {
        float s = sqrt(tr + 1.0) * 2.0;
        q = vec4((c1.z - c2.y) / s, (c2.x - c0.z) / s, (c0.y - c1.x) / s, 0.25 * s);
    } else if (c0.x > c1.y && c0.x > c2.z) {
        float s = sqrt(1.0 + c0.x - c1.y - c2.z) * 2.0;
        q = vec4(0.25 * s, (c1.x + c0.y) / s, (c2.x + c0.z) / s, (c1.z - c2.y) / s);
    } else if (c1.y > c2.z) {
        float s = sqrt(1.0 + c1.y - c0.x - c2.z) * 2.0;
        q = vec4((c1.x + c0.y) / s, 0.25 * s, (c2.y + c1.z) / s, (c2.x - c0.z) / s);
    } else {
        float s = sqrt(1.0 + c2.z - c0.x - c1.y) * 2.0;
        q = vec4((c2.x + c0.z) / s, (c2.y + c1.z) / s, 0.25 * s, (c0.y - c1.x) / s);
    }
    return normalize(q);
}

// rotate v by the inverse of q
vec3 qrot_inv(vec4 q, vec3 v) {
    vec3 u = -q.xyz;
    return v + 2.0 * cross(u, cross(u, v) + q.w * v);
}

// column blocks (logs) lying along a segment take its axis
uint orient_column(uint b, vec3 d) {
    if (b == 0u || ((pc.blocks.b[b & 0xFFFu].flags >> 25) & 3u) != 1u) return b;
    vec3 a = abs(d);
    uint st = (a.y >= a.x && a.y >= a.z) ? 0u : (a.x >= a.z ? 1u : 2u);
    return (b & 0xFFFu) | (st << 12);
}

Prim pb_new(uint kind, uint block, uint prio) {
    Prim p;
    p.a = vec4(0.0);
    p.b = vec4(0.0);
    p.q = vec4(0.0, 0.0, 0.0, 1.0);
    p.n = vec4(0.0);
    p.m = uvec4(kind | (min(prio, 255u) << 8), block & 0xFFFFu, 0u, 0u);
    return p;
}

void pb_push(Prim p) {
    if (gPrimN < MAX_PRIMS) {
        PRIM(gPrimN) = p;
        gPrimN++;
    }
}

void pb_sphere(vec3 c, float r, uint block, uint prio) {
    Prim p = pb_new(PRIM_SPHERE, block, prio);
    p.a = vec4(xf_point(c), r * gXf.s);
    pb_push(p);
}

void pb_ellipsoid(vec3 c, vec3 r, float yaw, uint block, uint prio) {
    Prim p = pb_new(PRIM_ELLIPSOID, block, prio);
    p.a = vec4(xf_point(c), 0.0);
    p.b = vec4(r * gXf.s, 0.0);
    p.q = quat_from_mat(gXf.m * rot_y(radians(yaw)));
    pb_push(p);
}

void pb_capsule(vec3 a, vec3 b, float ra, float rb, uint block, uint prio) {
    vec3 wa = xf_point(a), wb = xf_point(b);
    Prim p = pb_new(PRIM_CAPSULE, orient_column(block, wb - wa), prio);
    p.a = vec4(wa, ra * gXf.s);
    p.b = vec4(wb, rb * gXf.s);
    pb_push(p);
}

// flat ends: from `a` to `b`, radius ra at a and rb at b (a cone when rb is 0)
void pb_cylinder(vec3 a, vec3 b, float ra, float rb, uint block, uint prio) {
    vec3 wa = xf_point(a), wb = xf_point(b);
    Prim p = pb_new(PRIM_CYLINDER, orient_column(block, wb - wa), prio);
    p.a = vec4(wa, ra * gXf.s);
    p.b = vec4(wb, rb * gXf.s);
    pb_push(p);
}

void pb_box(vec3 c, vec3 size, float yaw, uint block, uint prio) {
    Prim p = pb_new(PRIM_BOX, block, prio);
    p.a = vec4(xf_point(c), 0.0);
    p.b = vec4(size * 0.5 * gXf.s, 0.0);
    p.q = quat_from_mat(gXf.m * rot_y(radians(yaw)));
    pb_push(p);
}

// a single cell (fruit, flowers): snapped to the cell grid of prop space
void pb_point(vec3 c, uint block, uint prio) {
    vec3 w = xf_point(c);
    w = vec3(round(w.x), floor(w.y) + 0.5, round(w.z));
    Prim p = pb_new(PRIM_SPHERE, block, prio);
    p.a = vec4(w, 0.45);
    pb_push(p);
}

// ---- modifiers on the primitives emitted since `s` ----

void pb_set_op(uint s, uint op) {
    for (uint i = s; i < gPrimN; ++i) PRIM(i).m.x = (PRIM(i).m.x & ~0xF0u) | (op << 4);
}

void pb_set_block(uint s, uint block, uint prio, uint block2, float share) {
    for (uint i = s; i < gPrimN; ++i) {
        uint k = PRIM(i).m.x & 15u;
        uint b = (k == PRIM_CAPSULE || k == PRIM_CYLINDER) ? orient_column(block, PRIM(i).b.xyz - PRIM(i).a.xyz) : block;
        PRIM(i).m.x = (PRIM(i).m.x & ~0xFF00u) | (min(prio, 255u) << 8);
        PRIM(i).m.y = (b & 0xFFFFu) | ((block2 & 0xFFFFu) << 16);
        PRIM(i).n.w = block2 == 0u ? 0.0 : share;
    }
}

void pb_displace(uint s, float amount, float size, uint seed) {
    for (uint i = s; i < gPrimN; ++i) {
        PRIM(i).n.x += amount;
        PRIM(i).n.y = 1.0 / max(size, 0.25);
        PRIM(i).m.z ^= seed * 0x9E3779B9u + 0x7F4A7C15u;
    }
}

void pb_holes(uint s, float amount, uint seed) {
    for (uint i = s; i < gPrimN; ++i) {
        PRIM(i).n.z = clamp(amount, 0.0, 1.0);
        PRIM(i).m.z ^= seed * 0x85EBCA6Bu + 0x165667B1u;
    }
}

void pb_smooth(uint s, float k) {
    gGroupN = min(gGroupN + 1u, 0xFFFFu);
    for (uint i = s; i < gPrimN; ++i) {
        PRIM(i).m.x = (PRIM(i).m.x & 0xFFFFu) | (gGroupN << 16);
        PRIM(i).m.w = floatBitsToUint(max(k, 1e-3));
    }
}

// ---- skeletons ----------------------------------------------------------------------------

void sk_push(vec3 a, float ra, vec3 b, float rb, uint flags) {
    if (gSegN < MAX_SEGS) {
        SEG(gSegN) = Seg(vec4(a, ra), vec4(b, rb), flags);
        gSegN++;
    }
}

// the end of the chain starting at segment i (exclusive), within [i, e)
uint sk_chain_end(uint i, uint e) {
    uint j = i + 1u;
    while (j < e && (SEG(j).flags & SEG_CHAIN) == 0u) ++j;
    return j;
}

float sk_chain_length(uint i, uint j) {
    float L = 0.0;
    for (uint k = i; k < j; ++k) L += distance(SEG(k).a.xyz, SEG(k).b.xyz);
    return L;
}

// point, direction and radius at arc length d along the chain [i, j)
void sk_chain_point(uint i, uint j, float d, out vec3 pos, out vec3 dir, out float r) {
    pos = SEG(i).a.xyz;
    dir = vec3(0.0, 1.0, 0.0);
    r = SEG(i).a.w;
    for (uint k = i; k < j; ++k) {
        vec3 a = SEG(k).a.xyz, b = SEG(k).b.xyz;
        float l = distance(a, b);
        dir = l > 1e-4 ? (b - a) / l : dir;
        if (d <= l || k + 1u == j) {
            float t = l > 1e-4 ? clamp(d / l, 0.0, 1.0) : 0.0;
            pos = mix(a, b, t);
            r = mix(SEG(k).a.w, SEG(k).b.w, t);
            return;
        }
        d -= l;
    }
}

vec3 any_perp(vec3 d) {
    vec3 a = abs(d.y) < 0.9 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    return normalize(cross(d, a));
}

// A trunk from the ground up (local space). Returns the first segment; the skeleton is
// [first, gSegN).
uint tree_trunk(float hmin, float hmax, float r, float taper, float tcurve, float flare, float bend, float twist, float sink, uint segs, uint key) {
    uint s0 = gSegN;
    segs = clamp(segs, 1u, 16u);
    float h = mix(hmin, max(hmax, hmin), pr(key));
    float dir = pr(key + 1u) * 6.2831853;
    vec2 bd = vec2(cos(dir), sin(dir));
    float ph = pr(key + 2u) * 6.2831853;
    vec3 prev = vec3(0.0);
    float prevR = 0.0;
    for (uint i = 0u; i <= segs; ++i) {
        float t = float(i) / float(segs);
        float y = -sink + (h + sink) * t;
        vec2 off = bd * bend * t * t + vec2(sin(t * 5.3 + ph), cos(t * 4.1 + ph * 1.7)) * twist * t;
        vec3 p = vec3(off.x, y, off.y);
        float above = max(y, 0.0);
        float rr = r * mix(1.0, taper, pow(t, tcurve)) + flare * pow(max(0.0, 1.0 - above / 2.5), 2.0);
        if (i > 0u) sk_push(prev, prevR, p, rr, (i == 1u ? SEG_CHAIN : 0u) | (i == segs ? SEG_TIP : 0u));
        prev = p;
        prevR = rr;
    }
    return s0;
}

// Branches from the chains of [ps, pe). The output skeleton is a copy of the parent followed by
// every new branch (so leaves can sit on the trunk's tip too); returns its first segment.
uint tree_branches(uint ps, uint pe, uint count, uint levels, float start, float end, float angle, float len, float falloff, float radius, float droop, bool elbow, uint key) {
    uint s0 = gSegN;
    for (uint i = ps; i < pe; ++i) sk_push(SEG(i).a.xyz, SEG(i).a.w, SEG(i).b.xyz, SEG(i).b.w, SEG(i).flags);
    uint ls = s0, le = gSegN;
    for (uint lv = 0u; lv < clamp(levels, 1u, 3u); ++lv) {
        uint ns = gSegN;
        float lf = pow(falloff, float(lv));
        uint cnt = max(1u, uint(round(float(count) * (lv == 0u ? 1.0 : falloff))));
        uint i = ls;
        while (i < le) {
            uint j = sk_chain_end(i, le);
            float L = sk_chain_length(i, j);
            for (uint c = 0u; c < cnt; ++c) {
                uint rk = key + lv * 7919u + i * 131u + c * 17u;
                float f = mix(start, end, (float(c) + 0.5 + (pr(rk) - 0.5) * 0.7) / float(cnt));
                vec3 pos, tan;
                float pr_;
                sk_chain_point(i, j, f * L, pos, tan, pr_);
                float az = float(c) * 2.3999632 + pr(rk + 1u) * 6.2831853;
                vec3 s1 = any_perp(tan);
                vec3 s2 = cross(tan, s1);
                vec3 side = s1 * cos(az) + s2 * sin(az);
                float ang = radians(angle) * mix(0.85, 1.15, pr(rk + 2u));
                vec3 dir = normalize(tan * cos(ang) + side * sin(ang));
                float bl = max(len * L * lf * mix(0.8, 1.2, pr(rk + 3u)), 1.0);
                float r0 = max(min(radius * lf, pr_), 0.3);
                float r1 = max(r0 * 0.75, 0.3);
                vec3 mid, tip;
                if (elbow) {
                    vec3 flat_ = normalize(vec3(dir.x, 0.0, dir.z) + vec3(1e-4, 0.0, 0.0));
                    mid = pos + flat_ * bl;
                    tip = mid - vec3(0.0, droop, 0.0);
                } else {
                    mid = pos + dir * bl * 0.55 - vec3(0.0, droop * 0.3, 0.0);
                    tip = pos + dir * bl - vec3(0.0, droop, 0.0);
                }
                uint lvl = min(lv + 1u, 15u);
                sk_push(pos, r0, mid, mix(r0, r1, 0.5), lvl | SEG_CHAIN);
                sk_push(mid, mix(r0, r1, 0.5), tip, r1, lvl | SEG_TIP);
            }
            i = j;
        }
        ls = ns;
        le = gSegN;
    }
    return s0;
}

// Capsules along the segments [s, e) that `from` and later make (branches skip the copied parent)
void tree_wood(uint s, uint e, uint block, uint prio) {
    for (uint i = s; i < e; ++i) pb_capsule(SEG(i).a.xyz, SEG(i).b.xyz, SEG(i).a.w, SEG(i).b.w, block, prio);
}

void leaf_cluster(uint shape, vec3 c, float r, float ys, vec3 dir, uint block, uint prio, uint key) {
    if (shape == 0u) {
        pb_sphere(c, r, block, prio);
    } else if (shape == 1u) {
        pb_ellipsoid(c, vec3(r, r * ys, r), pr(key) * 360.0, block, prio);
    } else if (shape == 2u) {
        // conifer layer: a cone around the stem, its base a little below
        float h = max(r * ys * 1.6, 1.0);
        pb_cylinder(c - vec3(0.0, h * 0.35, 0.0), c + vec3(0.0, h * 0.65, 0.0), r, 0.0, block, prio);
    } else if (shape == 3u) {
        pb_ellipsoid(c, vec3(r, max(r * ys * 0.35, 0.6), r), pr(key) * 360.0, block, prio);
    } else {
        // palm fronds: arms drooping from the crown
        uint arms = 6u + uint(pr(key) * 3.0);
        float a0 = pr(key + 1u) * 6.2831853;
        for (uint k = 0u; k < arms; ++k) {
            float a = a0 + float(k) * 6.2831853 / float(arms) + (pr(key + 2u + k) - 0.5) * 0.4;
            vec3 d = vec3(cos(a), 0.0, sin(a));
            vec3 m = c + d * r * 0.5 + vec3(0.0, 0.4, 0.0);
            vec3 t = c + d * r - vec3(0.0, r * ys * 0.5, 0.0);
            pb_capsule(c, m, 0.6, 0.6, block, prio);
            pb_capsule(m, t, 0.6, 0.5, block, prio);
        }
        pb_sphere(c, 1.2, block, prio);
    }
}

void tree_leaves(uint s, uint e, uint shape, uint at, float radius, float var, float ys, float yoff, float start, float spacing, float topScale, float density, float bumps, uint block, uint prio, uint key) {
    uint p0 = gPrimN;
    if (at == 0u) {
        for (uint i = s; i < e; ++i) {
            if ((SEG(i).flags & SEG_TIP) == 0u) continue;
            vec3 d = normalize(SEG(i).b.xyz - SEG(i).a.xyz + vec3(0.0, 1e-4, 0.0));
            float r = radius * mix(1.0 - var, 1.0 + var, pr(key + i * 7u));
            leaf_cluster(shape, SEG(i).b.xyz + vec3(0.0, yoff, 0.0), r, ys, d, block, prio, key + i * 13u);
        }
    } else {
        uint i = s;
        while (i < e) {
            uint j = sk_chain_end(i, e);
            float L = sk_chain_length(i, j);
            float d0 = start * L;
            uint n = 0u;
            for (float d = d0; d <= L + 1e-3 && n < 24u; d += max(spacing, 0.5), ++n) {
                vec3 pos, dir;
                float rr;
                sk_chain_point(i, j, d, pos, dir, rr);
                float t = L - d0 > 1e-3 ? (d - d0) / (L - d0) : 0.0;
                float r = radius * mix(1.0, topScale, t) * mix(1.0 - var, 1.0 + var, pr(key + i * 7u + n));
                if (r >= 0.3) leaf_cluster(shape, pos + vec3(0.0, yoff, 0.0), r, ys, dir, block, prio, key + i * 13u + n * 3u);
            }
            i = j;
        }
    }
    uint seed = gInst * 0x2C1B3C6Du + key;
    for (uint i = p0; i < gPrimN; ++i) {
        PRIM(i).n.x = bumps;
        PRIM(i).n.y = 1.0 / 2.2;
        PRIM(i).n.z = 1.0 - density;
        PRIM(i).m.z = seed;
    }
}

void tree_roots(uint s, uint e, uint count, float len, float depth, float radius, uint block, uint prio, uint key) {
    if (s >= e) return;
    vec3 base = SEG(s).a.xyz;
    float br = SEG(s).a.w;
    base.y = 0.5;
    float a0 = pr(key) * 6.2831853;
    for (uint c = 0u; c < count; ++c) {
        float a = a0 + (float(c) + (pr(key + 1u + c) - 0.5) * 0.6) * 6.2831853 / float(count);
        vec3 d = vec3(cos(a), 0.0, sin(a));
        float l = len * mix(0.7, 1.2, pr(key + 40u + c));
        vec3 p0 = base + d * br * 0.4 + vec3(0.0, 0.6, 0.0);
        vec3 p1 = base + d * (br + l * 0.5) - vec3(0.0, depth * 0.3, 0.0);
        vec3 p2 = base + d * (br + l) - vec3(0.0, depth, 0.0);
        pb_capsule(p0, p1, radius * 1.25, radius, block, prio);
        pb_capsule(p1, p2, radius, max(radius * 0.6, 0.3), block, prio);
    }
}

void tree_vines(uint s, uint e, uint count, float spread, float lmin, float lmax, float yoff, uint block, uint prio, uint key) {
    uint tips = 0u;
    for (uint i = s; i < e; ++i) if ((SEG(i).flags & SEG_TIP) != 0u) ++tips;
    for (uint c = 0u; c < count; ++c) {
        // the tip this strand hangs from
        uint pick = tips > 0u ? c % tips : 0u, seen = 0u;
        vec3 top = e > s ? SEG(e - 1u).b.xyz : vec3(0.0);
        for (uint i = s; i < e; ++i) {
            if ((SEG(i).flags & SEG_TIP) == 0u) continue;
            if (seen++ == pick) { top = SEG(i).b.xyz; break; }
        }
        float a = pr(key + c * 3u) * 6.2831853;
        float r = spread * sqrt(mix(0.35, 1.0, pr(key + c * 3u + 1u)));
        vec3 p = top + vec3(cos(a) * r, yoff, sin(a) * r);
        float l = mix(lmin, max(lmax, lmin), pr(key + c * 3u + 2u));
        pb_capsule(p, p - vec3(0.0, l, 0.0), 0.5, 0.5, block, prio);
    }
}

void tree_fruit(uint s, uint e, float chance, uint per, float spread, uint block, uint prio, uint key) {
    for (uint i = s; i < e; ++i) {
        if ((SEG(i).flags & SEG_TIP) == 0u) continue;
        for (uint k = 0u; k < per; ++k) {
            uint rk = key + i * 31u + k * 5u;
            if (pr(rk) >= chance) continue;
            float a = pr(rk + 1u) * 6.2831853;
            float r = spread * pr(rk + 2u);
            pb_point(SEG(i).b.xyz + vec3(cos(a) * r, -1.0 - pr(rk + 3u) * spread * 0.5, sin(a) * r), block, prio);
        }
    }
}

// ---- evaluation ------------------------------------------------------------------------

// distance from p to a primitive (< 0 inside); `thin`: the least radius of segments (coarse LODs)
float prim_dist(Prim P, uint kind, vec3 p, float thin) {
    if (kind == PRIM_SPHERE) return length(p - P.a.xyz) - P.a.w;
    if (kind == PRIM_CAPSULE || kind == PRIM_CYLINDER) {
        vec3 pa = p - P.a.xyz, ba = P.b.xyz - P.a.xyz;
        float bb = max(dot(ba, ba), 1e-6);
        float h = dot(pa, ba) / bb;
        float hc = clamp(h, 0.0, 1.0);
        float r = max(mix(P.a.w, P.b.w, hc), thin);
        float radial = length(pa - ba * hc) - r;
        if (kind == PRIM_CAPSULE) return radial;
        float L = sqrt(bb);
        float axial = max(-h, h - 1.0) * L;
        return max(length(pa - ba * h) - r, axial);
    }
    vec3 l = qrot_inv(P.q, p - P.a.xyz);
    if (kind == PRIM_ELLIPSOID) {
        vec3 r = max(P.b.xyz, vec3(0.05));
        float k0 = length(l / r);
        float k1 = length(l / (r * r));
        return k1 > 1e-6 ? k0 * (k0 - 1.0) / k1 : -min(r.x, min(r.y, r.z));
    }
    vec3 q = abs(l) - P.b.xyz;
    return length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// rough bounding sphere of a primitive: centre (xyz) and radius (w)
vec4 prim_bound(Prim P, uint kind) {
    if (kind == PRIM_SPHERE) return P.a;
    if (kind == PRIM_CAPSULE || kind == PRIM_CYLINDER) return vec4((P.a.xyz + P.b.xyz) * 0.5, distance(P.a.xyz, P.b.xyz) * 0.5 + max(P.a.w, P.b.w));
    if (kind == PRIM_ELLIPSOID) return vec4(P.a.xyz, max(P.b.x, max(P.b.y, P.b.z)));
    return vec4(P.a.xyz, length(P.b.xyz));
}

// The block a prop puts in the cell at p (prop space), PROP_NOTHING to leave it, 0 for air.
uint prop_eval(vec3 p, uint n) {
    uint res = PROP_NOTHING;
    int best = -1;
    uint curGroup = 0u;
    float gd = 1e9;
    ivec3 cell = ivec3(floor(p + vec3(0.5, 0.0, 0.5)));
    // wood stays connected: a capsule covers the cells its axis passes, diagonals included
    float thin = 0.72 * gScale;
    // coarse LODs sample one point per big cell: other shapes grow by half a cell so canopies
    // don't fall between the samples (while the trunks, kept thick, stay)
    float grow = gLod > 0u ? 0.5 * gScale : 0.0;
    for (uint i = 0u; i < n; ++i) {
        Prim P = sPrim[i];
        uint kind = P.m.x & 15u;
        uint op = (P.m.x >> 4) & 15u;
        vec4 bs = prim_bound(P, kind);
        float reach = bs.w + P.n.x + thin + grow;
        vec3 dv = p - bs.xyz;
        if (dot(dv, dv) > reach * reach) {
            if (op == OP_KEEP) {
                res = PROP_NOTHING;
                best = -1;
            }
            continue;
        }
        // wood (capsules) stays at least one cell thick at any LOD; cones may end in a point
        float d = kind == PRIM_CAPSULE ? prim_dist(P, kind, p, thin) : prim_dist(P, kind, p, 0.0) - grow;
        if (P.n.x > 0.0) d += value3(p * P.n.y, P.m.z) * P.n.x;
        // holes and speckles hash the cell, so every chunk agrees; far away leaves stay solid
        if (P.n.z > 0.0 && gLod == 0u && rand3(cell, P.m.z ^ 0x51F2u) < P.n.z) d = max(d, 0.01);
        uint g = P.m.x >> 16;
        if (g != 0u) {
            if (g == curGroup) d = smoothmin(gd, d, uintBitsToFloat(P.m.w));
            curGroup = g;
            gd = d;
        } else {
            curGroup = 0u;
        }
        bool inside = d < 0.0;
        int prio = int((P.m.x >> 8) & 255u);
        if (op == OP_ADD) {
            if (inside && prio >= best) {
                res = (P.n.w > 0.0 && rand3(cell, P.m.z ^ 0xB10Cu) < P.n.w) ? (P.m.y >> 16) : (P.m.y & 0xFFFFu);
                best = prio;
            }
        } else if (op == OP_CARVE) {
            if (inside && prio >= best) {
                res = 0u;
                best = prio;
            }
        } else if (op == OP_REMOVE) {
            if (inside) {
                res = PROP_NOTHING;
                best = -1;
            }
        } else if (!inside) {
            res = PROP_NOTHING;
            best = -1;
        }
    }
    return res;
}
