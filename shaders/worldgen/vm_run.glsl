// World generator interpreter: the run loop. Instructions are five words [op | w << 8, a, b, c, d]
// (crates/worldgen/src/vm/mod.rs: `Op`). Operands are frame-relative, or absolute with VM_ABS.

#define VM_DEPTH 12

vec4 vm_ldf(uint a, uint n) {
    vec4 v = vec4(0.0);
    for (uint i = 0u; i < n; ++i) v[i] = uintBitsToFloat(R[a + i]);
    return v;
}
void vm_stf(uint a, vec4 v, uint n) {
    for (uint i = 0u; i < n; ++i) R[a + i] = floatBitsToUint(v[i]);
}
uvec4 vm_ldu(uint a, uint n) {
    uvec4 v = uvec4(0u);
    for (uint i = 0u; i < n; ++i) v[i] = R[a + i];
    return v;
}
void vm_stu(uint a, uvec4 v, uint n) {
    for (uint i = 0u; i < n; ++i) R[a + i] = v[i];
}

// dot / length / distance / normalize at the operand's exact width (as the compiled code does)
float vm_dot(vec4 x, vec4 y, uint n) { return n == 2u ? dot(x.xy, y.xy) : n == 3u ? dot(x.xyz, y.xyz) : n == 4u ? dot(x, y) : x.x * y.x; }
float vm_len(vec4 x, uint n) { return n == 2u ? length(x.xy) : n == 3u ? length(x.xyz) : n == 4u ? length(x) : abs(x.x); }
float vm_dist(vec4 x, vec4 y, uint n) { return n == 2u ? distance(x.xy, y.xy) : n == 3u ? distance(x.xyz, y.xyz) : n == 4u ? distance(x, y) : abs(x.x - y.x); }
vec4 vm_norm(vec4 x, uint n) { return n == 2u ? vec4(normalize(x.xy), 0.0, 0.0) : n == 3u ? vec4(normalize(x.xyz), 0.0) : n == 4u ? normalize(x) : vec4(sign(x.x), 0.0, 0.0, 0.0); }

void vm_run(uint entry, uint fb0) {
    uint ip = entry;
    uint fb = fb0;
    uint depth = 0u;
    uint retIp[VM_DEPTH];
    uint retFb[VM_DEPTH];
    for (uint guard = 0u; guard < 4000000u; ++guard) {
        uint o = vm_word(ip);
        uint a = vm_word(ip + 1u), b = vm_word(ip + 2u), c = vm_word(ip + 3u), d = vm_word(ip + 4u);
        uint w = (o >> 8) & 0xFFu;
        o &= 0xFFu;
        ip += 5u;
        // operand addresses
        uint A = (a & VM_ABS) != 0u ? (a & ~VM_ABS) : fb + a;
        uint B = (b & VM_ABS) != 0u ? (b & ~VM_ABS) : fb + b;
        uint C = (c & VM_ABS) != 0u ? (c & ~VM_ABS) : fb + c;
        uint D = (d & VM_ABS) != 0u ? (d & ~VM_ABS) : fb + d;
        if (o <= VM_SPLAT) {
            switch (o) {
                case VM_MOV: for (uint i = 0u; i < w; ++i) R[A + i] = R[B + i]; break;
                case VM_LDK: R[A] = b; if (w > 1u) R[A + 1u] = c; if (w > 2u) R[A + 2u] = d; break;
                case VM_LDI: { uint p = R[B] + c; for (uint i = 0u; i < d; ++i) R[A + i] = R[p + i]; } break;
                case VM_STI: { uint p = R[A] + c; for (uint i = 0u; i < d; ++i) R[p + i] = R[B + i]; } break;
                case VM_ADDR: R[A] = B; break;
                case VM_LEA: R[A] = R[B] + R[C] * d; break;
                case VM_SWZ: { uvec4 v = vm_ldu(B, 4u); uvec4 r; for (uint i = 0u; i < w; ++i) r[i] = v[(c >> (2u * i)) & 3u]; vm_stu(A, r, w); } break;
                case VM_SPLAT: { uint v = R[B]; for (uint i = 0u; i < w; ++i) R[A + i] = v; } break;
                default: break;
            }
            continue;
        }
        if (o <= VM_FCROSS) {
            vec4 x = vm_ldf(B, max(w, 1u)), y = vm_ldf(C, max(w, 1u));
            vec4 r = vec4(0.0);
            uint n = max(w, 1u);
            switch (o) {
                case VM_FADD: r = x + y; break;
                case VM_FSUB: r = x - y; break;
                case VM_FMUL: r = x * y; break;
                case VM_FDIV: r = x / y; break;
                case VM_FMOD: r = mod(x, y); break;
                case VM_FMIN: r = min(x, y); break;
                case VM_FMAX: r = max(x, y); break;
                case VM_FPOW: r = pow(x, y); break;
                case VM_FATAN2: r = atan(x, y); break;
                case VM_FSTEP: r = step(x, y); break;
                case VM_FNEG: r = -x; break;
                case VM_FABS: r = abs(x); break;
                case VM_FSIGN: r = sign(x); break;
                case VM_FFLOOR: r = floor(x); break;
                case VM_FCEIL: r = ceil(x); break;
                case VM_FFRACT: r = fract(x); break;
                case VM_FSQRT: r = sqrt(x); break;
                case VM_FINVSQRT: r = inversesqrt(x); break;
                case VM_FSIN: r = sin(x); break;
                case VM_FCOS: r = cos(x); break;
                case VM_FTAN: r = tan(x); break;
                case VM_FASIN: r = asin(x); break;
                case VM_FACOS: r = acos(x); break;
                case VM_FATAN: r = atan(x); break;
                case VM_FEXP: r = exp(x); break;
                case VM_FLOG: r = log(x); break;
                case VM_FEXP2: r = exp2(x); break;
                case VM_FLOG2: r = log2(x); break;
                case VM_FROUND: r = round(x); break;
                case VM_FTRUNC: r = trunc(x); break;
                case VM_FMIX: r = mix(x, y, vm_ldf(D, n)); break;
                case VM_FCLAMP: r = clamp(x, y, vm_ldf(D, n)); break;
                case VM_FSMOOTH: r = smoothstep(x, y, vm_ldf(D, n)); break;
                case VM_FDOT: r.x = vm_dot(x, y, n); n = 1u; break;
                case VM_FLEN: r.x = vm_len(x, n); n = 1u; break;
                case VM_FDIST: r.x = vm_dist(x, y, n); n = 1u; break;
                case VM_FNORM: r = vm_norm(x, n); break;
                case VM_FCROSS: r.xyz = cross(x.xyz, y.xyz); n = 3u; break;
                default: break;
            }
            vm_stf(A, r, n);
            continue;
        }
        if (o <= VM_ISIGN) {
            uint n = max(w, 1u);
            uvec4 x = vm_ldu(B, n), y = vm_ldu(C, n);
            ivec4 xi = ivec4(x), yi = ivec4(y);
            uvec4 r = uvec4(0u);
            switch (o) {
                case VM_XADD: r = x + y; break;
                case VM_XSUB: r = x - y; break;
                case VM_XMUL: r = x * y; break;
                case VM_IDIV: r = uvec4(xi / yi); break;
                case VM_UDIV: r = x / y; break;
                case VM_IMOD: r = uvec4(xi % yi); break;
                case VM_UMOD: r = x % y; break;
                case VM_AND: r = x & y; break;
                case VM_OR: r = x | y; break;
                case VM_XOR: r = x ^ y; break;
                case VM_SHL: r = x << y; break;
                case VM_ISHR: r = uvec4(xi >> yi); break;
                case VM_USHR: r = x >> y; break;
                case VM_IMIN: r = uvec4(min(xi, yi)); break;
                case VM_IMAX: r = uvec4(max(xi, yi)); break;
                case VM_UMIN: r = min(x, y); break;
                case VM_UMAX: r = max(x, y); break;
                case VM_ICLAMP: r = uvec4(clamp(xi, yi, ivec4(vm_ldu(D, n)))); break;
                case VM_UCLAMP: r = clamp(x, y, vm_ldu(D, n)); break;
                case VM_INEG: r = uvec4(-xi); break;
                case VM_INOT: r = ~x; break;
                case VM_IABS: r = uvec4(abs(xi)); break;
                case VM_ISIGN: r = uvec4(sign(xi)); break;
                default: break;
            }
            vm_stu(A, r, n);
            continue;
        }
        if (o <= VM_X2B) {
            uint n = max(w, 1u);
            uvec4 x = vm_ldu(B, n);
            uvec4 r = uvec4(0u);
            switch (o) {
                case VM_F2I: r = uvec4(ivec4(uintBitsToFloat(x))); break;
                case VM_F2U: r = uvec4(uintBitsToFloat(x)); break;
                case VM_I2F: r = floatBitsToUint(vec4(ivec4(x))); break;
                case VM_U2F: r = floatBitsToUint(vec4(x)); break;
                case VM_F2B: r = uvec4(notEqual(uintBitsToFloat(x), vec4(0.0))); break;
                case VM_X2B: r = uvec4(notEqual(x, uvec4(0u))); break;
                default: break;
            }
            vm_stu(A, r, n);
            continue;
        }
        if (o <= VM_LNOT) {
            uint n = max(w, 1u);
            bool r = false;
            if (o <= VM_FNE) {
                vec4 x = vm_ldf(B, n), y = vm_ldf(C, n);
                switch (o) {
                    case VM_FLT: r = x.x < y.x; break;
                    case VM_FLE: r = x.x <= y.x; break;
                    case VM_FEQ: r = all(equal(x, y)); break;
                    case VM_FNE: r = any(notEqual(x, y)); break;
                    default: break;
                }
            } else {
                uvec4 x = vm_ldu(B, n), y = vm_ldu(C, n);
                switch (o) {
                    case VM_ILT: r = int(x.x) < int(y.x); break;
                    case VM_ILE: r = int(x.x) <= int(y.x); break;
                    case VM_ULT: r = x.x < y.x; break;
                    case VM_ULE: r = x.x <= y.x; break;
                    case VM_XEQ: r = all(equal(x, y)); break;
                    case VM_XNE: r = any(notEqual(x, y)); break;
                    case VM_LNOT: r = x.x == 0u; break;
                    default: break;
                }
            }
            R[A] = r ? 1u : 0u;
            continue;
        }
        switch (o) {
            case VM_JMP: ip = b; break;
            case VM_JZ: if (R[A] == 0u) ip = b; break;
            case VM_JNZ: if (R[A] != 0u) ip = b; break;
            case VM_CALL:
                if (depth < VM_DEPTH) {
                    retIp[depth] = ip;
                    retFb[depth] = fb;
                    ++depth;
                    fb = C;
                    ip = b;
                }
                break;
            case VM_RET:
                if (depth == 0u) return;
                --depth;
                ip = retIp[depth];
                fb = retFb[depth];
                break;
            case VM_NAT: vm_native(b, A, C); break;
            default: break;
        }
    }
}
