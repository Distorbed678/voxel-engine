// World generator interpreter: registers and code access (see crates/worldgen/src/vm/mod.rs).
// The program (header + bytecode of the layer's generated code) is at pc.vmProgram.
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer VmCode { uint w[]; };

uint R[VM_REGS];

uint vm_word(uint i) { return VmCode(pc.vmProgram).w[i]; }

// zero every register (once per invocation, in wg_init): nothing depends on what a thread found
void vm_clear() { for (uint i = 0u; i < VM_REGS; ++i) R[i] = 0u; }
