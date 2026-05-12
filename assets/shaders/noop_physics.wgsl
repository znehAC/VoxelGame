@compute @workgroup_size(8, 8, 4)
fn physics_step(@builtin(global_invocation_id) gid: vec3u) {
    // NOOP: future cellular automata / physics simulation pass
}
