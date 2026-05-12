@compute @workgroup_size(8, 8, 1)
fn lighting_pass(@builtin(global_invocation_id) gid: vec3u) {
    // NOOP: future GI / light propagation pass
}
