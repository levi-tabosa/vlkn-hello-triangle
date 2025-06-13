// This module's purpose is to embed the compiled shader files directly into the executable.
// The paths are provided by the build.zig script.
const vert_shader_path = @import("vert_spv_path");
const frag_shader_path = @import("frag_spv_path");

pub const vert_code = @embedFile(vert_shader_path);
pub const frag_code = @embedFile(frag_shader_path);
