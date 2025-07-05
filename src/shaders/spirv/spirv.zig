// src/shaders/spirv/spirv.zig

pub const vs = @embedFile("./bin/vert.spv");
pub const fs = @embedFile("./bin/frag.spv");
pub const gui_vs = @embedFile("./bin/gui.vert.spv");
pub const gui_fs = @embedFile("./bin/gui.frag.spv");
pub const text_vs = @embedFile("./bin/text.vert.spv");
pub const text_fs = @embedFile("./bin/text.frag.spv");
