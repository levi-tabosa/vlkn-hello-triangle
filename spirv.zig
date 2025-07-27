const std = @import("std");

const spv_path_opt = @import("spv_path");

// Helper function
fn embed(comptime filename: []const u8) []const u8 {
    return @embedFile(spv_path_opt.out_dir ++ filename);
}

pub const gui_vert = embed("gui.vert.spv");
pub const gui_frag = embed("gui.frag.spv");

pub const text_vert = embed("text.vert.spv");
pub const text_frag = embed("text.frag.spv");

pub const triangle_vert = embed("triangle.vert.spv");
pub const triangle_frag = embed("triangle.frag.spv");

pub const example_vert = embed("example.vert.spv");
pub const example_frag = embed("example.frag.spv");

pub const test_vert = embed("test.vert.spv");
pub const test_frag = embed("test.frag.spv");
