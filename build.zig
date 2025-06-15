// build.zig
const std = @import("std");

// pub fn bilu(b: *std.Build) void {}

pub fn build(b: *std.Build) void {
    // const target = b.standardTargetOptions(.{});
    const target = b.graph.host;

    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "hello_triangle",
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/main.zig"),
    });

    // const example = b.addExecutable(.{
    //     .name = "hello_triangle_example",
    //     .target = target,
    //     .optimize = optimize,
    //     .root_source_file = b.path("example/example.zig"),
    // });

    const c_mod = b.addModule("c", .{
        .root_source_file = b.path("src/c/c.zig"),
        .target = target,
    });

    c_mod.linkSystemLibrary("vulkan", .{});
    c_mod.linkSystemLibrary("glfw", .{});

    // Link flags for GLFW on Linux
    if (target.result.os.tag == .linux) {
        c_mod.linkSystemLibrary("m", .{});
        c_mod.linkSystemLibrary("pthread", .{});
        c_mod.linkSystemLibrary("dl", .{});
        c_mod.linkSystemLibrary("X11", .{});
        c_mod.linkSystemLibrary("xcb", .{});
        c_mod.linkSystemLibrary("Xrandr", .{});
    }

    const spirv_mod = b.addModule("spirv", .{
        .root_source_file = b.path("src/shaders/spirv/spirv.zig"),
        .target = target,
    });

    exe.root_module.addImport("c", c_mod);
    exe.root_module.addImport("spirv", spirv_mod);

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    // if (b.args) |args| {
    //     run_cmd.addArgs(args);
    // }

    // Recompile shaders if they change
    // b.addSystemCommand(&.{
    //     "glslc src/shaders/code/triangle.vert -o src/shaders/vert.spv",
    //     "glslc src/shaders/code/triangle.frag -o src/shaders/frag.spv",
    // })
    //     .step.dependOn(b.getInstallStep());

    b.step("run", "run step").dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
}
