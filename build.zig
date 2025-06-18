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

    const example = b.addExecutable(.{
        .name = "hello_triangle_example",
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("example/example.zig"),
        .link_libc = true,
    });

    const c_mod = b.addModule("c_mod", .{
        .root_source_file = b.path("src/c/c.zig"),
        .target = target,
    });

    const c = b.addLibrary(.{
        .linkage = .dynamic,
        .root_module = c_mod,
        .name = "c",
    });

    example.linkLibrary(c);

    c.linkSystemLibrary("vulkan");
    c.linkSystemLibrary("glfw");

    // c_mod.linkSystemLibrary("vulkan", .{});
    // c_mod.linkSystemLibrary("glfw", .{});

    // // Link flags for GLFW on Linux
    if (target.result.os.tag == .linux) {
        c.linkSystemLibrary2("m", .{});
        c.linkSystemLibrary2("pthread", .{});
        c.linkSystemLibrary2("dl", .{});
        c.linkSystemLibrary2("X11", .{});
        c.linkSystemLibrary2("xcb", .{});
        c.linkSystemLibrary2("Xrandr", .{});
    }

    const spirv_mod = b.addModule("spirv", .{
        .root_source_file = b.path("src/shaders/spirv/spirv.zig"),
        .target = target,
    });

    exe.root_module.addImport("c", c_mod);
    example.root_module.addImport("c", c_mod);
    example.root_module.addImport("spirv", spirv_mod);
    exe.root_module.addImport("spirv", spirv_mod);

    b.installArtifact(exe);
    const example_install = b.addInstallArtifact(example, .{ .dest_dir = .{ .override = .{ .custom = "example/bin/" } } });

    const run_cmd = b.addRunArtifact(exe);
    // if (b.args) |args| {
    //     run_cmd.addArgs(args);
    // }

    const example_cmd = b.addRunArtifact(example);
    // if (b.args) |args| {
    //     run_cmd.addArgs(args);
    // }

    // Recompile shaders if they change
    // b.addSystemCommand(&.{
    //     "glslc src/shaders/code/triangle.vert -o src/shaders/vert.spv",
    //     "glslc src/shaders/code/triangle.frag -o src/shaders/frag.spv",
    // })
    //     .step.dependOn(b.getInstallStep());

    b.step("go", "example run step").dependOn(&example_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());
    example_cmd.step.dependOn(&example_install.step);
}
