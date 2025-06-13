const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "hello_triangle",
        .target = target,
        .optimize = optimize,
        .root_source_file = b.path("src/main.zig"),
    });

    exe.linkSystemLibrary("vulkan");
    exe.linkSystemLibrary("glfw");

    // Link flags for GLFW on Linux
    if (target.result.os.tag == .linux) {
        exe.linkSystemLibrary("m");
        exe.linkSystemLibrary("pthread");
        exe.linkSystemLibrary("dl");
        exe.linkSystemLibrary("X11");
        exe.linkSystemLibrary("xcb");
        exe.linkSystemLibrary("Xrandr");
        // exe.linkSystemLibrary("Xinerama");
        // exe.linkSystemLibrary("Xxf86vm");
        // exe.linkSystemLibrary("Xcursor");
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    b.step("run", "run step").dependOn(&run_cmd.step);
}
