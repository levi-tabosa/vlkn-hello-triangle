// build.zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    //TODO: Ember shader source files here
    // const shader_source_files = &[_][]const u8{
    //     "src/shaders/code/triangle.vert",
    //     "src/shaders/code/triangle.frag",
    // };
    // for (shader_source_files) |shader_file| {
    //     b.addSystemCommand(&.{
    //         "glslc",
    //         shader_file,
    //         "-o",
    //         std.fmt.allocPrint(b.allocator, "src/shaders/{}.spv", .{std.fs.path.basename(shader_file)}) catch unreachable,
    //     });
    // }
    // Add the executable target

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
    }

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
