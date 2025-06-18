// build.zig
const std = @import("std");

// helper
/// TODO: make accept libraryNames
fn linkVulkanAndGlfwLibs(exe: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    exe.linkSystemLibrary("vulkan");
    exe.linkSystemLibrary("glfw");
    if (target.result.os.tag == .linux) {
        exe.linkSystemLibrary("m");
        exe.linkSystemLibrary("pthread");
        exe.linkSystemLibrary("dl");
        exe.linkSystemLibrary("X11");
        exe.linkSystemLibrary("xcb");
        exe.linkSystemLibrary("Xrandr");
    }
}

pub fn build(b: *std.Build) !void {
    const target = b.graph.host;
    const optimize = b.standardOptimizeOption(.{});

    const allocator = std.heap.page_allocator;

    const executables = [_]struct {
        step_id: []const u8,
        source: []const u8,
        name: ?[]const u8 = null,
        description: ?[]const u8 = null,
        install_dir: ?[]const u8 = null,
    }{
        .{ .step_id = "triangle", .source = "src/main.zig" },
        .{ .step_id = "example", .source = "example/example.zig" },
        .{ .step_id = "glfw", .source = "example/glfw.zig" },
    };

    const spirv_mod = b.addModule("spirv", .{
        .root_source_file = b.path("src/shaders/spirv/spirv.zig"),
        .target = target,
    });

    for (executables) |exe_info| {
        const exe = b.addExecutable(.{
            .name = exe_info.name orelse try std.fmt.allocPrint(
                allocator,
                "{s}_name", // Formated name of the executable file
                .{exe_info.step_id},
            ),
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(exe_info.source),
        });
        linkVulkanAndGlfwLibs(exe, target);
        exe.root_module.addImport("spirv", spirv_mod);

        const install = b.addInstallArtifact(exe, .{
            .dest_dir = if (exe_info.install_dir) |dir|
                .{ .override = .{ .custom = dir } }
            else
                .default,
        });

        const cmd = b.addRunArtifact(exe);
        const step = b.step(
            exe_info.step_id,
            exe_info.description orelse try std.fmt.allocPrint(allocator, "{s} description", .{exe_info.step_id}),
        );
        step.dependOn(&cmd.step);
        cmd.step.dependOn(&install.step);
    }
}
