// build.zig
const std = @import("std");

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

    const executables = [_]struct {
        step_id: []const u8,
        source: []const u8,
        name: ?[]const u8 = null,
        description: ?[]const u8 = null,
        install_dir: ?[]const u8 = null,
    }{
        .{ .step_id = "triangle", .source = "src/main.zig" },
        .{ .step_id = "example", .source = "example/example.zig" },
        .{ .step_id = "test", .source = "example/test.zig" },
    };

    var all_shader_steps = std.ArrayList(*std.Build.Step).init(b.allocator);
    defer all_shader_steps.deinit();

    for (executables) |exe_info| {
        // --- DEFINE SHADER PATHS ---
        const shader_code_path = "src/shaders/code/";
        const vert_shader_source = b.fmt(shader_code_path ++ "{s}/{s}.vert", .{ exe_info.step_id, exe_info.step_id });
        const frag_shader_source = b.fmt(shader_code_path ++ "{s}/{s}.frag", .{ exe_info.step_id, exe_info.step_id });
        const vert_shader_output = b.fmt("spirv/bin/{s}.vert.spv", .{exe_info.step_id});
        const frag_shader_output = b.fmt("spirv/bin/{s}.frag.spv", .{exe_info.step_id});

        // --- COMPILE VERTEX SHADER ---
        const compile_vert_step = b.addSystemCommand(&.{ "glslc", vert_shader_source });
        compile_vert_step.addFileInput(b.path(vert_shader_source));
        try all_shader_steps.append(&compile_vert_step.step);

        const install_vert_shader = b.addInstallFile(
            compile_vert_step.addPrefixedOutputFileArg("-o", vert_shader_output),
            "../src/shaders/spirv/bin/vert.spv",
        );

        // --- COMPILE FRAGMENT SHADER ---
        const compile_frag_step = b.addSystemCommand(&.{ "glslc", frag_shader_source });
        compile_frag_step.addFileInput(b.path(frag_shader_source));
        try all_shader_steps.append(&compile_frag_step.step);

        const intall_frag_shader = b.addInstallFile(
            compile_frag_step.addPrefixedOutputFileArg("-o", frag_shader_output),
            "../src/shaders/spirv/bin/frag.spv",
        );

        // --- COMPILE EXECUTABLE ---
        const exe = b.addExecutable(.{
            .name = exe_info.name orelse exe_info.step_id,
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(exe_info.source),
        });

        // The spirv module gets the shaders for the executable.
        exe.root_module.addAnonymousImport("spirv", .{
            .root_source_file = b.path("src/shaders/spirv/spirv.zig"),
            .target = target,
        });

        // The executable's compile step must depend on its shaders being cached or recompiled.
        exe.step.dependOn(&install_vert_shader.step);
        exe.step.dependOn(&intall_frag_shader.step);

        // Conditionally links libraries based on the target.
        linkVulkanAndGlfwLibs(exe, target);

        // Installs the executable and sets up the run command.
        const install = b.addInstallArtifact(exe, .{
            .dest_dir = if (exe_info.install_dir) |dir|
                .{ .override = .{ .custom = dir } }
            else
                .default,
        });

        const cmd = b.addRunArtifact(exe);
        // Starting point for the build graph.
        const step = b.step(
            exe_info.step_id,
            exe_info.description orelse b.fmt("Run {s}", .{exe_info.step_id}),
        );

        step.dependOn(&cmd.step);
        // Runs immediately after the executable is built.
        cmd.step.dependOn(&install.step);
    }
}
