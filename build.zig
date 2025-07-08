const std = @import("std");

fn linkVulkanAndGlfwLibs(exe: *std.Build.Step.Compile, target: std.Build.ResolvedTarget) void {
    exe.linkSystemLibrary("vulkan");
    exe.linkSystemLibrary("glfw");
    exe.linkSystemLibrary("c");
    if (target.result.os.tag == .linux) {
        exe.linkSystemLibrary("m");
        exe.linkSystemLibrary("pthread");
        exe.linkSystemLibrary("dl");
        exe.linkSystemLibrary("X11");
        exe.linkSystemLibrary("xcb");
        exe.linkSystemLibrary("Xrandr");
    }
}

fn addShaderStep(
    b: *std.Build,
    glsl_path: []const u8,
    spv_path: []const u8,
    install_path: []const u8,
    all_shader_steps: *std.ArrayList(*std.Build.Step),
) !*std.Build.Step {
    const compile_step = b.addSystemCommand(&.{ "glslc", glsl_path });
    compile_step.addFileInput(b.path(glsl_path));
    try all_shader_steps.append(&compile_step.step);

    const install_step = b.addInstallFile(
        compile_step.addPrefixedOutputFileArg("-o", spv_path),
        install_path,
    );
    return &install_step.step;
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
        .{ .step_id = "triangle", .source = "example/main.zig" },
        .{ .step_id = "example", .source = "example/example.zig" },
        .{ .step_id = "test", .source = "example/test.zig" },
    };

    var all_shader_steps = std.ArrayList(*std.Build.Step).init(b.allocator);
    defer all_shader_steps.deinit();

    // Common shaders (GUI, text)
    const gui_vert_shader_source = "src/shaders/code/gui/gui.vert";
    const gui_frag_shader_source = "src/shaders/code/gui/gui.frag";
    const gui_vert_shader_output = "spirv/bin/gui.vert.spv";
    const gui_frag_shader_output = "spirv/bin/gui.frag.spv";

    const text_vert_shader_source = "src/shaders/code/text/text.vert";
    const text_frag_shader_source = "src/shaders/code/text/text.frag";
    const text_vert_shader_output = "spirv/bin/text3d.vert.spv";
    const text_frag_shader_output = "spirv/bin/text3d.frag.spv";

    const install_gui_vert_shader = try addShaderStep(b, gui_vert_shader_source, gui_vert_shader_output, "../src/shaders/spirv/bin/gui.vert.spv", &all_shader_steps);
    const install_gui_frag_shader = try addShaderStep(b, gui_frag_shader_source, gui_frag_shader_output, "../src/shaders/spirv/bin/gui.frag.spv", &all_shader_steps);
    const install_text_vert_shader = try addShaderStep(b, text_vert_shader_source, text_vert_shader_output, "../src/shaders/spirv/bin/text3d.vert.spv", &all_shader_steps);
    const install_text_frag_shader = try addShaderStep(b, text_frag_shader_source, text_frag_shader_output, "../src/shaders/spirv/bin/text3d.frag.spv", &all_shader_steps);

    for (executables) |exe_info| {
        const shader_code_path = "src/shaders/code/";
        const vert_shader_source = b.fmt(shader_code_path ++ "{s}/{s}.vert", .{ exe_info.step_id, exe_info.step_id });
        const frag_shader_source = b.fmt(shader_code_path ++ "{s}/{s}.frag", .{ exe_info.step_id, exe_info.step_id });
        const vert_shader_output = b.fmt("spirv/bin/{s}.vert.spv", .{exe_info.step_id});
        const frag_shader_output = b.fmt("spirv/bin/{s}.frag.spv", .{exe_info.step_id});

        const install_vert_shader = try addShaderStep(b, vert_shader_source, vert_shader_output, "../src/shaders/spirv/bin/vert.spv", &all_shader_steps);
        const install_frag_shader = try addShaderStep(b, frag_shader_source, frag_shader_output, "../src/shaders/spirv/bin/frag.spv", &all_shader_steps);

        // --- COMPILE EXECUTABLE ---
        const exe = b.addExecutable(.{
            .name = exe_info.name orelse exe_info.step_id,
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(exe_info.source),
        });

        exe.root_module.addAnonymousImport("c", .{
            .root_source_file = b.path("src/c/c.zig"),
            .target = target,
        });
        exe.root_module.addAnonymousImport("spirv", .{
            .root_source_file = b.path("src/shaders/spirv/spirv.zig"),
            .target = target,
        });
        exe.root_module.addAnonymousImport("font", .{
            .root_source_file = b.path("src/fonts/font.zig"),
            .target = target,
        });
        exe.root_module.addAnonymousImport("geometry", .{
            .root_source_file = b.path("src/scenes/geometry.zig"),
            .target = target,
        });
        exe.root_module.addAnonymousImport("util", .{
            .root_source_file = b.path("src/util/util.zig"),
            .target = target,
        });

        // The executable's compile step must depend on its shaders being cached or recompiled.
        exe.step.dependOn(install_vert_shader);
        exe.step.dependOn(install_frag_shader);
        exe.step.dependOn(install_gui_vert_shader);
        exe.step.dependOn(install_gui_frag_shader);
        exe.step.dependOn(install_text_vert_shader);
        exe.step.dependOn(install_text_frag_shader);

        linkVulkanAndGlfwLibs(exe, target);

        const install = b.addInstallArtifact(exe, .{
            .dest_dir = if (exe_info.install_dir) |dir|
                .{ .override = .{ .custom = dir } }
            else
                .default,
        });

        const cmd = b.addRunArtifact(exe);
        const step = b.step(
            exe_info.step_id,
            exe_info.description orelse b.fmt("Run {s}", .{exe_info.step_id}),
        );
        step.dependOn(&cmd.step);
        cmd.step.dependOn(&install.step);
    }
}
