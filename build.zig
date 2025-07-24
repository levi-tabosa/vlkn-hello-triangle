const std = @import("std");

// FIXED: This function now correctly uses `b.addRunArtifact`
// In build.zig

// In build.zig

fn addShaderStep(
    b: *std.Build,
    glslc_exe: *std.Build.Step.Compile,
    optimize: std.builtin.OptimizeMode, // Pass in the build mode
    glsl_path: []const u8,
    spv_path: []const u8,
    install_path: []const u8,
) !*std.Build.Step {
    const compile_step = b.addRunArtifact(glslc_exe);

    // Add optimization flags based on the build mode, as recommended by the README.
    switch (optimize) {
        .Debug => {},
        .ReleaseSafe, .ReleaseFast => compile_step.addArgs(&.{"--optimize-perf"}),
        .ReleaseSmall => compile_step.addArgs(&.{ "--optimize-perf", "--optimize-size" }),
    }
    compile_step.addArgs(&.{
        "--target", "Vulkan-1.3",
    });

    // The command format is: [options] <input> <output>

    // 1. Add the input file argument.
    compile_step.addFileArg(b.path(glsl_path));

    // 2. Add the output file argument. This is positional, NO "-o" FLAG.
    //    addOutputFileArg correctly handles this and gives us a handle to the artifact.
    const output_file_source = compile_step.addOutputFileArg(spv_path);

    // 3. Install the generated artifact to the desired location.
    const install_step = b.addInstallFile(output_file_source, install_path);

    return &install_step.step;
}

fn linkVulkanAndGlfwLibs(
    exe: *std.Build.Step.Compile,
    target: std.Build.ResolvedTarget,
    lib_glfw: *std.Build.Step.Compile,
    glfw_dep: *std.Build.Dependency,
    vk_headers_dep: *std.Build.Dependency,
) void {
    exe.linkLibrary(lib_glfw);
    exe.addIncludePath(glfw_dep.path("include"));
    exe.addIncludePath(vk_headers_dep.path("include"));
    exe.linkSystemLibrary("vulkan");

    if (target.result.os.tag == .windows) {
        exe.linkSystemLibrary("gdi32");
        exe.linkSystemLibrary("shell32");
    } else if (target.result.os.tag == .linux) {
        exe.linkSystemLibrary("m");
        exe.linkSystemLibrary("pthread");
        exe.linkSystemLibrary("dl");
        exe.linkSystemLibrary("X11");
        exe.linkSystemLibrary("xcb");
        exe.linkSystemLibrary("Xrandr");
        exe.linkSystemLibrary("Xinerama");
        exe.linkSystemLibrary("Xi");
        exe.linkSystemLibrary("Xcursor");
        exe.linkSystemLibrary("Xxf86vm");
    } else if (target.result.os.tag == .macos) {
        exe.linkFramework("Cocoa");
        exe.linkFramework("IOKit");
        exe.linkFramework("CoreFoundation");
    }
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    //================================================================================
    // Fetch and Build Dependencies
    //================================================================================

    const glfw_dep = b.dependency("glfw", .{ .target = target, .optimize = optimize });
    const vk_headers_dep = b.dependency("vulkan_headers", .{});

    // SOLUTION: Fetch the glslc dependency and get its pre-defined executable artifact.
    // This is much simpler and more reliable.
    const glslc_dep = b.dependency("glslc", .{ .target = target, .optimize = optimize });
    const glslc_exe = glslc_dep.artifact("shader_compiler");

    // --- Compile GLFW as a Static Library --- (This part remains the same)
    const lib_glfw = b.addStaticLibrary(.{ .name = "glfw", .target = target, .optimize = optimize });
    lib_glfw.linkLibC();
    lib_glfw.addIncludePath(glfw_dep.path("include"));
    lib_glfw.addCSourceFiles(.{
        .root = glfw_dep.path("src"),
        .files = &.{ "context.c", "init.c", "input.c", "monitor.c", "vulkan.c", "window.c", "osmesa_context.c", "platform.c", "egl_context.c", "null_init.c", "null_monitor.c", "null_joystick.c", "null_window.c" },
    });
    if (target.result.os.tag == .linux) {
        lib_glfw.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{ "x11_init.c", "x11_monitor.c", "x11_window.c", "xkb_unicode.c", "posix_time.c", "posix_thread.c", "posix_module.c", "posix_poll.c", "glx_context.c", "linux_joystick.c" } });
        lib_glfw.root_module.addCMacro("_GLFW_X11", "1");
        lib_glfw.linkSystemLibrary("X11");
        lib_glfw.linkSystemLibrary("Xrandr");
        lib_glfw.linkSystemLibrary("Xinerama");
        lib_glfw.linkSystemLibrary("Xi");
        lib_glfw.linkSystemLibrary("Xcursor");
        lib_glfw.linkSystemLibrary("Xxf86vm");
    }
    // (You would add .windows and .macos branches here as before)

    //================================================================================
    // Executables and Shaders
    //================================================================================

    // ... The rest of your build script can now remain exactly the same ...
    // ... It will correctly use the `glslc_exe` artifact we fetched. ...

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

    const gui_vert_shader_source = "src/shaders/code/gui/gui.vert";
    const gui_frag_shader_source = "src/shaders/code/gui/gui.frag";
    const gui_vert_shader_output = "spirv/bin/gui.vert.spv";
    const gui_frag_shader_output = "spirv/bin/gui.frag.spv";

    const text_vert_shader_source = "src/shaders/code/text/text.vert";
    const text_frag_shader_source = "src/shaders/code/text/text.frag";
    const text_vert_shader_output = "spirv/bin/text3d.vert.spv";
    const text_frag_shader_output = "spirv/bin/text3d.frag.spv";

    const install_gui_vert_shader = try addShaderStep(b, glslc_exe, optimize, gui_vert_shader_source, gui_vert_shader_output, "../src/shaders/spirv/bin/gui.vert.spv");
    const install_gui_frag_shader = try addShaderStep(b, glslc_exe, optimize, gui_frag_shader_source, gui_frag_shader_output, "../src/shaders/spirv/bin/gui.frag.spv");
    const install_text_vert_shader = try addShaderStep(b, glslc_exe, optimize, text_vert_shader_source, text_vert_shader_output, "../src/shaders/spirv/bin/text3d.vert.spv");
    const install_text_frag_shader = try addShaderStep(b, glslc_exe, optimize, text_frag_shader_source, text_frag_shader_output, "../src/shaders/spirv/bin/text3d.frag.spv");

    for (executables) |exe_info| {
        const shader_code_path = "src/shaders/code/";
        const vert_shader_source = b.fmt(shader_code_path ++ "{s}/{s}.vert", .{ exe_info.step_id, exe_info.step_id });
        const frag_shader_source = b.fmt(shader_code_path ++ "{s}/{s}.frag", .{ exe_info.step_id, exe_info.step_id });
        const vert_shader_output = b.fmt("spirv/bin/{s}.vert.spv", .{exe_info.step_id});
        const frag_shader_output = b.fmt("spirv/bin/{s}.frag.spv", .{exe_info.step_id});

        const install_vert_shader = try addShaderStep(b, glslc_exe, optimize, vert_shader_source, vert_shader_output, "../src/shaders/spirv/bin/vert.spv");
        const install_frag_shader = try addShaderStep(b, glslc_exe, optimize, frag_shader_source, frag_shader_output, "../src/shaders/spirv/bin/frag.spv");

        const exe = b.addExecutable(.{
            .name = exe_info.name orelse exe_info.step_id,
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(exe_info.source),
        });

        exe.root_module.addAnonymousImport("c", .{ .root_source_file = b.path("src/c/c.zig") });
        exe.root_module.addAnonymousImport("spirv", .{ .root_source_file = b.path("src/shaders/spirv/spirv.zig") });
        exe.root_module.addAnonymousImport("font", .{ .root_source_file = b.path("src/fonts/font.zig") });
        exe.root_module.addAnonymousImport("geometry", .{ .root_source_file = b.path("src/scenes/geometry.zig") });
        exe.root_module.addAnonymousImport("util", .{ .root_source_file = b.path("src/util/util.zig") });

        exe.step.dependOn(install_vert_shader);
        exe.step.dependOn(install_frag_shader);
        exe.step.dependOn(install_gui_vert_shader);
        exe.step.dependOn(install_gui_frag_shader);
        exe.step.dependOn(install_text_vert_shader);
        exe.step.dependOn(install_text_frag_shader);

        linkVulkanAndGlfwLibs(exe, target, lib_glfw, glfw_dep, vk_headers_dep);

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
