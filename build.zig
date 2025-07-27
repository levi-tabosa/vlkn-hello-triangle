const std = @import("std");

/// Compiles a GLSL shader to SPIR-V using shader-compiler targeting Vulkan-1.3
/// Returns a install step that associate with the artifact.
fn addShaderStep(
    b: *std.Build,
    glslc_exe: *std.Build.Step.Compile,
    optimize: std.builtin.OptimizeMode,
    source_path: []const u8,
    // We need the final output name, e.g., "gui.vert.spv".
    output_name: []const u8,
) *std.Build.Step {
    const compile_step = b.addRunArtifact(glslc_exe);

    // Options
    switch (optimize) {
        .Debug => {},
        .ReleaseSafe, .ReleaseFast => compile_step.addArgs(&.{"--optimize-perf"}),
        .ReleaseSmall => compile_step.addArgs(&.{ "--optimize-perf", "--optimize-size" }),
    }

    compile_step.addArgs(&.{ "--target", "Vulkan-1.3" });

    // Re-execute the compile step if the source file changes.
    compile_step.addFileArg(b.path(source_path));

    // Get a handle to the output artifact.
    // This will be a new handle if cache miss occurs.
    const output_file_source = compile_step.addOutputFileArg(output_name);

    const install_step = b.addInstallFile(
        output_file_source,
        b.fmt("shaders/{s}", .{output_name}),
    );

    return &install_step.step;
}

/// This function configures a `Module` with all necessary C dependencies.
/// Any executable importing this module will automatically inherit these settings.
fn configureVulkanAndGlfw(
    module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    lib_glfw: *std.Build.Step.Compile,
    glfw_dep: *std.Build.Dependency,
    vk_headers_dep: *std.Build.Dependency,
) void {
    module.linkLibrary(lib_glfw);
    module.addIncludePath(glfw_dep.path("include"));
    module.addIncludePath(vk_headers_dep.path("include"));
    module.linkSystemLibrary("vulkan", .{});

    // Platform-specific libraries
    switch (target.result.os.tag) {
        .windows => {
            module.linkSystemLibrary("gdi32", .{});
            module.linkSystemLibrary("shell32", .{});
        },
        .linux => {
            module.linkSystemLibrary("m", .{});
            module.linkSystemLibrary("pthread", .{});
            module.linkSystemLibrary("dl", .{});
            module.linkSystemLibrary("X11", .{});
            module.linkSystemLibrary("xcb", .{});
            module.linkSystemLibrary("Xrandr", .{});
            module.linkSystemLibrary("Xinerama", .{});
            module.linkSystemLibrary("Xi", .{});
            module.linkSystemLibrary("Xcursor", .{});
            module.linkSystemLibrary("Xxf86vm", .{});
        },
        .macos => {
            module.linkFramework("Cocoa", .{});
            module.linkFramework("IOKit", .{});
            module.linkFramework("CoreFoundation", .{});
        },
        else => {},
    }
}

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const glfw_dep = b.dependency("glfw", .{ .target = target, .optimize = optimize });
    const vk_headers_dep = b.dependency("vulkan_headers", .{});
    const glslc_dep = b.dependency("glslc", .{ .target = target, .optimize = optimize });
    const glslc_exe = glslc_dep.artifact("shader_compiler");

    // --- Compile GLFW as a Static Library ---
    const lib_glfw = b.addStaticLibrary(.{ .name = "glfw", .target = target, .optimize = optimize });
    lib_glfw.linkLibC();
    lib_glfw.addIncludePath(glfw_dep.path("include"));
    lib_glfw.addCSourceFiles(.{
        .root = glfw_dep.path("src"),
        .files = &.{
            "context.c",
            "init.c",
            "input.c",
            "monitor.c",
            "vulkan.c",
            "window.c",
            "osmesa_context.c",
            "platform.c",
            "egl_context.c",
            "null_init.c",
            "null_monitor.c",
            "null_joystick.c",
            "null_window.c",
        },
    });
    if (target.result.os.tag == .linux) {
        lib_glfw.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
            "x11_init.c",
            "x11_monitor.c",
            "x11_window.c",
            "xkb_unicode.c",
            "posix_time.c",
            "posix_thread.c",
            "posix_module.c",
            "posix_poll.c",
            "glx_context.c",
            "linux_joystick.c",
        } });
        lib_glfw.root_module.addCMacro("_GLFW_X11", "1");
        lib_glfw.linkSystemLibrary("X11");
        lib_glfw.linkSystemLibrary("Xrandr");
        lib_glfw.linkSystemLibrary("Xinerama");
        lib_glfw.linkSystemLibrary("Xi");
        lib_glfw.linkSystemLibrary("Xcursor");
        lib_glfw.linkSystemLibrary("Xxf86vm");
    } else if (target.result.os.tag == .windows) {
        lib_glfw.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
            "win32_init.c",
            "win32_joystick.c",
            "win32_monitor.c",
            "win32_time.c",
            "win32_thread.c",
            "win32_window.c",
        } });
        lib_glfw.root_module.addCMacro("_GLFW_WIN32", "1");
        lib_glfw.linkSystemLibrary("gdi32");
        lib_glfw.linkSystemLibrary("shell32");
    } else if (target.result.os.tag == .macos) {
        lib_glfw.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
            "cocoa_init.m",
            "cocoa_joystick.m",
            "cocoa_monitor.m",
            "cocoa_time.m",
            "cocoa_window.m",
        } });
        lib_glfw.root_module.addCMacro("_GLFW_COCOA", "1");
    }

    // This module is created and configured with all C dependencies.
    // Any executable that imports "c" will now automatically get all the correct
    // include paths and library links.
    const c_mod = b.createModule(.{
        .root_source_file = b.path("src/c/c.zig"),
        .target = target,
    });
    configureVulkanAndGlfw(c_mod, target, lib_glfw, glfw_dep, vk_headers_dep);

    const shaders = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "gui", .path = "src/shaders/code/gui" },
        .{ .name = "text", .path = "src/shaders/code/text" },
        .{ .name = "triangle", .path = "src/shaders/code/triangle" },
        .{ .name = "example", .path = "src/shaders/code/example" },
        .{ .name = "test", .path = "src/shaders/code/test" },
    };

    // Compile all shaders and collect their install steps.
    var shader_install_steps = std.ArrayList(*std.Build.Step).init(b.allocator);
    defer shader_install_steps.deinit();

    for (shaders) |shader| {
        const vert_source = b.fmt("{s}/{s}.vert", .{ shader.path, shader.name });
        const frag_source = b.fmt("{s}/{s}.frag", .{ shader.path, shader.name });
        const vert_output = b.fmt("{s}.vert.spv", .{shader.name});
        const frag_output = b.fmt("{s}.frag.spv", .{shader.name});

        const install_vert_step = addShaderStep(b, glslc_exe, optimize, vert_source, vert_output);
        const install_frag_step = addShaderStep(b, glslc_exe, optimize, frag_source, frag_output);

        try shader_install_steps.append(install_vert_step);
        try shader_install_steps.append(install_frag_step);
    }

    // TODO: add shader binaries as anonymous imports
    // and generate spirv.zig automatically.
    // This will have the same effect but binaries won't need
    // to be in the package directory.
    const spirv_options = b.addOptions();
    spirv_options.addOption(
        []const u8,
        "out_dir",
        b.fmt("{s}/shaders/", .{b.install_prefix}),
    );
    const spirv_mod = b.createModule(.{
        .root_source_file = b.path("spirv.zig"),
        .target = target,
    });

    spirv_mod.addOptions("shaders", spirv_options);

    // Define and Build Executables
    const execs = [_]struct { []const u8, []const u8 }{
        .{ "triangle", "example/main.zig" },
        .{ "example", "example/example.zig" },
        .{ "test", "example/test.zig" },
    };

    for (execs) |exe_info| {
        const exe_id, const src = exe_info;
        const exe = b.addExecutable(.{
            .name = exe_id,
            .target = target,
            .optimize = optimize,
            .root_source_file = b.path(src),
        });

        exe.root_module.addImport("c", c_mod);
        exe.root_module.addImport("spirv", spirv_mod);

        exe.root_module.addAnonymousImport("font", .{ .root_source_file = b.path("src/fonts/font.zig") });
        // TODO: Make this import a scene interface instead so scenes can be user code
        exe.root_module.addAnonymousImport("geometry", .{ .root_source_file = b.path("src/scenes/geometry.zig") });
        exe.root_module.addAnonymousImport("util", .{ .root_source_file = b.path("src/util/util.zig") });

        // Shaders should be installed before compiling the executable.
        for (shader_install_steps.items) |shader_step| {
            exe.step.dependOn(shader_step);
        }

        // For now we run it emediately after building.
        const install = b.addInstallArtifact(exe, .{});
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(&install.step);

        const run_step = b.step(exe_id, b.fmt("Run the {s} example", .{exe_id}));
        run_step.dependOn(&run_cmd.step);
    }
}
