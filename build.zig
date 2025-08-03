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
    compile_step.addFileArg(b.path(source_path));
    const output_file_source = compile_step.addOutputFileArg(output_name);
    const install_step = b.addInstallFile(
        output_file_source,
        b.fmt("shaders/{s}", .{output_name}),
    );
    return &install_step.step;
}

//// This function configures a `Module` with all necessary C dependencies.
/// Any executable importing this module will automatically inherit these settings.
fn configureVulkanAndGlfw(
    module: *std.Build.Module,
    target: std.Build.ResolvedTarget,
    glfw_lib: *std.Build.Step.Compile,
    glfw_dep: *std.Build.Dependency,
    vk_headers_dep: *std.Build.Dependency,
    vk_loader_dep: *std.Build.Dependency,
    vk_loader_lib: *std.Build.Step.Compile,
) void {
    module.linkLibrary(glfw_lib);
    module.addIncludePath(glfw_dep.path("include"));
    module.linkLibrary(vk_loader_lib);
    module.addIncludePath(vk_headers_dep.path("include"));
    module.addIncludePath(vk_loader_dep.path("include"));
    module.addIncludePath(vk_loader_dep.path("loader"));

    // Platform-specific libraries and source files
    switch (target.result.os.tag) {
        .windows => {
            module.linkSystemLibrary("gdi32", .{});
            module.linkSystemLibrary("shell32", .{});
            vk_loader_lib.root_module.addCMacro("_CRT_SECURE_NO_WARNINGS", "NULL");

            // Add the correct Windows-specific source files for the loader.
            vk_loader_lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{ "loader_windows.c", "dirent_on_windows.c", "wsi_win32.c" },
            });
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

            vk_loader_lib.root_module.addCMacro("SYSCONFDIR", "\"/etc\"");
            vk_loader_lib.root_module.addCMacro("FALLBACK_CONFIG_DIRS", "\"/etc/xdg\"");
            vk_loader_lib.root_module.addCMacro("FALLBACK_DATA_DIRS", "\"/usr/local/share:/usr/share\"");
            vk_loader_lib.root_module.addCMacro("HAVE_SYS_STAT_H", "1");
            vk_loader_lib.root_module.addCMacro("HAVE_XCB_H", "1");
            vk_loader_lib.root_module.addCMacro("HAVE_STDATOMIC_H", "1");

            // Add the correct Linux-specific source files for the loader.
            vk_loader_lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{ "loader_linux.c", "wsi.c" },
            });
        },
        .macos => {
            module.linkFramework("Cocoa", .{});
            module.linkFramework("IOKit", .{});
            module.linkFramework("CoreFoundation", .{});

            // For macOS, the Vulkan loader re-uses the Linux source file for loader discovery.
            // Zig automatically defines __APPLE__ when targeting macOS, which the source uses.
            vk_loader_lib.root_module.addCMacro("SYSCONFDIR", "\"/etc\"");
            vk_loader_lib.root_module.addCMacro("HAVE_SYS_STAT_H", "1");
            vk_loader_lib.root_module.addCMacro("HAVE_STDATOMIC_H", "1");

            vk_loader_lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{"loader_linux.c"},
            });
            // The macOS WSI file is Objective-C
            vk_loader_lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{
                    "wsi_metal.m",
                },
            });
        },
        else => @panic("Unsupported OS."),
    }
}
pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Declare dependencies and libs
    const glfw_dep = b.dependency("glfw", .{ .target = target, .optimize = optimize });
    const vk_headers_dep = b.dependency("vulkan_headers", .{ .target = target, .optimize = optimize });
    const vk_loader_dep = b.dependency("vulkan_loader", .{ .target = target, .optimize = optimize });
    const glslc_dep = b.dependency("glslc", .{ .target = target, .optimize = optimize });
    const glslc_exe = glslc_dep.artifact("shader_compiler");
    const lib_glfw = b.addStaticLibrary(.{ .name = "glfw", .target = target, .optimize = optimize });
    const lib_vulkan_loader = b.addStaticLibrary(.{ .name = "vulkan-loader", .target = target, .optimize = optimize });

    // Linking
    lib_glfw.linkLibC();
    lib_vulkan_loader.linkLibC();
    lib_glfw.addCSourceFiles(.{
        .root = glfw_dep.path("src"),
        .files = &.{
            "context.c",        "init.c",         "input.c",
            "monitor.c",        "vulkan.c",       "window.c",
            "osmesa_context.c", "platform.c",     "egl_context.c",
            "null_init.c",      "null_monitor.c", "null_joystick.c",
            "null_window.c",
        },
    });
    if (target.result.os.tag == .linux) {
        lib_glfw.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
            "x11_init.c",       "x11_monitor.c", "x11_window.c",
            "xkb_unicode.c",    "posix_time.c",  "posix_thread.c",
            "posix_module.c",   "posix_poll.c",  "glx_context.c",
            "linux_joystick.c",
        } });
        lib_glfw.root_module.addCMacro("_GLFW_X11", "1");
    } else if (target.result.os.tag == .windows) {
        lib_glfw.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
            "win32_init.c", "win32_joystick.c", "win32_monitor.c",
            "win32_time.c", "win32_thread.c",   "win32_window.c",
        } });
        lib_glfw.root_module.addCMacro("_GLFW_WIN32", "1");
        lib_glfw.linkSystemLibrary("gdi32");
        lib_glfw.linkSystemLibrary("shell32");
    } else if (target.result.os.tag == .macos) {
        lib_glfw.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
            "cocoa_init.m", "cocoa_joystick.m", "cocoa_monitor.m",
            "cocoa_time.m", "cocoa_window.m",
        } });
        lib_glfw.root_module.addCMacro("_GLFW_COCOA", "1");
    }

    // Generic source files, all located in the `loader` directory.
    lib_vulkan_loader.addCSourceFiles(.{
        .root = vk_loader_dep.path("loader"),
        .files = &.{
            "loader.c",           "allocation.c",         "unknown_function_handling.c", "trampoline.c",
            "terminator.c",       "wsi.c",                "log.c",                       "cJSON.c",
            "loader_json.c",      "loader_environment.c", "settings.c",                  "dev_ext_trampoline.c",
            "extension_manual.c", "debug_utils.c",        "gpa_helper.c",
        },
    });

    // Include paths
    lib_glfw.addIncludePath(glfw_dep.path("include"));
    lib_vulkan_loader.addIncludePath(vk_headers_dep.path("include"));
    lib_vulkan_loader.addIncludePath(vk_loader_dep.path("loader"));
    lib_vulkan_loader.addIncludePath(vk_loader_dep.path("loader/generated"));
    lib_vulkan_loader.root_module.addCMacro("VULKAN_LOADER_STATIC_LIB", "1");
    lib_vulkan_loader.root_module.addCMacro("FALLTHROUGH_SUPPORTED", "1");
    lib_vulkan_loader.root_module.addCMacro("VK_ENABLE_BETA_EXTENSIONS", "1");

    const c_mod = b.createModule(.{
        .root_source_file = b.path("src/c/c.zig"),
        .target = target,
    });
    configureVulkanAndGlfw(c_mod, target, lib_glfw, glfw_dep, vk_headers_dep, vk_loader_dep, lib_vulkan_loader);

    const shaders = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "gui", .path = "src/shaders/code/gui" },
        .{ .name = "text", .path = "src/shaders/code/text" },
        .{ .name = "triangle", .path = "src/shaders/code/triangle" },
        .{ .name = "example", .path = "src/shaders/code/example" },
        .{ .name = "test", .path = "src/shaders/code/test" },
    };

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

    const spirv_options = b.addOptions();
    spirv_options.addOption([]const u8, "out_dir", b.fmt("{s}/shaders/", .{b.install_prefix}));
    const spirv_mod = b.createModule(.{ .root_source_file = b.path("spirv.zig"), .target = target });
    spirv_mod.addOptions("shaders", spirv_options);

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
        exe.root_module.addAnonymousImport("png", .{ .root_source_file = b.path("src/png/png_helper.zig") });
        exe.root_module.addAnonymousImport("geometry", .{ .root_source_file = b.path("src/scenes/geometry.zig") });
        exe.root_module.addAnonymousImport("util", .{ .root_source_file = b.path("src/util/util.zig") });

        for (shader_install_steps.items) |shader_step| {
            exe.step.dependOn(shader_step);
        }

        const install = b.addInstallArtifact(exe, .{});
        const run_cmd = b.addRunArtifact(exe);
        run_cmd.step.dependOn(&install.step);

        const run_step = b.step(exe_id, b.fmt("Run the {s} example", .{exe_id}));
        run_step.dependOn(&run_cmd.step);
    }
}
