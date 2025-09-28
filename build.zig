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

fn configureGlfwLib(
    lib: *std.Build.Step.Compile,
    glfw_dep: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
    x11_headers_dep: *std.Build.Dependency,
) void {
    switch (target.result.os.tag) {
        .windows => {
            // Untested stub
            lib.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
                "win32_init.c", "win32_joystick.c", "win32_monitor.c",
                "win32_time.c", "win32_thread.c",   "win32_window.c",
            } });
            lib.root_module.addCMacro("_GLFW_WIN32", "1");
            lib.linkSystemLibrary("gdi32");
            lib.linkSystemLibrary("shell32");
        },
        .linux => {
            lib.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
                "x11_init.c",       "x11_monitor.c", "x11_window.c",
                "xkb_unicode.c",    "posix_time.c",  "posix_thread.c",
                "posix_module.c",   "posix_poll.c",  "glx_context.c",
                "linux_joystick.c",
            } });
            lib.root_module.addCMacro("_GLFW_X11", "1");
            lib.addIncludePath(x11_headers_dep.path(""));
        },
        .macos => {
            // Untested stub
            lib.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
                "cocoa_init.m", "cocoa_joystick.m", "cocoa_monitor.m",
                "cocoa_time.m", "cocoa_window.m",
            } });
            lib.root_module.addCMacro("_GLFW_COCOA", "1");
        },
        else => @panic("Unsupported OS."),
    }
}

fn configureVulkanLoaderLib(
    lib: *std.Build.Step.Compile,
    vk_loader_dep: *std.Build.Dependency,
    target: std.Build.ResolvedTarget,
    x11_headers_dep: *std.Build.Dependency,
) void {
    switch (target.result.os.tag) {
        .windows => {
            // Untested stub
            lib.root_module.addCMacro("_CRT_SECURE_NO_WARNINGS", "NULL");

            // Add the correct Windows-specific source files for the loader.
            lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{ "loader_windows.c", "dirent_on_windows.c", "wsi_win32.c" },
            });
        },
        .linux => {
            // Tell the loader sources we have these system headers / features
            lib.root_module.addCMacro("SYSCONFDIR", "\"/etc\"");
            lib.root_module.addCMacro("FALLBACK_CONFIG_DIRS", "\"/etc/xdg\"");
            lib.root_module.addCMacro("FALLBACK_DATA_DIRS", "\"/usr/local/share:/usr/share\"");
            lib.root_module.addCMacro("HAVE_SYS_STAT_H", "1");
            lib.root_module.addCMacro("HAVE_STDATOMIC_H", "1");
            lib.root_module.addCMacro("VK_USE_PLATFORM_XCB_KHR", "1");

            // Add these two lines (and optional _GNU_SOURCE) to make dladdr/Dl_info visible
            lib.root_module.addCMacro("HAVE_DLFCN_H", "1");
            lib.root_module.addCMacro("HAVE_DLADDR", "1");
            // optional, sometimes source expects GNU extensions:
            lib.root_module.addCMacro("_GNU_SOURCE", "1");

            // Add the correct Linux-specific source files for the loader.
            lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{ "loader_linux.c", "wsi.c" },
            });
            lib.addIncludePath(x11_headers_dep.path(""));
        },
        .macos => {
            // Untested stub
            // For macOS, the Vulkan loader re-uses the Linux source file for loader discovery.
            // Zig automatically defines __APPLE__ when targeting macOS, which the source uses.
            lib.root_module.addCMacro("SYSCONFDIR", "\"/etc\"");
            lib.root_module.addCMacro("HAVE_SYS_STAT_H", "1");
            lib.root_module.addCMacro("HAVE_STDATOMIC_H", "1");

            lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{"loader_linux.c"},
            });
            // The macOS WSI file is Objective-C
            lib.addCSourceFiles(.{
                .root = vk_loader_dep.path("loader"),
                .files = &.{
                    "wsi_metal.m",
                },
            });
        },
        else => @panic("Unsupported OS."),
    }
}

/// This function configures a `Module` with all necessary C dependencies.
/// Any executable importing this module will automatically inherit these settings.
fn configureModuleWithVulkanAndGlfw(
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

    // Platform-specific libraries
    switch (target.result.os.tag) {
        .windows => {
            // Untested stub
            module.linkSystemLibrary("gdi32", .{});
            module.linkSystemLibrary("shell32", .{});
        },
        .linux => {
            module.linkSystemLibrary("m", .{});
            module.linkSystemLibrary("pthread", .{});
            module.linkSystemLibrary("dl", .{});
        },
        .macos => {
            // Untested stub
            module.linkFramework("Cocoa", .{});
            module.linkFramework("IOKit", .{});
            module.linkFramework("CoreFoundation", .{});
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
    const x11_headers_dep = b.dependency("x11", .{ .target = target, .optimize = optimize });

    const root_mod_lib_glfw = b.addModule("glfw", .{
        .target = target,
        .optimize = optimize,
    });

    const root_mod_lib_vk_loader = b.addModule("vulkan-loader", .{
        .target = target,
        .optimize = optimize,
    });

    const lib_glfw = b.addLibrary(.{
        .name = "glfw",
        .root_module = root_mod_lib_glfw,
    });
    const lib_vulkan_loader = b.addLibrary(.{
        .name = "vulkan-loader",
        .root_module = root_mod_lib_vk_loader,
    });

    // Linking
    lib_glfw.linkLibC();
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
    lib_glfw.addIncludePath(glfw_dep.path("include"));

    lib_vulkan_loader.linkLibC();
    lib_vulkan_loader.addCSourceFiles(.{
        .root = vk_loader_dep.path("loader"),
        .files = &.{
            "loader.c",           "allocation.c",         "unknown_function_handling.c", "trampoline.c",
            "terminator.c",       "wsi.c",                "log.c",                       "cJSON.c",
            "loader_json.c",      "loader_environment.c", "settings.c",                  "dev_ext_trampoline.c",
            "extension_manual.c", "debug_utils.c",        "gpa_helper.c",
        },
    });

    // Include paths and essential macros for Vulkan loader
    lib_vulkan_loader.addIncludePath(vk_headers_dep.path("include"));
    lib_vulkan_loader.addIncludePath(vk_loader_dep.path("loader"));
    lib_vulkan_loader.addIncludePath(vk_loader_dep.path("loader/generated"));
    lib_vulkan_loader.root_module.addCMacro("VULKAN_LOADER_STATIC_LIB", "1");
    lib_vulkan_loader.root_module.addCMacro("FALLTHROUGH_SUPPORTED", "1");
    lib_vulkan_loader.root_module.addCMacro("VK_ENABLE_BETA_EXTENSIONS", "1");

    // OS-dependent configuration for libraries
    configureGlfwLib(lib_glfw, glfw_dep, target, x11_headers_dep);
    configureVulkanLoaderLib(lib_vulkan_loader, vk_loader_dep, target, x11_headers_dep);

    const c_mod = b.createModule(.{
        .root_source_file = b.path("src/c/c.zig"),
        .target = target,
        .optimize = optimize,
    });
    configureModuleWithVulkanAndGlfw(c_mod, target, lib_glfw, glfw_dep, vk_headers_dep, vk_loader_dep, lib_vulkan_loader);

    const shaders = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "gui", .path = "src/shaders/code/gui" },
        .{ .name = "text", .path = "src/shaders/code/text" },
        .{ .name = "triangle", .path = "src/shaders/code/triangle" },
        .{ .name = "example", .path = "src/shaders/code/example" },
        .{ .name = "test", .path = "src/shaders/code/test" },
    };

    var shader_install_steps = try std.ArrayList(*std.Build.Step).initCapacity(b.allocator, shaders.len * 2);
    defer shader_install_steps.deinit(b.allocator);

    for (shaders) |shader| {
        const vert_source = b.fmt("{s}/{s}.vert", .{ shader.path, shader.name });
        const frag_source = b.fmt("{s}/{s}.frag", .{ shader.path, shader.name });
        const vert_output = b.fmt("{s}.vert.spv", .{shader.name});
        const frag_output = b.fmt("{s}.frag.spv", .{shader.name});
        const install_vert_step = addShaderStep(b, glslc_exe, optimize, vert_source, vert_output);
        const install_frag_step = addShaderStep(b, glslc_exe, optimize, frag_source, frag_output);
        try shader_install_steps.append(b.allocator, install_vert_step);
        try shader_install_steps.append(b.allocator, install_frag_step);
    }

    const spirv_options = b.addOptions();
    spirv_options.addOption([]const u8, "out_dir", b.fmt("{s}/shaders/", .{b.install_prefix}));
    const spirv_mod = b.createModule(.{
        .root_source_file = b.path("spirv.zig"),
        .target = target,
        .optimize = optimize,
    });
    spirv_mod.addOptions("shaders", spirv_options);

    const execs = [_]struct { []const u8, []const u8 }{
        .{ "triangle", "example/main.zig" },
        .{ "example", "example/example.zig" },
        .{ "test", "example/test.zig" },
    };

    for (execs) |exe_info| {
        const exe_id, const src = exe_info;
        const exe_mod = b.addModule(b.fmt("{s} module", .{exe_id}), .{
            .root_source_file = b.path(src),
            .target = target,
            .optimize = optimize,
        });
        const exe = b.addExecutable(.{
            .name = exe_id,
            .root_module = exe_mod,
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
