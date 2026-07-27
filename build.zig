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

pub fn build(b: *std.Build) !void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const glfw_dep = b.dependency("glfw", .{ .target = target, .optimize = optimize });
    const vk_loader_dep = b.dependency("vulkan_loader", .{});
    const vk_loader_mod = b.addModule("Vulkan Loader Module", .{ .target = target, .optimize = optimize, .link_libcpp = true });

    const glslc_dep = b.dependency("glslc", .{ .target = target, .optimize = optimize });
    const glslc_exe = glslc_dep.artifact("mr_glsl");

    // Compile GLFW as a Static Library
    const glfw_mod = b.addModule("glfw", .{ .target = target, .optimize = optimize, .link_libc = true });
    const vk_headers_dep = b.dependency("vulkan_headers", .{});

    // Add include paths and source files for GLFW for all platforms
    glfw_mod.addIncludePath(glfw_dep.path("include"));
    glfw_mod.addCSourceFiles(.{
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
    // Platform-specific libraries
    switch (target.result.os.tag) {
        .windows => {
            glfw_mod.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
                "win32_init.c",
                "win32_joystick.c",
                "win32_monitor.c",
                "win32_time.c",
                "win32_thread.c",
                "win32_window.c",
            } });
            glfw_mod.addCMacro("_GLFW_WIN32", "1");
            glfw_mod.linkSystemLibrary("gdi32", .{});
            glfw_mod.linkSystemLibrary("shell32", .{});
        },
        .linux => {
            glfw_mod.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
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
            glfw_mod.addCMacro("_GLFW_X11", "1");
            glfw_mod.linkSystemLibrary("X11", .{});
            glfw_mod.linkSystemLibrary("Xrandr", .{});
            glfw_mod.linkSystemLibrary("Xinerama", .{});
            glfw_mod.linkSystemLibrary("Xi", .{});
            glfw_mod.linkSystemLibrary("Xcursor", .{});
            glfw_mod.linkSystemLibrary("Xxf86vm", .{});
        },
        .macos => {
            glfw_mod.addCSourceFiles(.{ .root = glfw_dep.path("src"), .files = &.{
                "cocoa_init.m",
                "cocoa_joystick.m",
                "cocoa_monitor.m",
                "cocoa_time.m",
                "cocoa_window.m",
            } });
            glfw_mod.addCMacro("_GLFW_COCOA", "1");
            glfw_mod.linkFramework("Cocoa", .{});
            glfw_mod.linkFramework("IOKit", .{});
            glfw_mod.linkFramework("CoreFoundation", .{});
        },
        else => {},
    }

    // Arquivos comuns (todas as plataformas)
    var vk_loader_files = try std.ArrayList([]const u8).initCapacity(b.allocator, 20);
    vk_loader_files.appendSlice(b.allocator, &.{
        "allocation.c",
        "asm_offset.c",
        "cJSON.c",
        "debug_utils.c",
        "dev_ext_trampoline.c",
        "extension_manual.c",
        "gpa_helper.c",
        "loader.c",
        "loader_environment.c",
        "loader_json.c",
        "log.c",
        "phys_dev_ext.c",
        "settings.c",
        "terminator.c",
        "trampoline.c",
        "unknown_function_handling.c",
        "wsi.c",
    }) catch @panic("OOM");

    if (target.result.os.tag == .linux) {
        vk_loader_files.append(b.allocator, "loader_linux.c") catch @panic("OOM");
    } else if (target.result.os.tag == .windows) {
        vk_loader_files.appendSlice(b.allocator, &.{
            "loader_windows.c",
            "dirent_on_windows.c",
        }) catch @panic("OOM");
    }

    vk_loader_mod.addCSourceFiles(.{
        .root = vk_loader_dep.path("loader"),
        .files = vk_loader_files.items,
    });

    vk_loader_mod.addIncludePath(vk_loader_dep.path("loader"));
    vk_loader_mod.addIncludePath(vk_loader_dep.path("loader/generated"));

    if (target.result.os.tag == .linux) {
        vk_loader_mod.addCMacro("VK_USE_PLATFORM_XCB_KHR", "1");
        vk_loader_mod.linkSystemLibrary("xcb", .{});
        vk_loader_mod.addCMacro("_GNU_SOURCE", "1");
        vk_loader_mod.addCMacro("VK_ENABLE_BETA_EXTENSIONS", "1");
        vk_loader_mod.addCMacro("SYSCONFDIR", "\"/etc\"");
        vk_loader_mod.addCMacro("FALLBACK_CONFIG_DIRS", "\"/etc/xdg\"");
        vk_loader_mod.addCMacro("FALLBACK_DATA_DIRS", "\"/usr/local/share:/usr/share\"");
        glfw_mod.linkSystemLibrary("X11-xcb", .{});
        //vk_miod.addCMacro("VK_ENABLE_BETA_EXTENSIONS", "1");
    }

    const vk_loader_lib = b.addLibrary(.{ .name = "Vulkan Loader Library", .root_module = vk_loader_mod });

    const glfw_lib = b.addLibrary(.{
        .name = "glfw-library",
        .root_module = glfw_mod,
    });

    const shaders = [_]struct { name: []const u8, path: []const u8 }{
        .{ .name = "gui", .path = "src/shaders/code/gui" },
        .{ .name = "text", .path = "src/shaders/code/text" },
        .{ .name = "triangle", .path = "src/shaders/code/triangle" },
        .{ .name = "example", .path = "src/shaders/code/example" },
        .{ .name = "test", .path = "src/shaders/code/test" },
    };

    // Compile all shaders and collect their install steps.
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

    // Define and Build Demos
    const execs = [_]struct { []const u8, []const u8 }{
        .{ "triangle", "example/main.zig" },
        .{ "example", "example/example.zig" },
        .{ "test", "example/test.zig" },
    };

    for (execs) |exe_info| {
        const exe_id, const src = exe_info;
        const module = b.addModule(b.fmt("{s}_mod", .{exe_id}), .{
            .optimize = optimize,
            .target = target,
            .root_source_file = b.path(src),
        });

        const exe = b.addExecutable(.{
            .name = exe_id,
            .root_module = module,
        });

        module.addImport("spirv", spirv_mod);
        module.addAnonymousImport("font", .{ .root_source_file = b.path("src/fonts/font.zig") });
        module.addAnonymousImport("png", .{ .root_source_file = b.path("src/png/png_helper.zig") });
        // TODO: Make this import a scene interface instead so scenes can be user code
        module.addAnonymousImport("geometry", .{ .root_source_file = b.path("src/scenes/geometry.zig") });
        module.addAnonymousImport("util", .{ .root_source_file = b.path("src/util/util.zig") });
        // Shaders should be installed before compiling the executable.
        module.addIncludePath(glfw_dep.path("include"));
        module.linkLibrary(glfw_lib);
        module.linkLibrary(vk_loader_lib);
        module.addIncludePath(.{ .dependency = .{ .dependency = vk_headers_dep, .sub_path = "include" } });

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
