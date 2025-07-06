// test.zig
const std = @import("std");
const assert = std.debug.assert;
const spirv = @import("spirv");
const scene = @import("geometry");
const font = @import("font");
const gui = @import("gui/gui.zig");
const text3d = @import("text3d/text3d.zig");
const fps_tracker = @import("fps_tracker/performance_tracker.zig");
//TODO: remove
const rand = std.Random;
const c = @import("c").c;

const Allocator = std.mem.Allocator;

// --- Shader Bytecode ---
const vert_shader_bin = spirv.vs;
const frag_shader_bin = spirv.fs;

const Vertex = scene.V3;
const Scene = scene.Scene;

// --- Error Checking ---
pub fn vkCheck(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) return;

    std.log.err("Vulkan call failed with code: {}", .{result});
    return switch (result) {
        c.VK_INCOMPLETE => error.VulkanIncomplete,
        c.VK_ERROR_DEVICE_LOST => error.VulkanDeviceLost,
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY, c.VK_ERROR_OUT_OF_HOST_MEMORY => error.VulkanMemoryAllocationFailed,
        c.VK_ERROR_LAYER_NOT_PRESENT => error.VulkanLayerMissing,
        c.VK_ERROR_INITIALIZATION_FAILED => error.VulkanInitFailed,
        c.VK_ERROR_FORMAT_NOT_SUPPORTED => error.VulkanUnsupportedFormat,
        c.VK_ERROR_UNKNOWN => error.VulkanUnknown,
        c.VK_ERROR_SURFACE_LOST_KHR => error.VulkanSurfaceLost,
        c.VK_ERROR_NATIVE_WINDOW_IN_USE_KHR => error.NativeWindowInUse,
        c.VK_SUBOPTIMAL_KHR => error.VulkanSuboptimalKHR,
        c.VK_ERROR_OUT_OF_POOL_MEMORY => error.VulkanPoolOutOfMemory,
        else => error.VulkanDefault,
    };
}

pub fn checkGlfw(result: c_int) !void { // TODO: maybe remove
    if (result == c.GLFW_TRUE) return;

    std.log.err("Glfw call failed with code: {}", .{result});
    return switch (result) {
        c.GLFW_PLATFORM_ERROR => error.GlfwPlatformError,
        else => error.GlfwDefault,
    };
}

// --- Application Constants ---
const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;

// --- Vertex Definition ---
pub fn getVertexBindingDescription() c.VkVertexInputBindingDescription {
    return .{
        .binding = 0,
        .stride = @sizeOf(Vertex),
        .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
    };
}

pub fn getVertexAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
    return .{ .{
        .binding = 0,
        .location = 0,
        .format = c.VK_FORMAT_R32G32B32_SFLOAT,
        .offset = @offsetOf(Vertex, "pos"),
    }, .{
        .binding = 0,
        .location = 1,
        .format = c.VK_FORMAT_R32G32B32_SFLOAT,
        .offset = @offsetOf(Vertex, "offset"),
    }, .{
        .binding = 0,
        .location = 2,
        .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = @offsetOf(Vertex, "color"),
    } };
}

pub const WorldTxtVertex = struct {
    pos: @Vector(3, f32),
    uv: @Vector(2, f32),
    color: @Vector(4, f32),
    model: @Vector(16, f32),
    fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    fn getAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
        return .{
            .{
                .binding = 0,
                .location = 0,
                .format = c.VK_FORMAT_R32G32B32_SFLOAT,
                .offset = @offsetOf(Vertex, "pos"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(Vertex, "uv"),
            },
            .{
                .binding = 0,
                .location = 2,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(Vertex, "color"),
            },
            // location 3,4,5,6: model (mat4) - A mat4 is 4 consecutive vec4s
            .{ .binding = 0, .location = 3, .format = c.VK_FORMAT_R32G32B32A32_SFLOAT, .offset = @offsetOf(WorldTxtVertex, "model") + @sizeOf([4]f32) * 0 },
            .{ .binding = 0, .location = 4, .format = c.VK_FORMAT_R32G32B32A32_SFLOAT, .offset = @offsetOf(WorldTxtVertex, "model") + @sizeOf([4]f32) * 1 },
            .{ .binding = 0, .location = 5, .format = c.VK_FORMAT_R32G32B32A32_SFLOAT, .offset = @offsetOf(WorldTxtVertex, "model") + @sizeOf([4]f32) * 2 },
            .{ .binding = 0, .location = 6, .format = c.VK_FORMAT_R32G32B32A32_SFLOAT, .offset = @offsetOf(WorldTxtVertex, "model") + @sizeOf([4]f32) * 3 },
        };
    }
};

pub const WorldTxtRenderer = struct {
    const Self = @This();

    vk_ctx: *.VulkanContext,
    pipeline: Pipeline = undefined,
    pipeline_layout: PipelineLayout = undefined,

    descriptor_set_layout: c.VkDescriptorSetLayout = undefined,
    descriptor_pool: c.VkDescriptorPool = undefined,
    descriptor_set: c.VkDescriptorSet = undefined,
    sampler: c.VkSampler = undefined,
    texture: Image = undefined,
    texture_view: c.VkImageView = undefined,
    vertex_buffer: Buffer = undefined,
    index_buffer: Buffer = undefined,
    mapped_vertices: [*]WorldTxtVertex = undefined,
    mapped_indices: [*]u32 = undefined,
    vertex_count: u32 = 0,
    index_count: u32 = 0,
    font: gui.Font = undefined,

    const MAX_VERTICES = 8192;
    const MAX_INDICES = MAX_VERTICES * 3 / 2;

    pub fn init(vk_ctx: *.VulkanContext, render_pass: RenderPass, swapchain: Swapchain) !Self {
        var self: Self = .{
            .vk_ctx = vk_ctx,
            .font = .init(vk_ctx.allocator),
        };

        try self.font.loadFNT(font.periclesW01_fnt);
        try self.createUnifiedTextureAndSampler(vk_ctx.allocator, font.periclesW01_png);
        try self.createDescriptors();
        try self.createBuffers(vk_ctx);
        try self.createPipeline(render_pass, swapchain);
        return self;
    }

    // Reuse createUnifiedTextureAndSampler, createDescriptors, createBuffers from GuiRenderer
    // with adjustments for Text3DVertex in createBuffers

    pub fn beginFrame(self: *Self) void {
        self.vertex_count = 0;
        self.index_count = 0;
    }

    pub fn endFrame(self: *Self, cmd_buffer: c.VkCommandBuffer, view_projection: [16]f32) void {
        if (self.index_count == 0) return;

        c.vkCmdBindPipeline(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.handle);
        c.vkCmdBindDescriptorSets(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, 1, &self.descriptor_set, 0, null);
        const offset: c.VkDeviceSize = 0;
        c.vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &self.vertex_buffer.handle, &offset);
        c.vkCmdBindIndexBuffer(cmd_buffer, self.index_buffer.handle, 0, c.VK_INDEX_TYPE_UINT32);

        c.vkCmdPushConstants(cmd_buffer, self.pipeline_layout.handle, c.VK_SHADER_STAGE_VERTEX_BIT, 0, @sizeOf([16]f32), &view_projection);
        c.vkCmdDrawIndexed(cmd_buffer, self.index_count, 1, 0, 0, 0);
    }

    pub fn drawText3D(self: *Self, text: []const u8, transform: [16]f32, color: [4]f32, font_scale: f32) void {
        const M = math.matFromArray(transform); // Convert to matrix (assuming a math library)
        var current_x: f32 = 0;

        for (text) |char_code| {
            if (self.vertex_count + 4 > MAX_VERTICES or self.index_count + 6 > MAX_INDICES) return;

            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;

            // Local positions with font_scale (text on XY plane)
            const x0 = (current_x + glyph.xoffset) * font_scale;
            const y0 = glyph.yoffset * font_scale;
            const x1 = x0 + @as(f32, @floatFromInt(glyph.width)) * font_scale;
            const y1 = y0 + @as(f32, @floatFromInt(glyph.height)) * font_scale;

            const local_positions = [4][3]f32{
                .{ x0, y0, 0 },
                .{ x1, y0, 0 },
                .{ x1, y1, 0 },
                .{ x0, y1, 0 },
            };

            // Transform to world space
            var world_positions: [4][3]f32 = undefined;
            for (0..4) |i| {
                const local_vec = math.vec4(local_positions[i][0], local_positions[i][1], local_positions[i][2], 1.0);
                const world_vec = math.mulMat4Vec4(M, local_vec); // Matrix multiplication
                world_positions[i] = .{ world_vec[0], world_vec[1], world_vec[2] };
            }

            // UV coordinates
            const p0 = @as(f32, @floatFromInt(glyph.x)) / self.font.scale_w;
            const v0 = @as(f32, @floatFromInt(glyph.y)) / self.font.scale_h;
            const p1 = p0 + (@as(f32, @floatFromInt(glyph.width)) / self.font.scale_w);
            const v1 = v0 + (@as(f32, @floatFromInt(glyph.height)) / self.font.scale_h);

            // Add vertices
            const v_idx = self.vertex_count;
            self.mapped_vertices[v_idx + 0] = .{ .pos = world_positions[0], .uv = .{ p0, v0 }, .color = color };
            self.mapped_vertices[v_idx + 1] = .{ .pos = world_positions[1], .uv = .{ p1, v0 }, .color = color };
            self.mapped_vertices[v_idx + 2] = .{ .pos = world_positions[2], .uv = .{ p1, v1 }, .color = color };
            self.mapped_vertices[v_idx + 3] = .{ .pos = world_positions[3], .uv = .{ u0, v1 }, .color = color };

            // Indices
            self.mapped_indices[self.index_count + 0] = v_idx;
            self.mapped_indices[self.index_count + 1] = v_idx + 1;
            self.mapped_indices[self.index_count + 2] = v_idx + 2;
            self.mapped_indices[self.index_count + 3] = v_idx;
            self.mapped_indices[self.index_count + 4] = v_idx + 2;
            self.mapped_indices[self.index_count + 5] = v_idx + 3;

            self.vertex_count += 4;
            self.index_count += 6;

            current_x += glyph.xadvance;
        }
    }
};

const UniformBufferObject = extern struct {
    view_matrix: [16]f32,
    perspective_matrix: [16]f32,
    padding: [128]u8 = undefined,
};

// --- Callbacks and Window ---
const Callbacks = struct {
    fn cbCursorPos(wd: ?*c.GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
        const app: *App = @alignCast(@ptrCast(c.glfwGetWindowUserPointer(wd orelse return) orelse return));
        app.gui_renderer.handleCursorPos(xpos, ypos);
        // NEW: Forward to the input context as well
        app.wd_ctx.handleCursorPos(xpos, ypos);
    }

    fn cbMouseButton(wd: ?*c.GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.C) void {
        const app: *App = @alignCast(@ptrCast(c.glfwGetWindowUserPointer(wd orelse return) orelse return));
        app.gui_renderer.handleMouseButton(button, action, mods);
        app.wd_ctx.handleMouseButton(button, action);
    }

    fn cbKey(wd: ?*c.GLFWwindow, key: c_int, code: c_int, action: c_int, mods: c_int) callconv(.C) void {
        const app: *App = @alignCast(@ptrCast(c.glfwGetWindowUserPointer(wd orelse return) orelse return));
        app.wd_ctx.handleKey(key, action, mods);

        // TODO: Change
        if (action == c.GLFW_PRESS) {
            app.scene.setGridResolution(@intCast(code)) catch unreachable;
            app.updateVertexBuffer() catch @panic("Update VB failed");
        }
    }

    fn cbFramebufferResize(wd: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
        const app: *App = @alignCast(@ptrCast(c.glfwGetWindowUserPointer(wd orelse return) orelse return));
        app.window.size.x = width;
        app.window.size.y = height;
        app.framebuffer_resized = true;
    }

    fn cbScroll(wd: ?*c.GLFWwindow, xoffset: f64, yoffset: f64) callconv(.C) void {
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        const app: *App = @alignCast(@ptrCast(user_ptr));

        app.wd_ctx.handleScroll(xoffset, yoffset);
    }
};

fn addLineCallback(ptr: *anyopaque) void {
    const app: *App = @alignCast(@ptrCast(ptr));
    std.log.info("Button clicked: Adding a new line...", .{});
    var prng = rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
    const random = prng.random();
    app.scene.addLine(
        .{ 0, 0, 0 },
        .{
            random.float(f32) * 2.0 - 1.0,
            random.float(f32) * 2.0 - 1.0,
            random.float(f32) * 2.0 - 1.0,
        },
    ) catch unreachable;
    app.updateVertexBuffer() catch unreachable;
}

fn clearLinesCallback(ptr: *anyopaque) void {
    const app: *App = @alignCast(@ptrCast(ptr));
    std.log.info("Button clicked: Clearing lines.", .{});
    app.scene.clear();
}

fn quitCallback(ptr: *anyopaque) void {
    const app: *App = @alignCast(@ptrCast(ptr));
    std.log.info("Button clicked: Quitting.", .{});
    c.glfwSetWindowShouldClose(app.window.handle, 1);
}

const Window = struct {
    const Self = @This();

    handle: ?*c.GLFWwindow = undefined,
    size: struct {
        const Self = @This();
        x: c_int,
        y: c_int,
    },

    pub fn init(user_ptr: ?*anyopaque, width: c_int, height: c_int, title: [*c]const u8, monitor: ?*c.GLFWmonitor, share: ?*c.GLFWwindow) !Self {
        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_TRUE);
        const handle = c.glfwCreateWindow(width, height, title, monitor, share) orelse return error.GlfwCreateWindowFailed;
        c.glfwSetWindowUserPointer(handle, user_ptr);

        _ = c.glfwSetCursorPosCallback(handle, Callbacks.cbCursorPos);
        _ = c.glfwSetMouseButtonCallback(handle, Callbacks.cbMouseButton);
        _ = c.glfwSetKeyCallback(handle, Callbacks.cbKey);
        _ = c.glfwSetFramebufferSizeCallback(handle, Callbacks.cbFramebufferResize);
        _ = c.glfwSetScrollCallback(handle, Callbacks.cbScroll);
        return .{ .handle = handle, .size = .{ .x = width, .y = height } };
    }

    pub fn deinit(self: *Self) void {
        c.glfwDestroyWindow(self.handle);
    }

    pub fn minimized(self: Self) bool {
        return c.glfwGetWindowAttrib(self.handle, c.GLFW_ICONIFIED) == 1;
    }
};

pub const WindowContext = struct {
    const Self = @This();

    // Raw State from GLFW
    last_cursor_x: f64 = -1.0,
    last_cursor_y: f64 = -1.0,
    cursor_dx: f64 = 0,
    cursor_dy: f64 = 0,
    scroll_dy: f64 = 0,
    scroll_changed: bool = false,
    left_mouse_down: bool = false,
    ctrl_down: bool = false,

    // --- Public API for App ---

    /// Call this at the start of each frame to reset per-frame state.
    pub fn beginFrame(self: *Self) void {
        self.cursor_dx = 0;
        self.cursor_dy = 0;
        self.scroll_dy = 0;
        self.scroll_changed = false;
    }

    /// Processes camera movement based on its state.
    /// Returns true if the camera was updated.
    /// TODO: Improve
    pub fn processCameraInput(self: Self, s: *Scene) bool {
        var updated = false;

        if (self.ctrl_down) {
            const fov_change = @as(f32, @floatCast(self.scroll_dy)) * 4;
            s.camera.adjustFov(fov_change);
            updated = true;
        } else if (self.scroll_changed) {
            s.camera.adjustRadius(@as(f32, @floatCast(self.scroll_dy)) * 2);
            updated = true;
        }
        if (self.left_mouse_down) {
            // Scale mouse delta to a reasonable radian value.
            const pitch_change = @as(f32, @floatCast(self.cursor_dy)) * 0.005;
            const yaw_change = @as(f32, @floatCast(self.cursor_dx)) * 0.005;
            s.camera.adjustPitchYaw(pitch_change, yaw_change);
            updated = true;
        }

        return updated;
    }

    // --- Callback Handlers (called by GLFW callbacks) ---
    pub fn handleCursorPos(self: *Self, x: f64, y: f64) void {
        // On first event, just store position to avoid a large jump.
        if (self.last_cursor_x == -1.0) {
            self.last_cursor_x = x;
            self.last_cursor_y = y;
            return;
        }

        self.cursor_dx = x - self.last_cursor_x;
        self.cursor_dy = y - self.last_cursor_y;

        self.last_cursor_x = x;
        self.last_cursor_y = y;
    }

    pub fn handleMouseButton(self: *Self, button: c_int, action: c_int) void {
        if (button == c.GLFW_MOUSE_BUTTON_LEFT) {
            self.left_mouse_down = (action == c.GLFW_PRESS);
        }
    }

    pub fn handleKey(self: *Self, key: c_int, action: c_int, mods: c_int) void {
        if (mods & c.GLFW_MOD_CONTROL != 0) self.ctrl_down = !self.ctrl_down;

        _ = action;
        _ = key;
    }

    pub fn handleScroll(self: *Self, _: f64, yoffset: f64) void {
        self.scroll_changed = true;
        self.scroll_dy = yoffset;
    }
};

// --- Core Vulkan Component Structs ---

const Instance = struct {
    const Self = @This();

    handle: c.VkInstance = undefined,

    pub fn init() !Self {
        var self = Instance{};
        const app_info = c.VkApplicationInfo{
            .pApplicationName = "Vulkan Line App",
            .applicationVersion = c.VK_MAKE_API_VERSION(0, 1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = c.VK_MAKE_API_VERSION(0, 1, 0, 0),
            .apiVersion = c.VK_API_VERSION_1_0,
        };
        var extension_count: u32 = 0;
        const required_extensions_ptr = c.glfwGetRequiredInstanceExtensions(&extension_count);
        const required_extensions = required_extensions_ptr[0..extension_count];
        const create_info = c.VkInstanceCreateInfo{
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = @intCast(required_extensions.len),
            .ppEnabledExtensionNames = required_extensions.ptr,
        };
        try vkCheck(c.vkCreateInstance(&create_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self) void {
        c.vkDestroyInstance(self.handle, null);
    }
};

const Surface = struct {
    const Self = @This();

    handle: c.VkSurfaceKHR = undefined,

    pub fn init(instance: Instance, window: Window) !Self {
        assert(instance.handle != null and window.handle != null);
        var self = Self{};
        try vkCheck(c.glfwCreateWindowSurface(instance.handle, window.handle, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroySurfaceKHR(vk_ctx.instance.handle, self.handle, null);
    }
};

const PhysicalDevice = struct {
    const Self = @This();

    handle: c.VkPhysicalDevice = undefined,
    q_family_idx: u32 = undefined,

    pub fn init(allocator: Allocator, instance: Instance, surface: Surface) !Self {
        assert(instance.handle != null and surface.handle != null);
        var self = Self{};
        var physical_device_count: u32 = 0;
        try vkCheck(c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, null));
        const physical_devices = try allocator.alloc(c.VkPhysicalDevice, physical_device_count);
        defer allocator.free(physical_devices);
        try vkCheck(c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, physical_devices.ptr));

        // TODO: A more robust device selection mechanism
        self.handle = physical_devices[0];

        var q_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_count, null);
        const q_family_props = try allocator.alloc(c.VkQueueFamilyProperties, q_count);
        defer allocator.free(q_family_props);
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_count, q_family_props.ptr);

        for (q_family_props, 0..) |prop, i| {
            if (prop.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                var support: c.VkBool32 = c.VK_FALSE;
                try vkCheck(c.vkGetPhysicalDeviceSurfaceSupportKHR(self.handle, @intCast(i), surface.handle, &support));
                if (support == c.VK_TRUE) {
                    self.q_family_idx = @intCast(i);
                    return self;
                }
            }
        }
        return error.NoSuitableQueueFamily;
    }

    pub fn findMemoryType(self: Self, type_filter: u32, properties: c.VkMemoryPropertyFlags) !u32 {
        var mem_props: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(self.handle, &mem_props);
        const set: u64 = 1;
        var i: u5 = 0;
        while (i < mem_props.memoryTypeCount) : (i += 1) {
            if ((type_filter & set << i) != 0 and (mem_props.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        return error.MissingMemoryType;
    }

    pub fn findSupportedFormat(
        self: Self,
        candidates: []const c.VkFormat,
        tiling: c.VkImageTiling,
        features: c.VkFormatFeatureFlags,
    ) !c.VkFormat {
        for (candidates) |format| {
            var props: c.VkFormatProperties = undefined;
            c.vkGetPhysicalDeviceFormatProperties(self.handle, format, &props);

            if (tiling == c.VK_IMAGE_TILING_LINEAR and (props.linearTilingFeatures & features) == features) {
                return format;
            } else if (tiling == c.VK_IMAGE_TILING_OPTIMAL and (props.optimalTilingFeatures & features) == features) {
                return format;
            }
        }
        return error.NoSuitableFormatFound;
    }

    // Helper for the depth format
    pub fn findDepthFormat(self: Self) !c.VkFormat {
        return findSupportedFormat(
            self,
            &.{ c.VK_FORMAT_D32_SFLOAT, c.VK_FORMAT_D32_SFLOAT_S8_UINT, c.VK_FORMAT_D24_UNORM_S8_UINT },
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT,
        );
    }
};

const Device = struct {
    const Self = @This();

    handle: c.VkDevice = undefined,
    physical: PhysicalDevice,

    pub fn init(physical_device: PhysicalDevice) !Self {
        var self = Self{ .physical = physical_device };
        const queue_priority: f32 = 1.0;
        const queue_create_info = c.VkDeviceQueueCreateInfo{
            .queueFamilyIndex = physical_device.q_family_idx,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        var device_features = c.VkPhysicalDeviceFeatures{};
        const device_extensions = [_][*:0]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        const create_info = c.VkDeviceCreateInfo{
            .pEnabledFeatures = &device_features,
            .pQueueCreateInfos = &queue_create_info,
            .queueCreateInfoCount = 1,
            .ppEnabledExtensionNames = &device_extensions,
            .enabledExtensionCount = device_extensions.len,
        };
        try vkCheck(c.vkCreateDevice(physical_device.handle, &create_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self) void {
        c.vkDestroyDevice(self.handle, null);
    }
};

const Queue = struct {
    const Self = @This();

    handle: c.VkQueue = undefined,

    pub fn init(device: Device) !Self {
        var self = Self{};
        c.vkGetDeviceQueue(device.handle, device.physical.q_family_idx, 0, &self.handle);
        return self;
    }

    pub fn deinit(_: *Self) void {} // TODO: remove

    // pub fn submit(self: *Self, ) // TODO: create
};

// TODO: Improve this abstraction to be used in the UI render logic
const CommandBuffer = struct {
    const Self = @This();
    handle: c.VkCommandBuffer = undefined,

    pub fn allocate(vk_ctx: *VulkanContext, is_primary: bool) !Self {
        var self = Self{};
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .commandPool = vk_ctx.command_pool.handle,
            .level = if (is_primary) c.VK_COMMAND_BUFFER_LEVEL_PRIMARY else c.VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            .commandBufferCount = 1,
        };
        try vkCheck(c.vkAllocateCommandBuffers(vk_ctx.device.handle, &alloc_info, &self.handle));
        return self;
    }

    pub fn begin(self: *Self, flags: c.VkCommandBufferUsageFlags) !void {
        const begin_info = c.VkCommandBufferBeginInfo{
            .flags = flags,
        };
        try vkCheck(c.vkBeginCommandBuffer(self.handle, &begin_info));
    }

    pub fn end(self: *Self) !void {
        try vkCheck(c.vkEndCommandBuffer(self.handle));
    }

    fn beginSingleTimeCommands(vk_ctx: *VulkanContext, is_primary: bool) !Self {
        var self = Self{};

        const alloc_info = c.VkCommandBufferAllocateInfo{
            .level = if (is_primary) c.VK_COMMAND_BUFFER_LEVEL_PRIMARY else c.VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            .commandPool = vk_ctx.command_pool.handle,
            .commandBufferCount = 1,
        };
        try vkCheck(c.vkAllocateCommandBuffers(vk_ctx.device.handle, &alloc_info, &self.handle));

        const begin_info = c.VkCommandBufferBeginInfo{ .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        try vkCheck(c.vkBeginCommandBuffer(self.handle, &begin_info));

        return self;
    }

    fn endSingleTimeCommands(self: Self, vk_ctx: *VulkanContext) !void {
        try vkCheck(c.vkEndCommandBuffer(self.handle));

        const submit_info = c.VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &self.handle,
        };
        try vkCheck(c.vkQueueSubmit(vk_ctx.graphics_queue.handle, 1, &submit_info, null));
        try vkCheck(c.vkQueueWaitIdle(vk_ctx.graphics_queue.handle));

        c.vkFreeCommandBuffers(vk_ctx.device.handle, vk_ctx.command_pool.handle, 1, &self.handle);
    }
};

const CommandPool = struct {
    const Self = @This();

    handle: c.VkCommandPool = undefined,

    pub fn init(device: Device) !Self {
        var self = Self{};
        var cmd_pool_info = c.VkCommandPoolCreateInfo{
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device.physical.q_family_idx,
        };
        try vkCheck(c.vkCreateCommandPool(device.handle, &cmd_pool_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyCommandPool(vk_ctx.device.handle, self.handle, null);
    }
};

pub const Swapchain = struct {
    const Self = @This();
    handle: c.VkSwapchainKHR = undefined,
    image_format: c.VkSurfaceFormatKHR = undefined,
    extent: c.VkExtent2D = undefined,
    images: []c.VkImage = undefined,
    image_views: []c.VkImageView = undefined,
    owner: c.VkDevice,

    pub fn init(vk_ctx: *VulkanContext) !Self {
        var self = Self{ .owner = vk_ctx.device.handle };

        // TODO: Abstract this under `PhysicalDevice`
        var fmt_count: u32 = 0;
        try vkCheck(c.vkGetPhysicalDeviceSurfaceFormatsKHR(vk_ctx.physical_device.handle, vk_ctx.surface.handle, &fmt_count, null));
        const fmts = try vk_ctx.allocator.alloc(c.VkSurfaceFormatKHR, fmt_count);
        defer vk_ctx.allocator.free(fmts);
        try vkCheck(c.vkGetPhysicalDeviceSurfaceFormatsKHR(vk_ctx.physical_device.handle, vk_ctx.surface.handle, &fmt_count, fmts.ptr));
        self.image_format = fmts[0];

        var present_modes_count: u32 = undefined;
        try vkCheck(c.vkGetPhysicalDeviceSurfacePresentModesKHR(vk_ctx.physical_device.handle, vk_ctx.surface.handle, &present_modes_count, null));
        const present_modes = try vk_ctx.allocator.alloc(c.VkPresentModeKHR, present_modes_count);
        defer vk_ctx.allocator.free(present_modes);
        try vkCheck(c.vkGetPhysicalDeviceSurfacePresentModesKHR(vk_ctx.physical_device.handle, vk_ctx.surface.handle, &present_modes_count, present_modes.ptr));
        var present_mode: c_uint = c.VK_PRESENT_MODE_FIFO_KHR;
        for (present_modes) |mode| {
            if (mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
                present_mode = c.VK_PRESENT_MODE_MAILBOX_KHR;
                break;
            }
        }

        var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
        try vkCheck(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vk_ctx.physical_device.handle, vk_ctx.surface.handle, &capabilities));
        self.extent = capabilities.currentExtent;

        var image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 and image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }

        const create_info = c.VkSwapchainCreateInfoKHR{
            .surface = vk_ctx.surface.handle,
            .minImageCount = image_count,
            .imageFormat = self.image_format.format,
            .imageColorSpace = self.image_format.colorSpace,
            .imageExtent = capabilities.currentExtent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = present_mode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
        };
        try vkCheck(c.vkCreateSwapchainKHR(vk_ctx.device.handle, &create_info, null, &self.handle));

        // TODO: Refactor rest of this function
        var img_count: u32 = undefined;
        try vkCheck(c.vkGetSwapchainImagesKHR(vk_ctx.device.handle, self.handle, &img_count, null));
        self.images = try vk_ctx.allocator.alloc(c.VkImage, img_count);
        try vkCheck(c.vkGetSwapchainImagesKHR(vk_ctx.device.handle, self.handle, &img_count, self.images.ptr));

        self.image_views = try vk_ctx.allocator.alloc(c.VkImageView, image_count);
        for (self.images, 0..) |image, i| {
            const info = c.VkImageViewCreateInfo{
                .image = image,
                .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
                .format = self.image_format.format,
                .components = .{},
                .subresourceRange = .{
                    .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            try vkCheck(c.vkCreateImageView(vk_ctx.device.handle, &info, null, &self.image_views[i]));
        }

        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        vk_ctx.allocator.free(self.images);
        vk_ctx.allocator.free(self.image_views);
        c.vkDestroySwapchainKHR(self.owner, self.handle, null);
    }
};

pub const Image = struct {
    handle: c.VkImage,
    memory: c.VkDeviceMemory,
    width: u32,
    height: u32,
    format: c.VkFormat,

    /// Creates a VkImage and allocates its memory.
    pub fn create(
        vk_ctx: *VulkanContext,
        width: u32,
        height: u32,
        format: c.VkFormat,
        tiling: c.VkImageTiling,
        usage: c.VkImageUsageFlags,
        properties: c.VkMemoryPropertyFlags,
    ) !Image {
        const image_info = c.VkImageCreateInfo{
            .imageType = c.VK_IMAGE_TYPE_2D,
            .extent = .{ .width = width, .height = height, .depth = 1 },
            .mipLevels = 1,
            .arrayLayers = 1,
            .format = format,
            .tiling = tiling,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .usage = usage,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };

        // TODO: Refactor this under `Device` abstraction
        var image_handle: c.VkImage = undefined;
        try vkCheck(c.vkCreateImage(vk_ctx.device.handle, &image_info, null, &image_handle));

        var mem_reqs: c.VkMemoryRequirements = undefined;
        c.vkGetImageMemoryRequirements(vk_ctx.device.handle, image_handle, &mem_reqs);

        const mem_type_index = try vk_ctx.physical_device.findMemoryType(mem_reqs.memoryTypeBits, properties);

        const alloc_info = c.VkMemoryAllocateInfo{
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type_index,
        };

        var image_memory: c.VkDeviceMemory = undefined;
        try vkCheck(c.vkAllocateMemory(vk_ctx.device.handle, &alloc_info, null, &image_memory));

        try vkCheck(c.vkBindImageMemory(vk_ctx.device.handle, image_handle, image_memory, 0));

        return Image{
            .handle = image_handle,
            .memory = image_memory,
            .width = width,
            .height = height,
            .format = format,
        };
    }

    pub fn deinit(self: *Image, vk_ctx: *VulkanContext) void {
        c.vkDestroyImage(vk_ctx.device.handle, self.handle, null);
        c.vkFreeMemory(vk_ctx.device.handle, self.memory, null);
    }

    /// Creates a VkImageView for this image.
    pub fn createView(
        self: Image,
        vk_ctx: *VulkanContext,
        aspect_flags: c.VkImageAspectFlags,
    ) !c.VkImageView {
        const view_info = c.VkImageViewCreateInfo{
            .image = self.handle,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = self.format,
            .subresourceRange = .{
                .aspectMask = aspect_flags,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        var view: c.VkImageView = undefined;
        try vkCheck(c.vkCreateImageView(vk_ctx.device.handle, &view_info, null, &view));
        return view;
    }

    /// Transitions the image from one layout to another using a barrier.
    /// TODO: Refactor to use vocabullary types and methods
    pub fn transitionLayout(
        self: Image,
        vk_ctx: *VulkanContext,
        old_layout: c.VkImageLayout,
        new_layout: c.VkImageLayout,
    ) !void {
        const command_buffer = try CommandBuffer.beginSingleTimeCommands(vk_ctx, true);

        var barrier = c.VkImageMemoryBarrier{
            .oldLayout = old_layout,
            .newLayout = new_layout,
            .srcQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .dstQueueFamilyIndex = c.VK_QUEUE_FAMILY_IGNORED,
            .image = self.handle,
            .subresourceRange = .{
                .aspectMask = if (self.hasDepthComponent()) c.VK_IMAGE_ASPECT_DEPTH_BIT else c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };

        var source_stage: c.VkPipelineStageFlags = undefined;
        var destination_stage: c.VkPipelineStageFlags = undefined;
        source_stage = c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;

        // Determine pipeline stages and access masks based on layout transition
        // TODO: if-elses maybe unnecessary
        if (old_layout == c.VK_IMAGE_LAYOUT_UNDEFINED and new_layout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;

            source_stage = c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
        } else if (old_layout == c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL and new_layout == c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
            barrier.srcAccessMask = c.VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = c.VK_ACCESS_SHADER_READ_BIT;

            source_stage = c.VK_PIPELINE_STAGE_TRANSFER_BIT;
            destination_stage = c.VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
        } else if (old_layout == c.VK_IMAGE_LAYOUT_UNDEFINED and new_layout == c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL) {
            barrier.srcAccessMask = 0;
            barrier.dstAccessMask = c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | c.VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

            source_stage = c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
            destination_stage = c.VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        } else {
            return error.UnsupportedLayoutTransition;
        }

        c.vkCmdPipelineBarrier(command_buffer.handle, source_stage, destination_stage, 0, 0, null, 0, null, 1, &barrier);

        try command_buffer.endSingleTimeCommands(vk_ctx);
    }

    /// Copies data from a buffer to this image.
    pub fn copyFromBuffer(self: Image, vk_ctx: *VulkanContext, buffer: Buffer) !void {
        const command_buffer = try CommandBuffer.beginSingleTimeCommands(vk_ctx, true);

        const region = c.VkBufferImageCopy{
            .bufferOffset = 0,
            .bufferRowLength = 0,
            .bufferImageHeight = 0,
            .imageSubresource = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .mipLevel = 0,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
            .imageOffset = .{ .x = 0, .y = 0, .z = 0 },
            .imageExtent = .{ .width = self.width, .height = self.height, .depth = 1 },
        };

        c.vkCmdCopyBufferToImage(command_buffer.handle, buffer.handle, self.handle, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

        try command_buffer.endSingleTimeCommands(vk_ctx);
    }

    pub fn hasDepthComponent(self: Image) bool {
        return self.format == c.VK_FORMAT_D32_SFLOAT or self.format == c.VK_FORMAT_D32_SFLOAT_S8_UINT or self.format == c.VK_FORMAT_D24_UNORM_S8_UINT;
    }
};

const DepthBuffer = struct {
    const Self = @This();

    image: Image, // You'll need an Image struct that holds VkImage and VkDeviceMemory
    view: c.VkImageView,
    format: c.VkFormat,

    pub fn init(vk_ctx: *VulkanContext, width: u32, height: u32) !Self {
        const format = try vk_ctx.physical_device.findDepthFormat();

        // This is a simplified version of image creation.
        // A full implementation would be a function `createImage(...)`.
        const image = try Image.create(
            vk_ctx,
            width,
            height,
            format,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );

        const view = try image.createView(vk_ctx, c.VK_IMAGE_ASPECT_DEPTH_BIT);

        // Important: Transition the image layout so it's ready for use as a depth attachment
        try image.transitionLayout(vk_ctx, c.VK_IMAGE_LAYOUT_UNDEFINED, c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        return .{ .image = image, .view = view, .format = format };
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyImageView(vk_ctx.device.handle, self.view, null);
        self.image.deinit(vk_ctx);
    }
};

pub const RenderPass = struct {
    const Self = @This();
    handle: c.VkRenderPass = undefined,
    framebuffer: []c.VkFramebuffer = undefined,
    owner: c.VkDevice,

    pub fn init(vk_ctx: *VulkanContext, swapchain: *Swapchain) !Self {
        var self = Self{ .owner = vk_ctx.device.handle };

        // Essencial for framebuffer color baking
        const color_attachment = c.VkAttachmentDescription{
            .format = swapchain.image_format.format,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
            .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        };
        const color_attachment_ref = c.VkAttachmentReference{
            .attachment = 0,
            .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        };

        var subpass_dep = c.VkSubpassDependency{
            .srcSubpass = c.VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        // Depth setup routine avoids far clipping

        const depth_format = try vk_ctx.physical_device.findDepthFormat();
        const depth_attachment = c.VkAttachmentDescription{
            .format = depth_format,
            .samples = c.VK_SAMPLE_COUNT_1_BIT,
            .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR, // Clear depth at the start of the pass
            .storeOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE, // We don't need to save depth after
            .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
            .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
            .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
            .finalLayout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        // Create an attachment reference for the subpass
        const depth_attachment_ref = c.VkAttachmentReference{
            .attachment = 1, // Color is 0, Depth is 1
            .layout = c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        };

        const subpass = c.VkSubpassDescription{
            .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_ref,
            .pDepthStencilAttachment = &depth_attachment_ref, // Set this!
        };

        // RenderPass create info
        const attachments = [_]c.VkAttachmentDescription{ color_attachment, depth_attachment };
        const create_info = c.VkRenderPassCreateInfo{
            .attachmentCount = attachments.len,
            .pAttachments = &attachments,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &subpass_dep,
        };

        try vkCheck(c.vkCreateRenderPass(vk_ctx.device.handle, &create_info, null, &self.handle));

        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        self.deinitFramebuffer(vk_ctx.allocator);
        c.vkDestroyRenderPass(self.owner, self.handle, null);
    }

    pub fn initFrameBuffer(self: *Self, vk_ctx: *VulkanContext, swapchain: *Swapchain, depth_view: c.VkImageView) !void {
        self.framebuffer = try vk_ctx.allocator.alloc(c.VkFramebuffer, swapchain.image_views.len);
        for (swapchain.image_views, 0..) |iv, i| {
            const attachments = [_]c.VkImageView{ iv, depth_view }; // Color and depth
            const f_buffer_create_info = c.VkFramebufferCreateInfo{
                .renderPass = self.handle,
                .attachmentCount = attachments.len, // 2
                .pAttachments = &attachments,
                .width = swapchain.extent.width,
                .height = swapchain.extent.height,
                .layers = 1,
            };
            try vkCheck(c.vkCreateFramebuffer(self.owner, &f_buffer_create_info, null, &self.framebuffer[i]));
        }
    }

    pub fn deinitFramebuffer(self: *Self, allocator: Allocator) void {
        for (self.framebuffer) |fb| {
            c.vkDestroyFramebuffer(self.owner, fb, null);
        }
        allocator.free(self.framebuffer);
    }
};

const SyncObjects = struct {
    const Self = @This();
    img_available_semaphore: Semaphore,
    render_ended_semaphore: Semaphore,
    in_flight_fence: Fence,

    pub fn init(vk_ctx: *VulkanContext) !Self {
        return .{
            .img_available_semaphore = try Semaphore.init(vk_ctx),
            .render_ended_semaphore = try Semaphore.init(vk_ctx),
            .in_flight_fence = try Fence.init(vk_ctx),
        };
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        self.img_available_semaphore.deinit(vk_ctx);
        self.render_ended_semaphore.deinit(vk_ctx);
        self.in_flight_fence.deinit(vk_ctx);
    }
};

const Semaphore = struct {
    const Self = @This();
    handle: c.VkSemaphore = undefined,

    pub fn init(vk_ctx: *VulkanContext) !Self {
        var self = Self{};
        try vkCheck(c.vkCreateSemaphore(vk_ctx.device.handle, &.{}, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroySemaphore(vk_ctx.device.handle, self.handle, null);
    }
};

const Fence = struct {
    const Self = @This();
    handle: c.VkFence = undefined,

    pub fn init(vk_ctx: *VulkanContext) !Self {
        var self = Self{};
        const fence_create_info = c.VkFenceCreateInfo{ .flags = c.VK_FENCE_CREATE_SIGNALED_BIT };
        try vkCheck(c.vkCreateFence(vk_ctx.device.handle, &fence_create_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyFence(vk_ctx.device.handle, self.handle, null);
    }
};

pub const Buffer = struct {
    const Self = @This();

    handle: c.VkBuffer = undefined,
    memory: c.VkDeviceMemory = undefined,
    size: u64,

    pub fn init(vk_ctx: *VulkanContext, size: u64, usage: c.VkBufferUsageFlags, properties: c.VkMemoryPropertyFlags) !Self {
        var self = Self{ .size = size };
        var buffer_info = c.VkBufferCreateInfo{
            .size = size,
            .usage = usage,
            // Not shared between queues
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };
        try vkCheck(c.vkCreateBuffer(vk_ctx.device.handle, &buffer_info, null, &self.handle));

        var mem_reqs: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(vk_ctx.device.handle, self.handle, &mem_reqs);
        const mem_type_index = try vk_ctx.physical_device.findMemoryType(mem_reqs.memoryTypeBits, properties);
        var alloc_info = c.VkMemoryAllocateInfo{
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type_index,
        };
        try vkCheck(c.vkAllocateMemory(vk_ctx.device.handle, &alloc_info, null, &self.memory));
        try vkCheck(c.vkBindBufferMemory(vk_ctx.device.handle, self.handle, self.memory, 0));
        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyBuffer(vk_ctx.device.handle, self.handle, null);
        c.vkFreeMemory(vk_ctx.device.handle, self.memory, null);
    }

    pub fn map(self: *Self, vk_ctx: *VulkanContext, T: type) ![*]T {
        var data_ptr: ?*anyopaque = undefined;
        try vkCheck(c.vkMapMemory(vk_ctx.device.handle, self.memory, 0, self.size, 0, &data_ptr));
        return @ptrCast(@alignCast(data_ptr.?));
    }

    pub fn unmap(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkUnmapMemory(vk_ctx.device.handle, self.memory);
    }

    ///TODO: Use CommandBuffer vocabullary type methods
    pub fn copyTo(self: *Self, vk_ctx: *VulkanContext, dst_buffer: Buffer) !void {
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandPool = vk_ctx.command_pool.handle,
            .commandBufferCount = 1,
        };
        var command_buffer: c.VkCommandBuffer = undefined;
        try vkCheck(c.vkAllocateCommandBuffers(vk_ctx.device.handle, &alloc_info, &command_buffer));
        defer c.vkFreeCommandBuffers(vk_ctx.device.handle, vk_ctx.command_pool.handle, 1, &command_buffer);

        const begin_info = c.VkCommandBufferBeginInfo{ .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT };
        try vkCheck(c.vkBeginCommandBuffer(command_buffer, &begin_info));

        const copy_region = c.VkBufferCopy{ .size = self.size };
        c.vkCmdCopyBuffer(command_buffer, self.handle, dst_buffer.handle, 1, &copy_region);

        try vkCheck(c.vkEndCommandBuffer(command_buffer));

        const submit_info = c.VkSubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
        };

        try vkCheck(c.vkQueueSubmit(vk_ctx.graphics_queue.handle, 1, &submit_info, null));
        try vkCheck(c.vkQueueWaitIdle(vk_ctx.graphics_queue.handle));
    }
};

// TODO: Pass in PushConstantType
pub const PushConstantRange = struct {
    const Self = @This();

    handle: c.VkPushConstantRange,

    pub fn init(T: type) Self {
        return .{ .handle = .{
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT,
            .offset = 0,
            .size = @sizeOf(T),
        } };
    }
};

const DescriptorType = enum(u8) {
    Uniform,
    CombinedImageSampler,
    pub fn getVkType(@"type": DescriptorType) c_uint {
        return switch (@"type") {
            .Uniform => c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .CombinedImageSampler => c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
        };
    }
};

pub const DescriptorSetLayout = struct {
    const Self = @This();
    handle: c.VkDescriptorSetLayout = undefined,

    pub fn init(vk_ctx: *VulkanContext, @"type": DescriptorType) !Self {
        var self = Self{};
        var ubo_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = @"type".getVkType(),
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT | c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        var layout_info = c.VkDescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &ubo_layout_binding };
        try vkCheck(c.vkCreateDescriptorSetLayout(vk_ctx.device.handle, &layout_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyDescriptorSetLayout(vk_ctx.device.handle, self.handle, null);
    }
};

pub const DescriptorPool = struct {
    const Self = @This();
    handle: c.VkDescriptorPool = undefined,

    pub fn init(vk_ctx: *VulkanContext, @"type": DescriptorType) !Self {
        var self = Self{};
        var pool_size = c.VkDescriptorPoolSize{ .type = @"type".getVkType(), .descriptorCount = 1 };
        var pool_info = c.VkDescriptorPoolCreateInfo{
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
            .maxSets = 1,
        };
        try vkCheck(c.vkCreateDescriptorPool(vk_ctx.device.handle, &pool_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyDescriptorPool(vk_ctx.device.handle, self.handle, null);
    }

    pub fn allocateSet(self: Self, vk_ctx: *VulkanContext, layout: DescriptorSetLayout) !c.VkDescriptorSet {
        var set_alloc_info = c.VkDescriptorSetAllocateInfo{
            .descriptorPool = self.handle,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout.handle,
        };
        var set: c.VkDescriptorSet = undefined;
        try vkCheck(c.vkAllocateDescriptorSets(vk_ctx.device.handle, &set_alloc_info, &set));
        return set;
    }

    pub fn updateSet(_: Self, vk_ctx: *VulkanContext, set: c.VkDescriptorSet, buffer: Buffer, ubo_type: type) void {
        var desc_buffer_info = c.VkDescriptorBufferInfo{ .buffer = buffer.handle, .offset = 0, .range = @sizeOf(ubo_type) };
        var desc_write = c.VkWriteDescriptorSet{
            .dstSet = set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = DescriptorType.Uniform.getVkType(),
            .descriptorCount = 1,
            .pBufferInfo = &desc_buffer_info,
        };
        c.vkUpdateDescriptorSets(vk_ctx.device.handle, 1, &desc_write, 0, null);
    }
};

pub const ShaderModule = struct {
    const Self = @This();
    handle: c.VkShaderModule = undefined,
    owner: c.VkDevice, // TODO: REMOVE

    pub fn init(allocator: Allocator, device_handle: c.VkDevice, code: []const u8) !Self {
        var self = Self{ .owner = device_handle };
        const aligned_code = try allocator.alignedAlloc(u32, @alignOf(u32), code.len / @sizeOf(u32));
        defer allocator.free(aligned_code);
        @memcpy(std.mem.sliceAsBytes(aligned_code), code);
        var create_info = c.VkShaderModuleCreateInfo{ .codeSize = code.len, .pCode = aligned_code.ptr };
        try vkCheck(c.vkCreateShaderModule(device_handle, &create_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self) void {
        c.vkDestroyShaderModule(self.owner, self.handle, null);
    }
};

pub const PipelineLayout = struct {
    const Self = @This();

    handle: c.VkPipelineLayout = undefined,

    pub fn init(
        vk_ctx: *VulkanContext,
        create_info: c.VkPipelineLayoutCreateInfo,
    ) !Self {
        var self = Self{};
        try vkCheck(c.vkCreatePipelineLayout(vk_ctx.device.handle, &create_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyPipelineLayout(vk_ctx.device.handle, self.handle, null);
    }
};

pub const Pipeline = struct {
    const Self = @This();

    handle: c.VkPipeline = undefined,

    pub fn init(vk_ctx: *VulkanContext, render_pass: RenderPass, layout: PipelineLayout, swapchain: Swapchain) !Self {
        var self = Self{};

        var vert_shader_module = try ShaderModule.init(vk_ctx.allocator, vk_ctx.device.handle, vert_shader_bin);
        defer vert_shader_module.deinit();
        var frag_shader_module = try ShaderModule.init(vk_ctx.allocator, vk_ctx.device.handle, frag_shader_bin);
        defer frag_shader_module.deinit();

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
            .{ .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vert_shader_module.handle, .pName = "main" },
            .{ .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_shader_module.handle, .pName = "main" },
        };
        const binding_description = getVertexBindingDescription();
        const attribute_descriptions = getVertexAttributeDescriptions();
        const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_description,
            .vertexAttributeDescriptionCount = attribute_descriptions.len,
            .pVertexAttributeDescriptions = &attribute_descriptions,
        };
        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
            .topology = c.VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            .primitiveRestartEnable = c.VK_FALSE,
        };
        const viewport = c.VkViewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(swapchain.extent.width),
            .height = @floatFromInt(swapchain.extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };
        const scissor = c.VkRect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = swapchain.extent,
        };
        var viewport_state = c.VkPipelineViewportStateCreateInfo{
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };
        var rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_BACK_BIT,
            .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = c.VK_FALSE,
        };
        var multisampling = c.VkPipelineMultisampleStateCreateInfo{
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
        };
        var color_blend_attachment = c.VkPipelineColorBlendAttachmentState{
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = c.VK_FALSE,
        };
        var color_blending = c.VkPipelineColorBlendStateCreateInfo{
            .logicOpEnable = c.VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
        };
        var depth_stencil = c.VkPipelineDepthStencilStateCreateInfo{
            .depthTestEnable = c.VK_TRUE,
            .depthWriteEnable = c.VK_TRUE,
            // GREATER or GREATER_OR_EQUAL
            .depthCompareOp = c.VK_COMPARE_OP_GREATER_OR_EQUAL,
            .depthBoundsTestEnable = c.VK_FALSE,
            .stencilTestEnable = c.VK_FALSE,
        };

        var pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &color_blending,
            .pDepthStencilState = &depth_stencil,
            .layout = layout.handle,
            .renderPass = render_pass.handle,
            .subpass = 0,
        };

        try vkCheck(c.vkCreateGraphicsPipelines(vk_ctx.device.handle, null, 1, &pipeline_info, null, &self.handle));
        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyPipeline(vk_ctx.device.handle, self.handle, null);
    }
};

// --- VULKAN CONTEXT ---
pub const VulkanContext = struct {
    const Self = @This();
    allocator: Allocator,
    instance: Instance,
    surface: Surface,
    physical_device: PhysicalDevice,
    device: Device,
    graphics_queue: Queue,
    command_pool: CommandPool,

    pub fn init(allocator: Allocator, window: Window) !Self {
        const instance = try Instance.init();
        const surface = try Surface.init(instance, window);
        const physical_device = try PhysicalDevice.init(allocator, instance, surface);
        const device = try Device.init(physical_device);
        const graphics_queue = try Queue.init(device);
        const command_pool = try CommandPool.init(device);

        return .{
            .allocator = allocator,
            .instance = instance,
            .surface = surface,
            .physical_device = physical_device,
            .device = device,
            .graphics_queue = graphics_queue,
            .command_pool = command_pool,
        };
    }

    pub fn deinit(self: *Self) void {
        // Destroy in reverse order of creation
        self.command_pool.deinit(self);
        self.graphics_queue.deinit();
        self.device.deinit();
        self.surface.deinit(self);
        self.instance.deinit();
    }
};

// --- MAIN APPLICATION STRUCT ---
pub const App = struct {
    const Self = @This();

    allocator: Allocator,
    scene: Scene,
    window: Window,
    vk_ctx: *VulkanContext,
    gui_renderer: gui.GuiRenderer,
    text_renderer: text3d.Text3DRenderer,
    main_ui: gui.UI,
    wd_ctx: WindowContext,

    // Vulkan objects that depend on the swapchain (and are recreated)
    depth_buffer: DepthBuffer,
    swapchain: Swapchain,
    render_pass: RenderPass,
    descriptor_layout: DescriptorSetLayout,
    pipeline_layout: PipelineLayout,
    pipeline: Pipeline,

    // Other Vulkan objects
    vertex_buffer: Buffer,
    uniform_buffer: Buffer,
    descriptor_pool: DescriptorPool,
    descriptor_set: c.VkDescriptorSet,
    command_buffer: CommandBuffer,
    sync: SyncObjects,

    framebuffer_resized: bool = false,
    perf: fps_tracker.PerformanceTracker,

    /// Caller owns memory
    pub fn init(allocator: Allocator) !*Self {
        const window = try Window.init(null, WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan Line App", null, null);
        const vk_ctx = try allocator.create(VulkanContext);
        vk_ctx.* = try VulkanContext.init(allocator, window);

        const app = try allocator.create(App);
        app.allocator = allocator;
        app.window = window;
        app.vk_ctx = vk_ctx;
        app.scene = try Scene.init(allocator, 20);
        app.wd_ctx = .{};
        app.main_ui = gui.UI.init(allocator);
        app.perf = fps_tracker.PerformanceTracker.init();
        try app.initUi();

        try app.initVulkanResources();

        c.glfwSetWindowUserPointer(app.window.handle, app);
        return app;
    }

    // Initialize Vulkan resources after the context is created
    fn initVulkanResources(self: *Self) !void {
        self.swapchain = try Swapchain.init(self.vk_ctx);
        self.depth_buffer = try DepthBuffer.init(self.vk_ctx, self.swapchain.extent.width, self.swapchain.extent.height);
        self.descriptor_layout = try DescriptorSetLayout.init(self.vk_ctx, .Uniform);
        self.descriptor_pool = try DescriptorPool.init(self.vk_ctx, .Uniform);
        self.descriptor_set = try self.descriptor_pool.allocateSet(self.vk_ctx, self.descriptor_layout);
        self.command_buffer = try CommandBuffer.allocate(self.vk_ctx, true);
        self.sync = try SyncObjects.init(self.vk_ctx);

        try self.initVertexBuffer();
        try self.initUniformBuffer();

        self.descriptor_pool.updateSet(self.vk_ctx, self.descriptor_set, self.uniform_buffer, UniformBufferObject);

        self.render_pass = try RenderPass.init(self.vk_ctx, &self.swapchain);
        try self.render_pass.initFrameBuffer(self.vk_ctx, &self.swapchain, self.depth_buffer.view);
        self.pipeline_layout = try PipelineLayout.init(self.vk_ctx, .{
            .pSetLayouts = &[_]c.VkDescriptorSetLayout{self.descriptor_layout.handle},
            .setLayoutCount = 1,
        });
        self.pipeline = try Pipeline.init(self.vk_ctx, self.render_pass, self.pipeline_layout, self.swapchain);

        // Initialize GUI at the end
        self.gui_renderer = try gui.GuiRenderer.init(self.vk_ctx, self.render_pass, self.swapchain);
        self.text_renderer = try text3d.Text3DRenderer.init(self.vk_ctx, self.render_pass, self.descriptor_layout, self.swapchain);
    }

    ///
    pub fn initUi(self: *Self) !void {
        try self.main_ui.addButton(.{
            .x = 10,
            .y = 10,
            .width = 150,
            .height = 30,
            .on_click = addLineCallback,
            .data = .{
                .button = .{
                    .text = "Add Line",
                },
            },
        });

        try self.main_ui.addButton(.{
            .x = 10,
            .y = 50,
            .width = 150,
            .height = 30,
            .on_click = clearLinesCallback,
            .data = .{
                .button = .{
                    .text = "Clear Lines",
                },
            },
        });

        try self.main_ui.addButton(.{
            .x = 10,
            .y = 550,
            .width = 150,
            .height = 30,
            .on_click = quitCallback,
            .data = .{
                .button = .{
                    .text = "Quit",
                },
            },
        });
    }

    pub fn deinit(self: *Self) void {
        // Wait for device to be idle before cleaning up
        _ = c.vkDeviceWaitIdle(self.vk_ctx.device.handle);

        self.main_ui.deinit();
        self.gui_renderer.deinit();
        self.text_renderer.deinit();
        self.cleanupSwapchain();

        self.sync.deinit(self.vk_ctx);
        self.vertex_buffer.deinit(self.vk_ctx);
        self.uniform_buffer.deinit(self.vk_ctx);
        self.descriptor_layout.deinit(self.vk_ctx);
        self.descriptor_pool.deinit(self.vk_ctx);

        self.vk_ctx.deinit();
        self.scene.deinit(self.allocator);
        self.window.deinit();
        self.allocator.destroy(self.vk_ctx);
    }

    fn cleanupSwapchain(self: *Self) void {
        self.gui_renderer.destroyPipeline();
        self.pipeline.deinit(self.vk_ctx);
        self.pipeline_layout.deinit(self.vk_ctx);
        self.render_pass.deinit(self.vk_ctx);
        self.depth_buffer.deinit(self.vk_ctx);
        self.swapchain.deinit(self.vk_ctx);
    }

    pub fn run(self: *Self) !void {
        while (c.glfwWindowShouldClose(self.window.handle) == 0) {
            self.perf.beginFrame();
            self.gui_renderer.beginFrame();
            self.text_renderer.beginFrame();
            self.wd_ctx.beginFrame();

            c.glfwPollEvents();

            if (self.wd_ctx.processCameraInput(&self.scene)) {
                try self.updateUniformBuffer();
            }

            if (self.window.minimized()) {
                c.glfwWaitEvents();
                continue;
            }

            const transform1 = scene.Transform.new(.{ .position = .{ -5, 5, 0 } }).toMatrix();
            self.text_renderer.drawText("Hello 3D World!", transform1, .{ 1.0, 0.8, 0.2, 1.0 }, 1.0);

            // Example 2: Text rotating around the Y axis

            const time = @as(f32, @floatFromInt(@mod(std.time.milliTimestamp(), 10000))) / 1000.0;
            const transform2 = scene.Transform.new(.{
                .position = .{ 0, 2, -5 },
                .rotation = scene.Quat.fromAxisAngle(.{ 0, 1, 0 }, time * 0.5),
            }).toMatrix();
            self.text_renderer.drawText("Rotating Text", transform2, .{ 0.2, 1.0, 0.8, 1.0 }, 4.0);

            self.gui_renderer.processAndDrawUi(self, &self.main_ui);

            self.perf.endFrame();
            const fps = std.fmt.allocPrint(self.vk_ctx.allocator, "fps: {d:1}", .{self.perf.avg_fps}) catch unreachable;
            defer self.vk_ctx.allocator.free(fps);

            self.gui_renderer.drawText(
                fps,
                10.0,
                @as(f32, @floatFromInt(self.window.size.y)) - 80.0,
                .{ 1.0, 0.0, 0.5, 1.0 },
                0.8,
            );

            try self.draw();
        }
    }

    fn draw(self: *Self) !void {
        try vkCheck(c.vkWaitForFences(self.vk_ctx.device.handle, 1, &self.sync.in_flight_fence.handle, c.VK_TRUE, std.math.maxInt(u64)));

        var image_index: u32 = 0;
        const acquire_result = c.vkAcquireNextImageKHR(self.vk_ctx.device.handle, self.swapchain.handle, std.math.maxInt(u64), self.sync.img_available_semaphore.handle, null, &image_index);

        if (acquire_result == c.VK_ERROR_OUT_OF_DATE_KHR) {
            try self.recreateSwapchain();
            return;
        } else if (acquire_result != c.VK_SUCCESS and acquire_result != c.VK_SUBOPTIMAL_KHR) {
            try vkCheck(acquire_result);
        }

        try vkCheck(c.vkResetFences(self.vk_ctx.device.handle, 1, &self.sync.in_flight_fence.handle));
        try vkCheck(c.vkResetCommandBuffer(self.command_buffer.handle, 0));
        try self.recordCommandBuffer(image_index);

        const wait_semaphores = [_]c.VkSemaphore{self.sync.img_available_semaphore.handle};
        const wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const signal_semaphores = [_]c.VkSemaphore{self.sync.render_ended_semaphore.handle};
        const submit_info = c.VkSubmitInfo{
            .waitSemaphoreCount = wait_semaphores.len,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffer.handle,
            .signalSemaphoreCount = signal_semaphores.len,
            .pSignalSemaphores = &signal_semaphores,
        };
        try vkCheck(c.vkQueueSubmit(self.vk_ctx.graphics_queue.handle, 1, &submit_info, self.sync.in_flight_fence.handle));

        const swapchains = [_]c.VkSwapchainKHR{self.swapchain.handle};
        const present_info = c.VkPresentInfoKHR{
            .waitSemaphoreCount = signal_semaphores.len,
            .pWaitSemaphores = &signal_semaphores,
            .swapchainCount = swapchains.len,
            .pSwapchains = &swapchains,
            .pImageIndices = &image_index,
        };
        const present_result = c.vkQueuePresentKHR(self.vk_ctx.graphics_queue.handle, &present_info);

        if (present_result == c.VK_ERROR_OUT_OF_DATE_KHR or present_result == c.VK_SUBOPTIMAL_KHR or self.framebuffer_resized) {
            self.framebuffer_resized = false;
            try self.recreateSwapchain();
        } else {
            try vkCheck(present_result);
        }
    }

    fn initVertexBuffer(self: *Self) !void {
        const buffer_size = @sizeOf(Vertex) * self.scene.getTotalVertexCount();
        if (buffer_size == 0) return;

        var staging_buffer = try Buffer.init(self.vk_ctx, buffer_size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, Vertex);
        defer staging_buffer.unmap(self.vk_ctx);

        const mapped_slice = data_ptr[0..self.scene.getTotalVertexCount()];
        const lines_offset = self.scene.axis.len + self.scene.grid.len;

        @memcpy(mapped_slice[0..self.scene.axis.len], &self.scene.axis);
        @memcpy(mapped_slice[self.scene.axis.len..lines_offset], self.scene.grid);
        @memcpy(mapped_slice[lines_offset .. lines_offset + self.scene.lines.items.len], self.scene.lines.items);
        self.vertex_buffer = try Buffer.init(self.vk_ctx, @sizeOf(Vertex) * 1024 * 1024, c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        try staging_buffer.copyTo(self.vk_ctx, self.vertex_buffer);
    }

    fn updateVertexBuffer(self: *Self) !void {
        const buffer_size = @sizeOf(Vertex) * self.scene.getTotalVertexCount();
        if (buffer_size == 0) return;

        var staging_buffer = try Buffer.init(self.vk_ctx, buffer_size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, Vertex);
        defer staging_buffer.unmap(self.vk_ctx);

        const mapped_slice = data_ptr[0 .. self.scene.axis.len + self.scene.grid.len + self.scene.lines.items.len];
        const lines_offset = self.scene.axis.len + self.scene.grid.len;
        @memcpy(mapped_slice[0..self.scene.axis.len], &self.scene.axis);
        @memcpy(mapped_slice[self.scene.axis.len..lines_offset], self.scene.grid);
        @memcpy(mapped_slice[lines_offset .. lines_offset + self.scene.lines.items.len], self.scene.lines.items);

        try staging_buffer.copyTo(self.vk_ctx, self.vertex_buffer);
    }

    fn initUniformBuffer(self: *Self) !void {
        const ubo_size = @sizeOf(UniformBufferObject);
        self.uniform_buffer = try Buffer.init(self.vk_ctx, ubo_size, c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        try self.updateUniformBuffer();
    }

    fn updateUniformBuffer(self: *Self) !void {
        const aspect_ratio = @as(f32, @floatFromInt(self.window.size.x)) / @as(f32, @floatFromInt(self.window.size.y));
        const ubo = UniformBufferObject{
            .view_matrix = self.scene.camera.viewMatrix(),
            .perspective_matrix = self.scene.camera.projectionMatrix(aspect_ratio),
        };

        const data_ptr = try self.uniform_buffer.map(self.vk_ctx, UniformBufferObject);
        defer self.uniform_buffer.unmap(self.vk_ctx);
        data_ptr[0] = ubo;
    }

    fn recreateSwapchain(self: *Self) !void {
        try vkCheck(c.vkDeviceWaitIdle(self.vk_ctx.device.handle));
        self.cleanupSwapchain();

        self.swapchain = try Swapchain.init(self.vk_ctx);
        self.depth_buffer = try DepthBuffer.init(self.vk_ctx, self.swapchain.extent.width, self.swapchain.extent.height);
        self.render_pass = try RenderPass.init(self.vk_ctx, &self.swapchain);
        try self.render_pass.initFrameBuffer(self.vk_ctx, &self.swapchain, self.depth_buffer.view);
        self.pipeline_layout = try PipelineLayout.init(self.vk_ctx, .{
            .pSetLayouts = &self.descriptor_layout.handle,
            .setLayoutCount = 1,
        });
        self.pipeline = try Pipeline.init(self.vk_ctx, self.render_pass, self.pipeline_layout, self.swapchain);
        try self.gui_renderer.createPipeline(self.render_pass, self.swapchain);
    }

    pub fn recordCommandBuffer(self: *Self, image_index: u32) !void {
        const begin_info = c.VkCommandBufferBeginInfo{};
        try vkCheck(c.vkBeginCommandBuffer(self.command_buffer.handle, &begin_info));

        const clear_values = [_]c.VkClearValue{
            // Color attachment clear value
            .{ .color = .{ .float32 = .{ 0.1, 0.1, 0.1, 1.0 } } },
            // Depth attachment clear value
            .{ .depthStencil = .{ .depth = 0.0, .stencil = 0 } },
        };

        const render_pass_info = c.VkRenderPassBeginInfo{
            .renderPass = self.render_pass.handle,
            .framebuffer = self.render_pass.framebuffer[image_index],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swapchain.extent,
            },
            .clearValueCount = clear_values.len,
            .pClearValues = &clear_values,
        };

        c.vkCmdBeginRenderPass(self.command_buffer.handle, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdBindPipeline(self.command_buffer.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.handle);
        const vertex_buffers = [_]c.VkBuffer{self.vertex_buffer.handle};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(self.command_buffer.handle, 0, 1, &vertex_buffers, &offsets);
        c.vkCmdBindDescriptorSets(self.command_buffer.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, 1, &self.descriptor_set, 0, null);
        c.vkCmdDraw(self.command_buffer.handle, @intCast(self.scene.getTotalVertexCount()), 1, 0, 0);
        self.text_renderer.endFrame(self.command_buffer.handle, self.descriptor_set);
        self.gui_renderer.endFrame(
            self.command_buffer.handle,
            @floatFromInt(self.window.size.x),
            @floatFromInt(self.window.size.y),
        );
        c.vkCmdEndRenderPass(self.command_buffer.handle);
        try vkCheck(c.vkEndCommandBuffer(self.command_buffer.handle));
    }
};

pub fn main() !void {
    try checkGlfw(c.glfwInit());
    defer c.glfwTerminate();

    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var app = try App.init(allocator);
    defer allocator.destroy(app);
    defer app.deinit();

    try app.run();
}
pub const math = struct {
    pub fn vec4(x: f32, y: f32, z: f32, w: f32) @Vector(4, f32) {
        return .{ x, y, z, w };
    }

    pub fn matFromArray(arr: [16]f32) [4][4]f32 {
        return .{
            .{ arr[0], arr[1], arr[2], arr[3] },
            .{ arr[4], arr[5], arr[6], arr[7] },
            .{ arr[8], arr[9], arr[10], arr[11] },
            .{ arr[12], arr[13], arr[14], arr[15] },
        };
    }

    pub fn mulMat4Vec4(m: [4][4]f32, v: @Vector(4, f32)) @Vector(4, f32) {
        var result: @Vector(4, f32) = .{ 0, 0, 0, 0 };
        inline for (0..4) |i| {
            result[i] = m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + m[i][3] * v[3];
        }
        return result;
    }
};
