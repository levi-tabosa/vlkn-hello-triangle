// src/main.zig

const std = @import("std");

const spirv = @import("spirv");
const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("vulkan/vulkan.h");
    @cInclude("GLFW/glfw3.h");
});

// --- Shader Bytecode ---
// We embed the compiled SPIR-V files directly into the executable.
// This makes distribution easier as we don't need to ship the .spv files.
const vert_shader_code = spirv.vs;
const frag_shader_code = spirv.fs;

fn checkVk(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) {
        return;
    }
    std.log.err("Vulkan call failed with code: {}", .{result});

    return switch (result) {
        c.VK_INCOMPLETE => error.VulkanIncomplete,
        c.VK_ERROR_DEVICE_LOST => error.VulkanDeviceLost,
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY, c.VK_ERROR_OUT_OF_HOST_MEMORY => error.VulkanMemoryAllocationFailed,
        c.VK_ERROR_LAYER_NOT_PRESENT => error.VulkanLayerMissing,
        c.VK_ERROR_INITIALIZATION_FAILED => error.VulkanInitFailed,
        c.VK_ERROR_FORMAT_NOT_SUPPORTED => error.VulkanUnsupportedFormat,
        c.VK_ERROR_UNKNOWN => error.VulkanUnknown,
        else => error.VulkanDefault,
    };
}

fn checkGlfw(result: c_int) !void {
    if (result == c.GLFW_TRUE) {
        return;
    }
    std.log.err("Glfw call failed with code: {}", .{result});
    // TODO: improve error handling
    return switch (result) {
        c.GLFW_PLATFORM_ERROR => error.GlfwPlatformError,
        else => error.GlfwDefault,
    };
}

// --- Application Constants ---
const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;

// --- Vertex Definition ---
// This struct defines the layout of our vertex data in memory.
// It must match the `layout(location = ...)` in the vertex shader.
const Vertex = extern struct {
    pos: @Vector(3, f32),
    offset: @Vector(3, f32) = .{ 0, 0, 0 }, // This is an optional offset vector, can be used for transformations.
    color: @Vector(4, f32) = .{ 1, 1, 1, 1 }, // Default color is white.

    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    pub fn getAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
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

    pub fn add(a: Vertex, b: Vertex) Vertex {
        return .{ .pos = .{
            a.pos[0] + b.pos[0],
            a.pos[1] + b.pos[1],
            a.pos[2] + b.pos[2],
        } };
    }

    pub fn subtract(a: Vertex, b: Vertex) Vertex {
        return .{ .pos = .{
            a.pos[0] - b.pos[0],
            a.pos[1] - b.pos[1],
            a.pos[2] - b.pos[2],
        } };
    }

    pub fn dot(a: Vertex, b: Vertex) f32 {
        return a.pos[0] * b.pos[0] + a.pos[1] * b.pos[1] + a.pos[2] * b.pos[2];
    }

    pub fn cross(a: Vertex, b: Vertex) Vertex {
        return .{ .pos = .{
            a.pos[1] * b.pos[2] - a.pos[2] * b.pos[1],
            a.pos[2] * b.pos[0] - a.pos[0] * b.pos[2],
            a.pos[0] * b.pos[1] - a.pos[1] * b.pos[0],
        } };
    }

    pub fn normalize(v: Vertex) Vertex {
        const length = std.math.sqrt(
            v.pos[0] * v.pos[0] + v.pos[1] * v.pos[1] + v.pos[2] * v.pos[2],
        );
        return .{ .pos = .{
            v.pos[0] / length,
            v.pos[1] / length,
            v.pos[2] / length,
        } };
    }
};

const UniformBufferObject = extern struct {
    view_matrix: [16]f32, // 4x4 matrix for the view transformation
    perspective_matrix: [16]f32, // 4x4 matrix for the perspective projection
};

// --- Vertex Data ---
// These are the two 3D vectors that define our line.

const Callbacks = struct {
    fn cbCursorPos(wd: ?*c.GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
        // const app: *App = @alignCast(@ptrCast(c.glfwGetWindowUserPointer(wd)));
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse return;
        const app: *App = @alignCast(@ptrCast(user_ptr));

        const ndc_x = @as(f32, @floatCast(xpos)) / @as(f32, @floatFromInt(WINDOW_WIDTH)) * 2.0 - 1.0;
        // Y conversion from screen space to NDC space.
        const ndc_y = @as(f32, @floatCast(ypos)) / @as(f32, @floatFromInt(WINDOW_HEIGHT)) * 2.0 - 1.0;

        // Convert NDC coordinates to radians for pitch and yaw.
        const pitch = -ndc_y * std.math.pi / 2.0; // Invert Y-axis for camera
        const yaw = ndc_x * std.math.pi; // X-axis maps to yaw
        // _ = pitch;
        // _ = yaw;
        app.scene.setPitchYaw(pitch, yaw);
    }
};

const Scene = struct {
    const Self = @This();

    pitch: f32 = 0.5,
    yaw: f32 = 0.2,
    view_matrix: [16]f32,
    camera: Camera,
    axis: [6]Vertex,
    grid: []Vertex,

    pub fn init(allocator: std.mem.Allocator) !Self {
        var camera = Camera.init(.{ .pos = .{ 2, 2, 2 } }, 13);

        return .{
            .axis = .{
                // X-axis (Red)
                .{ .pos = .{ -5.0, 0.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } }, .{ .pos = .{ 5.0, 0.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } },
                // Y-axis (Green)
                .{ .pos = .{ 0.0, -5.0, 0.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } }, .{ .pos = .{ 0.0, 5.0, 0.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } },
                // Z-axis (Blue)
                .{ .pos = .{ 0.0, 0.0, -5.0 }, .color = .{ 0.0, 0.0, 1.0, 1.0 } }, .{ .pos = .{ 0.0, 0.0, 5.0 }, .color = .{ 0.0, 0.0, 1.0, 1.0 } },
            },
            .grid = try createGrid(allocator, 20),
            .camera = camera,
            .view_matrix = camera.viewMatrix(),
        };
        // return self;
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.grid);
    }

    pub fn setPitchYaw(self: *Self, pitch: f32, yaw: f32) void {
        self.pitch = pitch;
        self.yaw = yaw;
        self.updateViewMatrix();
    }

    pub fn updateViewMatrix(self: *Self) void {
        if (self.camera.radius) |r| {
            self.camera.pos = .{
                .pos = .{
                    r * @cos(self.yaw) * @cos(self.pitch),
                    r * @sin(self.yaw) * @cos(self.pitch),
                    r * @sin(self.pitch),
                },
            };
        } else {
            self.camera.target = Vertex.add(self.camera.pos, Vertex{ .pos = .{
                @cos(-self.yaw) * @cos(self.pitch),
                @sin(-self.yaw) * @cos(self.pitch),
                @sin(self.pitch),
            } });
        }

        self.view_matrix = self.camera.viewMatrix();
    }

    fn createGrid(allocator: std.mem.Allocator, resolution: u32) ![]Vertex {
        const j: i32 = @intCast(resolution / 2);
        const upperLimit = j;
        var i: i32 = -j;
        var grid = allocator.alloc(Vertex, resolution * 4) catch unreachable;
        const fixed: f32 = @floatFromInt(j);

        while (i < upperLimit) : (i += 1) {
            const idx: f32 = @as(f32, @floatFromInt(i));
            const index = @as(usize, @intCast((i + j) * 4));
            grid[index] = Vertex{ .pos = .{ idx, fixed, 0.0 }, .color = .{ idx, fixed, 0.0, 1.0 } };
            grid[index + 1] = Vertex{ .pos = .{ idx, -fixed, 0.0 }, .color = .{ idx, -fixed, 0.0, 1.0 } };
            grid[index + 2] = Vertex{ .pos = .{ fixed, idx, 0.0 }, .color = .{ fixed, idx, 0.0, 1.0 } };
            grid[index + 3] = Vertex{ .pos = .{ -fixed, idx, 0.0 }, .color = .{ -fixed, idx, 0.0, 1.0 } };
        }

        return grid;
    }
};

const Camera = struct {
    const Self = @This();

    pos: Vertex,
    target: Vertex = .{ .pos = .{ 0, 0, 0 } },
    up: Vertex = .{ .pos = .{ 0, 0, 1 } }, // Z is up
    radius: ?f32 = null,
    shape: [8]Vertex,

    pub fn init(pos: Vertex, radius: ?f32) Self {
        const half_edge_len = 0.05;
        const cube: [8]Vertex = .{
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] - half_edge_len } },
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] - half_edge_len } },
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] - half_edge_len } },
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] - half_edge_len } },
        };
        return .{
            .pos = pos,
            .radius = radius,
            .shape = cube,
        };
    }

    pub fn viewMatrix(self: Self) [16]f32 {
        const z_axis = Vertex.normalize(Vertex.subtract(self.pos, self.target));
        const x_axis = Vertex.normalize(Vertex.cross(self.up, z_axis));
        const y_axis = Vertex.cross(z_axis, x_axis);

        return .{
            x_axis.pos[0],                 y_axis.pos[0],                 z_axis.pos[0],                 0.0,
            x_axis.pos[1],                 y_axis.pos[1],                 z_axis.pos[1],                 0.0,
            x_axis.pos[2],                 y_axis.pos[2],                 z_axis.pos[2],                 0.0,
            -Vertex.dot(x_axis, self.pos), -Vertex.dot(y_axis, self.pos), -Vertex.dot(z_axis, self.pos), 1.0,
        };
    }
};

// --- Main Application Struct ---
// It's good practice to hold all our Vulkan handles and state in a single struct.
const App = struct {
    const Self = @This();

    alloc_callbacks: ?*c.VkAllocationCallbacks = null,
    allocator: std.mem.Allocator,

    // GLFW handle for the window.
    window: ?*c.GLFWwindow = null,

    // Core Vulkan handles
    instance: c.VkInstance = null,
    physical_device: c.VkPhysicalDevice = undefined,
    device: c.VkDevice = null,
    graphics_queue: c.VkQueue = undefined,

    // Window surface (the bridge between Vulkan and the window system).
    surface: c.VkSurfaceKHR = null,
    surface_format: c.VkSurfaceFormatKHR = .{},
    present_mode: c.VkPresentModeKHR = 0,

    // Swapchain for presenting images to the screen.
    swapchain: c.VkSwapchainKHR = null,
    swapchain_images: []c.VkImage = &.{},
    swapchain_image_views: []c.VkImageView = &.{},
    swapchain_extent: c.VkExtent2D = undefined,

    // The Graphics Pipeline
    render_pass: c.VkRenderPass = undefined,
    descriptor_set_layout: c.VkDescriptorSetLayout = undefined,
    descriptor_pool: c.VkDescriptorPool = undefined,
    descriptor_set: c.VkDescriptorSet = undefined,
    pipeline_layout: c.VkPipelineLayout = undefined,
    graphics_pipeline: c.VkPipeline = undefined,

    // Framebuffers (one for each swapchain image).
    framebuffers: []c.VkFramebuffer = undefined,

    // Command submission
    command_pool: c.VkCommandPool = undefined,
    command_buffer: c.VkCommandBuffer = undefined, // We'll use one command buffer for simplicity.

    // Synchronization objects to coordinate CPU and GPU.
    image_available_semaphore: c.VkSemaphore = undefined,
    render_finished_semaphore: c.VkSemaphore = undefined,
    in_flight_fence: c.VkFence = undefined,

    // GPU buffer for our vertex data.
    vertex_buffer: c.VkBuffer = undefined,
    vertex_buffer_memory: c.VkDeviceMemory = undefined,
    uniform_buffer: c.VkBuffer = undefined,
    uniform_buffer_memory: c.VkDeviceMemory = undefined,

    // Scene with vertex data
    scene: Scene,

    fn initWindow(self: *Self) !void {
        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_NO_API);

        self.window = c.glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan Line", null, null);
        if (self.window == null) return error.GlfwCreateWindowFailed;
        c.glfwSetWindowUserPointer(self.window, self);

        _ = c.glfwSetCursorPosCallback(self.window, Callbacks.cbCursorPos);
    }

    fn initVulkan(app: *Self) !void {
        try app.createInstance();
        try app.createSurface();
        try app.pickPhysicalDevice();
        try app.createLogicalDeviceAndQueues();
        try app.createSwapchain();
        try app.createImageViews();
        try app.createVertexBuffer();
        try app.createUniformBuffer();
        try app.createDescriptorSetLayout();
        try app.createDescriptorPoolAndSets();
        try app.createRenderPass();
        try app.createGraphicsPipeline();
        try app.createFramebuffers();
        try app.createCommandPool();
        try app.createCommandBuffer();
        try app.createSyncObjects();
    }

    // The main application loop.
    fn run(app: *Self) !void {
        while (c.glfwWindowShouldClose(app.window) == 0) {
            c.glfwPollEvents();
            try app.drawFrame();
        }
        // Wait for the GPU to finish all operations before we start cleaning up.
        try checkVk(c.vkDeviceWaitIdle(app.device));
    }

    // The cleanup function, destroying Vulkan objects in reverse order of creation.
    fn cleanup(app: *Self) void {
        app.scene.deinit(app.allocator);
        // Destroy synchronization objects
        c.vkDestroySemaphore(app.device, app.image_available_semaphore, app.alloc_callbacks);
        c.vkDestroySemaphore(app.device, app.render_finished_semaphore, app.alloc_callbacks);
        c.vkDestroyFence(app.device, app.in_flight_fence, app.alloc_callbacks);

        // Destroy buffers and memory
        c.vkDestroyBuffer(app.device, app.vertex_buffer, app.alloc_callbacks);
        c.vkDestroyBuffer(app.device, app.uniform_buffer, app.alloc_callbacks);

        c.vkFreeMemory(app.device, app.vertex_buffer_memory, app.alloc_callbacks);
        c.vkFreeMemory(app.device, app.uniform_buffer_memory, app.alloc_callbacks);

        // Destroy command pool (which also frees command buffers)
        c.vkDestroyCommandPool(app.device, app.command_pool, app.alloc_callbacks);

        // Destroy framebuffers
        for (app.framebuffers) |fb| {
            c.vkDestroyFramebuffer(app.device, fb, app.alloc_callbacks);
        }
        app.allocator.free(app.framebuffers);

        // Destroy pipeline and related objects
        c.vkDestroyPipeline(app.device, app.graphics_pipeline, app.alloc_callbacks);
        c.vkDestroyPipelineLayout(app.device, app.pipeline_layout, app.alloc_callbacks);
        c.vkDestroyRenderPass(app.device, app.render_pass, app.alloc_callbacks);
        c.vkDestroyDescriptorPool(app.device, app.descriptor_pool, app.alloc_callbacks);
        c.vkDestroyDescriptorSetLayout(app.device, app.descriptor_set_layout, app.alloc_callbacks);

        // Destroy image views
        for (app.swapchain_image_views) |iv| {
            c.vkDestroyImageView(app.device, iv, app.alloc_callbacks);
        }
        app.allocator.free(app.swapchain_image_views);
        app.allocator.free(app.swapchain_images);

        // Destroy swapchain and surface
        c.vkDestroySwapchainKHR(app.device, app.swapchain, app.alloc_callbacks);
        c.vkDestroySurfaceKHR(app.instance, app.surface, app.alloc_callbacks);

        // Destroy logical and physical devices
        c.vkDestroyDevice(app.device, app.alloc_callbacks);

        // Destroy Vulkan instance
        c.vkDestroyInstance(app.instance, app.alloc_callbacks);

        // Destroy window
        c.glfwDestroyWindow(app.window);
    }

    // --- Vulkan Setup Functions ---

    fn createInstance(app: *Self) !void {
        const app_info = c.VkApplicationInfo{
            .pApplicationName = "Vulkan Line App",
            .applicationVersion = c.VK_MAKE_API_VERSION(0, 1, 0, 0),
            .pEngineName = "No Engine",
            .engineVersion = c.VK_MAKE_API_VERSION(0, 1, 0, 0),
            .apiVersion = c.VK_API_VERSION_1_0,
        };

        // Get the extensions required by GLFW to interface with the window system.
        var extension_count: u32 = 0;
        const required_extensions_ptr = c.glfwGetRequiredInstanceExtensions(&extension_count);
        for (0..extension_count) |i| {
            std.log.info("Required extension: {s}", .{std.mem.span(required_extensions_ptr[i])});
        }
        const required_extensions = required_extensions_ptr[0..extension_count];

        const create_info = c.VkInstanceCreateInfo{
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = @intCast(required_extensions.len),
            .ppEnabledExtensionNames = required_extensions.ptr,
        };

        try checkVk(c.vkCreateInstance(&create_info, app.alloc_callbacks, &app.instance));
    }

    fn createSurface(app: *Self) !void {
        try checkVk(c.glfwCreateWindowSurface(app.instance, app.window, app.alloc_callbacks, &app.surface));
    }

    fn pickPhysicalDevice(app: *Self) !void {
        var device_count: u32 = 0;
        // 1. First call to get the count
        try checkVk(c.vkEnumeratePhysicalDevices(app.instance, &device_count, null));

        if (device_count == 0) {
            std.log.err("Failed to find GPUs with Vulkan support!", .{});
            return error.NoGpuFound;
        }

        // 2. Allocate space and make the second call to get the handles
        const devices = try app.allocator.alloc(c.VkPhysicalDevice, device_count);
        defer app.allocator.free(devices);
        try checkVk(c.vkEnumeratePhysicalDevices(app.instance, &device_count, devices.ptr));
        if (device_count > 1) std.log.info(
            "More than one GPU physical device found, picking first: {any}",
            .{devices[0]},
        );

        app.physical_device = devices[0];
    }

    /// Helper function
    fn findQueueFamilies(
        allocator: std.mem.Allocator,
        phys_device: c.VkPhysicalDevice,
        surface: c.VkSurfaceKHR,
    ) !u32 {
        var queue_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_count, null);
        const queue_family_properties = try allocator.alloc(c.VkQueueFamilyProperties, queue_count);
        defer allocator.free(queue_family_properties);

        c.vkGetPhysicalDeviceQueueFamilyProperties(
            phys_device,
            &queue_count,
            queue_family_properties.ptr,
        );

        for (queue_family_properties, 0..queue_count) |family_prop, i| {
            // We need a queue that supports graphics operations.
            if (family_prop.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                // We also need a queue that can present to our surface.
                var present_support: c.VkBool32 = c.VK_FALSE;
                try checkVk(c.vkGetPhysicalDeviceSurfaceSupportKHR(phys_device, @intCast(i), surface, &present_support));

                if (present_support == c.VK_TRUE) {
                    return @intCast(i);
                }
            }
        }

        return error.NoSuitableQueueFamily;
    }

    fn createLogicalDeviceAndQueues(app: *Self) !void {
        const queue_family_index = try findQueueFamilies(app.allocator, app.physical_device, app.surface);
        const queue_priority: f32 = 1.0;
        const queue_create_info = c.VkDeviceQueueCreateInfo{
            .queueFamilyIndex = queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        var device_features = c.VkPhysicalDeviceFeatures{};
        // We need to enable the swapchain extension.
        const device_extensions = [_][*:0]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        const create_info = c.VkDeviceCreateInfo{
            .pQueueCreateInfos = &queue_create_info,
            .queueCreateInfoCount = 1,
            .pEnabledFeatures = &device_features,
            .enabledExtensionCount = device_extensions.len,
            .ppEnabledExtensionNames = &device_extensions,
        };

        try checkVk(c.vkCreateDevice(
            app.physical_device,
            &create_info,
            app.alloc_callbacks,
            &app.device,
        ));

        // Get a handle to the graphics queue.
        c.vkGetDeviceQueue(app.device, queue_family_index, 0, &app.graphics_queue);
    }

    ///  TODO: refactor
    fn createSwapchain(self: *Self) !void {
        // --- Choose Swapchain Settings ---
        var fmt_count: u32 = undefined;
        try checkVk(c.vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.physical_device,
            self.surface,
            &fmt_count,
            null,
        ));
        const fmts = self.allocator.alloc(c.VkSurfaceFormatKHR, fmt_count) catch @panic("OOM");
        defer self.allocator.free(fmts);
        try checkVk(c.vkGetPhysicalDeviceSurfaceFormatsKHR(
            self.physical_device,
            self.surface,
            &fmt_count,
            fmts.ptr,
        ));
        self.surface_format = fmts[0]; // Choose a format, B8G8R8A8_SRGB is common.

        var present_modes_count: u32 = undefined;
        try checkVk(c.vkGetPhysicalDeviceSurfacePresentModesKHR(
            self.physical_device,
            self.surface,
            &present_modes_count,
            null,
        ));
        const present_modes = self.allocator.alloc(c.VkPresentModeKHR, present_modes_count) catch @panic("OOM");
        defer self.allocator.free(present_modes);
        try checkVk(c.vkGetPhysicalDeviceSurfacePresentModesKHR(
            self.physical_device,
            self.surface,
            &present_modes_count,
            present_modes.ptr,
        ));

        self.present_mode = c.VK_PRESENT_MODE_FIFO_KHR; // V-Sync, guaranteed to be available.
        for (present_modes) |mode| {
            if (mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
                self.present_mode = c.VK_PRESENT_MODE_MAILBOX_KHR; // Triple buffering, better for latency.
                break;
            }
        }

        var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
        try checkVk(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(self.physical_device, self.surface, &capabilities));
        self.swapchain_extent = capabilities.currentExtent;

        var image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 and image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }

        const create_info = c.VkSwapchainCreateInfoKHR{
            .surface = self.surface,
            .minImageCount = image_count,
            .imageFormat = self.surface_format.format,
            .imageColorSpace = self.surface_format.colorSpace,
            .imageExtent = self.swapchain_extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = self.present_mode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
        };
        try checkVk(c.vkCreateSwapchainKHR(
            self.device,
            &create_info,
            self.alloc_callbacks,
            &self.swapchain,
        ));

        var img_count: u32 = undefined;
        try checkVk(c.vkGetSwapchainImagesKHR(
            self.device,
            self.swapchain,
            &img_count,
            null,
        ));
        const swapchain_images = self.allocator.alloc(c.VkImage, img_count) catch @panic("OOM");
        try checkVk(c.vkGetSwapchainImagesKHR(
            self.device,
            self.swapchain,
            &img_count,
            swapchain_images.ptr,
        ));
        self.swapchain_images = swapchain_images;
    }

    fn createImageViews(app: *Self) !void {
        app.swapchain_image_views = app.allocator.alloc(c.VkImageView, app.swapchain_images.len) catch @panic("OOM");
        for (app.swapchain_images, 0..) |image, i| {
            const create_info = c.VkImageViewCreateInfo{
                .image = image,
                .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
                .format = app.surface_format.format,
                .components = .{}, // Use default .r, .g, .b, .a mapping
                .subresourceRange = .{
                    .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                    .baseMipLevel = 0,
                    .levelCount = 1,
                    .baseArrayLayer = 0,
                    .layerCount = 1,
                },
            };
            try checkVk(c.vkCreateImageView(
                app.device,
                &create_info,
                app.alloc_callbacks,
                &app.swapchain_image_views[i],
            ));
        }
    }

    fn createRenderPass(app: *Self) !void {
        var color_attachment = c.VkAttachmentDescription{
            .format = app.surface_format.format,
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

        const subpass = c.VkSubpassDescription{
            .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_ref,
        };

        var subpass_dep = c.VkSubpassDependency{
            .srcSubpass = c.VK_SUBPASS_EXTERNAL,
            .dstSubpass = 0,
            .srcStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .srcAccessMask = 0,
            .dstStageMask = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
        };

        const create_info = c.VkRenderPassCreateInfo{
            .attachmentCount = 1,
            .pAttachments = &color_attachment,
            .subpassCount = 1,
            .pSubpasses = &subpass,
            .dependencyCount = 1,
            .pDependencies = &subpass_dep,
        };

        try checkVk(c.vkCreateRenderPass(
            app.device,
            &create_info,
            app.alloc_callbacks,
            &app.render_pass,
        ));
    }

    fn createDescriptorSetLayout(app: *Self) !void {
        var ubo_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            // FIX: The UBO is used by both the vertex shader (for matrices) and potentially
            // the fragment shader (for color). We must specify all stages that access it.
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT | c.VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = null,
        };

        var layout_info = c.VkDescriptorSetLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &ubo_layout_binding,
        };

        try checkVk(c.vkCreateDescriptorSetLayout(
            app.device,
            &layout_info,
            app.alloc_callbacks,
            &app.descriptor_set_layout,
        ));
    }

    fn createDescriptorPoolAndSets(self: *Self) !void {
        var pool_size = c.VkDescriptorPoolSize{
            .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
        };
        var pool_info = c.VkDescriptorPoolCreateInfo{
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
            .maxSets = 1,
        };

        try checkVk(c.vkCreateDescriptorPool(
            self.device,
            &pool_info,
            self.alloc_callbacks,
            &self.descriptor_pool,
        ));

        var set_alloc_info = c.VkDescriptorSetAllocateInfo{
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &self.descriptor_set_layout,
        };

        try checkVk(c.vkAllocateDescriptorSets(self.device, &set_alloc_info, &self.descriptor_set));

        var desc_buffer_info = c.VkDescriptorBufferInfo{
            .buffer = self.uniform_buffer,
            .offset = 0,
            .range = @sizeOf(UniformBufferObject),
        };

        var desc_write = c.VkWriteDescriptorSet{
            .dstSet = self.descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &desc_buffer_info,
        };
        c.vkUpdateDescriptorSets(self.device, 1, &desc_write, 0, null);

        const ubo = UniformBufferObject{
            .view_matrix = self.scene.view_matrix,
            .perspective_matrix = blk: {
                // FIX: This is the correct perspective matrix for Vulkan's coordinate system.
                // The original matrix was for OpenGL, which uses a different depth range [-1, 1]
                // and has a different Y-axis direction in its normalized device coordinates.
                // This matrix handles the depth range [0, 1] and flips the Y-axis.
                const fovy = std.math.degreesToRadians(45.0);
                const aspect = @as(f32, @floatFromInt(WINDOW_WIDTH)) / @as(f32, @floatFromInt(WINDOW_HEIGHT));
                const near = 0.1;
                const far = 100.0;
                const f = 1.0 / std.math.tan(fovy / 2.0);

                break :blk .{
                    f / aspect, 0, 0, 0,
                    0, -f, 0,                           0, // Note the -f to flip Y
                    0, 0,  far / (near - far),          -1,
                    0, 0,  (far * near) / (near - far), 0,
                };
            },
        };

        var ubo_data_ptr: ?*anyopaque = undefined;
        try checkVk(c.vkMapMemory(self.device, self.uniform_buffer_memory, 0, @sizeOf(UniformBufferObject), 0, &ubo_data_ptr));
        const mapped_ubo: *UniformBufferObject = @ptrCast(@alignCast(ubo_data_ptr.?));
        mapped_ubo.* = ubo;
        c.vkUnmapMemory(self.device, self.uniform_buffer_memory);
    }

    fn createShaderModule(allocator: std.mem.Allocator, device: c.VkDevice, code: []const u8, allocation_callbacks: ?*c.VkAllocationCallbacks) !c.VkShaderModule {
        std.debug.assert(code.len % 4 == 0);

        const aligned_code = try allocator.alignedAlloc(u32, @alignOf(u32), code.len / @sizeOf(u32));
        defer allocator.free(aligned_code);

        @memcpy(std.mem.sliceAsBytes(aligned_code), code);

        var create_info = c.VkShaderModuleCreateInfo{
            .codeSize = code.len,
            .pCode = aligned_code.ptr,
        };

        var shader_module: c.VkShaderModule = undefined;
        try checkVk(c.vkCreateShaderModule(device, &create_info, allocation_callbacks, &shader_module));
        return shader_module;
    }

    fn createGraphicsPipeline(app: *Self) !void {
        const vert_shader_module = try createShaderModule(app.allocator, app.device, vert_shader_code, app.alloc_callbacks);
        defer c.vkDestroyShaderModule(app.device, vert_shader_module, app.alloc_callbacks);
        const frag_shader_module = try createShaderModule(app.allocator, app.device, frag_shader_code, app.alloc_callbacks);
        defer c.vkDestroyShaderModule(app.device, frag_shader_module, app.alloc_callbacks);

        const vert_shader_stage_info = c.VkPipelineShaderStageCreateInfo{
            .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
            .module = vert_shader_module,
            .pName = "main",
        };
        const frag_shader_stage_info = c.VkPipelineShaderStageCreateInfo{
            .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
            .module = frag_shader_module,
            .pName = "main",
        };

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{ vert_shader_stage_info, frag_shader_stage_info };

        const binding_description = Vertex.getBindingDescription();
        const attribute_descriptions = Vertex.getAttributeDescriptions();

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
            .width = @floatFromInt(app.swapchain_extent.width),
            .height = @floatFromInt(app.swapchain_extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };

        const scissor = c.VkRect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = app.swapchain_extent,
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
            .cullMode = c.VK_CULL_MODE_NONE, // Use NONE for 2D or line geometry
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

        var pipeline_layout_info = c.VkPipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &app.descriptor_set_layout,
        };

        try checkVk(c.vkCreatePipelineLayout(
            app.device,
            &pipeline_layout_info,
            app.alloc_callbacks,
            &app.pipeline_layout,
        ));

        var pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &color_blending,
            .layout = app.pipeline_layout,
            .renderPass = app.render_pass,
            .subpass = 0,
        };

        try checkVk(c.vkCreateGraphicsPipelines(
            app.device,
            null,
            1,
            &pipeline_info,
            app.alloc_callbacks,
            &app.graphics_pipeline,
        ));
    }

    fn createFramebuffers(app: *Self) !void {
        app.framebuffers = try app.allocator.alloc(c.VkFramebuffer, app.swapchain_image_views.len);
        for (app.swapchain_image_views, 0..) |image_view, i| {
            const attachments = [_]c.VkImageView{image_view};
            const create_info = c.VkFramebufferCreateInfo{
                .renderPass = app.render_pass,
                .attachmentCount = attachments.len,
                .pAttachments = &attachments,
                .width = app.swapchain_extent.width,
                .height = app.swapchain_extent.height,
                .layers = 1,
            };
            try checkVk(c.vkCreateFramebuffer(app.device, &create_info, null, &app.framebuffers[i]));
        }
    }

    // --- Buffer Creation Helper ---
    fn findMemoryType(
        physical_device: c.VkPhysicalDevice,
        type_filter: u32,
        properties: c.VkMemoryPropertyFlags,
    ) !u32 {
        var mem_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

        const set: u64 = 1;
        var i: u5 = 0;
        while (i < mem_properties.memoryTypeCount) : (i += 1) {
            if ((type_filter & set << i) != 0 and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        return error.MissingMemoryType;
    }

    fn createBuffer(
        app: Self,
        size: c.VkDeviceSize,
        usage: c.VkBufferUsageFlags,
        properties: c.VkMemoryPropertyFlags,
    ) !struct { buffer: c.VkBuffer, memory: c.VkDeviceMemory } {
        var buffer_info = c.VkBufferCreateInfo{
            .size = size,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };
        var buffer: c.VkBuffer = undefined;
        try checkVk(c.vkCreateBuffer(app.device, &buffer_info, null, &buffer));

        var mem_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(app.device, buffer, &mem_requirements);

        var alloc_info = c.VkMemoryAllocateInfo{
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = try findMemoryType(app.physical_device, mem_requirements.memoryTypeBits, properties),
        };
        var memory: c.VkDeviceMemory = undefined;
        try checkVk(c.vkAllocateMemory(app.device, &alloc_info, null, &memory));

        try checkVk(c.vkBindBufferMemory(app.device, buffer, memory, 0));

        return .{ .buffer = buffer, .memory = memory };
    }

    fn createVertexBuffer(self: *Self) !void {
        // Calculate the total size needed for both axis and grid vertices.
        const total_vertex_count = self.scene.axis.len + self.scene.grid.len;
        const buffer_size = @sizeOf(Vertex) * total_vertex_count;

        if (buffer_size == 0) return; // Avoid creating a zero-size buffer

        const buffer = try self.createBuffer(
            buffer_size,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.vertex_buffer = buffer.buffer;
        self.vertex_buffer_memory = buffer.memory;

        // --- Copy vertex data to the buffer ---
        var data_ptr: ?*anyopaque = undefined;
        try checkVk(c.vkMapMemory(self.device, buffer.memory, 0, buffer_size, 0, &data_ptr));
        defer c.vkUnmapMemory(self.device, buffer.memory);

        const many_item_ptr: [*]Vertex = @ptrCast(@alignCast(data_ptr.?));
        const mapped_vertex_slice = many_item_ptr[0..total_vertex_count];

        // Copy the axis vertices to the beginning of the buffer.
        @memcpy(mapped_vertex_slice[0..self.scene.axis.len], &self.scene.axis);

        // Copy the grid vertices immediately after the axis vertices.
        const grid_offset = self.scene.axis.len;
        @memcpy(mapped_vertex_slice[grid_offset .. grid_offset + self.scene.grid.len], self.scene.grid);
    }

    fn updateVertexBuffers(self: *Self) !void {
        const buffer_size = @sizeOf(Vertex) * (self.scene.axis.len + self.scene.grid.len);

        var data_ptr: ?*anyopaque = undefined;
        try checkVk(c.vkMapMemory(self.device, self.vertex_buffer_memory, 0, buffer_size, 0, &data_ptr));
        defer c.vkUnmapMemory(self.device, self.vertex_buffer_memory);

        const mapped_vertex_slice: [*]Vertex = @ptrCast(@alignCast(data_ptr));
        @memcpy(mapped_vertex_slice[0..self.scene.axis.len], &self.scene.axis);
        const grid_offset = self.scene.axis.len;
        @memcpy(mapped_vertex_slice[grid_offset .. grid_offset + self.scene.grid.len], self.scene.grid);
    }

    fn createUniformBuffer(self: *Self) !void {
        // FIX: The buffer size must match the UniformBufferObject struct, not the Vertex struct.
        const ubo_size = @sizeOf(UniformBufferObject);
        const uniform_buffer_obj = try self.createBuffer(
            ubo_size,
            c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );

        self.uniform_buffer = uniform_buffer_obj.buffer;
        self.uniform_buffer_memory = uniform_buffer_obj.memory;
    }

    fn updateUniformBuffer(self: *Self) !void {
        const ubo = UniformBufferObject{
            .view_matrix = self.scene.view_matrix,
            .perspective_matrix = blk: {
                // FIX: This is the correct perspective matrix for Vulkan's coordinate system.
                //TODO: Internalize this better, maybe use a function to generate it.
                const fovy = std.math.degreesToRadians(45.0);
                const aspect = @as(f32, @floatFromInt(WINDOW_WIDTH)) / @as(f32, @floatFromInt(WINDOW_HEIGHT));
                const near = 0.1;
                const far = 100.0;
                const f = 1.0 / std.math.tan(fovy / 2.0);

                break :blk .{
                    f / aspect, 0, 0, 0,
                    0, -f, 0,                           0, // Note the -f to flip Y
                    0, 0,  far / (near - far),          -1,
                    0, 0,  (far * near) / (near - far), 0,
                };
            },
        };

        var ubo_data_ptr: ?*anyopaque = undefined;
        try checkVk(c.vkMapMemory(self.device, self.uniform_buffer_memory, 0, @sizeOf(UniformBufferObject), 0, &ubo_data_ptr));
        const mapped_ubo: *UniformBufferObject = @ptrCast(@alignCast(ubo_data_ptr.?));
        mapped_ubo.* = ubo;
        c.vkUnmapMemory(self.device, self.uniform_buffer_memory);
    }

    fn createCommandPool(self: *Self) !void {
        const queue_family_index = try findQueueFamilies(self.allocator, self.physical_device, self.surface);
        var cmd_pool_info = c.VkCommandPoolCreateInfo{
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_index,
        };

        try checkVk(c.vkCreateCommandPool(self.device, &cmd_pool_info, null, &self.command_pool));
    }

    fn createCommandBuffer(self: *Self) !void {
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .commandPool = self.command_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        try checkVk(c.vkAllocateCommandBuffers(self.device, &alloc_info, &self.command_buffer));
    }

    fn createSyncObjects(self: *Self) !void {
        const sync_create_info = c.VkSemaphoreCreateInfo{};
        const fence_create_info = c.VkFenceCreateInfo{
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        }; // Create signaled so first frame doesn't wait.

        try checkVk(c.vkCreateSemaphore(self.device, &sync_create_info, self.alloc_callbacks, &self.image_available_semaphore));
        try checkVk(c.vkCreateSemaphore(self.device, &sync_create_info, self.alloc_callbacks, &self.render_finished_semaphore));
        try checkVk(c.vkCreateFence(self.device, &fence_create_info, self.alloc_callbacks, &self.in_flight_fence));
    }

    // --- Per-frame Drawing Logic ---
    fn drawFrame(self: *Self) !void {
        // Wait for the previous frame's fence to be signaled (meaning it's done rendering).
        try checkVk(c.vkWaitForFences(self.device, 1, &self.in_flight_fence, c.VK_TRUE, std.math.maxInt(u64)));
        try self.updateUniformBuffer();
        try checkVk(c.vkResetFences(self.device, 1, &self.in_flight_fence));

        // Acquire an image from the swapchain.
        var image_index: u32 = 0;
        try checkVk(c.vkAcquireNextImageKHR(self.device, self.swapchain, std.math.maxInt(u64), self.image_available_semaphore, null, &image_index));

        // Reset and record the command buffer.
        try checkVk(c.vkResetCommandBuffer(self.command_buffer, 0));
        try self.recordCommandBuffer(image_index);

        // Submit the command buffer to the graphics queue.
        const wait_semaphores = [_]c.VkSemaphore{self.image_available_semaphore};
        const wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const signal_semaphores = [_]c.VkSemaphore{self.render_finished_semaphore};

        const submit_info = c.VkSubmitInfo{
            .waitSemaphoreCount = wait_semaphores.len,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffer,
            .signalSemaphoreCount = signal_semaphores.len,
            .pSignalSemaphores = &signal_semaphores,
        };
        try checkVk(c.vkQueueSubmit(self.graphics_queue, 1, &submit_info, self.in_flight_fence));

        // Present the image to the screen.
        const swapchains = [_]c.VkSwapchainKHR{self.swapchain};
        const present_info = c.VkPresentInfoKHR{
            .waitSemaphoreCount = signal_semaphores.len,
            .pWaitSemaphores = &signal_semaphores,
            .swapchainCount = swapchains.len,
            .pSwapchains = &swapchains,
            .pImageIndices = &image_index,
        };

        try checkVk(c.vkQueuePresentKHR(self.graphics_queue, &present_info));
    }

    fn recordCommandBuffer(self: *Self, image_index: u32) !void {
        const begin_info = c.VkCommandBufferBeginInfo{};
        try checkVk(c.vkBeginCommandBuffer(self.command_buffer, &begin_info));

        const _clear_color = c.VkClearValue{ .color = .{ .float32 = .{ 0.1, 0.1, 0.1, 1.0 } } };

        const render_pass_info = c.VkRenderPassBeginInfo{
            .renderPass = self.render_pass,
            .framebuffer = self.framebuffers[@intCast(image_index)],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swapchain_extent,
            },
            .clearValueCount = 1,
            .pClearValues = &_clear_color,
        };
        c.vkCmdBeginRenderPass(self.command_buffer, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);

        // Bind the graphics pipeline.
        c.vkCmdBindPipeline(self.command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.graphics_pipeline);

        // Bind the vertex buffer.
        const vertex_buffers = [_]c.VkBuffer{self.vertex_buffer};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(self.command_buffer, 0, 1, &vertex_buffers, &offsets);
        c.vkCmdBindDescriptorSets(
            self.command_buffer,
            c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline_layout,
            0,
            1,
            &self.descriptor_set,
            0,
            null,
        );

        c.vkCmdDraw(self.command_buffer, @intCast(self.scene.axis.len + self.scene.grid.len), 1, 0, 0);

        c.vkCmdEndRenderPass(self.command_buffer);
        try checkVk(c.vkEndCommandBuffer(self.command_buffer));
    }
};

// --- Zig Entry Point ---
// --- Entry Point ---
pub fn main() !void {
    try checkGlfw(c.glfwInit());
    defer c.glfwTerminate();

    var da: std.heap.DebugAllocator(.{}) = .init;
    defer std.log.info("memory leak: {}{any}\n", .{
        da.detectLeaks(),
        da.deinit(),
    });

    const allocator = da.allocator();

    const app = try allocator.create(App);
    defer allocator.destroy(app);

    app.* = .{
        .allocator = allocator,
        .scene = try Scene.init(allocator),
    };

    defer app.cleanup();

    try app.initWindow();
    try app.initVulkan();
    try app.run();
}
