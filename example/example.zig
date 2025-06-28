// test.zig

const std = @import("std");

const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const spirv = @import("spirv");
const scene = @import("geometry");
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

const Vertex = scene.V3;
const Scene = scene.Scene;

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
        c.VK_ERROR_SURFACE_LOST_KHR => error.VulkanSurfaceLost,
        c.VK_ERROR_NATIVE_WINDOW_IN_USE_KHR => error.NativeWindowInUse,
        c.VK_SUBOPTIMAL_KHR => error.VulkanSuboptimalKHR,
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

const UniformBufferObject = extern struct {
    view_matrix: [16]f32, // 4x4 matrix for the view transformation
    perspective_matrix: [16]f32, // 4x4 matrix for the perspective projection
};

// --- Vertex Data ---
// These are the two 3D vectors that define our line.

const Callbacks = struct {
    // https://www.glfw.org/docs/3.0/group__input
    fn cbCursorPos(wd: ?*c.GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        const app: *App = @alignCast(@ptrCast(user_ptr));

        const ndc_x = @as(f32, @floatCast(xpos)) / @as(f32, @floatFromInt(WINDOW_WIDTH)) * 2.0 - 1.0;
        // Y conversion from screen space to NDC space.
        const ndc_y = @as(f32, @floatCast(ypos)) / @as(f32, @floatFromInt(WINDOW_HEIGHT)) * 2.0 - 1.0;

        // Convert NDC coordinates to radians for pitch and yaw.
        const pitch = -ndc_y * std.math.pi / 2.0; // Invert Y-axis for camera
        const yaw = ndc_x * std.math.pi; // X-axis maps to yaw

        app.scene.setPitchYaw(pitch, yaw);
    }

    fn cbKey(wd: ?*c.GLFWwindow, char: c_int, code: c_int, btn: c_int, mods: c_int) callconv(.C) void {
        _ = wd;
        // const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        // const app: *App = @alignCast(@ptrCast(user_ptr));

        std.debug.print("{c}; code : {} action : {} mods : {} \n", .{ @as(u8, @intCast(char)), code, btn, mods });

        // app.scene.;
    }

    fn cbFramebufferResize(wd: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        const app: *App = @alignCast(@ptrCast(user_ptr));
        app.window.size.x = width;
        app.window.size.y = height;
        // We just set a flag here. The actual recreation will happen in the draw loop.
        app.framebuffer_resized = true;
    }
};

const Window = struct {
    const Self = @This();

    handle: ?*c.GLFWwindow = undefined,
    size: struct { x: c_int, y: c_int },

    pub fn init(user_ptr: ?*anyopaque, width: c_int, height: c_int, title: [*c]const u8, monitor: ?*c.GLFWmonitor, share: ?*c.GLFWwindow) !Self {
        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_TRUE);

        const handle = c.glfwCreateWindow(width, height, title, monitor, share);

        if (handle == null) return error.GlfwCreateWindowFailed;
        c.glfwSetWindowUserPointer(handle, user_ptr);
        _ = c.glfwSetCursorPosCallback(handle, Callbacks.cbCursorPos);
        _ = c.glfwSetKeyCallback(handle, Callbacks.cbKey);
        _ = c.glfwSetFramebufferSizeCallback(handle, Callbacks.cbFramebufferResize);

        return .{
            .handle = handle,
            .size = .{ .x = width, .y = height },
        };
    }

    pub fn deinit(self: *Self) void {
        c.glfwDestroyWindow(self.handle);
        self.handle = undefined;
    }

    pub fn minimized(self: Self) bool {
        return self.size.x == 0 or self.size.y == 0;
    }
};

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

        try checkVk(
            c.vkCreateInstance(&create_info, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyInstance(self.handle, null);
    }
};

const Surface = struct {
    const Self = @This();

    handle: c.VkSurfaceKHR = undefined,
    owner: c.VkInstance,

    pub fn init(instance: Instance, window: Window) !Self {
        assert(instance.handle != null and window.handle != null);
        var self = Self{
            .owner = instance.handle,
        };

        try checkVk(
            c.glfwCreateWindowSurface(instance.handle, window.handle, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroySurfaceKHR(self.owner, self.handle, null);
    }
};

const PhysicalDevice = struct {
    const Self = @This();

    handle: c.VkPhysicalDevice = undefined,
    q_family_idx: u32 = undefined,

    pub fn init(allocator: Allocator, instance: Instance, surface: Surface) !Self {
        assert(instance.handle != null and surface.handle != null);
        var self = Self{};
        // Pick physical device
        var physical_device_count: u32 = 0;
        try checkVk(
            c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, null),
        );

        const physical_devices = try allocator.alloc(c.VkPhysicalDevice, physical_device_count);
        defer allocator.free(physical_devices);
        try checkVk(
            c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, physical_devices.ptr),
        );

        self.handle = physical_devices[0];

        // Get queue family that supports graphics
        var q_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_count, null);

        const q_family_props = try allocator.alloc(c.VkQueueFamilyProperties, q_count);
        defer allocator.free(q_family_props);
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_count, q_family_props.ptr);

        for (q_family_props, 0..) |prop, i| {
            // We need a queue that supports graphics operations.
            if (prop.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                var support: c.VkBool32 = c.VK_FALSE;
                try checkVk(
                    c.vkGetPhysicalDeviceSurfaceSupportKHR(
                        self.handle,
                        @intCast(i),
                        surface.handle,
                        &support,
                    ),
                );

                if (support == c.VK_TRUE) {
                    self.q_family_idx = @intCast(i);
                    return self;
                }
            }
        }

        return error.NoSuitableQueueFamily;
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }

    pub fn findMemoryType(
        self: Self,
        type_filter: u32,
        properties: c.VkMemoryPropertyFlags,
    ) !u32 {
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

    pub fn findQueueFamilies(
        self: Self,
        allocator: std.mem.Allocator,
        surface: Surface,
    ) !u32 {
        assert(surface.handle != null);
        var queue_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self, &queue_count, null);
        const queue_family_properties = try allocator.alloc(c.VkQueueFamilyProperties, queue_count);
        defer allocator.free(queue_family_properties);

        c.vkGetPhysicalDeviceQueueFamilyProperties(
            self,
            &queue_count,
            queue_family_properties.ptr,
        );

        for (queue_family_properties, 0..queue_count) |family_prop, i| {
            // We need a queue that supports graphics operations.
            if (family_prop.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                // We also need a queue that can present to our surface.
                var present_support: c.VkBool32 = c.VK_FALSE;
                try checkVk(
                    c.vkGetPhysicalDeviceSurfaceSupportKHR(self, @intCast(i), surface.handle, &present_support),
                );

                if (present_support == c.VK_TRUE) {
                    return @intCast(i);
                }
            }
        }

        return error.NoSuitableQueueFamily;
    }
};

const Queue = struct {
    const Self = @This();

    handle: c.VkQueue = undefined,

    pub fn init(device: Device, physical_device: PhysicalDevice) !Self {
        assert(device.handle != null and physical_device.handle != null);
        var self = Self{
            .owner = device,
        };

        c.vkGetDeviceQueue(device.handle, physical_device.q_family_idx, 0, &self.handle);
    }

    pub fn deinit(self: *Self) void {
        _ = self;
    }
};

const Swapchain = struct {
    const Self = @This();

    handle: c.VkSwapchainKHR = undefined,
    image_format: c.VkSurfaceFormatKHR = undefined,
    extent: c.VkExtent2D = undefined,
    images: []c.VkImage = undefined,
    image_views: []c.VkImageView = undefined,
    owner: c.VkDevice,

    pub fn init(allocator: Allocator, device: Device, surface: Surface) !Self {
        assert(device.handle != null and surface.handle != null);
        var self = Self{
            .owner = device.handle,
        };

        var fmt_count: u32 = 0;
        try checkVk(c.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical.handle,
            surface.handle,
            &fmt_count,
            null,
        ));

        const fmts = try allocator.alloc(c.VkSurfaceFormatKHR, fmt_count);
        defer allocator.free(fmts);
        try checkVk(c.vkGetPhysicalDeviceSurfaceFormatsKHR(
            device.physical.handle,
            surface.handle,
            &fmt_count,
            fmts.ptr,
        ));

        self.image_format = fmts[0];

        var present_modes_count: u32 = undefined;
        try checkVk(c.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical.handle,
            surface.handle,
            &present_modes_count,
            null,
        ));
        const present_modes = allocator.alloc(c.VkPresentModeKHR, present_modes_count) catch @panic("OOM");
        defer allocator.free(present_modes);
        try checkVk(c.vkGetPhysicalDeviceSurfacePresentModesKHR(
            device.physical.handle,
            surface.handle,
            &present_modes_count,
            present_modes.ptr,
        ));

        var present_mode: c_uint = c.VK_PRESENT_MODE_FIFO_KHR; // V-Sync, guaranteed to be available.
        for (present_modes) |mode| {
            if (mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
                present_mode = c.VK_PRESENT_MODE_MAILBOX_KHR; // Triple buffering, better for latency.
                break;
            }
        }

        var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
        try checkVk(
            c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physical.handle, surface.handle, &capabilities),
        );
        // Get handle for `extent`
        self.extent = capabilities.currentExtent;

        // Image count is
        var image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 and image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }

        const create_info = c.VkSwapchainCreateInfoKHR{
            .surface = surface.handle,
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
        // Get swapchain handle
        try checkVk(
            c.vkCreateSwapchainKHR(device.handle, &create_info, null, &self.handle),
        );

        var img_count: u32 = undefined;
        try checkVk(
            c.vkGetSwapchainImagesKHR(device.handle, self.handle, &img_count, null),
        );

        self.images = try allocator.alloc(c.VkImage, img_count);
        try checkVk(
            c.vkGetSwapchainImagesKHR(device.handle, self.handle, &img_count, self.images.ptr),
        );

        self.image_views = allocator.alloc(c.VkImageView, image_count) catch @panic("OOM");
        for (self.images, 0..) |image, i| {
            const info = c.VkImageViewCreateInfo{
                .image = image,
                .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
                .format = self.image_format.format,
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
                device.handle,
                &info,
                null,
                &self.image_views[i],
            ));
        }

        return self;
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        allocator.free(self.images);
        allocator.free(self.image_views);
        c.vkDestroySwapchainKHR(self.owner, self.handle, null);
    }
};

const RenderPass = struct {
    const Self = @This();

    handle: c.VkRenderPass = undefined,
    framebuffer: []c.VkFramebuffer = undefined,
    owner: c.VkDevice,

    pub fn init(allocator: Allocator, device: Device, swapchain: Swapchain) !Self {
        assert(device.handle != null and swapchain.handle != null);
        var self = Self{
            .owner = device.handle,
        };

        var color_attachment = c.VkAttachmentDescription{
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

        try checkVk(
            c.vkCreateRenderPass(device.handle, &create_info, null, &self.handle),
        );

        try self.initFrameBuffer(allocator, swapchain);

        return self;
    }

    pub fn deinit(self: *Self, allocator: Allocator) void {
        self.deinitFramebuffer(allocator);
        c.vkDestroyRenderPass(self.owner, self.handle, null);
    }

    pub fn initFrameBuffer(self: *Self, allocator: Allocator, swapchain: Swapchain) !void {
        self.framebuffer = try allocator.alloc(c.VkFramebuffer, swapchain.image_views.len);
        for (swapchain.image_views, 0..) |iv, i| {
            const attachments = [_]c.VkImageView{iv};
            const f_buffer_create_info = c.VkFramebufferCreateInfo{
                .renderPass = self.handle,
                .attachmentCount = attachments.len,
                .pAttachments = &attachments,
                .width = swapchain.extent.width,
                .height = swapchain.extent.height,
                .layers = 1,
            };
            try checkVk(
                c.vkCreateFramebuffer(self.owner, &f_buffer_create_info, null, &self.framebuffer[i]),
            );
        }
    }

    pub fn deinitFramebuffer(self: *Self, allocator: Allocator) void {
        for (self.framebuffer) |fb| {
            c.vkDestroyFramebuffer(self.owner, fb, null);
        }
        allocator.free(self.framebuffer);
    }
};

const CommandPool = struct {
    const Self = @This();

    handle: c.VkCommandPool = undefined,
    owner: c.VkDevice,

    pub fn init(device: Device) !Self {
        var self = Self{
            .owner = device.handle,
        };

        var cmd_pool_info = c.VkCommandPoolCreateInfo{
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = device.physical.q_family_idx,
        };

        try checkVk(
            c.vkCreateCommandPool(device.handle, &cmd_pool_info, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyCommandPool(self.owner, self.handle, null);
    }

    pub fn allocateCommandBuffer(self: *Self) !CommandBuffer {
        assert(self.handle != null);
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .commandPool = self.handle,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        var handle: c.VkCommandBuffer = undefined;
        try checkVk(
            c.vkAllocateCommandBuffers(self.owner, &alloc_info, &handle),
        );

        return .{
            .handle = handle,
        };
    }
};

const CommandBuffer = struct {
    handle: c.VkCommandBuffer = undefined,
};

const BufferUsage = enum(u8) {
    Vertex,
    Uniform,

    pub fn getVkFlag(self: BufferUsage) c.VkBufferUsageFlags {
        return switch (self) {
            .Vertex => c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            .Uniform => c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        };
    }
};

const SyncObjects = struct {
    const Self = @This();

    img_available_semaphore: Semaphore,
    render_ended_semaphore: Semaphore,
    in_flight_fence: Fence,

    pub fn init(device: Device) !Self {
        return .{
            .img_available_semaphore = try Semaphore.init(device),
            .render_ended_semaphore = try Semaphore.init(device),
            .in_flight_fence = try Fence.init(device),
        };
    }

    pub fn deinit(self: *Self) void {
        self.img_available_semaphore.deinit();
        self.render_ended_semaphore.deinit();
        self.in_flight_fence.deinit();
    }
};

const Semaphore = struct {
    const Self = @This();

    handle: c.VkSemaphore = undefined,
    owner: c.VkDevice,

    pub fn init(device: Device) !Self {
        var self = Self{
            .owner = device.handle,
        };

        try checkVk(
            c.vkCreateSemaphore(device.handle, &.{}, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroySemaphore(self.owner, self.handle, null);
    }
};

const Fence = struct {
    const Self = @This();

    handle: c.VkFence = undefined,
    owner: c.VkDevice,

    pub fn init(device: Device) !Self {
        var self = Self{
            .owner = device.handle,
        };

        // Create signaled so first frame doesn't wait.
        // TODO: maybe accept bit as argument
        const fence_create_info = c.VkFenceCreateInfo{
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        };
        try checkVk(
            c.vkCreateFence(device.handle, &fence_create_info, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyFence(self.owner, self.handle, null);
    }
};

const Buffer = struct {
    const Self = @This();

    handle: c.VkBuffer = undefined,
    memory: c.VkDeviceMemory = undefined,
    usage: BufferUsage,
    owner: c.VkDevice,

    pub fn init(
        device: Device,
        size: c.VkDeviceSize,
        usage: BufferUsage,
        properties: c.VkMemoryPropertyFlags,
    ) !Self {
        var self = Self{
            .usage = usage,
            .owner = device.handle,
        };

        var buffer_info = c.VkBufferCreateInfo{
            .size = size,
            .usage = usage.getVkFlag(),
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };
        try checkVk(
            c.vkCreateBuffer(device.handle, &buffer_info, null, &self.handle),
        );

        var mem_reqs: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(device.handle, self.handle, &mem_reqs);

        const memory_type_index = try device.physical.findMemoryType(mem_reqs.memoryTypeBits, properties);
        var alloc_info = c.VkMemoryAllocateInfo{
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = memory_type_index,
        };

        try checkVk(
            c.vkAllocateMemory(device.handle, &alloc_info, null, &self.memory),
        );

        try checkVk(
            c.vkBindBufferMemory(device.handle, self.handle, self.memory, 0),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkFreeMemory(self.owner, self.memory, null);
        c.vkDestroyBuffer(self.owner, self.handle, null);
    }

    // pub fn getMappedSlice(self: Self) []Vertex {}
};

const DescriptorType = enum(u8) {
    Uniform,
    UniformDynamic,

    pub fn getVkType(@"type": DescriptorType) c_uint {
        return switch (@"type") {
            .Uniform => c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .UniformDynamic => c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
        };
    }
};

const DescriptorSetLayout = struct {
    const Self = @This();

    handle: c.VkDescriptorSetLayout = undefined,
    type: DescriptorType,
    owner: c.VkDevice,

    pub fn init(device: Device, @"type": DescriptorType) !Self {
        assert(device.handle != null);
        var self = Self{
            .owner = device.handle,
            .type = @"type",
        };
        var ubo_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = @"type".getVkType(),
            .descriptorCount = 1,
            // FIX: The UBO is used by both the vertex shader (for matrices) and potentially
            // the fragment shader (for color). We must specify all stages that access it.
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT | c.VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = null,
        };

        var layout_info = c.VkDescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &ubo_layout_binding,
        };

        try checkVk(
            c.vkCreateDescriptorSetLayout(device.handle, &layout_info, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyDescriptorSetLayout(self.owner, self.handle, null);
    }
};

// TODO: maybe remove on refactor if vocabulary type is unnecessary
const DescriptorSet = struct {
    handle: c.VkDescriptorSet = undefined,
    type: DescriptorType,
};

const DescriptorPool = struct {
    const Self = @This();

    handle: c.VkDescriptorPool = undefined,
    type: DescriptorType,
    owner: c.VkDevice,

    fn init(device: Device, layout: DescriptorSetLayout) !Self {
        assert(device.handle != null and layout.handle != null);
        var self = Self{
            .owner = device.handle,
            .type = layout.type,
        };

        var pool_size = c.VkDescriptorPoolSize{
            .type = layout.type.getVkType(),
            .descriptorCount = 1,
        };
        var pool_info = c.VkDescriptorPoolCreateInfo{
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
            .maxSets = 1,
        };

        try checkVk(
            c.vkCreateDescriptorPool(device.handle, &pool_info, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyDescriptorPool(self.owner, self.handle, null);
    }

    pub fn allocateDescriptorSet(self: Self, layout: DescriptorSetLayout) !DescriptorSet {
        assert(layout.handle != null);
        var set_alloc_info = c.VkDescriptorSetAllocateInfo{
            .descriptorPool = self.handle,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout.handle,
        };
        var descriptor_set: c.VkDescriptorSet = undefined;
        try checkVk(
            c.vkAllocateDescriptorSets(self.owner, &set_alloc_info, &descriptor_set),
        );

        return .{
            .handle = descriptor_set,
            .type = layout.type,
        };
    }

    /// Asserts that buffer is uniform type
    pub fn updateDescritorSets(
        self: *Self,
        buffer: Buffer,
        buffer_offset: u64,
        dst_set: DescriptorSet,
        ubo: type,
    ) void {
        //TODO: remove on ubo/ descriptor refactor
        assert(buffer.usage == .Uniform);

        var desc_buffer_info = c.VkDescriptorBufferInfo{
            .buffer = buffer.handle,
            .offset = buffer_offset,
            .range = @sizeOf(ubo),
        };

        var desc_write = c.VkWriteDescriptorSet{
            .dstSet = dst_set.handle,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = dst_set.type.getVkType(),
            .descriptorCount = 1,
            .pBufferInfo = &desc_buffer_info,
        };
        c.vkUpdateDescriptorSets(self.owner, 1, &desc_write, 0, null);
    }
};

const ShaderType = enum(u8) {
    Vertex,
    Fragment,
};

const ShaderModule = struct {
    const Self = @This();

    handle: c.VkShaderModule = undefined,
    owner: c.VkDevice,

    pub fn init(allocator: Allocator, device_handle: c.VkDevice, code: []const u8) !Self {
        var self = Self{ .owner = device_handle };
        std.debug.assert(code.len % 4 == 0);

        const aligned_code = try allocator.alignedAlloc(u32, @alignOf(u32), code.len / @sizeOf(u32));
        defer allocator.free(aligned_code);

        @memcpy(std.mem.sliceAsBytes(aligned_code), code);

        var create_info = c.VkShaderModuleCreateInfo{
            .codeSize = code.len,
            .pCode = aligned_code.ptr,
        };

        try checkVk(
            c.vkCreateShaderModule(device_handle, &create_info, null, &self.handle),
        );
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyShaderModule(self.owner, self.handle, null);
    }
};

const Shader = struct {
    const Self = @This();

    module: ShaderModule,
    type: ShaderType,

    pub fn init(module: ShaderModule, @"type": ShaderType) Self {
        return .{
            .module = module,
            .type = @"type",
        };
    }

    pub fn deinit(self: *Self) void {
        self.module.deinit();
    }

    pub fn info(self: Self) c.VkPipelineShaderStageCreateInfo {
        const stage = switch (self.type) {
            .Vertex => c.VK_SHADER_STAGE_VERTEX_BIT,
            .Fragment => c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        return .{
            .stage = @intCast(stage),
            .module = self.module.handle,
            .pName = "main",
        };
    }
};

const PipelineLayout = struct {
    const Self = @This();

    handle: c.VkPipelineLayout = undefined,
    owner: c.VkDevice,
    desc_set_layout: DescriptorSetLayout,

    pub fn init(device: Device, desc_set_layouts: DescriptorSetLayout) !Self {
        assert(device.handle != null and desc_set_layouts.handle != null);
        var self = Self{
            .owner = device.handle,
            .desc_set_layout = desc_set_layouts,
        };

        var pipeline_layout_info = c.VkPipelineLayoutCreateInfo{
            .setLayoutCount = 1,
            .pSetLayouts = &desc_set_layouts.handle,
        };

        try checkVk(
            c.vkCreatePipelineLayout(device.handle, &pipeline_layout_info, null, &self.handle),
        );

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyPipelineLayout(self.owner, self.handle, null);
    }
};

const Pipeline = struct {
    const Self = @This();

    handle: c.VkPipeline = undefined,
    owner: c.VkDevice,

    pub fn init(
        allocator: Allocator,
        device: Device,
        render_pass: RenderPass,
        layout: PipelineLayout,
        swapchain: Swapchain,
    ) !Self {
        assert(device.handle != null and render_pass.handle != null and layout.handle != null);
        var self = Self{
            .owner = device.handle,
        };

        var vert_shader = Shader.init(
            try ShaderModule.init(allocator, device.handle, vert_shader_code),
            .Vertex,
        );
        defer vert_shader.deinit();

        var frag_shader = Shader.init(
            try ShaderModule.init(allocator, device.handle, frag_shader_code),
            .Fragment,
        );
        defer frag_shader.deinit();

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{ vert_shader.info(), frag_shader.info() };

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

        var pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &color_blending,
            .layout = layout.handle,
            .renderPass = render_pass.handle,
            .subpass = 0,
        };

        try checkVk(c.vkCreateGraphicsPipelines(
            layout.owner,
            null,
            1,
            &pipeline_info,
            null,
            &self.handle,
        ));
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyPipeline(self.owner, self.handle, null);
    }
};

const Device = struct {
    const Self = @This();

    handle: c.VkDevice = undefined,
    graphics_queue: Queue = undefined,
    physical: *PhysicalDevice,

    pub fn init(physical_device: *PhysicalDevice) !Self {
        var self = Self{
            .graphics_queue = .{},
            .physical = physical_device,
        };

        const queue_priority: f32 = 1.0;
        const queue_create_info = c.VkDeviceQueueCreateInfo{
            .queueFamilyIndex = physical_device.q_family_idx,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        var device_features = c.VkPhysicalDeviceFeatures{};
        // We need to enable the swapchain extension.
        const device_extensions = [_][*:0]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        const create_info = c.VkDeviceCreateInfo{
            .pEnabledFeatures = &device_features,
            .pQueueCreateInfos = &queue_create_info,
            .queueCreateInfoCount = 1,
            .ppEnabledExtensionNames = &device_extensions,
            .enabledExtensionCount = device_extensions.len,
        };

        try checkVk(c.vkCreateDevice(
            physical_device.handle,
            &create_info,
            null,
            &self.handle,
        ));

        // Get a handle to the graphics queue.
        c.vkGetDeviceQueue(self.handle, physical_device.q_family_idx, 0, &self.graphics_queue.handle);

        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroyDevice(self.handle, null);
    }
};

const App = struct {
    const Self = @This();

    allocator: Allocator,
    scene: Scene,
    window: Window = undefined,
    instance: Instance = undefined,
    surface: Surface = undefined,
    physical_device: PhysicalDevice = undefined,
    device: Device = undefined,
    swapchain: Swapchain = undefined,
    vertex_buffer: Buffer = undefined,
    uniform_buffer: Buffer = undefined,
    descriptor_pool: DescriptorPool = undefined,
    render_pass: RenderPass = undefined,
    pipeline_layout: PipelineLayout = undefined,
    pipeline: Pipeline = undefined,
    descriptor_layout: DescriptorSetLayout = undefined,
    descriptor_set: DescriptorSet = undefined,
    command_pool: CommandPool = undefined,
    command_buffer: CommandBuffer = undefined,
    sync: SyncObjects = undefined,
    framebuffer_resized: bool = false,

    pub fn initWindow(self: *Self) !void {
        self.window = try Window.init(self, 800, 600, "legal", null, null);
    }

    pub fn initVulkan(self: *Self) !void {
        // Initialize instance
        self.instance = try Instance.init();

        // Initialize surface
        self.surface = try Surface.init(self.instance, self.window);

        // Initialize physical and logical devices
        self.physical_device = try PhysicalDevice.init(self.allocator, self.instance, self.surface);
        self.device = try Device.init(&self.physical_device);

        // Initialize swapchain
        self.swapchain = try Swapchain.init(self.allocator, self.device, self.surface);

        // Initialize descriptor set layout and pool
        self.descriptor_layout = try DescriptorSetLayout.init(self.device, .Uniform);
        self.descriptor_pool = try DescriptorPool.init(self.device, self.descriptor_layout);

        // Initialize descriptor sets
        self.descriptor_set = try self.descriptor_pool.allocateDescriptorSet(self.descriptor_layout);

        // Initialize buffers for the updateDescriptorSets call
        try self.initVertexBuffer();
        try self.initUniformBuffer();

        self.descriptor_pool.updateDescritorSets(
            self.uniform_buffer,
            0,
            self.descriptor_set,
            UniformBufferObject,
        );

        // Initialize render pass
        self.render_pass = try RenderPass.init(self.allocator, self.device, self.swapchain);

        // Initialize pipeline
        self.pipeline_layout = try PipelineLayout.init(self.device, self.descriptor_layout);
        self.pipeline = try Pipeline.init(
            self.allocator,
            self.device,
            self.render_pass,
            self.pipeline_layout,
            self.swapchain,
        );

        // Initialize command pool and buffer
        self.command_pool = try CommandPool.init(self.device);
        self.command_buffer = try self.command_pool.allocateCommandBuffer();

        self.sync = try SyncObjects.init(self.device);
    }

    pub fn deinit(self: *Self) void {
        self.scene.deinit(self.allocator);
        self.sync.deinit();
        self.command_pool.deinit();
        self.pipeline_layout.deinit();
        self.pipeline.deinit();
        self.render_pass.deinit(self.allocator);
        self.vertex_buffer.deinit();
        self.uniform_buffer.deinit();
        self.descriptor_layout.deinit();
        self.descriptor_pool.deinit(); // Needs to be the last descriptor resource destroyed
        self.swapchain.deinit(self.allocator);
        self.device.deinit();
        self.physical_device.deinit();
        self.surface.deinit();
        self.instance.deinit();
        self.window.deinit();
    }

    pub fn run(self: *Self) !void {
        while (c.glfwWindowShouldClose(self.window.handle) == 0) {
            c.glfwPollEvents();
            if (self.window.minimized()) {
                c.glfwWaitEvents();
                continue;
            }
            try self.draw();
        }
        // Wait for the GPU to finish all operations before we start cleaning up.
        try checkVk(
            c.vkDeviceWaitIdle(self.device.handle),
        );
    }

    fn draw(self: *Self) !void {
        // Wait for the previous frame's fence to be signaled (meaning it's done rendering).
        // TODO: Change timeout
        try checkVk(c.vkWaitForFences(
            self.device.handle,
            1,
            &self.sync.in_flight_fence.handle,
            c.VK_TRUE,
            std.math.maxInt(u64),
        ));

        try self.updateUniformBuffer();

        // Acquire an image from the swapchain.
        var image_index: u32 = 0;

        const acquire_result = c.vkAcquireNextImageKHR(
            self.device.handle,
            self.swapchain.handle,
            std.math.maxInt(u64),
            self.sync.img_available_semaphore.handle,
            null,
            &image_index,
        );

        if (acquire_result == c.VK_ERROR_OUT_OF_DATE_KHR) {
            // The swapchain is no longer compatible with the surface. Recreate it and try again next frame.
            try self.recreateSwapchain();
            return;
        } else if (acquire_result != c.VK_SUCCESS and acquire_result != c.VK_SUBOPTIMAL_KHR) {
            // For other errors, we propagate them.
            try checkVk(acquire_result);
        }

        try checkVk(
            c.vkResetFences(self.device.handle, 1, &self.sync.in_flight_fence.handle),
        );

        // Reset and record the command buffer.
        try checkVk(
            c.vkResetCommandBuffer(self.command_buffer.handle, 0),
        );
        try self.recordCommandBuffer(image_index);

        // Submit the command buffer to the graphics queue.
        const wait_semaphores = [_]c.VkSemaphore{
            self.sync.img_available_semaphore.handle,
        };
        const wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const signal_semaphores = [_]c.VkSemaphore{
            self.sync.render_ended_semaphore.handle,
        };

        const submit_info = c.VkSubmitInfo{
            .waitSemaphoreCount = wait_semaphores.len,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffer.handle,
            .signalSemaphoreCount = signal_semaphores.len,
            .pSignalSemaphores = &signal_semaphores,
        };
        try checkVk(c.vkQueueSubmit(
            self.device.graphics_queue.handle,
            1,
            &submit_info,
            self.sync.in_flight_fence.handle,
        ));

        // Present the image to the screen.
        const swapchains = [_]c.VkSwapchainKHR{self.swapchain.handle};
        const present_info = c.VkPresentInfoKHR{
            .waitSemaphoreCount = signal_semaphores.len,
            .pWaitSemaphores = &signal_semaphores,
            .swapchainCount = swapchains.len,
            .pSwapchains = &swapchains,
            .pImageIndices = &image_index,
        };

        const present_result = c.vkQueuePresentKHR(self.device.graphics_queue.handle, &present_info);

        if (present_result == c.VK_ERROR_OUT_OF_DATE_KHR or present_result == c.VK_SUBOPTIMAL_KHR) {
            self.framebuffer_resized = false;
            try self.recreateSwapchain();
        } else {
            try checkVk(present_result);
        }
    }

    /// Helper function
    fn initVertexBuffer(self: *Self) !void {
        // Calculate the total size needed for both axis and grid vertices.
        const total_vertex_count = self.scene.axis.len + self.scene.grid.len;
        const buffer_size = @sizeOf(Vertex) * total_vertex_count;

        if (buffer_size == 0) return; // Avoid creating a zero-size buffer

        self.vertex_buffer = try Buffer.init(
            self.device,
            buffer_size,
            .Vertex,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );

        // --- Copy vertex data to the buffer ---
        var data_ptr: ?*anyopaque = undefined;
        try checkVk(
            c.vkMapMemory(self.device.handle, self.vertex_buffer.memory, 0, buffer_size, 0, &data_ptr),
        );
        defer c.vkUnmapMemory(self.device.handle, self.vertex_buffer.memory);

        const many_item_ptr: [*]Vertex = @ptrCast(@alignCast(data_ptr.?));
        const mapped_vertex_slice = many_item_ptr[0..total_vertex_count];

        // Copy the axis vertices to the beginning of the buffer.
        @memcpy(mapped_vertex_slice[0..self.scene.axis.len], &self.scene.axis);

        // Copy the grid vertices immediately after the axis vertices.
        const grid_offset = self.scene.axis.len;
        @memcpy(mapped_vertex_slice[grid_offset .. grid_offset + self.scene.grid.len], self.scene.grid);
    }

    /// Helper function
    fn initUniformBuffer(self: *Self) !void {
        // FIX: The buffer size must match the UniformBufferObject struct, not the Vertex struct.
        const ubo_size = @sizeOf(UniformBufferObject);
        self.uniform_buffer = try Buffer.init(
            self.device,
            ubo_size,
            .Uniform,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
    }

    fn updateUniformBuffer(self: *Self) !void {
        const ubo = UniformBufferObject{
            .view_matrix = self.scene.view_matrix,
            .perspective_matrix = blk: {
                // FIX: This is the correct perspective matrix for Vulkan's coordinate system.
                //TODO: Internalize this better, maybe use a function to generate it.
                const fovy = std.math.degreesToRadians(45.0);
                // TODO: Get these values from input and served window dimentions
                const aspect: f32 = @as(f32, @floatFromInt(self.window.size.x)) / @as(f32, @floatFromInt(self.window.size.y));
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
        try checkVk(c.vkMapMemory(self.device.handle, self.uniform_buffer.memory, 0, @sizeOf(UniformBufferObject), 0, &ubo_data_ptr));
        const mapped_ubo: *UniformBufferObject = @ptrCast(@alignCast(ubo_data_ptr.?));
        mapped_ubo.* = ubo;
        c.vkUnmapMemory(self.device.handle, self.uniform_buffer.memory);
    }

    fn recreateSwapchain(self: *Self) !void {
        // Wait until the device is idle before we start destroying resources.
        try checkVk(c.vkDeviceWaitIdle(self.device.handle));

        // --- 1. Destroy old resources ---
        // Destroy in reverse order of creation.
        self.pipeline.deinit();
        self.pipeline_layout.deinit();
        self.render_pass.deinitFramebuffer(self.allocator);
        self.swapchain.deinit(self.allocator);

        // --- 2. Recreate resources with new properties ---
        self.swapchain = try Swapchain.init(self.allocator, self.device, self.surface);
        try self.render_pass.initFrameBuffer(self.allocator, self.swapchain);
        self.pipeline_layout = try PipelineLayout.init(self.device, self.descriptor_layout);
        self.pipeline = try Pipeline.init(
            self.allocator,
            self.device,
            self.render_pass,
            self.pipeline_layout,
            self.swapchain,
        );
    }

    pub fn recordCommandBuffer(self: *Self, image_index: u32) !void {
        const begin_info = c.VkCommandBufferBeginInfo{};
        try checkVk(
            c.vkBeginCommandBuffer(self.command_buffer.handle, &begin_info),
        );

        const clear_color = c.VkClearValue{ .color = .{ .float32 = .{ 0.1, 0.1, 0.1, 1.0 } } };

        const render_pass_info = c.VkRenderPassBeginInfo{
            .renderPass = self.render_pass.handle,
            .framebuffer = self.render_pass.framebuffer[image_index],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swapchain.extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };
        c.vkCmdBeginRenderPass(
            self.command_buffer.handle,
            &render_pass_info,
            c.VK_SUBPASS_CONTENTS_INLINE,
        );

        // Bind the graphics pipeline.
        c.vkCmdBindPipeline(
            self.command_buffer.handle,
            c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline.handle,
        );

        // Bind the vertex buffer.
        const vertex_buffers = [_]c.VkBuffer{self.vertex_buffer.handle};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(self.command_buffer.handle, 0, 1, &vertex_buffers, &offsets);
        c.vkCmdBindDescriptorSets(
            self.command_buffer.handle,
            c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            self.pipeline_layout.handle,
            0,
            1,
            &self.descriptor_set.handle,
            0,
            null,
        );

        c.vkCmdDraw(self.command_buffer.handle, @intCast(self.scene.axis.len + self.scene.grid.len), 1, 0, 0);

        c.vkCmdEndRenderPass(self.command_buffer.handle);
        try checkVk(c.vkEndCommandBuffer(self.command_buffer.handle));
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

    var app = try allocator.create(App);
    defer allocator.destroy(app);

    app.* = .{
        .allocator = allocator,
        .scene = try Scene.init(allocator, 20),
    };

    defer app.deinit();

    try app.initWindow();
    try app.initVulkan();

    try app.run();
}
