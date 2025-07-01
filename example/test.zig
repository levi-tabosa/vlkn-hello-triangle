// test.zig
const std = @import("std");
const assert = std.debug.assert;
const spirv = @import("spirv");
const scene = @import("geometry");
const gui = @import("./gui/gui.zig");
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

const UniformBufferObject = extern struct {
    view_matrix: [16]f32,
    perspective_matrix: [16]f32,
    padding: [128]u8 = undefined,
};

// --- Callbacks and Window ---
const Callbacks = struct {
    const Self = @This();
    fn cbCursorPos(wd: ?*c.GLFWwindow, xpos: f64, ypos: f64) callconv(.C) void {
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        const app: *App = @alignCast(@ptrCast(user_ptr));

        const ndc_x = @as(f32, @floatCast(xpos)) / @as(f32, @floatFromInt(WINDOW_WIDTH)) * 2.0 - 1.0;
        const ndc_y = @as(f32, @floatCast(ypos)) / @as(f32, @floatFromInt(WINDOW_HEIGHT)) * 2.0 - 1.0;

        const pitch = -ndc_y * std.math.pi / 2.0;
        const yaw = ndc_x * std.math.pi;

        app.scene.setPitchYaw(pitch, yaw);
        app.updateUniformBuffer() catch unreachable;
        app.gui_ctx.handleCursorPos(xpos, ypos);
    }

    fn cbMouseButton(wd: ?*c.GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.C) void {
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        const app: *App = @alignCast(@ptrCast(user_ptr));
        app.gui_ctx.handleMouseButton(button, action, mods);
    }

    fn cbKey(wd: ?*c.GLFWwindow, char: c_int, code: c_int, btn: c_int, mods: c_int) callconv(.C) void {
        _ = char;
        _ = btn;
        _ = mods;
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        const app: *App = @alignCast(@ptrCast(user_ptr));

        app.scene.setGridResolution(@intCast(code)) catch unreachable;
        // app.vertex_buffer.deinit(app.vk_ctx);
        app.updateVertexBuffer() catch unreachable;
    }

    fn cbFramebufferResize(wd: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.C) void {
        const user_ptr = c.glfwGetWindowUserPointer(wd orelse return) orelse @panic("No window user ptr");
        const app: *App = @alignCast(@ptrCast(user_ptr));
        app.window.size.x = width;
        app.window.size.y = height;

        app.framebuffer_resized = true;
        app.updateUniformBuffer() catch @panic("resize callback error on update UB");
    }
};

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
        const handle = c.glfwCreateWindow(width, height, title, monitor, share);
        if (handle == null) return error.GlfwCreateWindowFailed;
        c.glfwSetWindowUserPointer(handle, user_ptr);
        _ = c.glfwSetCursorPosCallback(handle, Callbacks.cbCursorPos);
        _ = c.glfwSetMouseButtonCallback(handle, Callbacks.cbMouseButton);
        _ = c.glfwSetKeyCallback(handle, Callbacks.cbKey);
        _ = c.glfwSetFramebufferSizeCallback(handle, Callbacks.cbFramebufferResize);
        return .{ .handle = handle, .size = .{ .x = width, .y = height } };
    }

    pub fn deinit(self: *Self) void {
        c.glfwDestroyWindow(self.handle);
        self.handle = undefined;
    }

    pub fn minimized(self: Self) bool {
        return c.glfwGetWindowAttrib(self.handle, c.GLFW_ICONIFIED) == 1;
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
    pub fn deinit(_: *Self) void {} //TODO: remove

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
        self.physical_device.deinit();
        self.surface.deinit(self);
        self.instance.deinit();
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

    pub fn init(vk_ctx: *VulkanContext) !Self {
        var self = Self{ .owner = vk_ctx.device.handle };

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

pub const RenderPass = struct {
    const Self = @This();
    handle: c.VkRenderPass = undefined,
    framebuffer: []c.VkFramebuffer = undefined,
    owner: c.VkDevice,

    pub fn init(vk_ctx: *VulkanContext, swapchain: *Swapchain) !Self {
        var self = Self{ .owner = vk_ctx.device.handle };

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
        try vkCheck(c.vkCreateRenderPass(vk_ctx.device.handle, &create_info, null, &self.handle));

        try self.initFrameBuffer(vk_ctx, swapchain);
        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        self.deinitFramebuffer(vk_ctx.allocator);
        c.vkDestroyRenderPass(self.owner, self.handle, null);
    }

    pub fn initFrameBuffer(self: *Self, vk_ctx: *VulkanContext, swapchain: *Swapchain) !void {
        self.framebuffer = try vk_ctx.allocator.alloc(c.VkFramebuffer, swapchain.image_views.len);
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
        var buffer_info = c.VkBufferCreateInfo{ .size = size, .usage = usage, .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE };
        try vkCheck(c.vkCreateBuffer(vk_ctx.device.handle, &buffer_info, null, &self.handle));

        var mem_reqs: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(vk_ctx.device.handle, self.handle, &mem_reqs);
        const mem_type_index = try vk_ctx.physical_device.findMemoryType(mem_reqs.memoryTypeBits, properties);
        var alloc_info = c.VkMemoryAllocateInfo{ .allocationSize = mem_reqs.size, .memoryTypeIndex = mem_type_index };
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

        const submit_info = c.VkSubmitInfo{ .commandBufferCount = 1, .pCommandBuffers = &command_buffer };
        try vkCheck(c.vkQueueSubmit(vk_ctx.graphics_queue.handle, 1, &submit_info, null));
        try vkCheck(c.vkQueueWaitIdle(vk_ctx.graphics_queue.handle));
    }
};

pub const PushConstantRange = struct {
    const Self = @This();

    handle: c.VkPushConstantRange,

    pub fn init(T: type) Self {
        return .{ .handle = .{
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT | c.VK_SHADER_STAGE_FRAGMENT_BIT,
            .offset = 0,
            .size = @sizeOf(T),
        } };
    }
};

const DescriptorType = enum(u8) {
    Uniform,
    pub fn getVkType(@"type": DescriptorType) c_uint {
        return switch (@"type") {
            .Uniform => c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        };
    }
};

const DescriptorSetLayout = struct {
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

const DescriptorPool = struct {
    const Self = @This();
    handle: c.VkDescriptorPool = undefined,

    pub fn init(vk_ctx: *VulkanContext, @"type": DescriptorType) !Self {
        var self = Self{};
        var pool_size = c.VkDescriptorPoolSize{ .type = @"type".getVkType(), .descriptorCount = 1 };
        var pool_info = c.VkDescriptorPoolCreateInfo{ .poolSizeCount = 1, .pPoolSizes = &pool_size, .maxSets = 1 };
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
    owner: c.VkDevice,

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
        desc_set_layout: ?DescriptorSetLayout,
        push_consts_range: ?PushConstantRange,
    ) !Self {
        assert(desc_set_layout != null or push_consts_range != null);
        var self = Self{};
        var pipeline_layout_info = if (desc_set_layout) |l|
            c.VkPipelineLayoutCreateInfo{
                .setLayoutCount = 1,
                .pSetLayouts = &l.handle,
            }
        else
            c.VkPipelineLayoutCreateInfo{
                .pushConstantRangeCount = 1,
                .pPushConstantRanges = &push_consts_range.?.handle,
            };
        try vkCheck(c.vkCreatePipelineLayout(vk_ctx.device.handle, &pipeline_layout_info, null, &self.handle));
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
        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{ .topology = c.VK_PRIMITIVE_TOPOLOGY_LINE_LIST, .primitiveRestartEnable = c.VK_FALSE };
        const viewport = c.VkViewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(swapchain.extent.width),
            .height = @floatFromInt(swapchain.extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };
        const scissor = c.VkRect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = swapchain.extent };
        var viewport_state = c.VkPipelineViewportStateCreateInfo{ .viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor };
        var rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_NONE,
            .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = c.VK_FALSE,
        };
        var multisampling = c.VkPipelineMultisampleStateCreateInfo{ .sampleShadingEnable = c.VK_FALSE, .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT };
        var color_blend_attachment = c.VkPipelineColorBlendAttachmentState{ .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT, .blendEnable = c.VK_FALSE };
        var color_blending = c.VkPipelineColorBlendStateCreateInfo{ .logicOpEnable = c.VK_FALSE, .attachmentCount = 1, .pAttachments = &color_blend_attachment };
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

        try vkCheck(c.vkCreateGraphicsPipelines(vk_ctx.device.handle, null, 1, &pipeline_info, null, &self.handle));
        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyPipeline(vk_ctx.device.handle, self.handle, null);
    }
};

// --- MAIN APPLICATION STRUCT ---
const App = struct {
    const Self = @This();
    allocator: Allocator,
    scene: Scene,
    window: Window,
    vk_ctx: *VulkanContext,
    gui_ctx: gui.GuiContext,

    // Vulkan objects that depend on the swapchain (and are recreated)
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

    pub fn init(allocator: Allocator) !*Self {
        const window = try Window.init(null, WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan Line App", null, null);
        const vk_ctx = try allocator.create(VulkanContext);
        vk_ctx.* = try VulkanContext.init(allocator, window);

        const app = try allocator.create(App);
        app.allocator = allocator;
        app.window = window;
        app.vk_ctx = vk_ctx;
        app.scene = try Scene.init(allocator, 20);

        try app.initVulkanResources();

        c.glfwSetWindowUserPointer(app.window.handle, app);
        return app;
    }

    // Initialize Vulkan resources after the context is created
    fn initVulkanResources(self: *Self) !void {
        self.swapchain = try Swapchain.init(self.vk_ctx);
        self.descriptor_layout = try DescriptorSetLayout.init(self.vk_ctx, .Uniform);
        self.descriptor_pool = try DescriptorPool.init(self.vk_ctx, .Uniform);
        self.descriptor_set = try self.descriptor_pool.allocateSet(self.vk_ctx, self.descriptor_layout);
        self.command_buffer = try CommandBuffer.allocate(self.vk_ctx, true);
        self.sync = try SyncObjects.init(self.vk_ctx);

        try self.initVertexBuffer();
        try self.initUniformBuffer();

        self.descriptor_pool.updateSet(self.vk_ctx, self.descriptor_set, self.uniform_buffer, UniformBufferObject);

        self.render_pass = try RenderPass.init(self.vk_ctx, &self.swapchain);
        self.pipeline_layout = try PipelineLayout.init(self.vk_ctx, self.descriptor_layout, null);
        self.pipeline = try Pipeline.init(self.vk_ctx, self.render_pass, self.pipeline_layout, self.swapchain);

        // Initialize GUI at the end
        self.gui_ctx = try gui.GuiContext.init(self.vk_ctx, self.render_pass);
    }

    pub fn deinit(self: *Self) void {
        // Wait for device to be idle before cleaning up
        _ = c.vkDeviceWaitIdle(self.vk_ctx.device.handle);

        self.gui_ctx.deinit(); // Deinit GUI resources
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
        self.gui_ctx.destroyPipeline();
        self.pipeline.deinit(self.vk_ctx);
        self.pipeline_layout.deinit(self.vk_ctx);
        self.render_pass.deinit(self.vk_ctx);
        self.swapchain.deinit(self.vk_ctx);
    }

    pub fn run(self: *Self) !void {
        while (c.glfwWindowShouldClose(self.window.handle) == 0) {
            c.glfwPollEvents();
            self.gui_ctx.beginFrame();
            if (self.window.minimized()) {
                c.glfwWaitEvents();
                continue;
            }
            // Draw the UI Widgets.
            if (self.gui_ctx.button(10, 10, 150, 30)) {
                var prng = rand.DefaultPrng.init(@intCast(std.time.milliTimestamp()));
                const random = prng.random();
                try self.scene.addVector(
                    .{ 0, 0, 0 },
                    .{
                        random.float(f32) * 2.0 - 1.0,
                        random.float(f32) * 2.0 - 1.0,
                        random.float(f32) * 2.0 - 1.0,
                    },
                );
                try self.updateVertexBuffer();
            }
            if (self.gui_ctx.button(10, 50, 150, 30)) {
                // Example: Quit the application
                c.glfwSetWindowShouldClose(self.window.handle, 1);
            }
            // self.gui_ctx.draw();
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
        // c.vkWa
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
        const fovy = std.math.degreesToRadians(90.0);
        const aspect = @as(f32, @floatFromInt(self.window.size.x)) / @as(f32, @floatFromInt(self.window.size.y));
        const near = 0.1;
        const far = 100.0;
        const f = 1.0 / std.math.tan(fovy / 2.0);
        const ubo = UniformBufferObject{
            .view_matrix = self.scene.view_matrix,
            .perspective_matrix = .{
                f / aspect, 0,  0,                           0,
                0,          -f, 0,                           0,
                0,          0,  far / (near - far),          -1,
                0,          0,  (far * near) / (near - far), 0,
            },
        };

        const data_ptr = try self.uniform_buffer.map(self.vk_ctx, UniformBufferObject);
        defer self.uniform_buffer.unmap(self.vk_ctx);
        data_ptr[0] = ubo;
    }

    fn recreateSwapchain(self: *Self) !void {
        try vkCheck(c.vkDeviceWaitIdle(self.vk_ctx.device.handle));

        self.cleanupSwapchain(); // This now correctly calls gui_context.destroyPipeline()

        self.swapchain = try Swapchain.init(self.vk_ctx);
        self.render_pass = try RenderPass.init(self.vk_ctx, &self.swapchain);
        self.pipeline_layout = try PipelineLayout.init(self.vk_ctx, self.descriptor_layout, null);
        self.pipeline = try Pipeline.init(self.vk_ctx, self.render_pass, self.pipeline_layout, self.swapchain);

        // Recreate ONLY the GUI pipeline with the new render pass
        try self.gui_ctx.createPipeline(self.render_pass);
    }

    pub fn recordCommandBuffer(self: *Self, image_index: u32) !void {
        const begin_info = c.VkCommandBufferBeginInfo{};
        try vkCheck(c.vkBeginCommandBuffer(self.command_buffer.handle, &begin_info));

        const clear_color = c.VkClearValue{ .color = .{ .float32 = .{ 0.1, 0.1, 0.1, 1.0 } } };
        const render_pass_info = c.VkRenderPassBeginInfo{
            .renderPass = self.render_pass.handle,
            .framebuffer = self.render_pass.framebuffer[image_index],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = self.swapchain.extent },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };
        c.vkCmdBeginRenderPass(self.command_buffer.handle, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdBindPipeline(self.command_buffer.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.handle);
        const vertex_buffers = [_]c.VkBuffer{self.vertex_buffer.handle};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(self.command_buffer.handle, 0, 1, &vertex_buffers, &offsets);
        c.vkCmdBindDescriptorSets(self.command_buffer.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, 1, &self.descriptor_set, 0, null);
        c.vkCmdDraw(self.command_buffer.handle, @intCast(self.scene.getTotalVertexCount()), 1, 0, 0);
        self.gui_ctx.endFrame(
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

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var app = try App.init(allocator);
    defer allocator.destroy(app);
    defer app.deinit();

    try app.run();
}
