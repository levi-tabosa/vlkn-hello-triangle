// vulkan.zig — Vulkan primitive types and context.
// Extracted from example/test.zig.
const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

pub const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("vulkan/vulkan.h");
    @cInclude("GLFW/glfw3.h");
});

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

pub fn checkGlfw(result: c_int) !void {
    if (result == c.GLFW_TRUE) return;

    std.log.err("Glfw call failed with code: {}", .{result});
    return switch (result) {
        c.GLFW_PLATFORM_ERROR => error.GlfwPlatformError,
        else => error.GlfwDefault,
    };
}

pub fn getVertexBindingDescription(V: type) c.VkVertexInputBindingDescription {
    return .{
        .binding = 0,
        .stride = @sizeOf(V),
        .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
    };
}

pub fn getVertexAttributeDescriptions(V: type) [3]c.VkVertexInputAttributeDescription {
    return .{ .{
        .binding = 0,
        .location = 0,
        .format = c.VK_FORMAT_R32G32B32_SFLOAT,
        .offset = @offsetOf(V, "coor"),
    }, .{
        .binding = 0,
        .location = 1,
        .format = c.VK_FORMAT_R32G32B32_SFLOAT,
        .offset = @offsetOf(V, "offset"),
    }, .{
        .binding = 0,
        .location = 2,
        .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
        .offset = @offsetOf(V, "color"),
    } };
}

// --- Vocabulary Types ---

pub const Instance = struct {
    const Self = @This();

    handle: c.VkInstance = undefined,

    pub fn init() !Self {
        var self = Instance{};
        const app_info = c.VkApplicationInfo{
            .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
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
            .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
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

pub const Surface = struct {
    const Self = @This();

    handle: c.VkSurfaceKHR = undefined,

    pub fn init(instance: Instance, window_handle: *c.GLFWwindow) !Self {
        assert(instance.handle != null);
        var self = Self{};
        try vkCheck(c.glfwCreateWindowSurface(instance.handle, window_handle, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroySurfaceKHR(vk_ctx.instance.handle, self.handle, null);
    }
};

pub const PhysicalDevice = struct {
    const Self = @This();

    handle: c.VkPhysicalDevice = undefined,
    graphics_q_family: u32 = undefined,
    present_q_family: u32 = undefined,
    transfer_q_family: u32 = undefined,

    pub fn init(allocator: Allocator, instance: Instance, surface: Surface) !Self {
        assert(instance.handle != null and surface.handle != null);
        var self = Self{};

        // Enumerate physical devices
        var physical_device_count: u32 = 0;
        try vkCheck(c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, null));
        if (physical_device_count == 0) return error.NoPhysicalDevicesFound;
        const physical_devices = try allocator.alloc(c.VkPhysicalDevice, physical_device_count);
        defer allocator.free(physical_devices);
        try vkCheck(c.vkEnumeratePhysicalDevices(instance.handle, &physical_device_count, physical_devices.ptr));

        // Select the best physical device based on a scoring system
        var best_device: ?c.VkPhysicalDevice = null;
        var best_score: i32 = -1;

        for (physical_devices) |device| {
            var properties: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(device, &properties);

            // Score devices: prefer discrete GPUs, then integrated, etc.
            var score: i32 = 0;
            switch (properties.deviceType) {
                c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU => score += 1000,
                c.VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU => score += 500,
                else => score += 100, // Other types (e.g., virtual, CPU) get lower priority
            }

            // Add more criteria here if needed (e.g., required features, memory size)
            if (score > best_score) {
                best_device = device;
                best_score = score;
            }
        }

        if (best_device == null) return error.NoSuitablePhysicalDevice;
        self.handle = best_device.?;

        // Get queue family properties
        var q_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_count, null);
        const q_family_props = try allocator.alloc(c.VkQueueFamilyProperties, q_count);
        defer allocator.free(q_family_props);
        c.vkGetPhysicalDeviceQueueFamilyProperties(self.handle, &q_count, q_family_props.ptr);

        var graphics_idx: ?u32 = null;
        var present_idx: ?u32 = null;
        var transfer_idx: ?u32 = null;

        // Find a dedicated transfer queue
        for (q_family_props, 0..) |prop, i| {
            const idx: u32 = @intCast(i);
            if ((prop.queueFlags & c.VK_QUEUE_TRANSFER_BIT != 0) and (prop.queueFlags & c.VK_QUEUE_GRAPHICS_BIT == 0)) {
                transfer_idx = idx;
                break; // Found the ideal one, stop searching
            }
        }

        // Find graphics, present, and a fallback transfer queue
        for (q_family_props, 0..) |prop, i| {
            const idx: u32 = @intCast(i);

            // Graphics
            if (prop.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                if (graphics_idx == null) graphics_idx = idx;
            }

            // Present queue
            var support: c.VkBool32 = c.VK_FALSE;
            try vkCheck(c.vkGetPhysicalDeviceSurfaceSupportKHR(self.handle, idx, surface.handle, &support));
            if (support == c.VK_TRUE) {
                if (present_idx == null) {
                    present_idx = idx;
                }
            }

            // If we didn't find a dedicated transfer queue, use any queue that supports transfer
            if (transfer_idx == null and (prop.queueFlags & c.VK_QUEUE_TRANSFER_BIT != 0)) {
                transfer_idx = idx;
            }

            if (graphics_idx != null and present_idx != null and transfer_idx != null) {
                break;
            }
        }

        if (graphics_idx == null) return error.NoGraphicsQueueFamily;
        if (present_idx == null) return error.NoPresentQueueFamily;
        if (transfer_idx == null) return error.NoTransferQueueFamily;

        self.graphics_q_family = graphics_idx.?;
        self.present_q_family = present_idx.?;
        self.transfer_q_family = transfer_idx.?;

        // Log the queue info
        std.log.info("Queue Families Found: Graphics={d}, Present={d}, Transfer={d}", .{ self.graphics_q_family, self.present_q_family, self.transfer_q_family });

        return self;
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

pub const Device = struct {
    const Self = @This();

    handle: c.VkDevice = undefined,
    physical: PhysicalDevice,

    pub fn init(allocator: Allocator, physical_device: PhysicalDevice) !Self {
        var self = Self{ .physical = physical_device };
        const queue_priority: f32 = 1.0;

        // Use a BitSet to find the unique queue family indices
        var unique_queue_families: std.bit_set.StaticBitSet(3) = .initEmpty();
        unique_queue_families.set(physical_device.graphics_q_family);
        unique_queue_families.set(physical_device.present_q_family);
        unique_queue_families.set(physical_device.transfer_q_family);

        // Create a VkDeviceQueueCreateInfo for each unique family
        var queue_create_infos = try std.ArrayList(c.VkDeviceQueueCreateInfo).initCapacity(allocator, unique_queue_families.count());
        defer queue_create_infos.deinit(allocator);

        var it = unique_queue_families.iterator(.{});
        while (it.next()) |family_index| {
            try queue_create_infos.append(allocator, .{
                .sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                .queueFamilyIndex = @intCast(family_index),
                .queueCount = 1,
                .pQueuePriorities = &queue_priority,
            });
        }

        var device_features = c.VkPhysicalDeviceFeatures{};
        const device_extensions = [_][*:0]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        const create_info = c.VkDeviceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pEnabledFeatures = &device_features,
            .pQueueCreateInfos = queue_create_infos.items.ptr,
            .queueCreateInfoCount = @intCast(queue_create_infos.items.len),
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

pub const Queue = struct {
    const Self = @This();

    handle: c.VkQueue = undefined,

    /// The real queue is managed by the Api so no need for deinit call
    pub fn init(device: Device, family_index: u32) !Self {
        var self = Self{};
        c.vkGetDeviceQueue(device.handle, family_index, 0, &self.handle);
        return self;
    }

    pub fn submit(self: *const Self, command_buffer: CommandBuffer, fence: c.VkFence) !void {
        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer.handle,
        };

        try vkCheck(c.vkQueueSubmit(self.handle, 1, &submit_info, fence));
    }

    pub fn waitIdle(self: Self) !void {
        try vkCheck(c.vkQueueWaitIdle(self.handle));
    }
};

// TODO: Improve this abstraction to be used in the UI render logic
pub const CommandBuffer = struct {
    const Self = @This();
    handle: c.VkCommandBuffer = undefined,

    pub fn allocate(vk_ctx: *VulkanContext, command_pool: CommandPool, is_primary: bool) !Self {
        var self = Self{};
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .commandPool = command_pool.handle,
            .level = if (is_primary) c.VK_COMMAND_BUFFER_LEVEL_PRIMARY else c.VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            .commandBufferCount = 1,
        };
        try vkCheck(c.vkAllocateCommandBuffers(vk_ctx.device.handle, &alloc_info, &self.handle));
        return self;
    }

    pub fn free(self: *const Self, vk_ctx: *VulkanContext, command_pool: CommandPool) void {
        c.vkFreeCommandBuffers(vk_ctx.device.handle, command_pool.handle, 1, &self.handle);
    }

    pub fn begin(self: *Self, flags: c.VkCommandBufferUsageFlags) !void {
        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = flags,
        };
        try vkCheck(c.vkBeginCommandBuffer(self.handle, &begin_info));
    }

    pub fn end(self: *Self) !void {
        try vkCheck(c.vkEndCommandBuffer(self.handle));
    }

    pub fn beginSingleTimeCommands(vk_ctx: *VulkanContext, command_pool: CommandPool, is_primary: bool) !Self {
        var self = Self{};

        const alloc_info = c.VkCommandBufferAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .level = if (is_primary) c.VK_COMMAND_BUFFER_LEVEL_PRIMARY else c.VK_COMMAND_BUFFER_LEVEL_SECONDARY,
            .commandPool = command_pool.handle,
            .commandBufferCount = 1,
        };
        try vkCheck(c.vkAllocateCommandBuffers(vk_ctx.device.handle, &alloc_info, &self.handle));

        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        try vkCheck(c.vkBeginCommandBuffer(self.handle, &begin_info));

        return self;
    }

    pub fn endSingleTimeCommands(self: Self, vk_ctx: *VulkanContext, command_pool: CommandPool, queue: Queue) !void {
        try vkCheck(c.vkEndCommandBuffer(self.handle));

        try queue.submit(self, null);
        try queue.waitIdle();
        self.free(vk_ctx, command_pool);
    }
};

pub const CommandPool = struct {
    const Self = @This();

    handle: c.VkCommandPool = undefined,

    pub fn init(device: Device, queue_family_index: u32) !Self {
        var self = Self{};
        var cmd_pool_info = c.VkCommandPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_index,
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
            .sType = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
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
        errdefer vk_ctx.allocator.free(self.images);
        try vkCheck(c.vkGetSwapchainImagesKHR(vk_ctx.device.handle, self.handle, &img_count, self.images.ptr));

        self.image_views = try vk_ctx.allocator.alloc(c.VkImageView, image_count);
        errdefer vk_ctx.allocator.free(self.image_views);
        for (self.images, 0..) |image, i| {
            const info = c.VkImageViewCreateInfo{
                .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
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
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
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
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
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
        const command_buffer = try CommandBuffer.beginSingleTimeCommands(vk_ctx, vk_ctx.command_pool, true);

        var barrier = c.VkImageMemoryBarrier{
            .sType = c.VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER,
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

        try command_buffer.endSingleTimeCommands(vk_ctx, vk_ctx.command_pool, vk_ctx.graphics_queue);
    }

    /// Copies data from a buffer to this image.
    pub fn copyFromBuffer(self: Image, vk_ctx: *VulkanContext, buffer: Buffer) !void {
        const command_buffer = try CommandBuffer.beginSingleTimeCommands(vk_ctx, vk_ctx.transfer_command_pool, true);

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

        try command_buffer.endSingleTimeCommands(vk_ctx, vk_ctx.transfer_command_pool, vk_ctx.transfer_queue);
    }

    pub fn hasDepthComponent(self: Image) bool {
        return self.format == c.VK_FORMAT_D32_SFLOAT or self.format == c.VK_FORMAT_D32_SFLOAT_S8_UINT or self.format == c.VK_FORMAT_D24_UNORM_S8_UINT;
    }
};

pub const DepthBuffer = struct {
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
        // try image.transitionLayout(vk_ctx, c.VK_IMAGE_LAYOUT_UNDEFINED, c.VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

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
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
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
                .sType = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
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

pub const SyncObjects = struct {
    const Self = @This();
    img_available_s: c.VkSemaphore = undefined,
    render_ended_s: c.VkSemaphore = undefined,
    in_flight_f: c.VkFence = undefined,

    pub fn init(vk_ctx: *VulkanContext) !Self {
        var self: Self = .{};
        const sem_info = c.VkSemaphoreCreateInfo{ .sType = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        try vkCheck(c.vkCreateSemaphore(vk_ctx.device.handle, &sem_info, null, &self.img_available_s));
        try vkCheck(c.vkCreateSemaphore(vk_ctx.device.handle, &sem_info, null, &self.render_ended_s));

        const fence_create_info = c.VkFenceCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        };
        try vkCheck(c.vkCreateFence(vk_ctx.device.handle, &fence_create_info, null, &self.in_flight_f));
        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroySemaphore(vk_ctx.device.handle, self.img_available_s, null);
        c.vkDestroySemaphore(vk_ctx.device.handle, self.render_ended_s, null);
        c.vkDestroyFence(vk_ctx.device.handle, self.in_flight_f, null);
    }
};

pub const Buffer = struct {
    const Self = @This();

    handle: c.VkBuffer = undefined,
    memory: c.VkDeviceMemory = undefined,
    size: u64,

    pub fn init(vk_ctx: *VulkanContext, size: u64, usage: c.VkBufferUsageFlags, properties: c.VkMemoryPropertyFlags) !Self {
        var self = Self{
            .size = size,
        };

        var buffer_info = c.VkBufferCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .size = size,
            .usage = usage,
            // Not shared between queues
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };

        if (vk_ctx.physical_device.graphics_q_family != vk_ctx.physical_device.transfer_q_family) {
            const queue_family_indices = [_]u32{
                vk_ctx.physical_device.graphics_q_family,
                vk_ctx.physical_device.transfer_q_family,
            };
            buffer_info.sharingMode = c.VK_SHARING_MODE_CONCURRENT;
            buffer_info.queueFamilyIndexCount = queue_family_indices.len;
            buffer_info.pQueueFamilyIndices = &queue_family_indices;
        }
        try vkCheck(c.vkCreateBuffer(vk_ctx.device.handle, &buffer_info, null, &self.handle));

        var mem_reqs: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(vk_ctx.device.handle, self.handle, &mem_reqs);
        const mem_type_index = try vk_ctx.physical_device.findMemoryType(mem_reqs.memoryTypeBits, properties);
        var alloc_info = c.VkMemoryAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
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

    pub fn copyTo(self: *Self, vk_ctx: *VulkanContext, dst_buffer: Buffer) !void {
        var cmdbuffer = try CommandBuffer.allocate(vk_ctx, vk_ctx.transfer_command_pool, true);

        defer cmdbuffer.free(vk_ctx, vk_ctx.transfer_command_pool);

        try cmdbuffer.begin(c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);

        const copy_region = c.VkBufferCopy{ .size = self.size };
        c.vkCmdCopyBuffer(cmdbuffer.handle, self.handle, dst_buffer.handle, 1, &copy_region);

        try cmdbuffer.end();
        try vk_ctx.graphics_queue.submit(cmdbuffer, null);

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

pub const DescriptorType = enum(u8) {
    Uniform,
    CombinedImageSampler,
    Storage,
    pub fn getVkType(@"type": DescriptorType) c_uint {
        return switch (@"type") {
            .Uniform => c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .CombinedImageSampler => c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .Storage => c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        };
    }
};

pub const DescriptorSetLayoutArgs = struct { type: c.VkDescriptorType, stage_flags: c.VkShaderStageFlags };

pub const DescriptorSetLayout = struct {
    const Self = @This();

    handle: c.VkDescriptorSetLayout = undefined,

    pub fn init(vk_ctx: *VulkanContext, args: []const DescriptorSetLayoutArgs) !Self {
        var self = Self{};
        const bindings = try vk_ctx.allocator.alloc(c.VkDescriptorSetLayoutBinding, args.len);
        defer vk_ctx.allocator.free(bindings);

        for (args, 0..) |arg, i| {
            bindings[i] = .{
                .binding = @intCast(i),
                .descriptorType = arg.type,
                .descriptorCount = 1,
                .stageFlags = arg.stage_flags,
            };
        }

        var layout_info = c.VkDescriptorSetLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = @intCast(bindings.len),
            .pBindings = bindings.ptr,
        };
        try vkCheck(c.vkCreateDescriptorSetLayout(vk_ctx.device.handle, &layout_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyDescriptorSetLayout(vk_ctx.device.handle, self.handle, null);
    }
};

// Expressive union for descriptor info
pub const DescriptorWriteInfo = union(enum) {
    UniformBuffer: c.VkDescriptorBufferInfo,
    StorageBuffer: c.VkDescriptorBufferInfo,
    CombinedImageSampler: c.VkDescriptorImageInfo,
};

pub const DescriptorWrite = struct {
    binding: u32,
    info: DescriptorWriteInfo,
};

pub const DescriptorSet = struct {
    handle: c.VkDescriptorSet = undefined,

    pub fn update(
        self: *const DescriptorSet,
        vk_ctx: *VulkanContext,
        writes: []const DescriptorWrite,
    ) !void {
        const desc_writes = try vk_ctx.allocator.alloc(c.VkWriteDescriptorSet, writes.len);
        defer vk_ctx.allocator.free(desc_writes);

        // We need to store the info structs because pBufferInfo/pImageInfo are pointers.
        // Their lifetime must exceed the vkUpdateDescriptorSets call.
        var buffer_infos = try vk_ctx.allocator.alloc(c.VkDescriptorBufferInfo, writes.len);
        defer vk_ctx.allocator.free(buffer_infos);
        var image_infos = try vk_ctx.allocator.alloc(c.VkDescriptorImageInfo, writes.len);
        defer vk_ctx.allocator.free(image_infos);

        for (writes, 0..) |write, i| {
            desc_writes[i] = .{
                .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .dstSet = self.handle,
                .dstBinding = write.binding,
                .dstArrayElement = 0,
                .descriptorCount = 1,
            };

            switch (write.info) {
                .UniformBuffer => |info| {
                    buffer_infos[i] = info;
                    desc_writes[i].descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
                    desc_writes[i].pBufferInfo = &buffer_infos[i];
                },
                .StorageBuffer => |info| {
                    buffer_infos[i] = info;
                    desc_writes[i].descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    desc_writes[i].pBufferInfo = &buffer_infos[i];
                },
                .CombinedImageSampler => |info| {
                    image_infos[i] = info;
                    desc_writes[i].descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    desc_writes[i].pImageInfo = &image_infos[i];
                },
            }
        }

        c.vkUpdateDescriptorSets(vk_ctx.device.handle, @intCast(desc_writes.len), desc_writes.ptr, 0, null);
    }
};

pub const DescriptorPoolArgs = struct {
    type: c.VkDescriptorType,
    count: u32,
};

pub const DescriptorPool = struct {
    const Self = @This();
    handle: c.VkDescriptorPool = undefined,

    pub fn init(vk_ctx: *VulkanContext, max_sets: u32, pool_sizes_args: []const DescriptorPoolArgs) !Self {
        var self = Self{};
        const sizes = try vk_ctx.allocator.alloc(c.VkDescriptorPoolSize, pool_sizes_args.len);
        defer vk_ctx.allocator.free(sizes);

        for (pool_sizes_args, 0..) |arg, i| {
            sizes[i] = .{
                .type = arg.type,
                .descriptorCount = arg.count,
            };
        }

        var pool_info = c.VkDescriptorPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .poolSizeCount = @intCast(sizes.len),
            .pPoolSizes = sizes.ptr,
            .maxSets = max_sets,
        };
        try vkCheck(c.vkCreateDescriptorPool(vk_ctx.device.handle, &pool_info, null, &self.handle));
        return self;
    }

    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyDescriptorPool(vk_ctx.device.handle, self.handle, null);
    }

    /// Allocates multiple sets from multiple layouts into a caller-owned buffer
    pub fn allocateSets(self: Self, vk_ctx: *VulkanContext, layouts: []const DescriptorSetLayout, out_sets: []DescriptorSet) !void {
        std.debug.assert(layouts.len == out_sets.len);

        const layout_handles = try vk_ctx.allocator.alloc(c.VkDescriptorSetLayout, layouts.len);
        defer vk_ctx.allocator.free(layout_handles);

        const set_handles = try vk_ctx.allocator.alloc(c.VkDescriptorSet, out_sets.len);
        defer vk_ctx.allocator.free(set_handles);

        for (layouts, 0..) |l, i| {
            layout_handles[i] = l.handle;
        }

        var set_alloc_info = c.VkDescriptorSetAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = self.handle,
            .descriptorSetCount = @intCast(layouts.len),
            .pSetLayouts = layout_handles.ptr,
        };
        try vkCheck(c.vkAllocateDescriptorSets(vk_ctx.device.handle, &set_alloc_info, set_handles.ptr));

        for (out_sets, 0..) |*set, i| {
            set.handle = set_handles[i];
        }
    }

    /// A convenience wrapper for the common case of allocating a single set
    pub fn allocateSet(self: Self, vk_ctx: *VulkanContext, layout: DescriptorSetLayout) !DescriptorSet {
        var set: DescriptorSet = .{};
        // Create a slice of length 1 that points to the 'set' variable on the stack.
        try self.allocateSets(vk_ctx, &.{layout}, (&set)[0..1]);
        return set;
    }
};

pub const ShaderModule = struct {
    const Self = @This();
    handle: c.VkShaderModule = undefined,

    pub fn init(allocator: Allocator, vk_ctx: *VulkanContext, code: []const u8) !Self {
        var self = Self{};
        const aligned_code = try allocator.alignedAlloc(u32, std.mem.Alignment.of(u32), code.len / @sizeOf(u32));
        defer allocator.free(aligned_code);
        @memcpy(std.mem.sliceAsBytes(aligned_code), code);
        var create_info = c.VkShaderModuleCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = code.len,
            .pCode = aligned_code.ptr,
        };
        try vkCheck(c.vkCreateShaderModule(vk_ctx.device.handle, &create_info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyShaderModule(vk_ctx.device.handle, self.handle, null);
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
        var info = create_info;
        info.sType = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        try vkCheck(c.vkCreatePipelineLayout(vk_ctx.device.handle, &info, null, &self.handle));
        return self;
    }
    pub fn deinit(self: *Self, vk_ctx: *VulkanContext) void {
        c.vkDestroyPipelineLayout(vk_ctx.device.handle, self.handle, null);
    }
};

pub const Pipeline = struct {
    const Self = @This();

    handle: c.VkPipeline = undefined,

    pub fn init(vk_ctx: *VulkanContext, render_pass: RenderPass, layout: PipelineLayout, Vertex: type, vert_shader_bin: []const u8, frag_shader_bin: []const u8) !Self {
        var self = Self{};

        var vert_shader_module = try ShaderModule.init(vk_ctx.allocator, vk_ctx, vert_shader_bin);
        defer vert_shader_module.deinit(vk_ctx);
        var frag_shader_module = try ShaderModule.init(vk_ctx.allocator, vk_ctx, frag_shader_bin);
        defer frag_shader_module.deinit(vk_ctx);

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
            .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vert_shader_module.handle, .pName = "main" },
            .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_shader_module.handle, .pName = "main" },
        };
        const binding_description = getVertexBindingDescription(Vertex);
        const attribute_descriptions = getVertexAttributeDescriptions(Vertex);
        const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_description,
            .vertexAttributeDescriptionCount = attribute_descriptions.len,
            .pVertexAttributeDescriptions = &attribute_descriptions,
        };
        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = c.VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
            .primitiveRestartEnable = c.VK_FALSE,
        };

        var rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_BACK_BIT,
            .frontFace = c.VK_FRONT_FACE_COUNTER_CLOCKWISE,
            .depthBiasEnable = c.VK_FALSE,
        };
        var multisampling = c.VkPipelineMultisampleStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
        };
        var color_blend_attachment = c.VkPipelineColorBlendAttachmentState{
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = c.VK_FALSE,
        };
        var color_blending = c.VkPipelineColorBlendStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = c.VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
        };
        var depth_stencil = c.VkPipelineDepthStencilStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
            .depthTestEnable = c.VK_TRUE,
            .depthWriteEnable = c.VK_TRUE,
            // GREATER or GREATER_OR_EQUAL
            .depthCompareOp = c.VK_COMPARE_OP_GREATER_OR_EQUAL,
            .depthBoundsTestEnable = c.VK_FALSE,
            .stencilTestEnable = c.VK_FALSE,
        };

        var viewport_state = c.VkPipelineViewportStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
            .viewportCount = 1,
            .scissorCount = 1,
        };

        const dynamic_states = [_]c.VkDynamicState{
            c.VK_DYNAMIC_STATE_VIEWPORT,
            c.VK_DYNAMIC_STATE_SCISSOR,
        };

        const dynamic_states_create_info = c.VkPipelineDynamicStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
            .dynamicStateCount = dynamic_states.len,
            .pDynamicStates = &dynamic_states,
        };

        var pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pDynamicState = &dynamic_states_create_info,
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
    present_queue: Queue,
    transfer_queue: Queue,
    command_pool: CommandPool,
    transfer_command_pool: CommandPool,

    pub fn init(allocator: Allocator, window_handle: *c.GLFWwindow) !Self {
        const instance = try Instance.init();
        const surface = try Surface.init(instance, window_handle);
        const physical_device = try PhysicalDevice.init(allocator, instance, surface);
        const device = try Device.init(allocator, physical_device);
        const graphics_queue = try Queue.init(device, physical_device.graphics_q_family);
        const present_queue = try Queue.init(device, physical_device.present_q_family);
        const transfer_queue = try Queue.init(device, physical_device.transfer_q_family);
        const command_pool = try CommandPool.init(device, physical_device.graphics_q_family);
        const transfer_command_pool = try CommandPool.init(device, physical_device.transfer_q_family);

        return .{
            .allocator = allocator,
            .instance = instance,
            .surface = surface,
            .physical_device = physical_device,
            .device = device,
            .graphics_queue = graphics_queue,
            .present_queue = present_queue,
            .transfer_queue = transfer_queue,
            .command_pool = command_pool,
            .transfer_command_pool = transfer_command_pool,
        };
    }

    pub fn deinit(self: *Self) void {
        // Destroy in reverse order of creation
        self.command_pool.deinit(self);
        self.device.deinit();
        self.surface.deinit(self);
        self.instance.deinit();
    }
};
