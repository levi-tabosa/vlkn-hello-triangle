const std = @import("std");
const math = std.math;
const C = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("GLFW/glfw3.h");
});

const AppError = error{
    GlfwInitFailed,
    GlfwWindowFailed,
    VulkanInstanceFailed,
    VulkanSurfaceFailed,
    NoSuitableGpu,
    VulkanDeviceFailed,
    VulkanSwapchainFailed,
    VulkanImageViewsFailed,
    VulkanRenderPassFailed,
    VulkanCommandPoolFailed,
    VulkanFramebuffersFailed,
    VulkanSyncObjectsFailed,
    VulkanCommandBuffersFailed,
    VulkanDrawFailed,
};

// Helper function to check Vulkan results and return a Zig error.
fn checkVk(result: C.VkResult) !void {
    if (result != C.VK_SUCCESS) {
        // In a real app, you'd want to map VkResult to more specific Zig errors.
        std.log.err("Vulkan call failed with code: {}", .{result});
        return error.VulkanError;
    }
}

pub fn main() !void {
    // 1. Initialize GLFW and create a window
    if (C.glfwInit() == C.GLFW_FALSE) return AppError.GlfwInitFailed;
    defer C.glfwTerminate();

    C.glfwWindowHint(C.GLFW_CLIENT_API, C.GLFW_NO_API);
    const window = C.glfwCreateWindow(800, 600, "Zig + Vulkan", null, null) orelse return AppError.GlfwWindowFailed;
    defer C.glfwDestroyWindow(window);

    // 2. Create Vulkan Instance
    var app_info = C.VkApplicationInfo{
        .sType = C.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Zig Vulkan App",
        .applicationVersion = C.VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = C.VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = C.VK_API_VERSION_1_0,
    };

    var extension_count: u32 = 0;
    const extension_names_ptr = C.glfwGetRequiredInstanceExtensions(&extension_count);
    const extension_names = extension_names_ptr[0..extension_count];

    var create_info = C.VkInstanceCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extension_names.ptr,
    };

    var instance: C.VkInstance = undefined;
    try checkVk(C.vkCreateInstance(&create_info, null, &instance));
    defer C.vkDestroyInstance(instance, null);

    // 3. Create Window Surface
    var surface: C.VkSurfaceKHR = undefined;
    // This function connects Vulkan to the window created by GLFW.
    try checkVk(C.glfwCreateWindowSurface(instance, window, null, &surface));
    defer C.vkDestroySurfaceKHR(instance, surface, null);

    // 4. Pick Physical Device (GPU)
    var device_count: u32 = 0;
    _ = C.vkEnumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) return AppError.NoSuitableGpu;

    const devices = try std.heap.page_allocator.alloc(C.VkPhysicalDevice, device_count);
    defer std.heap.page_allocator.free(devices);
    _ = C.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

    // For this simple example, we pick the first available device.
    // A real app would query properties and features to find the *best* device.
    const physical_device = devices[0];

    // 5. Create Logical Device and Queues
    // Find a queue family that supports graphics operations.
    var queue_family_count: u32 = 0;
    C.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try std.heap.page_allocator.alloc(C.VkQueueFamilyProperties, queue_family_count);
    defer std.heap.page_allocator.free(queue_families);
    C.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptr);

    var graphics_family_index: u32 = std.math.maxInt(u32);
    for (queue_families, 0..) |family, i| {
        if (family.queueFlags & C.VK_QUEUE_GRAPHICS_BIT != 0) {
            graphics_family_index = @intCast(i);
            break;
        }
    }
    if (graphics_family_index == std.math.maxInt(u32)) return AppError.NoSuitableGpu;

    const queue_priority: f32 = 1.0;
    var queue_create_info = C.VkDeviceQueueCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = graphics_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const p_swapchain_ext = C.VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    const device_extensions_names = [_]?*const u8{
        &p_swapchain_ext[0],
    };
    var device_features: C.VkPhysicalDeviceFeatures = .{}; // Empty features for now
    var device_create_info = C.VkDeviceCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pQueueCreateInfos = &queue_create_info,
        .queueCreateInfoCount = 1,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = device_extensions_names.len,
        .ppEnabledExtensionNames = &device_extensions_names,
    };

    var device: C.VkDevice = undefined;
    try checkVk(C.vkCreateDevice(physical_device, &device_create_info, null, &device));
    defer C.vkDestroyDevice(device, null);

    var graphics_queue: C.VkQueue = undefined;
    C.vkGetDeviceQueue(device, graphics_family_index, 0, &graphics_queue);

    // 6. Create Swap Chain
    var capabilities: C.VkSurfaceCapabilitiesKHR = undefined;
    _ = C.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities);

    const surface_format = C.VkSurfaceFormatKHR{
        .format = C.VK_FORMAT_B8G8R8A8_SRGB,
        .colorSpace = C.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    };

    var swapchain_create_info = C.VkSwapchainCreateInfoKHR{
        .sType = C.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = capabilities.minImageCount,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = capabilities.currentExtent,
        .imageArrayLayers = 1,
        .imageUsage = C.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = C.VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = C.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = C.VK_PRESENT_MODE_FIFO_KHR, // Guaranteed to be available
        .clipped = C.VK_TRUE,
        .oldSwapchain = null,
    };

    var swapchain: C.VkSwapchainKHR = undefined;
    try checkVk(C.vkCreateSwapchainKHR(device, &swapchain_create_info, null, &swapchain));
    defer C.vkDestroySwapchainKHR(device, swapchain, null);

    // Get swap chain images
    var image_count: u32 = 0;
    _ = C.vkGetSwapchainImagesKHR(device, swapchain, &image_count, null);
    const swapchain_images = try std.heap.page_allocator.alloc(C.VkImage, image_count);
    defer std.heap.page_allocator.free(swapchain_images);
    _ = C.vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.ptr);

    // 7. Create Image Views
    var swapchain_image_views = try std.heap.page_allocator.alloc(C.VkImageView, image_count);
    defer {
        for (swapchain_image_views) |view| C.vkDestroyImageView(device, view, null);
        std.heap.page_allocator.free(swapchain_image_views);
    }
    for (swapchain_images, 0..) |image, i| {
        var iv_create_info = C.VkImageViewCreateInfo{
            .sType = C.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = C.VK_IMAGE_VIEW_TYPE_2D,
            .format = surface_format.format,
            .subresourceRange = .{
                .aspectMask = C.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        try checkVk(C.vkCreateImageView(device, &iv_create_info, null, &swapchain_image_views[i]));
    }

    // 8. Create Render Pass (tells Vulkan how to render)
    var color_attachment = C.VkAttachmentDescription{
        .format = surface_format.format,
        .samples = C.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = C.VK_ATTACHMENT_LOAD_OP_CLEAR, // Clear the framebuffer before drawing
        .storeOp = C.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = C.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = C.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = C.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = C.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, // Image will be presented to the screen
    };

    var color_attachment_ref = C.VkAttachmentReference{
        .attachment = 0,
        .layout = C.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    var subpass = C.VkSubpassDescription{
        .pipelineBindPoint = C.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
    };

    var rp_create_info = C.VkRenderPassCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
    };

    var render_pass: C.VkRenderPass = undefined;
    try checkVk(C.vkCreateRenderPass(device, &rp_create_info, null, &render_pass));
    defer C.vkDestroyRenderPass(device, render_pass, null);

    // 9. Create Framebuffers
    var swapchain_framebuffers = try std.heap.page_allocator.alloc(C.VkFramebuffer, image_count);
    defer {
        for (swapchain_framebuffers) |fb| C.vkDestroyFramebuffer(device, fb, null);
        std.heap.page_allocator.free(swapchain_framebuffers);
    }
    for (swapchain_image_views, 0..) |view, i| {
        var fb_create_info = C.VkFramebufferCreateInfo{
            .sType = C.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = capabilities.currentExtent.width,
            .height = capabilities.currentExtent.height,
            .layers = 1,
        };
        try checkVk(C.vkCreateFramebuffer(device, &fb_create_info, null, &swapchain_framebuffers[i]));
    }

    // 10. Create Command Pool and Command Buffers
    var pool_info = C.VkCommandPoolCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = C.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_family_index,
    };
    var command_pool: C.VkCommandPool = undefined;
    try checkVk(C.vkCreateCommandPool(device, &pool_info, null, &command_pool));
    defer C.vkDestroyCommandPool(device, command_pool, null);

    var alloc_info = C.VkCommandBufferAllocateInfo{
        .sType = C.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = C.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    var command_buffer: C.VkCommandBuffer = undefined;
    try checkVk(C.vkAllocateCommandBuffers(device, &alloc_info, &command_buffer));

    // 11. Create Synchronization Objects (Semaphores and Fences)
    var sync_create_info = C.VkSemaphoreCreateInfo{ .sType = C.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    var fence_create_info = C.VkFenceCreateInfo{ .sType = C.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = C.VK_FENCE_CREATE_SIGNALED_BIT };

    var image_available_semaphore: C.VkSemaphore = undefined;
    var render_finished_semaphore: C.VkSemaphore = undefined;
    var in_flight_fence: C.VkFence = undefined;

    try checkVk(C.vkCreateSemaphore(device, &sync_create_info, null, &image_available_semaphore));
    try checkVk(C.vkCreateSemaphore(device, &sync_create_info, null, &render_finished_semaphore));
    try checkVk(C.vkCreateFence(device, &fence_create_info, null, &in_flight_fence));
    defer C.vkDestroySemaphore(device, image_available_semaphore, null);
    defer C.vkDestroySemaphore(device, render_finished_semaphore, null);
    defer C.vkDestroyFence(device, in_flight_fence, null);

    // 12. Main Loop
    const start_time = std.time.milliTimestamp();
    while (C.glfwWindowShouldClose(window) == 0) {
        C.glfwPollEvents();

        // -- DRAW FRAME --
        // Wait for the previous frame to finish
        _ = C.vkWaitForFences(device, 1, &in_flight_fence, C.VK_TRUE, std.math.maxInt(u64));
        _ = C.vkResetFences(device, 1, &in_flight_fence);

        // Acquire an image from the swap chain
        var image_index: u32 = undefined;
        _ = C.vkAcquireNextImageKHR(device, swapchain, std.math.maxInt(u64), image_available_semaphore, null, &image_index);

        // Record command buffer
        _ = C.vkResetCommandBuffer(command_buffer, 0);
        var cmd_begin_info = C.VkCommandBufferBeginInfo{ .sType = C.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        _ = C.vkBeginCommandBuffer(command_buffer, &cmd_begin_info);

        const current_time = std.time.milliTimestamp();
        const time_val = @as(f32, @floatFromInt(current_time - start_time)) / 1000.0;

        var clear_color = C.VkClearValue{
            .color = .{ .float32 = .{
                0.5 + 0.5 * math.sin(time_val),
                0.5 + 0.5 * math.sin(time_val + 2.0 * math.pi / 3.0),
                0.5 + 0.5 * math.sin(time_val + 4.0 * math.pi / 3.0),
                1.0,
            } },
        };

        var rp_begin_info = C.VkRenderPassBeginInfo{
            .sType = C.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = swapchain_framebuffers[image_index],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };

        C.vkCmdBeginRenderPass(command_buffer, &rp_begin_info, C.VK_SUBPASS_CONTENTS_INLINE);
        // Drawing commands would go here (e.g., vkCmdDraw)
        C.vkCmdEndRenderPass(command_buffer);
        try checkVk(C.vkEndCommandBuffer(command_buffer));

        // Submit the command buffer
        var submit_info = C.VkSubmitInfo{
            .sType = C.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &image_available_semaphore,
            .pWaitDstStageMask = &@as(C.VkPipelineStageFlags, C.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_finished_semaphore,
        };
        try checkVk(C.vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fence));

        // Present the image to the screen
        var present_info = C.VkPresentInfoKHR{
            .sType = C.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
        };
        _ = C.vkQueuePresentKHR(graphics_queue, &present_info);
    }

    // Wait for the device to finish all operations before cleanup
    _ = C.vkDeviceWaitIdle(device);
}
