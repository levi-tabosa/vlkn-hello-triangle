// main.zig (with mouse control)

const std = @import("std");
const spirv = @import("spirv");
const math = std.math;
const c = @import("c").imports;

const vert_shader_code = spirv.triangle_vert;
const frag_shader_code = spirv.triangle_frag;

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
    VulkanPipelineFailed,
    VulkanDescriptorSetLayoutFailed,
    VulkanBufferCreationFailed,
    VulkanMemoryAllocationFailed,
    MissingMemoryType,
};

fn checkVk(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) {
        return;
    }
    std.log.err("Vulkan call failed with code: {}", .{result});
    switch (result) {
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY, c.VK_ERROR_OUT_OF_HOST_MEMORY => return AppError.VulkanMemoryAllocationFailed,
        else => return AppError.VulkanDrawFailed,
    }
}

const vertices = [_]Vertex{
    // Make the triangle a bit smaller so its center follows the cursor
    .{ .pos = .{ 0.0, -0.15 } },
    .{ .pos = .{ -0.15, 0.15 } },
    .{ .pos = .{ 0.15, 0.15 } },
};

// This struct is already correct
const UniformBufferObject = extern struct {
    color: [4]f32,
    offset: [2]f32,
};

// ... (No changes to helper functions: findMemoryType, createBuffer, createShaderModule) ...
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
    return AppError.MissingMemoryType;
}

fn createBuffer(
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    size: c.VkDeviceSize,
    usage: c.VkBufferUsageFlags,
    properties: c.VkMemoryPropertyFlags,
    allocator: std.mem.Allocator,
) !struct { buffer: c.VkBuffer, memory: c.VkDeviceMemory } {
    _ = allocator;
    var buffer_info = c.VkBufferCreateInfo{
        .size = size,
        .usage = usage,
        .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buffer: c.VkBuffer = undefined;
    try checkVk(c.vkCreateBuffer(device, &buffer_info, null, &buffer));

    var mem_requirements: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    var alloc_info = c.VkMemoryAllocateInfo{
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = try findMemoryType(physical_device, mem_requirements.memoryTypeBits, properties),
    };
    var memory: c.VkDeviceMemory = undefined;
    try checkVk(c.vkAllocateMemory(device, &alloc_info, null, &memory));

    try checkVk(c.vkBindBufferMemory(device, buffer, memory, 0));

    return .{ .buffer = buffer, .memory = memory };
}

fn createShaderModule(allocator: std.mem.Allocator, device: c.VkDevice, code: []const u8) !c.VkShaderModule {
    std.debug.assert(code.len % 4 == 0);
    const aligned_code_slice = try allocator.alignedAlloc(u32, 4, code.len / @sizeOf(u32));
    defer allocator.free(aligned_code_slice);
    @memcpy(std.mem.sliceAsBytes(aligned_code_slice), code);

    var create_info = c.VkShaderModuleCreateInfo{
        .codeSize = code.len,
        .pCode = aligned_code_slice.ptr,
    };

    var shader_module: c.VkShaderModule = undefined;
    try checkVk(c.vkCreateShaderModule(device, &create_info, null, &shader_module));
    return shader_module;
}

pub fn main() !void {
    var da: std.heap.DebugAllocator(.{}) = .init;
    defer _ = da.deinit();

    var allocator = da.allocator();

    // ... (No changes to Vulkan setup from steps 1-8) ...
    if (c.glfwInit() == c.GLFW_FALSE) return AppError.GlfwInitFailed;
    defer c.glfwTerminate();

    c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
    const window = c.glfwCreateWindow(800, 600, "Zig + Vulkan Triangle", null, null) orelse return AppError.GlfwWindowFailed;
    defer c.glfwDestroyWindow(window);

    var app_info = c.VkApplicationInfo{
        .pApplicationName = "Zig Vulkan App",
        .applicationVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "No Engine",
        .engineVersion = c.VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = c.VK_API_VERSION_1_0,
    };

    var extension_count: u32 = 0;
    const extension_names_ptr = c.glfwGetRequiredInstanceExtensions(&extension_count);
    const extension_names = extension_names_ptr[0..extension_count];

    var create_info_instance = c.VkInstanceCreateInfo{
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extension_names.ptr,
    };

    var instance: c.VkInstance = undefined;
    try checkVk(c.vkCreateInstance(&create_info_instance, null, &instance));
    defer c.vkDestroyInstance(instance, null);

    var surface: c.VkSurfaceKHR = undefined;
    try checkVk(c.glfwCreateWindowSurface(instance, window, null, &surface));
    defer c.vkDestroySurfaceKHR(instance, surface, null);

    var device_count: u32 = 0;
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) return AppError.NoSuitableGpu;

    const devices = try allocator.alloc(c.VkPhysicalDevice, device_count);
    defer allocator.free(devices);
    try checkVk(c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr));
    const physical_device = devices[0];

    var queue_family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptr);

    var graphics_family_index: u32 = std.math.maxInt(u32);
    for (queue_families, 0..) |family, i| {
        if (family.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
            graphics_family_index = @intCast(i);
            break;
        }
    }
    if (graphics_family_index == std.math.maxInt(u32)) return AppError.NoSuitableGpu;

    const queue_priority: f32 = 1.0;
    var queue_create_info = c.VkDeviceQueueCreateInfo{
        .queueFamilyIndex = graphics_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const p_swapchain_ext = c.VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    const device_extensions_names = [_]?*const u8{&p_swapchain_ext[0]};
    var device_features: c.VkPhysicalDeviceFeatures = .{};
    var device_create_info = c.VkDeviceCreateInfo{
        .pQueueCreateInfos = &queue_create_info,
        .queueCreateInfoCount = 1,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = device_extensions_names.len,
        .ppEnabledExtensionNames = &device_extensions_names,
    };

    var device: c.VkDevice = undefined;
    try checkVk(c.vkCreateDevice(physical_device, &device_create_info, null, &device));
    defer c.vkDestroyDevice(device, null);

    var graphics_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, graphics_family_index, 0, &graphics_queue);

    var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
    try checkVk(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities));

    const surface_format = c.VkSurfaceFormatKHR{
        .format = c.VK_FORMAT_B8G8R8A8_SRGB,
        .colorSpace = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR,
    };

    var swapchain_create_info = c.VkSwapchainCreateInfoKHR{
        .surface = surface,
        .minImageCount = capabilities.minImageCount,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = capabilities.currentExtent,
        .imageArrayLayers = 1,
        .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = c.VK_PRESENT_MODE_FIFO_KHR,
        .clipped = c.VK_TRUE,
        .oldSwapchain = null,
    };

    var swapchain: c.VkSwapchainKHR = undefined;
    try checkVk(c.vkCreateSwapchainKHR(device, &swapchain_create_info, null, &swapchain));
    defer c.vkDestroySwapchainKHR(device, swapchain, null);

    var image_count: u32 = 0;
    try checkVk(c.vkGetSwapchainImagesKHR(device, swapchain, &image_count, null));
    const swapchain_images = try allocator.alloc(c.VkImage, image_count);
    defer allocator.free(swapchain_images);
    try checkVk(c.vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.ptr));

    var swapchain_image_views = try allocator.alloc(c.VkImageView, image_count);
    defer {
        for (swapchain_image_views) |view| c.vkDestroyImageView(device, view, null);
        allocator.free(swapchain_image_views);
    }
    for (swapchain_images, 0..) |image, i| {
        var iv_create_info = c.VkImageViewCreateInfo{
            .image = image,
            .viewType = c.VK_IMAGE_VIEW_TYPE_2D,
            .format = surface_format.format,
            .subresourceRange = .{
                .aspectMask = c.VK_IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        try checkVk(c.vkCreateImageView(device, &iv_create_info, null, &swapchain_image_views[i]));
    }

    var color_attachment = c.VkAttachmentDescription{
        .format = surface_format.format,
        .samples = c.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = c.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = c.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = c.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = c.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };
    var color_attachment_ref = c.VkAttachmentReference{
        .attachment = 0,
        .layout = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };
    var subpass = c.VkSubpassDescription{
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
    var rp_create_info = c.VkRenderPassCreateInfo{
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        .dependencyCount = 1,
        .pDependencies = &subpass_dep,
    };
    var render_pass: c.VkRenderPass = undefined;
    try checkVk(c.vkCreateRenderPass(device, &rp_create_info, null, &render_pass));
    defer c.vkDestroyRenderPass(device, render_pass, null);

    var ubo_layout_binding = c.VkDescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT | c.VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = null,
    };

    var layout_info = c.VkDescriptorSetLayoutCreateInfo{
        .bindingCount = 1,
        .pBindings = &ubo_layout_binding,
    };

    var descriptor_set_layout: c.VkDescriptorSetLayout = undefined;
    try checkVk(c.vkCreateDescriptorSetLayout(device, &layout_info, null, &descriptor_set_layout));
    defer c.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, null);

    // ... (No changes to pipeline creation or other setup) ...
    const vert_module = try createShaderModule(allocator, device, vert_shader_code);
    defer c.vkDestroyShaderModule(device, vert_module, null);
    const frag_module = try createShaderModule(allocator, device, frag_shader_code);
    defer c.vkDestroyShaderModule(device, frag_module, null);

    const vert_shader_stage_info = c.VkPipelineShaderStageCreateInfo{
        .stage = c.VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_module,
        .pName = "main",
    };
    const frag_shader_stage_info = c.VkPipelineShaderStageCreateInfo{
        .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_module,
        .pName = "main",
    };
    const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{ vert_shader_stage_info, frag_shader_stage_info };

    var binding_desc = Vertex.getBindingDescription();
    var attrib_desc = Vertex.getAttributeDescriptions();
    var vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_desc,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &attrib_desc,
    };

    var input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
        .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = c.VK_FALSE,
    };
    var viewport = c.VkViewport{
        .x = 0.0,
        .y = 0.0,
        .width = @as(f32, @floatFromInt(capabilities.currentExtent.width)),
        .height = @as(f32, @floatFromInt(capabilities.currentExtent.height)),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };
    var scissor = c.VkRect2D{
        .offset = .{ .x = 0, .y = 0 },
        .extent = capabilities.currentExtent,
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
    var pipeline_layout_info = c.VkPipelineLayoutCreateInfo{
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var pipeline_layout: c.VkPipelineLayout = undefined;
    try checkVk(c.vkCreatePipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout));
    defer c.vkDestroyPipelineLayout(device, pipeline_layout, null);
    var pipeline_info = c.VkGraphicsPipelineCreateInfo{
        .stageCount = shader_stages.len,
        .pStages = &shader_stages,
        .pVertexInputState = &vertex_input_info,
        .pInputAssemblyState = &input_assembly,
        .pViewportState = &viewport_state,
        .pRasterizationState = &rasterizer,
        .pMultisampleState = &multisampling,
        .pColorBlendState = &color_blending,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
    };
    var graphics_pipeline: c.VkPipeline = undefined;
    try checkVk(c.vkCreateGraphicsPipelines(device, null, 1, &pipeline_info, null, &graphics_pipeline));
    defer c.vkDestroyPipeline(device, graphics_pipeline, null);

    var swapchain_framebuffers = try allocator.alloc(c.VkFramebuffer, image_count);
    defer {
        for (swapchain_framebuffers) |fb| c.vkDestroyFramebuffer(device, fb, null);
        allocator.free(swapchain_framebuffers);
    }
    for (swapchain_image_views, 0..) |view, i| {
        var fb_create_info = c.VkFramebufferCreateInfo{
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = capabilities.currentExtent.width,
            .height = capabilities.currentExtent.height,
            .layers = 1,
        };
        try checkVk(c.vkCreateFramebuffer(device, &fb_create_info, null, &swapchain_framebuffers[i]));
    }

    const vertex_buffer_size = @sizeOf(Vertex) * vertices.len;
    const vertex_buffer_obj = try createBuffer(
        device,
        physical_device,
        vertex_buffer_size,
        c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer c.vkDestroyBuffer(device, vertex_buffer_obj.buffer, null);
    defer c.vkFreeMemory(device, vertex_buffer_obj.memory, null);

    var data_ptr: ?*anyopaque = undefined;
    try checkVk(c.vkMapMemory(device, vertex_buffer_obj.memory, 0, vertex_buffer_size, 0, &data_ptr));
    const mapped_memory: [*]Vertex = @ptrCast(@alignCast(data_ptr.?));
    @memcpy(mapped_memory[0..vertices.len], &vertices);
    c.vkUnmapMemory(device, vertex_buffer_obj.memory);

    const ubo_size = @sizeOf(UniformBufferObject);
    const uniform_buffer_obj = try createBuffer(
        device,
        physical_device,
        ubo_size,
        c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer c.vkDestroyBuffer(device, uniform_buffer_obj.buffer, null);
    defer c.vkFreeMemory(device, uniform_buffer_obj.memory, null);

    var pool_size = c.VkDescriptorPoolSize{
        .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
    };
    var pool_info = c.VkDescriptorPoolCreateInfo{
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
        .maxSets = 1,
    };
    var descriptor_pool: c.VkDescriptorPool = undefined;
    try checkVk(c.vkCreateDescriptorPool(device, &pool_info, null, &descriptor_pool));
    defer c.vkDestroyDescriptorPool(device, descriptor_pool, null);

    var set_alloc_info = c.VkDescriptorSetAllocateInfo{
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var descriptor_set: c.VkDescriptorSet = undefined;
    try checkVk(c.vkAllocateDescriptorSets(device, &set_alloc_info, &descriptor_set));

    var buffer_info_desc = c.VkDescriptorBufferInfo{
        .buffer = uniform_buffer_obj.buffer,
        .offset = 0,
        .range = ubo_size,
    };
    var descriptor_write = c.VkWriteDescriptorSet{
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &buffer_info_desc,
    };
    c.vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, null);

    var cmd_pool_info = c.VkCommandPoolCreateInfo{
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_family_index,
    };
    var command_pool: c.VkCommandPool = undefined;
    try checkVk(c.vkCreateCommandPool(device, &cmd_pool_info, null, &command_pool));
    defer c.vkDestroyCommandPool(device, command_pool, null);

    var cmd_alloc_info = c.VkCommandBufferAllocateInfo{
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    var command_buffer: c.VkCommandBuffer = undefined;
    try checkVk(c.vkAllocateCommandBuffers(device, &cmd_alloc_info, &command_buffer));

    var sync_create_info = c.VkSemaphoreCreateInfo{};
    var fence_create_info = c.VkFenceCreateInfo{ .flags = c.VK_FENCE_CREATE_SIGNALED_BIT };

    var image_available_semaphore: c.VkSemaphore = undefined;
    var render_finished_semaphore: c.VkSemaphore = undefined;
    var in_flight_fence: c.VkFence = undefined;

    try checkVk(c.vkCreateSemaphore(device, &sync_create_info, null, &image_available_semaphore));
    try checkVk(c.vkCreateSemaphore(device, &sync_create_info, null, &render_finished_semaphore));
    try checkVk(c.vkCreateFence(device, &fence_create_info, null, &in_flight_fence));
    defer c.vkDestroySemaphore(device, image_available_semaphore, null);
    defer c.vkDestroySemaphore(device, render_finished_semaphore, null);
    defer c.vkDestroyFence(device, in_flight_fence, null);

    // Main Loop
    const start_time = std.time.milliTimestamp();
    while (c.glfwWindowShouldClose(window) == 0) {
        c.glfwPollEvents();

        _ = c.vkWaitForFences(device, 1, &in_flight_fence, c.VK_TRUE, std.math.maxInt(u64));
        _ = c.vkResetFences(device, 1, &in_flight_fence);

        var image_index: u32 = undefined;
        _ = c.vkAcquireNextImageKHR(device, swapchain, std.math.maxInt(u64), image_available_semaphore, null, &image_index);

        _ = c.vkResetCommandBuffer(command_buffer, 0);
        var cmd_begin_info = c.VkCommandBufferBeginInfo{};
        _ = c.vkBeginCommandBuffer(command_buffer, &cmd_begin_info);

        // --- (CHANGE 1) --- Get window size and mouse position
        var win_width: c_int = 0;
        var win_height: c_int = 0;
        c.glfwGetWindowSize(window, &win_width, &win_height);

        var mouse_x: f64 = 0;
        var mouse_y: f64 = 0;
        c.glfwGetCursorPos(window, &mouse_x, &mouse_y);
        const cb: c.GLFWmousebuttonfun = struct {
            fn callback(wd: ?*c.GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.C) void {
                var mx: f64 = 0;
                var my: f64 = 0;
                c.glfwGetCursorPos(wd, &mx, &my);
                std.debug.print("pos : {}x {}y", .{ mx, my });
                _ = button; // Unused parameter
                _ = action; // Unused parameter
                _ = mods; // Unused parameter
            }
        }.callback;
        _ = c.glfwSetMouseButtonCallback(window, cb);

        // --- (CHANGE 2) --- Convert screen coordinates to NDC
        // X: [0, width] -> [0, 2] -> [-1, 1]
        const ndc_x = @as(f32, @floatCast(mouse_x)) / @as(f32, @floatFromInt(win_width)) * 2.0 - 1.0;
        // Y: [0, height] -> [0, 2] -> [1, -1] (Y is inverted between screen and NDC)
        const ndc_y = (@as(f32, @floatCast(mouse_y)) / @as(f32, @floatFromInt(win_height)) * 2.0) - 1.0;

        // Update UBO data
        const current_time = std.time.milliTimestamp();
        const time_val = @as(f32, @floatFromInt(current_time - start_time)) / 1000.0;
        const ubo = UniformBufferObject{
            // Keep the color animation
            .color = .{
                0.5 + 0.5 * math.sin(time_val),
                0.5 + 0.5 * math.sin(time_val + 2.0 * math.pi / 3.0),
                0.5 + 0.5 * math.sin(time_val + 4.0 * math.pi / 3.0),
                1.0,
            },
            // --- (CHANGE 3) --- Use the mouse position for the offset
            .offset = .{
                ndc_x,
                ndc_y,
            },
        };

        var ubo_data_ptr: ?*anyopaque = undefined;
        try checkVk(c.vkMapMemory(device, uniform_buffer_obj.memory, 0, ubo_size, 0, &ubo_data_ptr));
        const mapped_ubo: *UniformBufferObject = @ptrCast(@alignCast(ubo_data_ptr.?));
        mapped_ubo.* = ubo;
        c.vkUnmapMemory(device, uniform_buffer_obj.memory);

        // --- (NO CHANGE 4) --- The rest of the render loop is identical
        var clear_color = c.VkClearValue{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } };
        var rp_begin_info = c.VkRenderPassBeginInfo{
            .renderPass = render_pass,
            .framebuffer = swapchain_framebuffers[image_index],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };

        c.vkCmdBeginRenderPass(command_buffer, &rp_begin_info, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        const vertex_buffers = [_]c.VkBuffer{vertex_buffer_obj.buffer};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffers, &offsets);
        c.vkCmdBindDescriptorSets(
            command_buffer,
            c.VK_PIPELINE_BIND_POINT_GRAPHICS,
            pipeline_layout,
            0,
            1,
            &descriptor_set,
            0,
            null,
        );

        c.vkCmdDraw(command_buffer, vertices.len, 1, 0, 0);

        c.vkCmdEndRenderPass(command_buffer);
        try checkVk(c.vkEndCommandBuffer(command_buffer));

        var submit_info = c.VkSubmitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &image_available_semaphore,
            .pWaitDstStageMask = &@as(c.VkPipelineStageFlags, c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_finished_semaphore,
        };
        try checkVk(c.vkQueueSubmit(graphics_queue, 1, &submit_info, in_flight_fence));

        var present_info = c.VkPresentInfoKHR{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
        };
        _ = c.vkQueuePresentKHR(graphics_queue, &present_info);
    }

    _ = c.vkDeviceWaitIdle(device);
}

const Vertex = extern struct {
    pos: [2]f32,
    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Vertex),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }
    pub fn getAttributeDescriptions() [1]c.VkVertexInputAttributeDescription {
        return .{.{
            .binding = 0,
            .location = 0,
            .format = c.VK_FORMAT_R32G32_SFLOAT,
            .offset = @offsetOf(Vertex, "pos"),
        }};
    }
};
