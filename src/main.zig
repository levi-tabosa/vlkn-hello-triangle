const std = @import("std");
const builtin = @import("builtin");

const math = std.math;
const spirv = @import("spirv");
const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("vulkan/vulkan.h");
    @cInclude("GLFW/glfw3.h");
});

const frag_shader_code = spirv.frag_code;
const vert_shader_code = spirv.vert_code;

fn checkVk(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) {
        return;
    }
    std.log.err("Vulkan call failed with code: {}", .{result});
    // TODO: improve error handling
    switch (result) {
        c.VK_INCOMPLETE => {
            std.log.warn("Vulkan call incomplete: {}", .{result});
            return;
        },
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY, c.VK_ERROR_OUT_OF_HOST_MEMORY => return error.VulkanMemoryAllocationFailed,
        c.VK_ERROR_LAYER_NOT_PRESENT => return error.VulkanLayerMissing,
        c.VK_ERROR_INITIALIZATION_FAILED => return error.VulkanInitFailed,
        c.VK_ERROR_FORMAT_NOT_SUPPORTED => return error.VulkanUnsupportedFormat,
        c.VK_ERROR_DRAW_FAILED => return error.VulkanDrawFailed,
        c.VK_ERROR_UNKNOWN => return error.VulkanUnknown,
        else => return error.Default, // General error for drawing operations
    }
}

const vertices = [_]Vertex{
    .{ .pos = .{ 0.0, -0.8 } },
    .{ .pos = .{ -0.8, 0.8 } },
    .{ .pos = .{ 0.8, 0.8 } },
};

const UniformBufferObject = extern struct {
    color: [4]f32,
};

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

// --- NEW --- Helper to create a buffer (for vertices or uniforms)
fn createBuffer(
    device: c.VkDevice,
    physical_device: c.VkPhysicalDevice,
    size: c.VkDeviceSize,
    usage: c.VkBufferUsageFlags,
    properties: c.VkMemoryPropertyFlags,
    allocator: std.mem.Allocator,
) !struct { buffer: c.VkBuffer, memory: c.VkDeviceMemory } {
    _ = allocator; // Unused in this function, but can be used for custom memory management
    var buffer_info = c.VkBufferCreateInfo{
        .sType = c.STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = c.SHARING_MODE_EXCLUSIVE,
    };
    var buffer: c.VkBuffer = undefined;
    try checkVk(c.vkCreateBuffer(device, &buffer_info, null, &buffer));

    var mem_requirements: c.VkMemoryRequirements = undefined;
    c.vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    var alloc_info = c.VkMemoryAllocateInfo{
        .sType = c.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = try findMemoryType(physical_device, mem_requirements.memoryTypeBits, properties),
    };
    var memory: c.VkDeviceMemory = undefined;
    try checkVk(c.vkAllocateMemory(device, &alloc_info, null, &memory));

    _ = c.vkBindBufferMemory(device, buffer, memory, 0);

    return .{ .buffer = buffer, .memory = memory };
}

fn createShaderModule(allocator: std.mem.Allocator, device: c.VkDevice, code: []const u8) !c.VkShaderModule {
    std.debug.assert(code.len % 4 == 0);

    // SPIR-V needs to be aligned to a 4-byte boundary.
    // The safest way is to allocate new memory with the required alignment
    // and copy the embedded data into it.
    const aligned_code = try allocator.alignedAlloc(u32, @alignOf(u32), code.len / @sizeOf(u32));
    defer allocator.free(aligned_code);

    @memcpy(std.mem.sliceAsBytes(aligned_code), code);

    var create_info = c.ShaderModuleCreateInfo{
        .sType = c.STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.len,
        .pCode = aligned_code.ptr,
    };

    var shader_module: c.ShaderModule = undefined;
    try checkVk(c.vkCreateShaderModule(device, &create_info, null, &shader_module));
    return shader_module;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Initialize GLFW and create a window
    if (c.glfwInit() == c.GLFW_FALSE) return error.GlfwInitFailed;
    defer c.glfwTerminate();

    c.glfwWindowHint(c.CLIENT_API, c.NO_API);
    const window = c.glfwCreateWindow(800, 600, "Zig + Vulkan Triangle", null, null) orelse return error.GlfwWindowFailed;
    defer c.glfwDestroyWindow(window);

    // 2. Create Vulkan Instance

    var app_info = c.ApplicationInfo{
        .sType = c.STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Zig Vulkan App",
        .applicationVersion = c.MAKE_VERSION(0, 1, 0),
        .pEngineName = "No Engine",
        .engineVersion = c.MAKE_VERSION(0, 1, 0),
        .apiVersion = c.API_VERSION,
    };

    var extension_count: u32 = 0;
    const extension_names_ptr = c.glfwGetRequiredInstanceExtensions(&extension_count);
    const extension_names = extension_names_ptr[0..extension_count];

    var create_info = c.VkInstanceCreateInfo{
        .sType = c.STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extension_names.ptr,
    };

    var instance: c.VkInstance = undefined;
    try checkVk(c.vkCreateInstance(&create_info, null, &instance));
    defer c.vkDestroyInstance(instance, null);

    // 3. Create Window Surface
    var surface: c.VkSurfaceKHR = undefined;
    try checkVk(c.glfwCreateWindowSurface(instance, window, null, &surface));
    defer c.vkDestroySurfaceKHR(instance, surface, null);

    // 4. Pick Physical Device (GPU)
    var device_count: u32 = 0;
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) return error.NoSuitableGpu;

    const devices = try allocator.alloc(c.VkPhysicalDevice, device_count);
    defer allocator.free(devices);
    _ = c.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

    const physical_device = devices[0];

    // 5. Create Logical Device and Queues
    var queue_family_count: u32 = 0;
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try allocator.alloc(c.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptr);

    var graphics_family_index: u32 = std.math.maxInt(u32);
    for (queue_families, 0..) |family, i| {
        if (family.queueFlags & c.QUEUE_GRAPHICS_BIT != 0) {
            graphics_family_index = @intCast(i);
            break;
        }
    }
    if (graphics_family_index == std.math.maxInt(u32)) return error.NoSuitableGpu;

    const queue_priority: f32 = 1.0;
    var queue_create_info = c.VkDeviceQueueCreateInfo{
        .sType = c.STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = graphics_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const p_swapchain_ext = c.VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    const device_extensions = [_]?*const u8{
        &p_swapchain_ext[0], // passes the *null terminated string as *char
    };
    var device_features: c.VkPhysicalDeviceFeatures = .{};
    var device_create_info = c.VkDeviceCreateInfo{
        .sType = c.STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pQueueCreateInfos = &queue_create_info,
        .queueCreateInfoCount = 1,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = device_extensions.len,
        .ppEnabledExtensionNames = &device_extensions,
    };

    var device: c.VkDevice = undefined;
    try checkVk(c.vkCreateDevice(physical_device, &device_create_info, null, &device));
    defer c.vkDestroyDevice(device, null);

    var graphics_queue: c.VkQueue = undefined;
    c.vkGetDeviceQueue(device, graphics_family_index, 0, &graphics_queue);

    // 6. Create Swap Chain
    var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
    _ = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities);

    const surface_format = c.VkSurfaceFormatKHR{
        .format = c.FORMAT_B8G8R8A8_SRGB,
        .colorSpace = c.COLOR_SPACE_SRGB_NONLINEAR_KHR,
    };

    var swapchain_create_info = c.VkSwapchainCreateInfoKHR{
        .sType = c.STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
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

    var swapchain: c.SwapchainKHR = undefined;
    try checkVk(c.vkCreateSwapchainKHR(device, &swapchain_create_info, null, &swapchain));
    defer c.vkDestroySwapchainKHR(device, swapchain, null);

    var image_count: u32 = 0;
    _ = c.vkGetSwapchainImagesKHR(device, swapchain, &image_count, null);
    const swapchain_images = try allocator.alloc(c.Image, image_count);
    defer allocator.free(swapchain_images);
    _ = c.vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.ptr);

    // 7. Create Image Views
    var swapchain_image_views = try allocator.alloc(c.ImageView, image_count);
    defer {
        for (swapchain_image_views) |view| c.vkDestroyImageView(device, view, null);
        allocator.free(swapchain_image_views);
    }
    for (swapchain_images, 0..) |image, i| {
        var iv_create_info = c.ImageViewCreateInfo{
            .sType = c.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = c.IMAGE_VIEW_TYPE_2D,
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

    // 8. Create Render Pass
    var color_attachment = c.AttachmentDescription{
        .format = surface_format.format,
        .samples = c.SAMPLE_COUNT_1_BIT,
        .loadOp = c.ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = c.ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = c.ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = c.ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = c.IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = c.IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    var color_attachment_ref = c.AttachmentReference{
        .attachment = 0,
        .layout = c.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    var subpass = c.VkSubpassDescription{
        .pipelineBindPoint = c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
    };

    var subpass_dep = c.VkSubpassDependency{
        .srcSubpass = c.SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = c.VkPIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = c.VkPIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = c.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    var rp_create_info = c.VkRenderPassCreateInfo{
        .sType = c.STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        // --- MODIFIED --- Add dependency for layout transitions
        .dependencyCount = 1,
        .pDependencies = &subpass_dep,
    };

    var render_pass: c.VkRenderPass = undefined;
    try checkVk(c.vkCreateRenderPass(device, &rp_create_info, null, &render_pass));
    defer c.vkDestroyRenderPass(device, render_pass, null);

    // --- NEW 9a. Create Descriptor Set Layout (for UBO) ---
    var ubo_layout_binding = c.VkDescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = c.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = c.SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = null,
    };

    var layout_info = c.VkDescriptorSetLayoutCreateInfo{
        .sType = c.STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &ubo_layout_binding,
    };

    var descriptor_set_layout: c.VkDescriptorSetLayout = undefined;
    try checkVk(c.vkCreateDescriptorSetLayout(device, &layout_info, null, &descriptor_set_layout));
    defer c.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, null);

    // --- NEW 9b. Create Graphics Pipeline ---
    const vert_module = try createShaderModule(allocator, device, vert_shader_code);
    defer c.vkDestroyShaderModule(device, vert_module, null);
    const frag_module = try createShaderModule(allocator, device, frag_shader_code);
    defer c.vkDestroyShaderModule(device, frag_module, null);

    const vert_shader_stage_info = c.VkPipelineShaderStageCreateInfo{
        .sType = c.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.SHADER_STAGE_VERTEX_BIT,
        .module = vert_module,
        .pName = "main",
    };
    const frag_shader_stage_info = c.VkPipelineShaderStageCreateInfo{
        .sType = c.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = c.SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_module,
        .pName = "main",
    };
    const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{ vert_shader_stage_info, frag_shader_stage_info };

    var binding_desc = Vertex.getBindingDescription();
    var attrib_desc = Vertex.getAttributeDescriptions();
    var vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
        .sType = c.STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_desc,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &attrib_desc,
    };

    var input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
        .sType = c.STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
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
    var scissor = c.vkRect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent };
    var viewport_state = c.VkPipelineViewportStateCreateInfo{
        .sType = c.STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
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
        .sType = c.STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = c.FALSE,
        .rasterizationSamples = c.SAMPLE_COUNT_1_BIT,
    };

    var color_blend_attachment = c.VkPipelineColorBlendAttachmentState{
        .colorWriteMask = c.COLOR_COMPONENT_R_BIT | c.COLOR_COMPONENT_G_BIT | c.COLOR_COMPONENT_B_BIT | c.COLOR_COMPONENT_A_BIT,
        .blendEnable = c.FALSE,
    };
    var color_blending = c.VkPipelineColorBlendStateCreateInfo{
        .sType = c.STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = c.FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    var pipeline_layout_info = c.VkPipelineLayoutCreateInfo{
        .sType = c.STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var pipeline_layout: c.VkPipelineLayout = undefined;
    try checkVk(c.vkCreatePipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout));
    defer c.vkDestroyPipelineLayout(device, pipeline_layout, null);

    var pipeline_info = c.VkGraphicsPipelineCreateInfo{
        .sType = c.STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
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

    // 10. Create Framebuffers
    var swapchain_framebuffers = try allocator.alloc(c.VkFramebuffer, image_count);
    defer {
        for (swapchain_framebuffers) |fb| c.vkDestroyFramebuffer(device, fb, null);
        allocator.free(swapchain_framebuffers);
    }
    for (swapchain_image_views, 0..) |view, i| {
        var fb_create_info = c.FramebufferCreateInfo{
            .sType = c.STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = capabilities.currentExtent.width,
            .height = capabilities.currentExtent.height,
            .layers = 1,
        };
        try checkVk(c.vkCreateFramebuffer(device, &fb_create_info, null, &swapchain_framebuffers[i]));
    }

    // --- NEW 11. Create Vertex Buffer ---
    const vertex_buffer_size = @sizeOf(Vertex) * vertices.len;
    const vertex_buffer_obj = try createBuffer(
        device,
        physical_device,
        vertex_buffer_size,
        c.VkBUFFER_USAGE_VERTEX_BUFFER_BIT,
        c.VkMEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VkMEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer c.vkDestroyBuffer(device, vertex_buffer_obj.buffer, null);
    defer c.vkFreeMemory(device, vertex_buffer_obj.memory, null);

    // Copy vertex data to the buffer
    var data_ptr: ?*anyopaque = undefined;
    _ = c.vkMapMemory(device, vertex_buffer_obj.memory, 0, vertex_buffer_size, 0, &data_ptr);
    const mapped_memory: [*]Vertex = @ptrCast(@alignCast(data_ptr.?));
    @memcpy(mapped_memory[0..vertices.len], &vertices);
    c.vkUnmapMemory(device, vertex_buffer_obj.memory);

    // --- NEW 12. Create Uniform Buffer, Descriptor Pool and Set ---
    const ubo_size = @sizeOf(UniformBufferObject);
    const uniform_buffer_obj = try createBuffer(
        device,
        physical_device,
        ubo_size,
        c.VkBUFFER_USAGE_UNIFORM_BUFFER_BIT,
        c.VkMEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VkMEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer c.vkDestroyBuffer(device, uniform_buffer_obj.buffer, null);
    defer c.vkFreeMemory(device, uniform_buffer_obj.memory, null);

    var pool_size = c.VkDescriptorPoolSize{
        .type = c.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, // We only need one descriptor
    };
    var pool_info = c.VkDescriptorPoolCreateInfo{
        .sType = c.STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
        .maxSets = 1, // We only need one set
    };
    var descriptor_pool: c.VkDescriptorPool = undefined;
    try checkVk(c.vkCreateDescriptorPool(device, &pool_info, null, &descriptor_pool));
    defer c.vkDestroyDescriptorPool(device, descriptor_pool, null);

    var set_alloc_info = c.VkDescriptorSetAllocateInfo{
        .sType = c.STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var descriptor_set: c.VkDescriptorSet = undefined;
    try checkVk(c.allocateDescriptorSets(device, &set_alloc_info, &descriptor_set));

    var buffer_info_desc = c.VkDescriptorBufferInfo{
        .buffer = uniform_buffer_obj.buffer,
        .offset = 0,
        .range = ubo_size,
    };
    var descriptor_write = c.VkWriteDescriptorSet{
        .sType = c.STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = c.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &buffer_info_desc,
    };
    c.updateDescriptorSets(device, 1, &descriptor_write, 0, null);

    // 13. Create Command Pool and Command Buffers
    var cmd_pool_info = c.VkCommandPoolCreateInfo{
        .sType = c.STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_family_index,
    };
    var command_pool: c.VkCommandPool = undefined;
    try checkVk(c.vkCreateCommandPool(device, &cmd_pool_info, null, &command_pool));
    defer c.vkDestroyCommandPool(device, command_pool, null);

    var cmd_alloc_info = c.VkCommandBufferAllocateInfo{
        .sType = c.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    var command_buffer: c.VkCommandBuffer = undefined;
    try checkVk(c.allocateCommandBuffers(device, &cmd_alloc_info, &command_buffer));

    // 14. Create Synchronization Objects (Semaphores and Fences)
    var sync_create_info = c.VkSemaphoreCreateInfo{ .sType = c.STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    var fence_create_info = c.VkFenceCreateInfo{
        .sType = c.STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
    };

    var image_available_semaphore: c.VkSemaphore = undefined;
    var render_finished_semaphore: c.VkSemaphore = undefined;
    var in_flight_fence: c.Fence = undefined;

    try checkVk(c.vkCreateSemaphore(device, &sync_create_info, null, &image_available_semaphore));
    try checkVk(c.vkCreateSemaphore(device, &sync_create_info, null, &render_finished_semaphore));
    try checkVk(c.vkCreateFence(device, &fence_create_info, null, &in_flight_fence));
    defer c.vkDestroySemaphore(device, image_available_semaphore, null);
    defer c.vkDestroySemaphore(device, render_finished_semaphore, null);
    defer c.vkDestroyFence(device, in_flight_fence, null);

    // 15. Main Loop
    const start_time = std.time.milliTimestamp();
    while (c.glfwWindowShouldClose(window) == 0) {
        c.glfwPollEvents();

        _ = c.vkWaitForFences(device, 1, &in_flight_fence, c.TRUE, std.math.maxInt(u64));
        _ = c.vkResetFences(device, 1, &in_flight_fence);

        var image_index: u32 = undefined;
        _ = c.vkAcquireNextImageKHR(device, swapchain, std.math.maxInt(u64), image_available_semaphore, null, &image_index);

        _ = c.vkResetCommandBuffer(command_buffer, 0);
        var cmd_begin_info = c.VkCommandBufferBeginInfo{ .sType = c.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        _ = c.vkBeginCommandBuffer(command_buffer, &cmd_begin_info);

        // --- MODIFIED --- Update uniform buffer with new color
        const current_time = std.time.milliTimestamp();
        const time_val = @as(f32, @floatFromInt(current_time - start_time)) / 1000.0;
        const ubo = UniformBufferObject{
            .color = .{
                0.5 + 0.5 * math.sin(time_val),
                0.5 + 0.5 * math.sin(time_val + 2.0 * math.pi / 3.0),
                0.5 + 0.5 * math.sin(time_val + 4.0 * math.pi / 3.0),
                1.0,
            },
        };
        var ubo_data_ptr: ?*anyopaque = undefined;
        _ = c.mapMemory(device, uniform_buffer_obj.memory, 0, ubo_size, 0, &ubo_data_ptr);
        const mapped_ubo: *UniformBufferObject = @ptrCast(@alignCast(ubo_data_ptr.?));
        mapped_ubo.* = ubo;
        c.vkUnmapMemory(device, uniform_buffer_obj.memory);

        // --- MODIFIED --- Render Pass now clears to black
        var clear_color = c.ClearValue{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } };

        var rp_begin_info = c.VkRenderPassBeginInfo{
            .sType = c.STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = swapchain_framebuffers[image_index],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };

        c.vkCmdBeginRenderPass(command_buffer, &rp_begin_info, c.SUBPASS_CONTENTS_INLINE);
        // --- NEW --- Drawing commands
        c.vkCmdBindPipeline(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        const vertex_buffers = [_]c.VkBuffer{vertex_buffer_obj.buffer};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffers, &offsets);

        c.vkCmdBindDescriptorSets(command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, null);

        c.vkCmdDraw(command_buffer, vertices.len, 1, 0, 0);

        c.vkCmdEndRenderPass(command_buffer);
        try checkVk(c.vkEndCommandBuffer(command_buffer));

        var submit_info = c.VkSubmitInfo{
            .sType = c.STRUCTURE_TYPE_SUBMIT_INFO,
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
            .sType = c.STRUCTURE_TYPE_PRESENT_INFO_KHR,
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
            .binding = 0, // We are using binding 0
            .stride = @sizeOf(Vertex), // The distance in bytes between two vertices
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX, // Move to the next data entry after each vertex
        };
    }

    // This function describes the attributes of a single vertex (e.g., its position).
    pub fn getAttributeDescriptions() [1]c.VkVertexInputAttributeDescription {
        return .{
            // Position attribute
            .{
                .binding = 0, // Data comes from the buffer in binding 0
                .location = 0, // This is `layout(location = 0)` in the vertex shader
                .format = c.VK_FORMAT_R32G32_SFLOAT, // The format is two 32-bit floats
                .offset = @offsetOf(Vertex, "pos"), // The offset of the 'pos' field in the struct
            },
        };
    }
};
