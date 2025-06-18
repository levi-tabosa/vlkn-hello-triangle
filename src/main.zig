const std = @import("std");
const spirv = @import("spirv");
const math = std.math;
const C = @cImport({
    @cInclude("vulkan/vulkan.h");
    @cDefine("GLFW_INCLUDE_VULKAN", "");
    @cInclude("GLFW/glfw3.h");
});

// --- NEW --- Import the embedded shaders
const vert_shader_code = spirv.vert_code;
const frag_shader_code = spirv.frag_code;

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

// Helper function to check Vulkan results
fn checkVk(result: C.VkResult) !void {
    if (result == C.VK_SUCCESS) {
        return;
    }
    std.log.err("Vulkan call failed with code: {}", .{result});
    // TODO: Map VkResult to specific errors.
    switch (result) {
        C.VK_ERROR_OUT_OF_DEVICE_MEMORY, C.VK_ERROR_OUT_OF_HOST_MEMORY => return AppError.VulkanMemoryAllocationFailed,
        else => return AppError.VulkanDrawFailed,
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
    physical_device: C.VkPhysicalDevice,
    type_filter: u32,
    properties: C.VkMemoryPropertyFlags,
) !u32 {
    var mem_properties: C.VkPhysicalDeviceMemoryProperties = undefined;
    C.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    const set: u64 = 1;
    var i: u5 = 0;
    while (i < mem_properties.memoryTypeCount) : (i += 1) {
        if ((type_filter & set << i) != 0 and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    return AppError.MissingMemoryType;
}

// --- NEW --- Helper to create a buffer (for vertices or uniforms)
fn createBuffer(
    device: C.VkDevice,
    physical_device: C.VkPhysicalDevice,
    size: C.VkDeviceSize,
    usage: C.VkBufferUsageFlags,
    properties: C.VkMemoryPropertyFlags,
    allocator: std.mem.Allocator,
) !struct { buffer: C.VkBuffer, memory: C.VkDeviceMemory } {
    _ = allocator; // Unused in this function, but can be used for custom memory management
    var buffer_info = C.VkBufferCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = C.VK_SHARING_MODE_EXCLUSIVE,
    };
    var buffer: C.VkBuffer = undefined;
    try checkVk(C.vkCreateBuffer(device, &buffer_info, null, &buffer));

    var mem_requirements: C.VkMemoryRequirements = undefined;
    C.vkGetBufferMemoryRequirements(device, buffer, &mem_requirements);

    var alloc_info = C.VkMemoryAllocateInfo{
        .sType = C.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = try findMemoryType(physical_device, mem_requirements.memoryTypeBits, properties),
    };
    var memory: C.VkDeviceMemory = undefined;
    try checkVk(C.vkAllocateMemory(device, &alloc_info, null, &memory));

    _ = C.vkBindBufferMemory(device, buffer, memory, 0);

    return .{ .buffer = buffer, .memory = memory };
}

// --- MODIFIED --- Helper to create a shader module from SPIR-V code
// Now takes an allocator to create a temporary, aligned copy of the shader code.
fn createShaderModule(allocator: std.mem.Allocator, device: C.VkDevice, code: []const u8) !C.VkShaderModule {
    // SPIR-V is a stream of 32-bit words, so the byte length must be a multiple of 4.
    std.debug.assert(code.len % 4 == 0);

    // This is the key fix: We allocate new memory with a guaranteed alignment of 4,
    // large enough to hold the shader code as a slice of u32s.
    const aligned_code_slice = try allocator.alignedAlloc(u32, 4, code.len / @sizeOf(u32));
    // We must free this temporary memory after the Vulkan call.
    defer allocator.free(aligned_code_slice);

    // Copy the bytes from the unaligned embedded file into our new aligned buffer.
    @memcpy(std.mem.sliceAsBytes(aligned_code_slice), code);

    var create_info = C.VkShaderModuleCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.len,
        // Now we can use the pointer from our aligned_code_slice, which is guaranteed to be correct.
        // No casting or alignment tricks are needed here because the pointer is already the correct type and alignment.
        .pCode = aligned_code_slice.ptr,
    };

    var shader_module: C.VkShaderModule = undefined;
    try checkVk(C.vkCreateShaderModule(device, &create_info, null, &shader_module));
    return shader_module;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Initialize GLFW and create a window
    if (C.glfwInit() == C.GLFW_FALSE) return AppError.GlfwInitFailed;
    defer C.glfwTerminate();

    C.glfwWindowHint(C.GLFW_CLIENT_API, C.GLFW_NO_API);
    const window = C.glfwCreateWindow(800, 600, "Zig + Vulkan Triangle", null, null) orelse return AppError.GlfwWindowFailed;
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
    try checkVk(C.glfwCreateWindowSurface(instance, window, null, &surface));
    defer C.vkDestroySurfaceKHR(instance, surface, null);

    // 4. Pick Physical Device (GPU)
    var device_count: u32 = 0;
    _ = C.vkEnumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) return AppError.NoSuitableGpu;

    const devices = try allocator.alloc(C.VkPhysicalDevice, device_count);
    defer allocator.free(devices);
    _ = C.vkEnumeratePhysicalDevices(instance, &device_count, devices.ptr);

    const physical_device = devices[0];

    // 5. Create Logical Device and Queues
    var queue_family_count: u32 = 0;
    C.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try allocator.alloc(C.VkQueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
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
    var device_features: C.VkPhysicalDeviceFeatures = .{};
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
        .presentMode = C.VK_PRESENT_MODE_FIFO_KHR,
        .clipped = C.VK_TRUE,
        .oldSwapchain = null,
    };

    var swapchain: C.VkSwapchainKHR = undefined;
    try checkVk(C.vkCreateSwapchainKHR(device, &swapchain_create_info, null, &swapchain));
    defer C.vkDestroySwapchainKHR(device, swapchain, null);

    var image_count: u32 = 0;
    _ = C.vkGetSwapchainImagesKHR(device, swapchain, &image_count, null);
    const swapchain_images = try allocator.alloc(C.VkImage, image_count);
    defer allocator.free(swapchain_images);
    _ = C.vkGetSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.ptr);

    // 7. Create Image Views
    var swapchain_image_views = try allocator.alloc(C.VkImageView, image_count);
    defer {
        for (swapchain_image_views) |view| C.vkDestroyImageView(device, view, null);
        allocator.free(swapchain_image_views);
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

    // 8. Create Render Pass
    var color_attachment = C.VkAttachmentDescription{
        .format = surface_format.format,
        .samples = C.VK_SAMPLE_COUNT_1_BIT,
        .loadOp = C.VK_ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = C.VK_ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = C.VK_ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = C.VK_ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = C.VK_IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = C.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
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

    var subpass_dep = C.VkSubpassDependency{
        .srcSubpass = C.VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = C.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = C.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = C.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    var rp_create_info = C.VkRenderPassCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        // --- MODIFIED --- Add dependency for layout transitions
        .dependencyCount = 1,
        .pDependencies = &subpass_dep,
    };

    var render_pass: C.VkRenderPass = undefined;
    try checkVk(C.vkCreateRenderPass(device, &rp_create_info, null, &render_pass));
    defer C.vkDestroyRenderPass(device, render_pass, null);

    // --- NEW 9a. Create Descriptor Set Layout (for UBO) ---
    var ubo_layout_binding = C.VkDescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = C.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = C.VK_SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = null,
    };

    var layout_info = C.VkDescriptorSetLayoutCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &ubo_layout_binding,
    };

    var descriptor_set_layout: C.VkDescriptorSetLayout = undefined;
    try checkVk(C.vkCreateDescriptorSetLayout(device, &layout_info, null, &descriptor_set_layout));
    defer C.vkDestroyDescriptorSetLayout(device, descriptor_set_layout, null);

    // --- NEW 9b. Create Graphics Pipeline ---
    const vert_module = try createShaderModule(allocator, device, vert_shader_code);
    defer C.vkDestroyShaderModule(device, vert_module, null);
    const frag_module = try createShaderModule(allocator, device, frag_shader_code);
    defer C.vkDestroyShaderModule(device, frag_module, null);

    const vert_shader_stage_info = C.VkPipelineShaderStageCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = C.VK_SHADER_STAGE_VERTEX_BIT,
        .module = vert_module,
        .pName = "main",
    };
    const frag_shader_stage_info = C.VkPipelineShaderStageCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = C.VK_SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_module,
        .pName = "main",
    };
    const shader_stages = [_]C.VkPipelineShaderStageCreateInfo{ vert_shader_stage_info, frag_shader_stage_info };

    var binding_desc = Vertex.getBindingDescription();
    var attrib_desc = Vertex.getAttributeDescriptions();
    var vertex_input_info = C.VkPipelineVertexInputStateCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_desc,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &attrib_desc,
    };

    var input_assembly = C.VkPipelineInputAssemblyStateCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = C.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = C.VK_FALSE,
    };

    var viewport = C.VkViewport{
        .x = 0.0,
        .y = 0.0,
        .width = @as(f32, @floatFromInt(capabilities.currentExtent.width)),
        .height = @as(f32, @floatFromInt(capabilities.currentExtent.height)),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };
    var scissor = C.VkRect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent };
    var viewport_state = C.VkPipelineViewportStateCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    var rasterizer = C.VkPipelineRasterizationStateCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = C.VK_FALSE,
        .rasterizerDiscardEnable = C.VK_FALSE,
        .polygonMode = C.VK_POLYGON_MODE_FILL,
        .lineWidth = 1.0,
        .cullMode = C.VK_CULL_MODE_BACK_BIT,
        .frontFace = C.VK_FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = C.VK_FALSE,
    };

    var multisampling = C.VkPipelineMultisampleStateCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = C.VK_FALSE,
        .rasterizationSamples = C.VK_SAMPLE_COUNT_1_BIT,
    };

    var color_blend_attachment = C.VkPipelineColorBlendAttachmentState{
        .colorWriteMask = C.VK_COLOR_COMPONENT_R_BIT | C.VK_COLOR_COMPONENT_G_BIT | C.VK_COLOR_COMPONENT_B_BIT | C.VK_COLOR_COMPONENT_A_BIT,
        .blendEnable = C.VK_FALSE,
    };
    var color_blending = C.VkPipelineColorBlendStateCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = C.VK_FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    var pipeline_layout_info = C.VkPipelineLayoutCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var pipeline_layout: C.VkPipelineLayout = undefined;
    try checkVk(C.vkCreatePipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout));
    defer C.vkDestroyPipelineLayout(device, pipeline_layout, null);

    var pipeline_info = C.VkGraphicsPipelineCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
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

    var graphics_pipeline: C.VkPipeline = undefined;
    try checkVk(C.vkCreateGraphicsPipelines(device, null, 1, &pipeline_info, null, &graphics_pipeline));
    defer C.vkDestroyPipeline(device, graphics_pipeline, null);

    // 10. Create Framebuffers
    var swapchain_framebuffers = try allocator.alloc(C.VkFramebuffer, image_count);
    defer {
        for (swapchain_framebuffers) |fb| C.vkDestroyFramebuffer(device, fb, null);
        allocator.free(swapchain_framebuffers);
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

    // --- NEW 11. Create Vertex Buffer ---
    const vertex_buffer_size = @sizeOf(Vertex) * vertices.len;
    const vertex_buffer_obj = try createBuffer(
        device,
        physical_device,
        vertex_buffer_size,
        C.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
        C.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | C.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer C.vkDestroyBuffer(device, vertex_buffer_obj.buffer, null);
    defer C.vkFreeMemory(device, vertex_buffer_obj.memory, null);

    // Copy vertex data to the buffer
    var data_ptr: ?*anyopaque = undefined;
    _ = C.vkMapMemory(device, vertex_buffer_obj.memory, 0, vertex_buffer_size, 0, &data_ptr);
    const mapped_memory: [*]Vertex = @ptrCast(@alignCast(data_ptr.?));
    @memcpy(mapped_memory[0..vertices.len], &vertices);
    C.vkUnmapMemory(device, vertex_buffer_obj.memory);

    // --- NEW 12. Create Uniform Buffer, Descriptor Pool and Set ---
    const ubo_size = @sizeOf(UniformBufferObject);
    const uniform_buffer_obj = try createBuffer(
        device,
        physical_device,
        ubo_size,
        C.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        C.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | C.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer C.vkDestroyBuffer(device, uniform_buffer_obj.buffer, null);
    defer C.vkFreeMemory(device, uniform_buffer_obj.memory, null);

    var pool_size = C.VkDescriptorPoolSize{
        .type = C.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, // We only need one descriptor
    };
    var pool_info = C.VkDescriptorPoolCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
        .maxSets = 1, // We only need one set
    };
    var descriptor_pool: C.VkDescriptorPool = undefined;
    try checkVk(C.vkCreateDescriptorPool(device, &pool_info, null, &descriptor_pool));
    defer C.vkDestroyDescriptorPool(device, descriptor_pool, null);

    var set_alloc_info = C.VkDescriptorSetAllocateInfo{
        .sType = C.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var descriptor_set: C.VkDescriptorSet = undefined;
    try checkVk(C.vkAllocateDescriptorSets(device, &set_alloc_info, &descriptor_set));

    var buffer_info_desc = C.VkDescriptorBufferInfo{
        .buffer = uniform_buffer_obj.buffer,
        .offset = 0,
        .range = ubo_size,
    };
    var descriptor_write = C.VkWriteDescriptorSet{
        .sType = C.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = C.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &buffer_info_desc,
    };
    C.vkUpdateDescriptorSets(device, 1, &descriptor_write, 0, null);

    // 13. Create Command Pool and Command Buffers
    var cmd_pool_info = C.VkCommandPoolCreateInfo{
        .sType = C.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = C.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_family_index,
    };
    var command_pool: C.VkCommandPool = undefined;
    try checkVk(C.vkCreateCommandPool(device, &cmd_pool_info, null, &command_pool));
    defer C.vkDestroyCommandPool(device, command_pool, null);

    var cmd_alloc_info = C.VkCommandBufferAllocateInfo{
        .sType = C.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = C.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    var command_buffer: C.VkCommandBuffer = undefined;
    try checkVk(C.vkAllocateCommandBuffers(device, &cmd_alloc_info, &command_buffer));

    // 14. Create Synchronization Objects (Semaphores and Fences)
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

    // 15. Main Loop
    const start_time = std.time.milliTimestamp();
    while (C.glfwWindowShouldClose(window) == 0) {
        C.glfwPollEvents();

        _ = C.vkWaitForFences(device, 1, &in_flight_fence, C.VK_TRUE, std.math.maxInt(u64));
        _ = C.vkResetFences(device, 1, &in_flight_fence);

        var image_index: u32 = undefined;
        _ = C.vkAcquireNextImageKHR(device, swapchain, std.math.maxInt(u64), image_available_semaphore, null, &image_index);

        _ = C.vkResetCommandBuffer(command_buffer, 0);
        var cmd_begin_info = C.VkCommandBufferBeginInfo{ .sType = C.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        _ = C.vkBeginCommandBuffer(command_buffer, &cmd_begin_info);

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
        _ = C.vkMapMemory(device, uniform_buffer_obj.memory, 0, ubo_size, 0, &ubo_data_ptr);
        const mapped_ubo: *UniformBufferObject = @ptrCast(@alignCast(ubo_data_ptr.?));
        mapped_ubo.* = ubo;
        C.vkUnmapMemory(device, uniform_buffer_obj.memory);

        // --- MODIFIED --- Render Pass now clears to black
        var clear_color = C.VkClearValue{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } };

        var rp_begin_info = C.VkRenderPassBeginInfo{
            .sType = C.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = swapchain_framebuffers[image_index],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };

        C.vkCmdBeginRenderPass(command_buffer, &rp_begin_info, C.VK_SUBPASS_CONTENTS_INLINE);
        // --- NEW --- Drawing commands
        C.vkCmdBindPipeline(command_buffer, C.VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        const vertex_buffers = [_]C.VkBuffer{vertex_buffer_obj.buffer};
        const offsets = [_]C.VkDeviceSize{0};
        C.vkCmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffers, &offsets);

        C.vkCmdBindDescriptorSets(command_buffer, C.VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, null);

        C.vkCmdDraw(command_buffer, vertices.len, 1, 0, 0);

        C.vkCmdEndRenderPass(command_buffer);
        try checkVk(C.vkEndCommandBuffer(command_buffer));

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

    _ = C.vkDeviceWaitIdle(device);
}

const Vertex = extern struct {
    pos: [2]f32,

    // This function describes how vertex data is spaced in memory.
    // It doesn't need an instance of a vertex, so there's no `self` parameter.
    pub fn getBindingDescription() C.VkVertexInputBindingDescription {
        return .{
            .binding = 0, // We are using binding 0
            .stride = @sizeOf(Vertex), // The distance in bytes between two vertices
            .inputRate = C.VK_VERTEX_INPUT_RATE_VERTEX, // Move to the next data entry after each vertex
        };
    }

    // This function describes the attributes of a single vertex (e.g., its position).
    pub fn getAttributeDescriptions() [1]C.VkVertexInputAttributeDescription {
        return .{
            // Position attribute
            .{
                .binding = 0, // Data comes from the buffer in binding 0
                .location = 0, // This is `layout(location = 0)` in the vertex shader
                .format = C.VK_FORMAT_R32G32_SFLOAT, // The format is two 32-bit floats
                .offset = @offsetOf(Vertex, "pos"), // The offset of the 'pos' field in the struct
            },
        };
    }
};
