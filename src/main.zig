const std = @import("std");
const math = std.math;
const spirv = @import("spirv");
const c = @import("c");
const vk = c.vk;
const glfw = c.glfw;
const frag_shader_code = spirv.frag_code;
const vert_shader_code = spirv.vert_code;

fn checkVk(result: vk.Result) !void {
    if (result == vk.SUCCESS) {
        return;
    }
    std.log.err("Vulkan call failed with code: {}", .{result});
    // TODO: improve error handling
    switch (result) {
        vk.ERR_OUT_OF_DEVICE_MEMORY, vk.ERR_OUT_OF_HOST_MEMORY => return error.VulkanMemoryAllocationFailed,
        vk.ERR_LAYER_NOT_PRESENT => return error.VulkanLayerMissing,
        vk.ERR_INITIALIZATION_FAILED => return error.VulkanInitFailed,
        vk.ERR_FORMAT_NOT_SUPPORTED => return error.VulkanUnsupportedFormat,
        vk.ERR_DRAW_FAILED => return error.VulkanDrawFailed,
        vk.ERR_UNKNOWN => return error.VulkanUnknown,
        else => return error.VulkanDefault, // General error for drawing operations
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
    physical_device: vk.PhysicalDevice,
    type_filter: u32,
    properties: vk.MemoryPropertyFlags,
) !u32 {
    var mem_properties: vk.PhysicalDeviceMemoryProperties = undefined;
    vk.getPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

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
    device: vk.Device,
    physical_device: vk.PhysicalDevice,
    size: vk.DeviceSize,
    usage: vk.BufferUsageFlags,
    properties: vk.MemoryPropertyFlags,
    allocator: std.mem.Allocator,
) !struct { buffer: vk.Buffer, memory: vk.DeviceMemory } {
    _ = allocator; // Unused in this function, but can be used for custom memory management
    var buffer_info = vk.BufferCreateInfo{
        .sType = vk.STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = size,
        .usage = usage,
        .sharingMode = vk.SHARING_MODE_EXCLUSIVE,
    };
    var buffer: vk.Buffer = undefined;
    try checkVk(vk.createBuffer(device, &buffer_info, null, &buffer));

    var mem_requirements: vk.MemoryRequirements = undefined;
    vk.getBufferMemoryRequirements(device, buffer, &mem_requirements);

    var alloc_info = vk.MemoryAllocateInfo{
        .sType = vk.STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = mem_requirements.size,
        .memoryTypeIndex = try findMemoryType(physical_device, mem_requirements.memoryTypeBits, properties),
    };
    var memory: vk.DeviceMemory = undefined;
    try checkVk(vk.allocateMemory(device, &alloc_info, null, &memory));

    _ = vk.bindBufferMemory(device, buffer, memory, 0);

    return .{ .buffer = buffer, .memory = memory };
}

fn createShaderModule(allocator: std.mem.Allocator, device: vk.Device, code: []const u8) !vk.ShaderModule {
    std.debug.assert(code.len % 4 == 0);

    // SPIR-V needs to be aligned to a 4-byte boundary.
    // The safest way is to allocate new memory with the required alignment
    // and copy the embedded data into it.
    const aligned_code = try allocator.alignedAlloc(u32, @alignOf(u32), code.len / @sizeOf(u32));
    defer allocator.free(aligned_code);

    @memcpy(std.mem.sliceAsBytes(aligned_code), code);

    var create_info = vk.ShaderModuleCreateInfo{
        .sType = vk.STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = code.len,
        .pCode = aligned_code.ptr,
    };

    var shader_module: vk.ShaderModule = undefined;
    try checkVk(vk.createShaderModule(device, &create_info, null, &shader_module));
    return shader_module;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // 1. Initialize GLFW and create a window
    if (glfw.init() == glfw.FALSE) return error.GlfwInitFailed;
    defer glfw.terminate();

    glfw.windowHint(glfw.CLIENT_API, glfw.NO_API);
    const window = glfw.createWindow(800, 600, "Zig + Vulkan Triangle", null, null) orelse return error.GlfwWindowFailed;
    defer glfw.destroyWindow(window);

    // 2. Create Vulkan Instance

    var app_info = vk.ApplicationInfo{
        .sType = vk.STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "Zig Vulkan App",
        .applicationVersion = vk.MAKE_VERSION(0, 1, 0),
        .pEngineName = "No Engine",
        .engineVersion = vk.MAKE_VERSION(0, 1, 0),
        .apiVersion = vk.API_VERSION,
    };

    var extension_count: u32 = 0;
    const extension_names_ptr = glfw.getRequiredInstanceExtensions(&extension_count);
    const extension_names = extension_names_ptr[0..extension_count];

    const validation_layers = [_]?*const u8{
        &("VK_LAYER_KHRONOS_validation\x00"[0]),
    };

    var create_info = vk.InstanceCreateInfo{
        .sType = vk.STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = extension_count,
        .ppEnabledExtensionNames = extension_names.ptr,
        .enabledLayerCount = validation_layers.len,
        .ppEnabledLayerNames = &validation_layers,
    };

    var instance: vk.Instance = undefined;
    try checkVk(vk.createInstance(&create_info, null, &instance));
    defer vk.destroyInstance(instance, null);

    // 3. Create Window Surface
    var surface: vk.SurfaceKHR = undefined;
    try checkVk(glfw.createWindowSurface(instance, window, null, &surface));
    defer vk.destroySurfaceKHR(instance, surface, null);

    // 4. Pick Physical Device (GPU)
    var device_count: u32 = 0;
    _ = vk.enumeratePhysicalDevices(instance, &device_count, null);
    if (device_count == 0) return error.NoSuitableGpu;

    const devices = try allocator.alloc(vk.PhysicalDevice, device_count);
    defer allocator.free(devices);
    _ = vk.enumeratePhysicalDevices(instance, &device_count, devices.ptr);

    const physical_device = devices[0];

    // 5. Create Logical Device and Queues
    var queue_family_count: u32 = 0;
    vk.getPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, null);
    const queue_families = try allocator.alloc(vk.QueueFamilyProperties, queue_family_count);
    defer allocator.free(queue_families);
    vk.getPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, queue_families.ptr);

    var graphics_family_index: u32 = std.math.maxInt(u32);
    for (queue_families, 0..) |family, i| {
        if (family.queueFlags & vk.QUEUE_GRAPHICS_BIT != 0) {
            graphics_family_index = @intCast(i);
            break;
        }
    }
    if (graphics_family_index == std.math.maxInt(u32)) return error.NoSuitableGpu;

    const queue_priority: f32 = 1.0;
    var queue_create_info = vk.DeviceQueueCreateInfo{
        .sType = vk.STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = graphics_family_index,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const p_swapchain_ext = vk.KHR_SWAPCHAIN_EXTENSION_NAME;
    const device_extensions = [_]?*const u8{
        &p_swapchain_ext[0], // passes the *null terminated string as *char
    };
    var device_features: vk.PhysicalDeviceFeatures = .{};
    var device_create_info = vk.DeviceCreateInfo{
        .sType = vk.STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pQueueCreateInfos = &queue_create_info,
        .queueCreateInfoCount = 1,
        .pEnabledFeatures = &device_features,
        .enabledExtensionCount = device_extensions.len,
        .ppEnabledExtensionNames = &device_extensions,
    };

    var device: vk.Device = undefined;
    try checkVk(vk.createDevice(physical_device, &device_create_info, null, &device));
    defer vk.destroyDevice(device, null);

    var graphics_queue: vk.Queue = undefined;
    vk.getDeviceQueue(device, graphics_family_index, 0, &graphics_queue);

    // 6. Create Swap Chain
    var capabilities: vk.SurfaceCapabilitiesKHR = undefined;
    _ = vk.getPhysicalDeviceSurfaceCapabilitiesKHR(physical_device, surface, &capabilities);

    const surface_format = vk.SurfaceFormatKHR{
        .format = vk.FORMAT_B8G8R8A8_SRGB,
        .colorSpace = vk.COLOR_SPACE_SRGB_NONLINEAR_KHR,
    };

    var swapchain_create_info = vk.SwapchainCreateInfoKHR{
        .sType = vk.STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
        .surface = surface,
        .minImageCount = capabilities.minImageCount,
        .imageFormat = surface_format.format,
        .imageColorSpace = surface_format.colorSpace,
        .imageExtent = capabilities.currentExtent,
        .imageArrayLayers = 1,
        .imageUsage = vk.IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
        .imageSharingMode = vk.SHARING_MODE_EXCLUSIVE,
        .preTransform = capabilities.currentTransform,
        .compositeAlpha = vk.COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
        .presentMode = vk.PRESENT_MODE_FIFO_KHR,
        .clipped = vk.TRUE,
        .oldSwapchain = null,
    };

    var swapchain: vk.SwapchainKHR = undefined;
    try checkVk(vk.createSwapchainKHR(device, &swapchain_create_info, null, &swapchain));
    defer vk.destroySwapchainKHR(device, swapchain, null);

    var image_count: u32 = 0;
    _ = vk.getSwapchainImagesKHR(device, swapchain, &image_count, null);
    const swapchain_images = try allocator.alloc(vk.Image, image_count);
    defer allocator.free(swapchain_images);
    _ = vk.getSwapchainImagesKHR(device, swapchain, &image_count, swapchain_images.ptr);

    // 7. Create Image Views
    var swapchain_image_views = try allocator.alloc(vk.ImageView, image_count);
    defer {
        for (swapchain_image_views) |view| vk.destroyImageView(device, view, null);
        allocator.free(swapchain_image_views);
    }
    for (swapchain_images, 0..) |image, i| {
        var iv_create_info = vk.ImageViewCreateInfo{
            .sType = vk.STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
            .image = image,
            .viewType = vk.IMAGE_VIEW_TYPE_2D,
            .format = surface_format.format,
            .subresourceRange = .{
                .aspectMask = vk.IMAGE_ASPECT_COLOR_BIT,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
        };
        try checkVk(vk.createImageView(device, &iv_create_info, null, &swapchain_image_views[i]));
    }

    // 8. Create Render Pass
    var color_attachment = vk.AttachmentDescription{
        .format = surface_format.format,
        .samples = vk.SAMPLE_COUNT_1_BIT,
        .loadOp = vk.ATTACHMENT_LOAD_OP_CLEAR,
        .storeOp = vk.ATTACHMENT_STORE_OP_STORE,
        .stencilLoadOp = vk.ATTACHMENT_LOAD_OP_DONT_CARE,
        .stencilStoreOp = vk.ATTACHMENT_STORE_OP_DONT_CARE,
        .initialLayout = vk.IMAGE_LAYOUT_UNDEFINED,
        .finalLayout = vk.IMAGE_LAYOUT_PRESENT_SRC_KHR,
    };

    var color_attachment_ref = vk.AttachmentReference{
        .attachment = 0,
        .layout = vk.IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
    };

    var subpass = vk.SubpassDescription{
        .pipelineBindPoint = vk.PIPELINE_BIND_POINT_GRAPHICS,
        .colorAttachmentCount = 1,
        .pColorAttachments = &color_attachment_ref,
    };

    var subpass_dep = vk.SubpassDependency{
        .srcSubpass = vk.SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .srcAccessMask = 0,
        .dstStageMask = vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = vk.ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
    };

    var rp_create_info = vk.RenderPassCreateInfo{
        .sType = vk.STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO,
        .attachmentCount = 1,
        .pAttachments = &color_attachment,
        .subpassCount = 1,
        .pSubpasses = &subpass,
        // --- MODIFIED --- Add dependency for layout transitions
        .dependencyCount = 1,
        .pDependencies = &subpass_dep,
    };

    var render_pass: vk.RenderPass = undefined;
    try checkVk(vk.createRenderPass(device, &rp_create_info, null, &render_pass));
    defer vk.destroyRenderPass(device, render_pass, null);

    // --- NEW 9a. Create Descriptor Set Layout (for UBO) ---
    var ubo_layout_binding = vk.DescriptorSetLayoutBinding{
        .binding = 0,
        .descriptorType = vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .stageFlags = vk.SHADER_STAGE_FRAGMENT_BIT,
        .pImmutableSamplers = null,
    };

    var layout_info = vk.DescriptorSetLayoutCreateInfo{
        .sType = vk.STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1,
        .pBindings = &ubo_layout_binding,
    };

    var descriptor_set_layout: vk.DescriptorSetLayout = undefined;
    try checkVk(vk.createDescriptorSetLayout(device, &layout_info, null, &descriptor_set_layout));
    defer vk.destroyDescriptorSetLayout(device, descriptor_set_layout, null);

    // --- NEW 9b. Create Graphics Pipeline ---
    const vert_module = try createShaderModule(allocator, device, vert_shader_code);
    defer vk.destroyShaderModule(device, vert_module, null);
    const frag_module = try createShaderModule(allocator, device, frag_shader_code);
    defer vk.destroyShaderModule(device, frag_module, null);

    const vert_shader_stage_info = vk.PipelineShaderStageCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = vk.SHADER_STAGE_VERTEX_BIT,
        .module = vert_module,
        .pName = "main",
    };
    const frag_shader_stage_info = vk.PipelineShaderStageCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .stage = vk.SHADER_STAGE_FRAGMENT_BIT,
        .module = frag_module,
        .pName = "main",
    };
    const shader_stages = [_]vk.PipelineShaderStageCreateInfo{ vert_shader_stage_info, frag_shader_stage_info };

    var binding_desc = Vertex.getBindingDescription();
    var attrib_desc = Vertex.getAttributeDescriptions();
    var vertex_input_info = vk.PipelineVertexInputStateCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &binding_desc,
        .vertexAttributeDescriptionCount = 1,
        .pVertexAttributeDescriptions = &attrib_desc,
    };

    var input_assembly = vk.PipelineInputAssemblyStateCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        .topology = vk.PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        .primitiveRestartEnable = vk.FALSE,
    };

    var viewport = vk.Viewport{
        .x = 0.0,
        .y = 0.0,
        .width = @as(f32, @floatFromInt(capabilities.currentExtent.width)),
        .height = @as(f32, @floatFromInt(capabilities.currentExtent.height)),
        .minDepth = 0.0,
        .maxDepth = 1.0,
    };
    var scissor = vk.Rect2D{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent };
    var viewport_state = vk.PipelineViewportStateCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        .viewportCount = 1,
        .pViewports = &viewport,
        .scissorCount = 1,
        .pScissors = &scissor,
    };

    var rasterizer = vk.PipelineRasterizationStateCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
        .depthClampEnable = vk.FALSE,
        .rasterizerDiscardEnable = vk.FALSE,
        .polygonMode = vk.POLYGON_MODE_FILL,
        .lineWidth = 1.0,
        .cullMode = vk.CULL_MODE_BACK_BIT,
        .frontFace = vk.FRONT_FACE_COUNTER_CLOCKWISE,
        .depthBiasEnable = vk.FALSE,
    };

    var multisampling = vk.PipelineMultisampleStateCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
        .sampleShadingEnable = vk.FALSE,
        .rasterizationSamples = vk.SAMPLE_COUNT_1_BIT,
    };

    var color_blend_attachment = vk.PipelineColorBlendAttachmentState{
        .colorWriteMask = vk.COLOR_COMPONENT_R_BIT | vk.COLOR_COMPONENT_G_BIT | vk.COLOR_COMPONENT_B_BIT | vk.COLOR_COMPONENT_A_BIT,
        .blendEnable = vk.FALSE,
    };
    var color_blending = vk.PipelineColorBlendStateCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
        .logicOpEnable = vk.FALSE,
        .attachmentCount = 1,
        .pAttachments = &color_blend_attachment,
    };

    var pipeline_layout_info = vk.PipelineLayoutCreateInfo{
        .sType = vk.STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var pipeline_layout: vk.PipelineLayout = undefined;
    try checkVk(vk.createPipelineLayout(device, &pipeline_layout_info, null, &pipeline_layout));
    defer vk.destroyPipelineLayout(device, pipeline_layout, null);

    var pipeline_info = vk.GraphicsPipelineCreateInfo{
        .sType = vk.STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
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

    var graphics_pipeline: vk.Pipeline = undefined;
    try checkVk(vk.createGraphicsPipelines(device, null, 1, &pipeline_info, null, &graphics_pipeline));
    defer vk.destroyPipeline(device, graphics_pipeline, null);

    // 10. Create Framebuffers
    var swapchain_framebuffers = try allocator.alloc(vk.Framebuffer, image_count);
    defer {
        for (swapchain_framebuffers) |fb| vk.destroyFramebuffer(device, fb, null);
        allocator.free(swapchain_framebuffers);
    }
    for (swapchain_image_views, 0..) |view, i| {
        var fb_create_info = vk.FramebufferCreateInfo{
            .sType = vk.STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO,
            .renderPass = render_pass,
            .attachmentCount = 1,
            .pAttachments = &view,
            .width = capabilities.currentExtent.width,
            .height = capabilities.currentExtent.height,
            .layers = 1,
        };
        try checkVk(vk.createFramebuffer(device, &fb_create_info, null, &swapchain_framebuffers[i]));
    }

    // --- NEW 11. Create Vertex Buffer ---
    const vertex_buffer_size = @sizeOf(Vertex) * vertices.len;
    const vertex_buffer_obj = try createBuffer(
        device,
        physical_device,
        vertex_buffer_size,
        vk.BUFFER_USAGE_VERTEX_BUFFER_BIT,
        vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.MEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer vk.destroyBuffer(device, vertex_buffer_obj.buffer, null);
    defer vk.freeMemory(device, vertex_buffer_obj.memory, null);

    // Copy vertex data to the buffer
    var data_ptr: ?*anyopaque = undefined;
    _ = vk.mapMemory(device, vertex_buffer_obj.memory, 0, vertex_buffer_size, 0, &data_ptr);
    const mapped_memory: [*]Vertex = @ptrCast(@alignCast(data_ptr.?));
    @memcpy(mapped_memory[0..vertices.len], &vertices);
    vk.unmapMemory(device, vertex_buffer_obj.memory);

    // --- NEW 12. Create Uniform Buffer, Descriptor Pool and Set ---
    const ubo_size = @sizeOf(UniformBufferObject);
    const uniform_buffer_obj = try createBuffer(
        device,
        physical_device,
        ubo_size,
        vk.BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        vk.MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.MEMORY_PROPERTY_HOST_COHERENT_BIT,
        allocator,
    );
    defer vk.destroyBuffer(device, uniform_buffer_obj.buffer, null);
    defer vk.freeMemory(device, uniform_buffer_obj.memory, null);

    var pool_size = vk.DescriptorPoolSize{
        .type = vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1, // We only need one descriptor
    };
    var pool_info = vk.DescriptorPoolCreateInfo{
        .sType = vk.STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .poolSizeCount = 1,
        .pPoolSizes = &pool_size,
        .maxSets = 1, // We only need one set
    };
    var descriptor_pool: vk.DescriptorPool = undefined;
    try checkVk(vk.createDescriptorPool(device, &pool_info, null, &descriptor_pool));
    defer vk.destroyDescriptorPool(device, descriptor_pool, null);

    var set_alloc_info = vk.DescriptorSetAllocateInfo{
        .sType = vk.STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = descriptor_pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &descriptor_set_layout,
    };
    var descriptor_set: vk.DescriptorSet = undefined;
    try checkVk(vk.allocateDescriptorSets(device, &set_alloc_info, &descriptor_set));

    var buffer_info_desc = vk.DescriptorBufferInfo{
        .buffer = uniform_buffer_obj.buffer,
        .offset = 0,
        .range = ubo_size,
    };
    var descriptor_write = vk.WriteDescriptorSet{
        .sType = vk.STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = descriptor_set,
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorType = vk.DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1,
        .pBufferInfo = &buffer_info_desc,
    };
    vk.updateDescriptorSets(device, 1, &descriptor_write, 0, null);

    // 13. Create Command Pool and Command Buffers
    var cmd_pool_info = vk.CommandPoolCreateInfo{
        .sType = vk.STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .flags = vk.COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
        .queueFamilyIndex = graphics_family_index,
    };
    var command_pool: vk.CommandPool = undefined;
    try checkVk(vk.createCommandPool(device, &cmd_pool_info, null, &command_pool));
    defer vk.destroyCommandPool(device, command_pool, null);

    var cmd_alloc_info = vk.CommandBufferAllocateInfo{
        .sType = vk.STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = command_pool,
        .level = vk.COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };
    var command_buffer: vk.CommandBuffer = undefined;
    try checkVk(vk.allocateCommandBuffers(device, &cmd_alloc_info, &command_buffer));

    // 14. Create Synchronization Objects (Semaphores and Fences)
    var sync_create_info = vk.SemaphoreCreateInfo{ .sType = vk.STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    var fence_create_info = vk.FenceCreateInfo{ .sType = vk.STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = vk.FENCE_CREATE_SIGNALED_BIT };

    var image_available_semaphore: vk.Semaphore = undefined;
    var render_finished_semaphore: vk.Semaphore = undefined;
    var in_flight_fence: vk.Fence = undefined;

    try checkVk(vk.createSemaphore(device, &sync_create_info, null, &image_available_semaphore));
    try checkVk(vk.createSemaphore(device, &sync_create_info, null, &render_finished_semaphore));
    try checkVk(vk.createFence(device, &fence_create_info, null, &in_flight_fence));
    defer vk.destroySemaphore(device, image_available_semaphore, null);
    defer vk.destroySemaphore(device, render_finished_semaphore, null);
    defer vk.destroyFence(device, in_flight_fence, null);

    // 15. Main Loop
    const start_time = std.time.milliTimestamp();
    while (vk.glfwWindowShouldClose(window) == 0) {
        vk.glfwPollEvents();

        _ = vk.waitForFences(device, 1, &in_flight_fence, vk.TRUE, std.math.maxInt(u64));
        _ = vk.resetFences(device, 1, &in_flight_fence);

        var image_index: u32 = undefined;
        _ = vk.acquireNextImageKHR(device, swapchain, std.math.maxInt(u64), image_available_semaphore, null, &image_index);

        _ = vk.resetCommandBuffer(command_buffer, 0);
        var cmd_begin_info = vk.CommandBufferBeginInfo{ .sType = vk.STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        _ = vk.beginCommandBuffer(command_buffer, &cmd_begin_info);

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
        _ = vk.mapMemory(device, uniform_buffer_obj.memory, 0, ubo_size, 0, &ubo_data_ptr);
        const mapped_ubo: *UniformBufferObject = @ptrCast(@alignCast(ubo_data_ptr.?));
        mapped_ubo.* = ubo;
        vk.unmapMemory(device, uniform_buffer_obj.memory);

        // --- MODIFIED --- Render Pass now clears to black
        var clear_color = vk.ClearValue{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } };

        var rp_begin_info = vk.RenderPassBeginInfo{
            .sType = vk.STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = render_pass,
            .framebuffer = swapchain_framebuffers[image_index],
            .renderArea = .{ .offset = .{ .x = 0, .y = 0 }, .extent = capabilities.currentExtent },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };

        vk.cmdBeginRenderPass(command_buffer, &rp_begin_info, vk.SUBPASS_CONTENTS_INLINE);
        // --- NEW --- Drawing commands
        vk.cmdBindPipeline(command_buffer, vk.PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

        const vertex_buffers = [_]vk.Buffer{vertex_buffer_obj.buffer};
        const offsets = [_]vk.DeviceSize{0};
        vk.cmdBindVertexBuffers(command_buffer, 0, 1, &vertex_buffers, &offsets);

        vk.cmdBindDescriptorSets(command_buffer, vk.PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 1, &descriptor_set, 0, null);

        vk.cmdDraw(command_buffer, vertices.len, 1, 0, 0);

        vk.cmdEndRenderPass(command_buffer);
        try checkVk(vk.endCommandBuffer(command_buffer));

        var submit_info = vk.SubmitInfo{
            .sType = vk.STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &image_available_semaphore,
            .pWaitDstStageMask = &@as(vk.PipelineStageFlags, vk.PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT),
            .commandBufferCount = 1,
            .pCommandBuffers = &command_buffer,
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_finished_semaphore,
        };
        try checkVk(vk.queueSubmit(graphics_queue, 1, &submit_info, in_flight_fence));

        var present_info = vk.PresentInfoKHR{
            .sType = vk.STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &render_finished_semaphore,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index,
        };
        _ = vk.queuePresentKHR(graphics_queue, &present_info);
    }

    _ = vk.deviceWaitIdle(device);
}

const Vertex = extern struct {
    pos: [2]f32,

    pub fn getBindingDescription() vk.VertexInputBindingDescription {
        return .{
            .binding = 0, // We are using binding 0
            .stride = @sizeOf(Vertex), // The distance in bytes between two vertices
            .inputRate = vk.VERTEX_INPUT_RATE_VERTEX, // Move to the next data entry after each vertex
        };
    }

    // This function describes the attributes of a single vertex (e.g., its position).
    pub fn getAttributeDescriptions() [1]vk.VertexInputAttributeDescription {
        return .{
            // Position attribute
            .{
                .binding = 0, // Data comes from the buffer in binding 0
                .location = 0, // This is `layout(location = 0)` in the vertex shader
                .format = vk.FORMAT_R32G32_SFLOAT, // The format is two 32-bit floats
                .offset = @offsetOf(Vertex, "pos"), // The offset of the 'pos' field in the struct
            },
        };
    }
};
