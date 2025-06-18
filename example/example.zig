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
const vert_shader_code = spirv.vert_code;
const frag_shader_code = spirv.frag_code;

fn checkVk(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) {
        return;
    }
    std.log.err("Vulkan call failed with code: {}", .{result});
    // TODO: improve error handling
    return switch (result) {
        c.VK_INCOMPLETE => error.VulkanIncomplete,
        c.VK_ERROR_DEVICE_LOST => error.VulkanDeviceLost,
        c.VK_ERROR_OUT_OF_DEVICE_MEMORY, c.VK_ERROR_OUT_OF_HOST_MEMORY => error.VulkanMemoryAllocationFailed,
        c.VK_ERROR_LAYER_NOT_PRESENT => error.VulkanLayerMissing,
        c.VK_ERROR_INITIALIZATION_FAILED => error.VulkanInitFailed,
        c.VK_ERROR_FORMAT_NOT_SUPPORTED => error.VulkanUnsupportedFormat,
        c.VK_ERROR_UNKNOWN => error.VulkanUnknown,
        else => error.VulkanDefault, // General error
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
    pos: @Vector(3, f32), // A 3-component vector for position

    // This function tells Vulkan how to interpret the Vertex struct.
    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0, // The index of the binding
            .stride = @sizeOf(Vertex), // The size of one vertex struct
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX, // Move to the next data entry after each vertex
        };
    }

    // This function describes the attributes within the vertex (just position for us).
    pub fn getAttributeDescriptions() [1]c.VkVertexInputAttributeDescription {
        return .{
            // Position attribute
            .{
                .binding = 0, // From which binding the data comes
                .location = 0, // Corresponds to `layout(location = 0)` in the shader
                .format = c.VK_FORMAT_R32G32B32_SFLOAT, // Format is a 3-component 32-bit float vector
                .offset = @offsetOf(Vertex, "pos"), // Offset of the 'pos' field
            },
        };
    }
};

// --- Vertex Data ---
// These are the two 3D vectors that define our line.
const vertices = [_]Vertex{
    .{ .pos = .{ -0.5, -0.5, 0.0 } },
    .{ .pos = .{ 0.5, 0.5, 0.0 } },
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
    instance: c.VkInstance = undefined,
    physical_device: c.VkPhysicalDevice = undefined,
    device: c.VkDevice = undefined,
    graphics_queue: c.VkQueue = undefined,

    // Window surface (the bridge between Vulkan and the window system).
    surface: c.VkSurfaceKHR = undefined,
    surface_format: c.VkSurfaceFormatKHR = undefined,
    present_mode: c.VkPresentModeKHR = undefined,

    // Swapchain for presenting images to the screen.
    swapchain: c.VkSwapchainKHR = undefined,
    swapchain_images: []c.VkImage = undefined,
    swapchain_image_views: []c.VkImageView = undefined,
    swapchain_extent: c.VkExtent2D = undefined,

    // The Graphics Pipeline
    render_pass: c.VkRenderPass = undefined,
    descriptor_set_layout: c.VkDescriptorSetLayout = undefined,
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

    // --- Entry Point ---
    pub fn main() !void {
        // Initialize GLFW
        try checkGlfw(c.glfwInit());
        defer c.glfwTerminate();

        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();

        var app = App{ .allocator = allocator };
        defer app.cleanup();

        try app.initWindow();
        try app.initVulkan();
        try app.run();
    }

    fn initWindow(self: *App) !void {
        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_NO_API);

        self.window = c.glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan Line", null, null);
        if (self.window == null) return error.GlfwCreateWindowFailed;
    }

    fn initVulkan(app: *App) !void {
        try app.createInstance();
        try app.createSurface();
        try app.pickPhysicalDevice();
        try app.createLogicalDevice();
        try app.createSwapchain();
        try app.createImageViews();
        try app.createRenderPass();
        try app.createGraphicsPipeline();
        try app.createFramebuffers();
        try app.createCommandPool();
        try app.createVertexBuffer();
        try app.createCommandBuffer();
        try app.createSyncObjects();
    }

    // The main application loop.
    fn run(app: *App) !void {
        while (c.glfwWindowShouldClose(app.window) != 0) {
            c.glfwPollEvents();
            try app.drawFrame();
        }
        // Wait for the GPU to finish all operations before we start cleaning up.
        try checkVk(c.vkDeviceWaitIdle(app.device));
    }

    // The cleanup function, destroying Vulkan objects in reverse order of creation.
    fn cleanup(app: *App) void {
        // Destroy synchronization objects
        c.vkDestroySemaphore(app.device, app.image_available_semaphore, app.alloc_callbacks);
        c.vkDestroySemaphore(app.device, app.render_finished_semaphore, app.alloc_callbacks);
        c.vkDestroyFence(app.device, app.in_flight_fence, app.alloc_callbacks);

        // Destroy buffers and memory
        c.vkDestroyBuffer(app.device, app.vertex_buffer, app.alloc_callbacks);
        c.vkFreeMemory(app.device, app.vertex_buffer_memory, app.alloc_callbacks);

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
        // c.vkDestroyDescriptorSetLayout(app.device, app.descriptor_set_layout, app.alloc_callbacks);

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

    fn createInstance(app: *App) !void {
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
        const required_extensions = required_extensions_ptr[0..extension_count];

        const create_info = c.VkInstanceCreateInfo{
            .pApplicationInfo = &app_info,
            .enabledExtensionCount = @intCast(required_extensions.len),
            .ppEnabledExtensionNames = required_extensions.ptr,
        };

        try checkVk(c.vkCreateInstance(&create_info, app.alloc_callbacks, &app.instance));
    }

    fn createSurface(app: *App) !void {
        try checkVk(c.glfwCreateWindowSurface(app.instance, app.window, app.alloc_callbacks, &app.surface));
    }

    fn pickPhysicalDevice(app: *App) !void {
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

        // Just pick the first GPU we find.
        app.physical_device = devices[0];

        var props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(app.physical_device, &props);
        std.debug.print("Using GPU: {s}\n", .{props.deviceName});
    }

    fn findQueueFamilies(allocator: std.mem.Allocator, phys_device: c.VkPhysicalDevice, surface: c.VkSurfaceKHR) !u32 {
        var queue_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_count, null);
        const queue_family_properties = try allocator.alloc(c.VkQueueFamilyProperties, queue_count);
        defer allocator.free(queue_family_properties);

        c.vkGetPhysicalDeviceQueueFamilyProperties(phys_device, &queue_count, queue_family_properties.ptr);

        for (queue_family_properties, 0..queue_count) |family_prop, i| {
            // We need a queue that supports graphics operations.
            if (family_prop.queueFlags & c.VK_QUEUE_GRAPHICS_BIT != 0) {
                // We also need a queue that can present to our surface.
                var present_support: c.VkBool32 = c.VK_FALSE;
                _ = c.vkGetPhysicalDeviceSurfaceSupportKHR(phys_device, @intCast(i), surface, &present_support);

                // *** THE FIX IS HERE ***
                if (present_support == c.VK_TRUE) {
                    return @intCast(i);
                }
            }
        }

        return error.NoSuitableQueueFamily;
    }

    fn createLogicalDevice(app: *App) !void {
        const queue_family_index = try findQueueFamilies(app.allocator, app.physical_device, app.surface);
        const queue_priority: f32 = 1.0;
        const queue_create_info = c.VkDeviceQueueCreateInfo{
            .queueFamilyIndex = queue_family_index,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        var device_features = c.VkPhysicalDeviceFeatures{}; // We don't need any special features.

        // We need to enable the swapchain extension.
        const device_extensions = [_][*:0]const u8{c.VK_KHR_SWAPCHAIN_EXTENSION_NAME};

        const create_info = c.VkDeviceCreateInfo{
            .pQueueCreateInfos = &queue_create_info,
            .queueCreateInfoCount = 1,
            .pEnabledFeatures = &device_features,
            .enabledExtensionCount = device_extensions.len,
            .ppEnabledExtensionNames = &device_extensions,
        };

        try checkVk(c.vkCreateDevice(app.physical_device, &create_info, app.alloc_callbacks, &app.device));

        // Get a handle to the graphics queue.
        c.vkGetDeviceQueue(app.device, queue_family_index, 0, &app.graphics_queue);
    }

    fn createSwapchain(app: *App) !void {
        // --- Choose Swapchain Settings ---
        var format_count: u32 = undefined;
        try checkVk(c.vkGetPhysicalDeviceSurfaceFormatsKHR(app.physical_device, app.surface, &format_count, null));
        const formats = app.allocator.alloc(c.VkSurfaceFormatKHR, format_count) catch @panic("OOM");
        defer app.allocator.free(formats);
        try checkVk(c.vkGetPhysicalDeviceSurfaceFormatsKHR(app.physical_device, app.surface, &format_count, formats.ptr));
        app.surface_format = formats[0]; // Choose a format, B8G8R8A8_SRGB is common.

        var present_modes_count: u32 = undefined;
        try checkVk(c.vkGetPhysicalDeviceSurfacePresentModesKHR(app.physical_device, app.surface, &present_modes_count, null));
        const present_modes = app.allocator.alloc(c.VkPresentModeKHR, present_modes_count) catch @panic("OOM");
        defer app.allocator.free(present_modes);
        try checkVk(c.vkGetPhysicalDeviceSurfacePresentModesKHR(app.physical_device, app.surface, &present_modes_count, present_modes.ptr));

        app.present_mode = c.VK_PRESENT_MODE_FIFO_KHR; // V-Sync, guaranteed to be available.
        for (present_modes) |mode| {
            if (mode == c.VK_PRESENT_MODE_MAILBOX_KHR) {
                app.present_mode = c.VK_PRESENT_MODE_MAILBOX_KHR; // Triple buffering, better for latency.
                break;
            }
        }

        var capabilities: c.VkSurfaceCapabilitiesKHR = undefined;
        try checkVk(c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR(app.physical_device, app.surface, &capabilities));
        app.swapchain_extent = capabilities.currentExtent;

        var image_count = capabilities.minImageCount + 1;
        if (capabilities.maxImageCount > 0 and image_count > capabilities.maxImageCount) {
            image_count = capabilities.maxImageCount;
        }

        const create_info = c.VkSwapchainCreateInfoKHR{
            .surface = app.surface,
            .minImageCount = image_count,
            .imageFormat = app.surface_format.format,
            .imageColorSpace = app.surface_format.colorSpace,
            .imageExtent = app.swapchain_extent,
            .imageArrayLayers = 1,
            .imageUsage = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
            .imageSharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
            .preTransform = capabilities.currentTransform,
            .compositeAlpha = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            .presentMode = app.present_mode,
            .clipped = c.VK_TRUE,
            .oldSwapchain = null,
        };
        var swapchain: c.VkSwapchainKHR = undefined;
        try checkVk(c.vkCreateSwapchainKHR(app.device, &create_info, app.alloc_callbacks, &swapchain));

        var img_count: u32 = undefined;
        try checkVk(c.vkGetSwapchainImagesKHR(app.device, swapchain, &img_count, null));
        const swapchain_images = app.allocator.alloc(c.VkImage, img_count) catch @panic("OOM");
        try checkVk(c.vkGetSwapchainImagesKHR(app.device, swapchain, &img_count, swapchain_images.ptr));
        app.swapchain_images = swapchain_images;
    }

    fn createImageViews(app: *App) !void {
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

    fn createRenderPass(app: *App) !void {
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

        try checkVk(c.vkCreateRenderPass(app.device, &create_info, app.alloc_callbacks, &app.render_pass));
    }

    fn createDescriptorSetLayout(app: *App) !void {
        var ubo_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
            .pImmutableSamplers = null,
        };

        var layout_info = c.VkDescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &ubo_layout_binding,
        };

        try checkVk(c.vkCreateDescriptorSetLayout(app.device, &layout_info, app.alloc_callbacks, &app.descriptor_set_layout));
    }

    fn createShaderModule(allocator: std.mem.Allocator, device: c.VkDevice, code: []const u8, allocation_callbacks: ?*c.VkAllocationCallbacks) !c.VkShaderModule {
        std.debug.assert(code.len % 4 == 0);

        // SPIR-V needs to be aligned to a 4-byte boundary.
        // The safest way is to allocate new memory with the required alignment
        // and copy the embedded data into it.
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

    fn createGraphicsPipeline(app: *App) !void {
        const vert_shader_module = try createShaderModule(app.allocator, app.device, vert_shader_code, app.alloc_callbacks);
        const frag_shader_module = try createShaderModule(app.allocator, app.device, frag_shader_code, app.alloc_callbacks);

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
            // !!! THIS IS THE KEY PART FOR DRAWING A LINE !!!
            // We tell Vulkan to interpret the vertex data as a list of lines.
            // Every 2 vertices will form a line.
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

        const viewport_state = c.VkPipelineViewportStateCreateInfo{
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

        const multisampling = c.VkPipelineMultisampleStateCreateInfo{
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
        };

        var color_blend_attachment = c.VkPipelineColorBlendAttachmentState{
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = c.VK_FALSE,
        };
        var color_blending: c.VkPipelineColorBlendStateCreateInfo = .{};
        color_blending.sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        color_blending.logicOpEnable = c.VK_FALSE;
        // color_blending.logicOp is now VK_LOGIC_OP_COPY (0) by default
        color_blending.attachmentCount = 1;
        color_blending.pAttachments = &color_blend_attachment;
        // color_blending.blendConstants are now {0,0,0,0} by default

        // For this simple example, we don't have any uniforms, so the layout is empty.
        const pipeline_layout_info = c.VkPipelineLayoutCreateInfo{
            .setLayoutCount = 0,
            // .pSetLayouts = &app.descriptor_set_layout,
            .pSetLayouts = null,
        };
        try checkVk(c.vkCreatePipelineLayout(app.device, &pipeline_layout_info, null, &app.pipeline_layout));

        var pipeline_info: c.VkGraphicsPipelineCreateInfo = .{}; // Zero-init this too!
        pipeline_info.sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO; // Don't forget sType
        pipeline_info.stageCount = shader_stages.len;
        pipeline_info.pStages = &shader_stages;
        pipeline_info.pVertexInputState = &vertex_input_info;
        pipeline_info.pInputAssemblyState = &input_assembly;
        // .pTessellationState is null
        pipeline_info.pViewportState = &viewport_state;
        pipeline_info.pRasterizationState = &rasterizer;
        pipeline_info.pMultisampleState = &multisampling;
        // .pDepthStencilState is null
        pipeline_info.pColorBlendState = &color_blending;
        // .pDynamicState is null
        pipeline_info.layout = app.pipeline_layout;
        pipeline_info.renderPass = app.render_pass;
        pipeline_info.subpass = 0;

        std.debug.print("graphics pipeline: {any}\ninfo: {any}\n", .{ app.graphics_pipeline, pipeline_info });
        try checkVk(c.vkCreateGraphicsPipelines(app.device, null, 1, &pipeline_info, app.alloc_callbacks, &app.graphics_pipeline));
    }

    fn createFramebuffers(app: *App) !void {
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

    fn createCommandPool(app: *App) !void {
        const queue_family_index = try findQueueFamilies(app.allocator, app.physical_device, app.surface);
        var cmd_pool_info = c.VkCommandPoolCreateInfo{
            .flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = queue_family_index,
        };

        try checkVk(c.vkCreateCommandPool(app.device, &cmd_pool_info, null, &app.command_pool));
    }

    // --- Buffer Creation Helper ---
    fn findMemoryType(app: *App, type_filter: u32, properties: c.VkMemoryPropertyFlags) !u32 {
        var mem_properties: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(app.physical_device, &mem_properties);
        var i: u32 = 0;
        while (i < mem_properties.memoryTypeCount) : (i += 1) {
            if ((type_filter & (@as(u32, 1) << @intCast(i))) != 0 and (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }
        return error.NoSuitableMemoryType;
    }

    fn createBuffer(
        app: *App,
        size: c.VkDeviceSize,
        usage: c.VkBufferUsageFlags,
        properties: c.VkMemoryPropertyFlags,
    ) !struct { buffer: c.VkBuffer, memory: c.VkDeviceMemory } {
        const buffer_info = c.VkBufferCreateInfo{
            .size = size,
            .usage = usage,
            .sharingMode = c.VK_SHARING_MODE_EXCLUSIVE,
        };
        var buffer: c.VkBuffer = undefined;
        try checkVk(c.vkCreateBuffer(app.device, &buffer_info, app.alloc_callbacks, &buffer));

        var mem_requirements: c.VkMemoryRequirements = undefined;
        c.vkGetBufferMemoryRequirements(app.device, buffer, &mem_requirements);

        const alloc_info = c.VkMemoryAllocateInfo{
            .allocationSize = mem_requirements.size,
            .memoryTypeIndex = try app.findMemoryType(mem_requirements.memoryTypeBits, properties),
        };

        var memory: c.VkDeviceMemory = undefined;
        try checkVk(c.vkAllocateMemory(app.device, &alloc_info, app.alloc_callbacks, &memory));
        errdefer c.vkFreeMemory(app.device, memory, null);

        try checkVk(c.vkBindBufferMemory(app.device, buffer, memory, 0));
        return .{ .buffer = buffer, .memory = memory };
    }

    fn createVertexBuffer(app: *App) !void {
        const buffer_size = @sizeOf(Vertex) * vertices.len;

        const buffer = try app.createBuffer(
            buffer_size,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        app.vertex_buffer = buffer.buffer;
        app.vertex_buffer_memory = buffer.memory;

        // Copy vertex data to the buffer
        var data_ptr: ?*anyopaque = undefined;
        try checkVk(c.vkMapMemory(app.device, buffer.memory, 0, buffer_size, 0, &data_ptr));
        defer c.vkUnmapMemory(app.device, buffer.memory);

        const mapped_vertex_slice = @as([*]Vertex, @ptrCast(@alignCast(data_ptr)));
        @memcpy(mapped_vertex_slice, &vertices);
    }

    fn createCommandBuffer(app: *App) !void {
        const alloc_info = c.VkCommandBufferAllocateInfo{
            .commandPool = app.command_pool,
            .level = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };
        try checkVk(c.vkAllocateCommandBuffers(app.device, &alloc_info, &app.command_buffer));
    }

    fn createSyncObjects(app: *App) !void {
        const semaphore_info = c.VkSemaphoreCreateInfo{};
        const fence_info = c.VkFenceCreateInfo{
            .flags = c.VK_FENCE_CREATE_SIGNALED_BIT,
        }; // Create signaled so first frame doesn't wait.

        try checkVk(c.vkCreateSemaphore(app.device, &semaphore_info, app.alloc_callbacks, &app.image_available_semaphore));
        try checkVk(c.vkCreateFence(app.device, &fence_info, app.alloc_callbacks, &app.in_flight_fence));
    }

    // --- Per-frame Drawing Logic ---
    fn drawFrame(app: *App) !void {
        // Wait for the previous frame's fence to be signaled (meaning it's done rendering).
        try checkVk(c.vkWaitForFences(app.device, 1, &app.in_flight_fence, c.VK_TRUE, std.math.maxInt(u64)));
        try checkVk(c.vkResetFences(app.device, 1, &app.in_flight_fence));

        // Acquire an image from the swapchain.
        var image_index: u32 = 0;
        try checkVk(c.vkAcquireNextImageKHR(app.device, app.swapchain, std.math.maxInt(u64), app.image_available_semaphore, null, &image_index));

        // Reset and record the command buffer.
        try checkVk(c.vkResetCommandBuffer(app.command_buffer, 0));
        try app.recordCommandBuffer(image_index);

        // Submit the command buffer to the graphics queue.
        const wait_semaphores = [_]c.VkSemaphore{app.image_available_semaphore};
        const wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const signal_semaphores = [_]c.VkSemaphore{app.render_finished_semaphore};

        const submit_info = c.VkSubmitInfo{
            .waitSemaphoreCount = wait_semaphores.len,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &app.command_buffer,
            .signalSemaphoreCount = signal_semaphores.len,
            .pSignalSemaphores = &signal_semaphores,
        };
        try checkVk(c.vkQueueSubmit(app.graphics_queue, 1, &submit_info, app.in_flight_fence));

        // Present the image to the screen.
        const swapchains = [_]c.VkSwapchainKHR{app.swapchain};
        const present_info = c.VkPresentInfoKHR{
            .waitSemaphoreCount = signal_semaphores.len,
            .pWaitSemaphores = &signal_semaphores,
            .swapchainCount = swapchains.len,
            .pSwapchains = &swapchains,
            .pImageIndices = &image_index,
        };

        _ = c.vkQueuePresentKHR(app.graphics_queue, &present_info);
    }

    fn recordCommandBuffer(app: *App, image_index: u32) !void {
        const begin_info = c.VkCommandBufferBeginInfo{};
        try checkVk(c.vkBeginCommandBuffer(app.command_buffer, &begin_info));

        const clear_color = c.VkClearValue{ .color = .{ .float32 = .{ 0.0, 0.0, 0.0, 1.0 } } };
        const render_pass_info = c.VkRenderPassBeginInfo{
            .renderPass = app.render_pass,
            .framebuffer = app.framebuffers[@intCast(image_index)],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = app.swapchain_extent,
            },
            .clearValueCount = 1,
            .pClearValues = &clear_color,
        };
        c.vkCmdBeginRenderPass(app.command_buffer, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);

        // Bind the graphics pipeline.
        c.vkCmdBindPipeline(app.command_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, app.graphics_pipeline);

        // Bind the vertex buffer.
        const vertex_buffers = [_]c.VkBuffer{app.vertex_buffer};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(app.command_buffer, 0, 1, &vertex_buffers, &offsets);
        // c.vkCmdBindDescriptorSets(
        //     app.command_buffer,
        //     c.VK_PIPELINE_BIND_POINT_GRAPHICS,
        //     app.pipeline_layout,
        //     0,
        //     1,
        //     &app.descriptor_set,
        //     0,
        //     null,
        // );

        // !!! DRAW COMMAND !!!
        // We tell Vulkan to draw `vertices.len` (which is 2) vertices.
        // Because the topology is `line_list`, this will draw one line.
        c.vkCmdDraw(app.command_buffer, vertices.len, 1, 0, 0);

        // End the render pass and command buffer recording.
        c.vkCmdEndRenderPass(app.command_buffer);
        try checkVk(c.vkEndCommandBuffer(app.command_buffer));
    }
};

// --- Zig Entry Point ---
pub fn main() !void {
    try App.main();
}
