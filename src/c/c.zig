// c.zig
// Vulkan + GLFW C bindings and Zig-friendly re-exports

const std = @import("std");

// Import C headers and defines
const c = @cImport({
    @cDefine("GLFW_INCLUDE_VULKAN", {});
    @cInclude("vulkan/vulkan.h");
    @cInclude("GLFW/glfw3.h");
});

// Vulkan namespace: constants, types, and functions
pub const vk = struct {
    // --- Constants ---
    pub const TRUE = c.VK_TRUE;
    pub const FALSE = c.VK_FALSE;
    pub const API_VERSION = c.VK_API_VERSION_1_0;

    // Result codes
    pub const SUCCESS = c.VK_SUCCESS;
    pub const NOT_READY = c.VK_NOT_READY;
    pub const TIMEOUT = c.VK_TIMEOUT;
    pub const EVENT_SET = c.VK_EVENT_SET;
    pub const EVENT_RESET = c.VK_EVENT_RESET;
    pub const INCOMPLETE = c.VK_INCOMPLETE;

    // Error codes
    pub const ERR_OUT_OF_HOST_MEMORY = c.VK_ERROR_OUT_OF_HOST_MEMORY;
    pub const ERR_OUT_OF_DEVICE_MEMORY = c.VK_ERROR_OUT_OF_DEVICE_MEMORY;
    pub const ERR_INITIALIZATION_FAILED = c.VK_ERROR_INITIALIZATION_FAILED;
    pub const ERR_DEVICE_LOST = c.VK_ERROR_DEVICE_LOST;
    pub const ERR_MEMORY_MAP_FAILED = c.VK_ERROR_MEMORY_MAP_FAILED;
    pub const ERR_LAYER_NOT_PRESENT = c.VK_ERROR_LAYER_NOT_PRESENT;
    pub const ERR_EXTENSION_NOT_PRESENT = c.VK_ERROR_EXTENSION_NOT_PRESENT;
    pub const ERR_FEATURE_NOT_PRESENT = c.VK_ERROR_FEATURE_NOT_PRESENT;
    pub const ERR_INCOMPATIBLE_DRIVER = c.VK_ERROR_INCOMPATIBLE_DRIVER;
    pub const ERR_TOO_MANY_OBJECTS = c.VK_ERROR_TOO_MANY_OBJECTS;
    pub const ERR_FORMAT_NOT_SUPPORTED = c.VK_ERROR_FORMAT_NOT_SUPPORTED;
    pub const ERR_DRAW_FAILED = c.VK_ERROR_DRAW_FAILED_KHR;

    // Structure types
    pub const STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO = c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    pub const STRUCTURE_TYPE_APPLICATION_INFO = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
    pub const STRUCTURE_TYPE_INSTANCE_CREATE_INFO = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    pub const STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    pub const STRUCTURE_TYPE_DEVICE_CREATE_INFO = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    pub const STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO = c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    pub const STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR = c.VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    pub const STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO = c.VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    pub const STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO = c.VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    pub const STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    pub const STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO = c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pub const STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pub const STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO = c.VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
    pub const STRUCTURE_TYPE_BUFFER_CREATE_INFO = c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    pub const STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pub const STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    pub const STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    pub const STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pub const STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    pub const STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO = c.VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    pub const STRUCTURE_TYPE_FENCE_CREATE_INFO = c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    pub const STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    pub const STRUCTURE_TYPE_SUBMIT_INFO = c.VK_STRUCTURE_TYPE_SUBMIT_INFO;
    pub const STRUCTURE_TYPE_PRESENT_INFO_KHR = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pub const STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    // Pipeline, queue, and buffer flags
    pub const QUEUE_GRAPHICS_BIT = c.VK_QUEUE_GRAPHICS_BIT;
    pub const BUFFER_USAGE_VERTEX_BUFFER_BIT = c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    pub const BUFFER_USAGE_UNIFORM_BUFFER_BIT = c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    pub const MEMORY_PROPERTY_HOST_COHERENT_BIT = c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    pub const MEMORY_PROPERTY_HOST_VISIBLE_BIT = c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    pub const COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pub const COMMAND_BUFFER_LEVEL_PRIMARY = c.VK_COMMAND_BUFFER_LEVEL_PRIMARY;

    // Formats, color, and image
    pub const FORMAT_B8G8R8A8_SRGB = c.VK_FORMAT_B8G8R8A8_SRGB;
    pub const FORMAT_R32G32_SFLOAT = c.VK_FORMAT_R32G32_SFLOAT;
    pub const COLOR_SPACE_SRGB_NONLINEAR_KHR = c.VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    pub const IMAGE_USAGE_COLOR_ATTACHMENT_BIT = c.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    pub const SHARING_MODE_EXCLUSIVE = c.VK_SHARING_MODE_EXCLUSIVE;
    pub const COMPOSITE_ALPHA_OPAQUE_BIT_KHR = c.VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    pub const PRESENT_MODE_FIFO_KHR = c.VK_PRESENT_MODE_FIFO_KHR;
    pub const IMAGE_VIEW_TYPE_2D = c.VK_IMAGE_VIEW_TYPE_2D;
    pub const IMAGE_ASPECT_COLOR_BIT = c.VK_IMAGE_ASPECT_COLOR_BIT;
    pub const SAMPLE_COUNT_1_BIT = c.VK_SAMPLE_COUNT_1_BIT;
    pub const ATTACHMENT_LOAD_OP_CLEAR = c.VK_ATTACHMENT_LOAD_OP_CLEAR;
    pub const ATTACHMENT_STORE_OP_STORE = c.VK_ATTACHMENT_STORE_OP_STORE;
    pub const ATTACHMENT_LOAD_OP_DONT_CARE = c.VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    pub const ATTACHMENT_STORE_OP_DONT_CARE = c.VK_ATTACHMENT_STORE_OP_DONT_CARE;
    pub const IMAGE_LAYOUT_UNDEFINED = c.VK_IMAGE_LAYOUT_UNDEFINED;
    pub const IMAGE_LAYOUT_PRESENT_SRC_KHR = c.VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    pub const IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL = c.VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    // Pipeline and render pass
    pub const PIPELINE_BIND_POINT_GRAPHICS = c.VK_PIPELINE_BIND_POINT_GRAPHICS;
    pub const SUBPASS_EXTERNAL = c.VK_SUBPASS_EXTERNAL;
    pub const PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT = c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    pub const ACCESS_COLOR_ATTACHMENT_WRITE_BIT = c.VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    pub const DESCRIPTOR_TYPE_UNIFORM_BUFFER = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pub const SHADER_STAGE_FRAGMENT_BIT = c.VK_SHADER_STAGE_FRAGMENT_BIT;
    pub const SHADER_STAGE_VERTEX_BIT = c.VK_SHADER_STAGE_VERTEX_BIT;
    pub const PRIMITIVE_TOPOLOGY_TRIANGLE_LIST = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    pub const CULL_MODE_BACK_BIT = c.VK_CULL_MODE_BACK_BIT;
    pub const POLYGON_MODE_FILL = c.VK_POLYGON_MODE_FILL;
    pub const COLOR_COMPONENT_R_BIT = c.VK_COLOR_COMPONENT_R_BIT;
    pub const COLOR_COMPONENT_G_BIT = c.VK_COLOR_COMPONENT_G_BIT;
    pub const COLOR_COMPONENT_B_BIT = c.VK_COLOR_COMPONENT_B_BIT;
    pub const COLOR_COMPONENT_A_BIT = c.VK_COLOR_COMPONENT_A_BIT;
    pub const VERTEX_INPUT_RATE_VERTEX = c.VK_VERTEX_INPUT_RATE_VERTEX;
    pub const FRONT_FACE_COUNTER_CLOCKWISE = c.VK_FRONT_FACE_COUNTER_CLOCKWISE;
    pub const KHR_SWAPCHAIN_EXTENSION_NAME = c.VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    pub const SUBPASS_CONTENTS_INLINE = c.VK_SUBPASS_CONTENTS_INLINE;
    pub const FENCE_CREATE_SIGNALED_BIT = c.VK_FENCE_CREATE_SIGNALED_BIT;

    // --- Types ---
    pub const Result = c.VkResult;
    pub const ApplicationInfo = c.VkApplicationInfo;
    pub const InstanceCreateInfo = c.VkInstanceCreateInfo;
    pub const Instance = c.VkInstance;
    pub const SurfaceKHR = c.VkSurfaceKHR;
    pub const SurfaceCapabilitiesKHR = c.VkSurfaceCapabilitiesKHR;
    pub const DeviceCreateInfo = c.VkDeviceCreateInfo;
    pub const Device = c.VkDevice;
    pub const DeviceSize = c.VkDeviceSize;
    pub const DeviceMemory = c.VkDeviceMemory;
    pub const DeviceQueueCreateInfo = c.VkDeviceQueueCreateInfo;
    pub const Queue = c.VkQueue;
    pub const PhysicalDevice = c.VkPhysicalDevice;
    pub const PhysicalDeviceMemoryProperties = c.VkPhysicalDeviceMemoryProperties;
    pub const PhysicalDeviceFeatures = c.VkPhysicalDeviceFeatures;
    pub const QueueFamilyProperties = c.VkQueueFamilyProperties;
    pub const MemoryPropertyFlags = c.VkMemoryPropertyFlags;
    pub const BufferCreateInfo = c.VkBufferCreateInfo;
    pub const Buffer = c.VkBuffer;
    pub const BufferUsageFlags = c.VkBufferUsageFlags;
    pub const SwapchainCreateInfoKHR = c.VkSwapchainCreateInfoKHR;
    pub const SwapchainKHR = c.VkSwapchainKHR;
    pub const Image = c.VkImage;
    pub const ImageViewCreateInfo = c.VkImageViewCreateInfo;
    pub const ImageView = c.VkImageView;
    pub const AttachmentDescription = c.VkAttachmentDescription;
    pub const ShaderModuleCreateInfo = c.VkShaderModuleCreateInfo;
    pub const ShaderModule = c.VkShaderModule;
    pub const PipelineShaderSemblyStateCreateInfo = c.VkPipelineInputAssemblyStateCreateInfo;
    pub const Viewport = c.VkViewport;
    pub const Rect2D = c.VkRect2D;
    pub const FramebufferCreateInfo = c.VkFramebufferCreateInfo;
    pub const Framebuffer = c.VkFramebuffer;
    pub const MemoryRequirements = c.VkMemoryRequirements;
    pub const MemoryAllocateInfo = c.VkMemoryAllocateInfo;
    pub const DescriptorPool = c.VkDescriptorPool;
    pub const DescriptorPoolSize = c.VkDescriptorPoolSize;
    pub const DescriptorPoolCreateInfo = c.VkDescriptorPoolCreateInfo;
    pub const DescriptorSet = c.VkDescriptorSet;
    pub const DescriptorSetAllocateInfo = c.VkDescriptorSetAllocateInfo;
    pub const DescriptorBufferInfo = c.VkDescriptorBufferInfo;
    pub const DescriptorSetLayout = c.VkDescriptorSetLayout;
    pub const DescriptorSetLayoutCreateInfo = c.VkDescriptorSetLayoutCreateInfo;
    pub const DescriptorSetLayoutBinding = c.VkDescriptorSetLayoutBinding;
    pub const WriteDescriptorSet = c.VkWriteDescriptorSet;
    pub const CommandPoolCreateInfo = c.VkCommandPoolCreateInfo;
    pub const CommandPool = c.VkCommandPool;
    pub const CommandBufferAllocateInfo = c.VkCommandBufferAllocateInfo;
    pub const CommandBuffer = c.VkCommandBuffer;
    pub const FenceCreateInfo = c.VkFenceCreateInfo;
    pub const Fence = c.VkFence;
    pub const SemaphoreCreateInfo = c.VkSemaphoreCreateInfo;
    pub const Semaphore = c.VkSemaphore;
    pub const RenderPassBeginInfo = c.VkRenderPassBeginInfo;
    pub const SubmitInfo = c.VkSubmitInfo;
    pub const PresentInfoKHR = c.VkPresentInfoKHR;
    pub const VertexInputBindingDescription = c.VkVertexInputBindingDescription;
    pub const VertexInputAttributeDescription = c.VkVertexInputAttributeDescription;
    pub const GraphicsPipelineCreateInfo = c.VkGraphicsPipelineCreateInfo;
    pub const PipeipelineLayoutCreateInfo = c.VkPipelineLayoutCreateInfo;
    pub const PipelineLayout = c.VkPipelineLayout;
    pub const Pipeline = c.VkPipeline;
    pub const PipelineStageFlags = c.VkPipelineStageFlags;
    pub const PipelineLayoutCreateInfo = c.VkPipelineLayoutCreateInfo;
    pub const PipelineColorBlendStateCreateInfo = c.VkPipelineColorBlendStateCreateInfo;
    pub const PipelineColorBlendAttachmentState = c.VkPipelineColorBlendAttachmentState;
    pub const PipelineMultisampleStateCreateInfo = c.VkPipelineMultisampleStateCreateInfo;
    pub const PipelineRasterizationStateCreateInfo = c.VkPipelineRasterizationStateCreateInfo;
    pub const PipelineViewportStateCreateInfo = c.VkPipelineViewportStateCreateInfo;
    pub const PipelineInputAssemblyStateCreateInfo = c.VkPipelineInputAssemblyStateCreateInfo;
    pub const PipelineShaderStageCreateInfo = c.VkPipelineShaderStageCreateInfo;
    pub const PipelineVertexInputStateCreateInfo = c.VkPipelineVertexInputStateCreateInfo;
    pub const RenderPass = c.VkRenderPass;
    pub const RenderPassCreateInfo = c.VkRenderPassCreateInfo;
    pub const SubpassDependency = c.VkSubpassDependency;
    pub const SubpassDescription = c.VkSubpassDescription;
    pub const AttachmentReference = c.VkAttachmentReference;
    pub const SurfaceFormatKHR = c.VkSurfaceFormatKHR;
    pub const ClearValue = c.VkClearValue;
    pub const CommandBufferBeginInfo = c.VkCommandBufferBeginInfo;

    // --- Functions ---
    pub const MAKE_VERSION = c.VK_MAKE_VERSION;
    pub const createInstance = c.vkCreateInstance;
    pub const destroyInstance = c.vkDestroyInstance;
    pub const destroySurfaceKHR = c.vkDestroySurfaceKHR;
    pub const enumeratePhysicalDevices = c.vkEnumeratePhysicalDevices;
    pub const getPhysicalDeviceQueueFamilyProperties = c.vkGetPhysicalDeviceQueueFamilyProperties;
    pub const getPhysicalDeviceMemoryProperties = c.vkGetPhysicalDeviceMemoryProperties;
    pub const getPhysicalDeviceSurfaceCapabilitiesKHR = c.vkGetPhysicalDeviceSurfaceCapabilitiesKHR;
    pub const createDevice = c.vkCreateDevice;
    pub const destroyDevice = c.vkDestroyDevice;
    pub const createBuffer = c.vkCreateBuffer;
    pub const destroyBuffer = c.vkDestroyBuffer;
    pub const getDeviceQueue = c.vkGetDeviceQueue;
    pub const createSwapchainKHR = c.vkCreateSwapchainKHR;
    pub const destroySwapchainKHR = c.vkDestroySwapchainKHR;
    pub const getSwapchainImagesKHR = c.vkGetSwapchainImagesKHR;
    pub const createImageView = c.vkCreateImageView;
    pub const destroyImageView = c.vkDestroyImageView;
    pub const createShaderModule = c.vkCreateShaderModule;
    pub const destroyShaderModule = c.vkDestroyShaderModule;
    pub const createPipelineLayout = c.vkCreatePipelineLayout;
    pub const destroyPipelineLayout = c.vkDestroyPipelineLayout;
    pub const createGraphicsPipelines = c.vkCreateGraphicsPipelines;
    pub const destroyPipeline = c.vkDestroyPipeline;
    pub const createFramebuffer = c.vkCreateFramebuffer;
    pub const destroyFramebuffer = c.vkDestroyFramebuffer;
    pub const getBufferMemoryRequirements = c.vkGetBufferMemoryRequirements;
    pub const allocateMemory = c.vkAllocateMemory;
    pub const mapMemory = c.vkMapMemory;
    pub const freeMemory = c.vkFreeMemory;
    pub const bindBufferMemory = c.vkBindBufferMemory;
    pub const createDescriptorPool = c.vkCreateDescriptorPool;
    pub const destroyDescriptorPool = c.vkDestroyDescriptorPool;
    pub const updateDescriptorSets = c.vkUpdateDescriptorSets;
    pub const createCommandPool = c.vkCreateCommandPool;
    pub const destroyCommandPool = c.vkDestroyCommandPool;
    pub const allocateCommandBuffers = c.vkAllocateCommandBuffers;
    pub const createSemaphore = c.vkCreateSemaphore;
    pub const destroySemaphore = c.vkDestroySemaphore;
    pub const createFence = c.vkCreateFence;
    pub const destroyFence = c.vkDestroyFence;
    pub const unmapMemory = c.vkUnmapMemory;
    pub const cmdBeginRenderPass = c.vkCmdBeginRenderPass;
    pub const cmdBindPipeline = c.vkCmdBindPipeline;
    pub const cmdBindVertexBuffers = c.vkCmdBindVertexBuffers;
    pub const cmdBindDescriptorSets = c.vkCmdBindDescriptorSets;
    pub const allocateDescriptorSets = c.vkAllocateDescriptorSets;
    pub const cmdDraw = c.vkCmdDraw;
    pub const cmdEndRenderPass = c.vkCmdEndRenderPass;
    pub const endCommandBuffer = c.vkEndCommandBuffer;
    pub const queueSubmit = c.vkQueueSubmit;
    pub const queuePresentKHR = c.vkQueuePresentKHR;
    pub const deviceWaitIdle = c.vkDeviceWaitIdle;
    pub const createRenderPass = c.vkCreateRenderPass;
    pub const destroyRenderPass = c.vkDestroyRenderPass;
    pub const createDescriptorSetLayout = c.vkCreateDescriptorSetLayout;
    pub const destroyDescriptorSetLayout = c.vkDestroyDescriptorSetLayout;
    pub const waitForFences = c.vkWaitForFences;
    pub const resetFences = c.vkResetFences;
    pub const acquireNextImageKHR = c.vkAcquireNextImageKHR;
    pub const resetCommandBuffer = c.vkResetCommandBuffer;
    pub const beginCommandBuffer = c.vkBeginCommandBuffer;
    pub const glfwWindowShouldClose = c.glfwWindowShouldClose;
    pub const glfwPollEvents = c.glfwPollEvents;
};

// GLFW namespace: constants and functions
pub const glfw = struct {
    pub const CLIENT_API = c.GLFW_CLIENT_API;
    pub const NO_API = c.GLFW_NO_API;
    pub const TRUE = c.GLFW_TRUE;
    pub const FALSE = c.GLFW_FALSE;
    pub const Window = c.GLFWwindow;

    pub const init = c.glfwInit;
    pub const terminate = c.glfwTerminate;
    pub const windowHint = c.glfwWindowHint;
    pub const createWindow = c.glfwCreateWindow;
    pub const destroyWindow = c.glfwDestroyWindow;
    pub const getRequiredInstanceExtensions = c.glfwGetRequiredInstanceExtensions;
    pub const createWindowSurface = c.glfwCreateWindowSurface;
    pub const pollEvents = c.glfwPollEvents;
    pub const windowShouldClose = c.glfwWindowShouldClose;
};
