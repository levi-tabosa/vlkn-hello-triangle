const std = @import("std");
const vk = @import("../test.zig"); // Vulkan context and helpers
const c = @import("c").c; // C imports for Vulkan
const font = @import("font");
const util = @import("util");

const gui_vert_shader_bin = @import("spirv").gui_vs;
const gui_frag_shader_bin = @import("spirv").gui_fs;

// Callback function type for button clicks
const OnClickFn = *const fn (app: *anyopaque) void;

// Widget data variants
const WidgetData = union(enum) {
    button: ButtonData,
    plain_text: PlainTextData,

    const ButtonData = struct {
        text: []u8,
        font_size: f32 = 20.0,
        on_click: ?OnClickFn = null,
    };

    const PlainTextData = struct {
        text: []u8,
        font_size: f32 = 20.0,
    };
};

const Rect = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
};
// Rectangle structure for widget positioning
const RelativeRect = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
};

// Widget definition
pub const Widget = struct {
    rel_rect: RelativeRect,
    background: @Vector(4, f32) = .{ 0, 0, 0, 1 },
    foreground: @Vector(4, f32) = .{ 1, 1, 1, 1 },
    data: WidgetData,
};

// UI manager for widgets and string pooling
pub const UI = struct {
    const Self = @This();

    widgets: std.ArrayList(Widget),
    string_pool: util.Pool([256]u8),

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .widgets = std.ArrayList(Widget).init(allocator),
            .string_pool = util.Pool([256]u8).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.string_pool.deinit();
        self.widgets.deinit();
    }

    pub fn addButton(
        self: *Self,
        rel_rect: RelativeRect,
        text: []const u8,
        bg: @Vector(4, f32),
        fg: @Vector(4, f32),
        on_click: ?OnClickFn,
    ) !void {
        const text_copy = try self.copyString(text);
        try self.widgets.append(.{
            .rel_rect = rel_rect,
            .background = bg,
            .foreground = fg,
            .data = .{ .button = .{ .text = text_copy, .on_click = on_click } },
        });
    }

    pub fn addPlainText(
        self: *Self,
        rect: RelativeRect,
        text: []const u8,
        background: @Vector(4, f32),
        foreground: @Vector(4, f32),
    ) !void {
        const text_copy = try self.copyString(text);
        try self.widgets.append(.{
            .rel_rect = rect,
            .background = background,
            .foreground = foreground,
            .data = .{ .plain_text = .{ .text = text_copy } },
        });
    }

    fn copyString(self: *Self, text: []const u8) ![]u8 {
        const buffer = try self.string_pool.new();
        const text_copy = buffer[0..text.len];
        @memcpy(text_copy, text);
        return text_copy;
    }
};

// Vertex structure for GUI rendering
const GuiVertex = extern struct {
    pos: [2]f32,
    uv: [2]f32,
    color: [4]f32,

    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(GuiVertex),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    pub fn getAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
        return .{
            .{
                .binding = 0,
                .location = 0,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(GuiVertex, "pos"),
            },
            .{
                .binding = 0,
                .location = 1,
                .format = c.VK_FORMAT_R32G32_SFLOAT,
                .offset = @offsetOf(GuiVertex, "uv"),
            },
            .{
                .binding = 0,
                .location = 2,
                .format = c.VK_FORMAT_R32G32B32A32_SFLOAT,
                .offset = @offsetOf(GuiVertex, "color"),
            },
        };
    }
};

// Mouse input state
const MouseState = struct {
    x: f64 = 0,
    y: f64 = 0,
    left_button_down: bool = false,
};

// GUI renderer for Vulkan
pub const GuiRenderer = struct {
    const Self = @This();
    const MAX_VERTICES = 8192;
    const MAX_INDICES = MAX_VERTICES * 3 / 2;

    vk_ctx: *vk.VulkanContext,
    pipeline: vk.Pipeline,
    pipeline_layout: vk.PipelineLayout,
    push_constants: vk.PushConstantRange,
    descriptor_set_layout: c.VkDescriptorSetLayout,
    descriptor_pool: c.VkDescriptorPool,
    descriptor_set: c.VkDescriptorSet,
    sampler: c.VkSampler,
    texture: vk.Image,
    texture_view: c.VkImageView,
    dummy_pixel_uv: [2]f32,
    vertex_buffer: vk.Buffer,
    index_buffer: vk.Buffer,
    mapped_vertices: [*]GuiVertex,
    mapped_indices: [*]u32,
    vertex_count: u32 = 0,
    index_count: u32 = 0,
    font: Font,
    mouse_state: MouseState = .{},
    active_id: u32 = 0,
    hot_id: u32 = 0,
    last_id: u32 = 0,
    png_handle: PngImage,

    pub fn init(vk_ctx: *vk.VulkanContext, render_pass: vk.RenderPass) !Self {
        var self = Self{
            .vk_ctx = vk_ctx,
            .pipeline = undefined,
            .pipeline_layout = undefined,
            .push_constants = vk.PushConstantRange.init([16]f32),
            .descriptor_set_layout = undefined,
            .descriptor_pool = undefined,
            .descriptor_set = undefined,
            .sampler = undefined,
            .texture = undefined,
            .texture_view = undefined,
            .dummy_pixel_uv = undefined,
            .vertex_buffer = undefined,
            .index_buffer = undefined,
            .mapped_vertices = undefined,
            .mapped_indices = undefined,
            .font = Font.init(vk_ctx.allocator),
            .png_handle = undefined,
        };

        try self.font.loadFNT(font.periclesW01_fnt);
        try self.createTextureAndSampler(vk_ctx.allocator, font.periclesW01_png);
        try self.createDescriptors();
        try self.createBuffers();
        try self.createPipeline(render_pass);
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroySampler(self.vk_ctx.device.handle, self.sampler, null);
        c.vkDestroyImageView(self.vk_ctx.device.handle, self.texture_view, null);
        self.texture.deinit(self.vk_ctx);
        self.vk_ctx.allocator.free(self.png_handle.pixels);
        c.vkDestroyDescriptorPool(self.vk_ctx.device.handle, self.descriptor_pool, null);
        c.vkDestroyDescriptorSetLayout(self.vk_ctx.device.handle, self.descriptor_set_layout, null);
        self.pipeline_layout.deinit(self.vk_ctx);
        self.pipeline.deinit(self.vk_ctx);
        self.vertex_buffer.unmap(self.vk_ctx);
        self.index_buffer.unmap(self.vk_ctx);
        self.vertex_buffer.deinit(self.vk_ctx);
        self.index_buffer.deinit(self.vk_ctx);
        self.font.deinit();
    }

    fn createTextureAndSampler(self: *Self, allocator: std.mem.Allocator, png_data: []const u8) !void {
        const image = try loadPngGrayscale(allocator, png_data);
        self.png_handle = image;

        const bytes_per_pixel = image.bit_depth / 8;
        const new_width = image.width;
        const new_height = image.height + 1;
        const image_size: u64 = @intCast(image.width * image.height * bytes_per_pixel);
        const total_size: u64 = @intCast(new_width * new_height * bytes_per_pixel);

        var staging_buffer = try vk.Buffer.init(
            self.vk_ctx,
            total_size,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, u8);
        @memcpy(data_ptr[0..image_size], image.pixels[0..image_size]);
        @memset(data_ptr[image_size..total_size], 0xFF);
        staging_buffer.unmap(self.vk_ctx);

        self.dummy_pixel_uv = .{
            0.5 / @as(f32, @floatFromInt(new_width)),
            (@as(f32, @floatFromInt(image.height)) + 0.5) / @as(f32, @floatFromInt(new_height)),
        };

        self.texture = try vk.Image.create(
            self.vk_ctx,
            new_width,
            new_height,
            if (image.bit_depth == 8) c.VK_FORMAT_R8_UNORM else c.VK_FORMAT_R16_UNORM,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_TRANSFER_DST_BIT | c.VK_IMAGE_USAGE_SAMPLED_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );

        try self.texture.transitionLayout(self.vk_ctx, c.VK_IMAGE_LAYOUT_UNDEFINED, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        try self.texture.copyFromBuffer(self.vk_ctx, staging_buffer);
        try self.texture.transitionLayout(self.vk_ctx, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        self.texture_view = try self.texture.createView(self.vk_ctx, c.VK_IMAGE_ASPECT_COLOR_BIT);

        const sampler_info = c.VkSamplerCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
            .magFilter = c.VK_FILTER_LINEAR,
            .minFilter = c.VK_FILTER_LINEAR,
            .addressModeU = c.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeV = c.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .addressModeW = c.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
            .borderColor = c.VK_BORDER_COLOR_INT_OPAQUE_BLACK,
            .unnormalizedCoordinates = c.VK_FALSE,
        };
        try vk.vkCheck(c.vkCreateSampler(self.vk_ctx.device.handle, &sampler_info, null, &self.sampler));
    }

    fn createDescriptors(self: *Self) !void {
        // var a = try vk.DescriptorSetLayout.init(self.vk_ctx, &.{.CombinedImageSampler});
        const sampler_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        const layout_info = c.VkDescriptorSetLayoutCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &sampler_layout_binding,
        };
        try vk.vkCheck(c.vkCreateDescriptorSetLayout(self.vk_ctx.device.handle, &layout_info, null, &self.descriptor_set_layout));

        const pool_size = c.VkDescriptorPoolSize{
            .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
        };
        const pool_info = c.VkDescriptorPoolCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
            .maxSets = 1,
        };
        try vk.vkCheck(c.vkCreateDescriptorPool(self.vk_ctx.device.handle, &pool_info, null, &self.descriptor_pool));

        const set_alloc_info = c.VkDescriptorSetAllocateInfo{
            .sType = c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &self.descriptor_set_layout,
        };
        try vk.vkCheck(c.vkAllocateDescriptorSets(self.vk_ctx.device.handle, &set_alloc_info, &self.descriptor_set));

        const image_info = c.VkDescriptorImageInfo{
            .imageLayout = c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .imageView = self.texture_view,
            .sampler = self.sampler,
        };
        const desc_write = c.VkWriteDescriptorSet{
            .sType = c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = self.descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .pImageInfo = &image_info,
        };
        c.vkUpdateDescriptorSets(self.vk_ctx.device.handle, 1, &desc_write, 0, null);
    }

    fn createBuffers(self: *Self) !void {
        const vtx_buffer_size = MAX_VERTICES * @sizeOf(GuiVertex);
        self.vertex_buffer = try vk.Buffer.init(
            self.vk_ctx,
            vtx_buffer_size,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_vertices = try self.vertex_buffer.map(self.vk_ctx, GuiVertex);

        const idx_buffer_size = MAX_INDICES * @sizeOf(u32);
        self.index_buffer = try vk.Buffer.init(
            self.vk_ctx,
            idx_buffer_size,
            c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_indices = try self.index_buffer.map(self.vk_ctx, u32);
    }

    pub fn createPipeline(self: *Self, render_pass: vk.RenderPass) !void {
        self.pipeline_layout = try vk.PipelineLayout.init(self.vk_ctx, .{
            .setLayoutCount = 1,
            .pSetLayouts = &self.descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &self.push_constants.handle,
        });

        var vert_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx.device.handle, gui_vert_shader_bin);
        defer vert_mod.deinit();
        var frag_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx.device.handle, gui_frag_shader_bin);
        defer frag_mod.deinit();

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
            .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vert_mod.handle, .pName = "main" },
            .{ .sType = c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_mod.handle, .pName = "main" },
        };

        const binding_desc = GuiVertex.getBindingDescription();
        const attrib_desc = GuiVertex.getAttributeDescriptions();
        const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_desc,
            .vertexAttributeDescriptionCount = attrib_desc.len,
            .pVertexAttributeDescriptions = &attrib_desc,
        };

        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
            .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = c.VK_FALSE,
        };

        const rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_NONE,
            .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
        };

        const multisampling = c.VkPipelineMultisampleStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
        };

        const color_blend_attachment = c.VkPipelineColorBlendAttachmentState{
            .colorWriteMask = c.VK_COLOR_COMPONENT_R_BIT | c.VK_COLOR_COMPONENT_G_BIT | c.VK_COLOR_COMPONENT_B_BIT | c.VK_COLOR_COMPONENT_A_BIT,
            .blendEnable = c.VK_TRUE,
            .srcColorBlendFactor = c.VK_BLEND_FACTOR_SRC_ALPHA,
            .dstColorBlendFactor = c.VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
            .colorBlendOp = c.VK_BLEND_OP_ADD,
            .srcAlphaBlendFactor = c.VK_BLEND_FACTOR_ONE,
            .dstAlphaBlendFactor = c.VK_BLEND_FACTOR_ZERO,
            .alphaBlendOp = c.VK_BLEND_OP_ADD,
        };

        const color_blending = c.VkPipelineColorBlendStateCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
            .logicOpEnable = c.VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
        };

        const pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .sType = c.VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &color_blending,
            .layout = self.pipeline_layout.handle,
            .renderPass = render_pass.handle,
            .subpass = 0,
        };

        try vk.vkCheck(c.vkCreateGraphicsPipelines(self.vk_ctx.device.handle, null, 1, &pipeline_info, null, &self.pipeline.handle));
    }

    pub fn beginFrame(self: *Self) void {
        self.vertex_count = 0;
        self.index_count = 0;
        self.last_id = 0;
        self.hot_id = 0;
    }

    pub fn endFrame(self: *Self, cmd_buffer: c.VkCommandBuffer, window_width: f32, window_height: f32) void {
        if (self.index_count == 0) return;

        c.vkCmdBindPipeline(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.handle);
        c.vkCmdBindDescriptorSets(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, 1, &self.descriptor_set, 0, null);
        const offset: c.VkDeviceSize = 0;
        c.vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &self.vertex_buffer.handle, &offset);
        c.vkCmdBindIndexBuffer(cmd_buffer, self.index_buffer.handle, 0, c.VK_INDEX_TYPE_UINT32);

        const ortho_projection = orthoProjectionMatrix(0, window_width, 0, window_height);
        c.vkCmdPushConstants(cmd_buffer, self.pipeline_layout.handle, c.VK_SHADER_STAGE_VERTEX_BIT, 0, @sizeOf([16]f32), &ortho_projection);

        c.vkCmdDrawIndexed(cmd_buffer, self.index_count, 1, 0, 0, 0);

        if (!self.mouse_state.left_button_down) {
            self.active_id = 0;
        }
    }

    fn orthoProjectionMatrix(left: f32, right: f32, top: f32, bottom: f32) [16]f32 {
        return .{
            2.0 / (right - left),             0.0,                              0.0, 0.0,
            0.0,                              2.0 / (bottom - top),             0.0, 0.0,
            0.0,                              0.0,                              1.0, 0.0,
            -(right + left) / (right - left), -(bottom + top) / (bottom - top), 0.0, 1.0,
        };
    }

    pub fn processAndDrawUi(self: *GuiRenderer, ui_to_draw: *const UI, app: *anyopaque, width: c_int, height: c_int) void {
        const window_width: f32 = @floatFromInt(width);
        const window_height: f32 = @floatFromInt(height);

        for (ui_to_draw.widgets.items) |widget| {
            self.last_id += 1;
            const id = self.last_id;
            // Compute absolute rect from relative rect
            const rect = Rect{
                .x = widget.rel_rect.x * window_width,
                .y = widget.rel_rect.y * window_height,
                .width = widget.rel_rect.width * window_width,
                .height = widget.rel_rect.height * window_height,
            };

            if (self.isMouseOver(rect)) {
                self.hot_id = id;
                if (self.active_id == 0 and self.mouse_state.left_button_down) {
                    self.active_id = id;
                }
            }

            // Use rect for drawing and mouse interaction
            switch (widget.data) {
                .button => |data| {
                    var bg_color = widget.background;
                    // Visual feedback for hot/active states
                    if (self.hot_id == id) {
                        if (self.active_id == id) {
                            // Button is being pressed
                            bg_color = .{ widget.background[0] * 0.5, widget.background[1] * 0.5, widget.background[2] * 0.5, 1.0 };
                        } else {
                            // Button is being hovered over
                            bg_color = .{ widget.background[0] * 1.2, widget.background[1] * 1.2, widget.background[2] * 1.2, 1.0 };
                        }
                    }

                    // Draw button background
                    self.drawRect(rect, widget.background);

                    // Compute text scale based on widget height
                    const desired_text_height = 0.6 * rect.height;
                    const scale = desired_text_height / self.font.line_height;

                    // Measure text width for centering
                    const text_width = self.measureText(data.text, scale);
                    const text_x = rect.x + @max((rect.width - text_width) / 2.0, 1.0);
                    const text_y = rect.y + @max((rect.height - desired_text_height) / 2.0, 1.0);

                    // Draw text
                    self.drawText(data.text, text_x, text_y, widget.foreground, scale);

                    // Mouse interaction
                    if (self.mouse_state.left_button_down == false and self.hot_id == id and self.active_id == id) {
                        if (data.on_click) |callback| {
                            callback(app);
                        }
                    }
                },
                .plain_text => |data| {
                    // Similar logic for plain text
                    const desired_text_height = 0.8 * rect.height;
                    const scale = desired_text_height / self.font.line_height;
                    self.drawText(data.text, rect.x, rect.y, widget.foreground, scale);
                },
            }
        }
    }

    fn isMouseOver(self: *GuiRenderer, rect: Rect) bool {
        const mouse_x = self.mouse_state.x;
        const mouse_y = self.mouse_state.y;
        return mouse_x >= rect.x and
            mouse_x < rect.x + rect.width and
            mouse_y >= rect.y and
            mouse_y < rect.y + rect.height;
    }
    fn drawRect(self: *Self, rect: Rect, color: [4]f32) void {
        if (self.vertex_count + 4 > MAX_VERTICES or self.index_count + 6 > MAX_INDICES) return;

        const v_idx = self.vertex_count;
        const indices = [_]u32{ v_idx, v_idx + 1, v_idx + 2, v_idx, v_idx + 2, v_idx + 3 };
        @memcpy(self.mapped_indices[self.index_count .. self.index_count + 6], &indices);

        const uv = self.dummy_pixel_uv; // Use correct white pixel UV
        self.mapped_vertices[v_idx + 0] = .{ .pos = .{ rect.x, rect.y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 1] = .{ .pos = .{ rect.x + rect.width, rect.y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 2] = .{ .pos = .{ rect.x + rect.width, rect.y + rect.height }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 3] = .{ .pos = .{ rect.x, rect.y + rect.height }, .uv = uv, .color = color };

        self.vertex_count += 4;
        self.index_count += 6;
    }

    fn drawText(self: *Self, text: []const u8, x_start: f32, y_start: f32, color: [4]f32, scale: f32) void {
        if (self.vertex_count + 4 > MAX_VERTICES or self.index_count + 6 > MAX_INDICES) {
            std.log.warn("Text rendering limit reached: vertices {}, indices {}", .{ self.vertex_count, self.index_count });
            return;
        }

        var current_x = x_start;
        var current_y = y_start;

        for (text) |char_code| {
            if (char_code == 170) break;
            if (char_code == '\n') {
                current_x = x_start;
                current_y += self.font.line_height * scale;
                continue;
            }

            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;

            const x0 = current_x + glyph.xoffset * scale;
            const y0 = current_y + glyph.yoffset * scale;
            const x1 = x0 + @as(f32, @floatFromInt(glyph.width)) * scale;
            const y1 = y0 + @as(f32, @floatFromInt(glyph.height)) * scale;

            const @"u0" = @as(f32, @floatFromInt(glyph.x)) / self.font.scale_w;
            const v0 = @as(f32, @floatFromInt(glyph.y)) / self.font.scale_h;
            const @"u1" = @"u0" + (@as(f32, @floatFromInt(glyph.width)) / self.font.scale_w);
            const v1 = v0 + (@as(f32, @floatFromInt(glyph.height)) / self.font.scale_h);

            const v_idx = self.vertex_count;
            const indices = [_]u32{ v_idx, v_idx + 1, v_idx + 2, v_idx, v_idx + 2, v_idx + 3 };
            @memcpy(self.mapped_indices[self.index_count .. self.index_count + 6], &indices);

            self.mapped_vertices[v_idx + 0] = .{ .pos = .{ x0, y0 }, .uv = .{ @"u0", v0 }, .color = color };
            self.mapped_vertices[v_idx + 1] = .{ .pos = .{ x1, y0 }, .uv = .{ @"u1", v0 }, .color = color };
            self.mapped_vertices[v_idx + 2] = .{ .pos = .{ x1, y1 }, .uv = .{ @"u1", v1 }, .color = color };
            self.mapped_vertices[v_idx + 3] = .{ .pos = .{ x0, y1 }, .uv = .{ @"u0", v1 }, .color = color };

            self.vertex_count += 4;
            self.index_count += 6;
            current_x += glyph.xadvance * scale;
        }
    }

    fn measureText(self: *const Self, text: []const u8, scale: f32) f32 {
        var width: f32 = 0;
        for (text) |char_code| {
            if (char_code == 170) break;
            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;
            width += glyph.xadvance * scale;
        }
        return width;
    }

    pub fn handleCursorPos(self: *Self, x: f64, y: f64) void {
        self.mouse_state.x = x;
        self.mouse_state.y = y;
    }

    pub fn handleMouseButton(self: *Self, btn: c_int, action: c_int, _: c_int) void {
        if (btn == c.GLFW_MOUSE_BUTTON_LEFT) {
            self.mouse_state.left_button_down = (action == c.GLFW_PRESS);
        }
    }
};

// PNG image data structure
pub const PngImage = struct {
    width: u32,
    height: u32,
    bit_depth: u8,
    pixels: []u8,
};

// PNG parsing errors
pub const PngParseError = error{
    FileTooShort,
    InvalidSignature,
    ChunkTooShort,
    MissingIhdr,
    InvalidIhdr,
    UnsupportedFormat,
    MissingIdat,
    MissingIend,
    DecompressionError,
    InvalidFilterType,
};

fn paethPredictor(j: i64, k: i64, l: i64) i64 {
    const p = j + k - l;
    const pj = @abs(p - j);
    const pk = @abs(p - k);
    const pl = @abs(p - l);
    return if (pj <= pk and pj <= pl) j else if (pk <= pl) k else l;
}

pub fn loadPngGrayscale(allocator: std.mem.Allocator, png_data: []const u8) PngParseError!PngImage {
    const png_signature = [_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
    if (png_data.len < 8 or !std.mem.eql(u8, png_data[0..8], &png_signature)) {
        return PngParseError.InvalidSignature;
    }

    var ihdr: ?struct { width: u32, height: u32, bit_depth: u8 } = null;
    var idat_stream = std.ArrayList(u8).init(allocator);
    defer idat_stream.deinit();

    var cursor: usize = 8;
    while (cursor < png_data.len) {
        if (cursor + 8 > png_data.len) return PngParseError.ChunkTooShort;
        const data_len = std.mem.readInt(u32, @ptrCast(&png_data[cursor]), .big);
        const chunk_type = png_data[cursor + 4 .. cursor + 8];
        cursor += 8;
        if (cursor + data_len + 4 > png_data.len) return PngParseError.ChunkTooShort;
        const chunk_data = png_data[cursor .. cursor + data_len];

        if (std.mem.eql(u8, chunk_type, "IHDR")) {
            if (data_len != 13) return PngParseError.InvalidIhdr;
            const width = std.mem.readInt(u32, @ptrCast(&chunk_data[0]), .big);
            const height = std.mem.readInt(u32, @ptrCast(&chunk_data[4]), .big);
            const bit_depth = chunk_data[8];
            const color_type = chunk_data[9];
            const interlace_method = chunk_data[12];
            if (color_type != 0 or interlace_method != 0 or bit_depth != 8 and bit_depth != 16) {
                return PngParseError.UnsupportedFormat;
            }
            ihdr = .{ .width = width, .height = height, .bit_depth = bit_depth };
        } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
            if (ihdr == null) return PngParseError.MissingIhdr;
            idat_stream.appendSlice(chunk_data) catch @panic("OOM");
        } else if (std.mem.eql(u8, chunk_type, "IEND")) {
            break;
        }
        cursor += data_len + 4; // Skip CRC
    }

    const header = ihdr orelse return PngParseError.MissingIhdr;
    if (idat_stream.items.len == 0) return PngParseError.MissingIdat;

    var decompressed_buffer = std.ArrayList(u8).init(allocator);
    defer decompressed_buffer.deinit();
    {
        var compressed_reader = std.io.fixedBufferStream(idat_stream.items);
        var decompressor = std.compress.zlib.decompressor(compressed_reader.reader());

        decompressor.reader().readAllArrayList(&decompressed_buffer, std.math.maxInt(usize)) catch @panic("OOM");
    }

    const filtered_scanlines = decompressed_buffer.items;
    const bytes_per_pixel = header.bit_depth / 8;
    const scanline_length = header.width * bytes_per_pixel;
    const expected_filtered_size = header.height * (1 + scanline_length);
    if (filtered_scanlines.len != expected_filtered_size) return PngParseError.DecompressionError;

    var final_pixels = allocator.alloc(u8, header.width * header.height * bytes_per_pixel) catch @panic("OOM");
    var prior_scanline: []const u8 = &[_]u8{};

    for (0..header.height) |y| {
        const filtered_offset = y * (1 + scanline_length);
        const filter_type = filtered_scanlines[filtered_offset];
        const current_filtered = filtered_scanlines[filtered_offset + 1 ..][0..scanline_length];
        const output_offset = y * scanline_length;
        const current_output = final_pixels[output_offset..][0..scanline_length];

        switch (header.bit_depth) {
            8 => {
                for (0..scanline_length) |x| {
                    const filt_x = current_filtered[x];
                    const recon_a = if (x > 0) current_output[x - 1] else 0;
                    const recon_b = if (y > 0) prior_scanline[x] else 0;
                    const recon_c = if (x > 0 and y > 0) prior_scanline[x - 1] else 0;
                    const predictor = switch (filter_type) {
                        0 => 0,
                        1 => recon_a,
                        2 => recon_b,
                        3 => @as(u8, @intCast((@as(u16, recon_a) + @as(u16, recon_b)) / 2)),
                        4 => @as(u8, @intCast(paethPredictor(@as(i64, recon_a), @as(i64, recon_b), @as(i64, recon_c)))),
                        else => return PngParseError.InvalidFilterType,
                    };
                    current_output[x] = @addWithOverflow(filt_x, predictor)[0];
                }
            },
            16 => {
                for (0..header.width) |px_idx| {
                    const x = px_idx * 2;
                    const filt_x = std.mem.readInt(u16, @ptrCast(current_filtered[x..]), .big);
                    const recon_a = if (x > 0) std.mem.readInt(u16, @ptrCast(current_output[x - 2 ..]), .big) else 0;
                    const recon_b = if (y > 0) std.mem.readInt(u16, @ptrCast(&prior_scanline[x..]), .big) else 0;
                    const recon_c = if (x > 0 and y > 0) std.mem.readInt(u16, @ptrCast(&prior_scanline[x - 2 ..]), .big) else 0;
                    const predictor = switch (filter_type) {
                        0 => 0,
                        1 => recon_a,
                        2 => recon_b,
                        3 => @as(u16, @intCast((@as(u32, recon_a) + @as(u32, recon_b)) / 2)),
                        4 => @as(u16, @intCast(paethPredictor(@as(i64, recon_a), @as(i64, recon_b), @as(i64, recon_c)))),
                        else => return PngParseError.InvalidFilterType,
                    };
                    const recon_pixel = @addWithOverflow(filt_x, predictor)[0];
                    std.mem.writeInt(u16, @ptrCast(current_output[x..]), recon_pixel, .big);
                }
            },
            else => unreachable,
        }
        prior_scanline = current_output;
    }

    return PngImage{
        .width = header.width,
        .height = header.height,
        .bit_depth = header.bit_depth,
        .pixels = final_pixels,
    };
}

// Font glyph structure
pub const Glyph = struct {
    id: u32 = 0,
    x: u32 = 0,
    y: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,
    xoffset: f32 = 0,
    yoffset: f32 = 0,
    xadvance: f32 = 0,
};

// Font data and glyph management
pub const Font = struct {
    allocator: std.mem.Allocator,
    glyphs: std.AutoHashMap(u32, Glyph),
    line_height: f32 = 0,
    base: f32 = 0,
    scale_w: f32 = 0,
    scale_h: f32 = 0,
    font_size: f32 = 0,

    pub fn init(allocator: std.mem.Allocator) Font {
        return .{
            .allocator = allocator,
            .glyphs = std.AutoHashMap(u32, Glyph).init(allocator),
        };
    }

    pub fn deinit(self: *Font) void {
        self.glyphs.deinit();
    }

    fn parseKeyValue(part: []const u8) ?struct { key: []const u8, value: []const u8 } {
        const eq_idx = std.mem.indexOfScalar(u8, part, '=') orelse return null;
        const key = part[0..eq_idx];
        var value = part[eq_idx + 1 ..];
        if (value.len >= 2 and value[0] == '"' and value[value.len - 1] == '"') {
            value = value[1 .. value.len - 1];
        }
        return .{ .key = key, .value = value };
    }

    pub fn loadFNT(self: *Font, data: []const u8) !void {
        var lines = std.mem.splitScalar(u8, data, '\n');
        while (lines.next()) |line| {
            const trimmed_line = std.mem.trim(u8, line, " \r");
            if (trimmed_line.len == 0) continue;

            var parts = std.mem.tokenizeScalar(u8, trimmed_line, ' ');
            const tag = parts.next() orelse continue;

            if (std.mem.eql(u8, tag, "info")) {
                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse continue;
                    if (std.mem.eql(u8, kv.key, "size")) {
                        self.font_size = try std.fmt.parseFloat(f32, kv.value);
                    }
                }
            } else if (std.mem.eql(u8, tag, "common")) {
                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse continue;
                    if (std.mem.eql(u8, kv.key, "lineHeight")) self.line_height = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "base")) self.base = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "scaleW")) self.scale_w = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "scaleH")) self.scale_h = try std.fmt.parseFloat(f32, kv.value);
                }
            } else if (std.mem.eql(u8, tag, "char")) {
                var glyph = Glyph{};
                var id_parsed = false;

                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse continue;
                    if (std.mem.eql(u8, kv.key, "id")) {
                        glyph.id = try std.fmt.parseInt(u32, kv.value, 10);
                        id_parsed = true;
                    } else if (std.mem.eql(u8, kv.key, "x")) {
                        glyph.x = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "y")) {
                        glyph.y = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "width")) {
                        glyph.width = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "height")) {
                        glyph.height = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "xoffset")) {
                        glyph.xoffset = try std.fmt.parseFloat(f32, kv.value);
                    } else if (std.mem.eql(u8, kv.key, "yoffset")) {
                        glyph.yoffset = try std.fmt.parseFloat(f32, kv.value);
                    } else if (std.mem.eql(u8, kv.key, "xadvance")) {
                        glyph.xadvance = try std.fmt.parseFloat(f32, kv.value);
                    }
                }

                if (id_parsed) {
                    try self.glyphs.put(glyph.id, glyph);
                }
            }
        }
    }
};
