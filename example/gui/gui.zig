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
    slider: SliderData,

    const ButtonData = struct {
        text: []u8,
        font_size: f32 = 20.0,
        on_click: ?OnClickFn = null,
    };

    const PlainTextData = struct {
        text: []u8,
        font_size: f32 = 20.0,
    };

    const SliderData = struct {
        min_value: f32,
        max_value: f32,
        value: f32,
        on_change: ?*const fn (app: *anyopaque, new_value: f32) void = null,
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

    pub fn addSlider(
        self: *Self,
        rel_rect: RelativeRect,
        min_value: f32,
        max_value: f32,
        initial_value: f32,
        bg: @Vector(4, f32),
        fg: @Vector(4, f32),
        on_change: ?*const fn (app: *anyopaque, new_value: f32) void,
    ) !void {
        try self.widgets.append(.{
            .rel_rect = rel_rect,
            .background = bg,
            .foreground = fg,
            .data = .{ .slider = .{
                .min_value = min_value,
                .max_value = max_value,
                .value = initial_value,
                .on_change = on_change,
            } },
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
    descriptor_set_layout: vk.DescriptorSetLayout,
    descriptor_pool: vk.DescriptorPool,
    descriptor_set: vk.DescriptorSet,
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

        try self.font.loadFNT(font.hermit_light_fnt);
        try self.createTextureAndSampler(vk_ctx.allocator, font.hermit_light_png);
        self.font.scale_h = @as(f32, @floatFromInt(self.png_handle.height + 1));
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
        self.descriptor_pool.deinit(self.vk_ctx);
        self.descriptor_set_layout.deinit(self.vk_ctx);
        self.pipeline_layout.deinit(self.vk_ctx);
        self.pipeline.deinit(self.vk_ctx);
        self.vertex_buffer.unmap(self.vk_ctx);
        self.index_buffer.unmap(self.vk_ctx);
        self.vertex_buffer.deinit(self.vk_ctx);
        self.index_buffer.deinit(self.vk_ctx);
        self.font.deinit();
    }

    fn createTextureAndSampler(self: *Self, allocator: std.mem.Allocator, png_data: []const u8) !void {
        const image = try loadPng(allocator, png_data);
        self.png_handle = image;

        const bytes_per_pixel = image.bit_depth / 8;
        std.debug.print("bytes per pixel: {}\n", .{bytes_per_pixel});
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
        self.descriptor_set_layout = try vk.DescriptorSetLayout.init(self.vk_ctx, &.{
            .{
                .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                .stage_flags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
            },
        });

        self.descriptor_pool = try vk.DescriptorPool.init(
            self.vk_ctx,
            1,
            &.{
                .{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .count = 1 },
            },
        );

        self.descriptor_set = try self.descriptor_pool.allocateSet(
            self.vk_ctx,
            self.descriptor_set_layout,
        );

        try self.descriptor_set.update(self.vk_ctx, &.{
            .{
                .binding = 0,
                .info = .{
                    .CombinedImageSampler = .{
                        .imageLayout = c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                        .imageView = self.texture_view,
                        .sampler = self.sampler,
                    },
                },
            },
        });
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
            .pSetLayouts = &self.descriptor_set_layout.handle,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &self.push_constants.handle,
        });

        var vert_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx, gui_vert_shader_bin);
        defer vert_mod.deinit(self.vk_ctx);
        var frag_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx, gui_frag_shader_bin);
        defer frag_mod.deinit(self.vk_ctx);

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
            .{ .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vert_mod.handle, .pName = "main" },
            .{ .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_mod.handle, .pName = "main" },
        };

        const binding_desc = GuiVertex.getBindingDescription();
        const attrib_desc = GuiVertex.getAttributeDescriptions();
        const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_desc,
            .vertexAttributeDescriptionCount = attrib_desc.len,
            .pVertexAttributeDescriptions = &attrib_desc,
        };

        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
            .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
            .primitiveRestartEnable = c.VK_FALSE,
        };

        const rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_NONE,
            .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
        };

        const multisampling = c.VkPipelineMultisampleStateCreateInfo{
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
            .logicOpEnable = c.VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
        };

        const pipeline_info = c.VkGraphicsPipelineCreateInfo{
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
        c.vkCmdBindDescriptorSets(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, 1, &self.descriptor_set.handle, 0, null);
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

        for (ui_to_draw.widgets.items) |*widget| {
            self.last_id += 1;
            const id = self.last_id;
            // Compute absolute rect from relative rect
            const rect = Rect{
                .x = widget.rel_rect.x * window_width,
                .y = widget.rel_rect.y * window_height,
                .width = widget.rel_rect.width * window_width,
                .height = widget.rel_rect.height * window_height,
            };

            // const is_mouse_over = ;
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
                .slider => |*data| {
                    // --- Drawing ---
                    // Draw Slider Track (the background line)
                    const track_color = .{ 0.3, 0.3, 0.3, 1.0 };
                    self.drawRect(rect, track_color);

                    // Calculate handle position
                    const slider_width = rect.width;
                    const handle_width: f32 = 10.0; // Width of the slider handle (pixels)
                    const normalized_value = (data.value - data.min_value) / (data.max_value - data.min_value);
                    const handle_x = rect.x + (slider_width - handle_width) * normalized_value;
                    const handle_rect = Rect{
                        .x = handle_x,
                        .y = rect.y,
                        .width = handle_width,
                        .height = rect.height,
                    };

                    // Choose a color for the handle: darker if active, lighter if hot, normal otherwise
                    var handle_color = widget.foreground;
                    if (self.active_id == id) {
                        handle_color = .{ handle_color[0] * 0.5, handle_color[1] * 0.5, handle_color[2] * 0.5, 1.0 };
                    } else if (self.hot_id == id) {
                        handle_color = .{ handle_color[0] * 1.2, handle_color[1] * 1.2, handle_color[2] * 1.2, 1.0 };
                    }

                    // Draw the Slider Handle
                    self.drawRect(handle_rect, handle_color); // or drawRoundedRect for nicer visuals

                    // --- Interaction ---

                    const is_over_handle = self.isMouseOverSliderHandle(rect, data.value, data.min_value, data.max_value);

                    if (is_over_handle) {
                        self.hot_id = id;
                        if (self.active_id == 0 and self.mouse_state.left_button_down) {
                            self.active_id = id;
                        }
                    }

                    // If the slider is active, update the value based on mouse position
                    if (self.active_id == id and self.mouse_state.left_button_down) {
                        // Calculate new value based on mouse X position
                        var new_value = data.min_value + ((self.mouse_state.x - rect.x) / slider_width) * (data.max_value - data.min_value);

                        // Clamp the value to the allowed range
                        new_value = @max(new_value, data.min_value);
                        new_value = @min(new_value, data.max_value);

                        // Update the value
                        data.value = @floatCast(new_value);

                        // Call the onChange callback, if provided
                        if (data.on_change) |callback| {
                            callback(app, @floatCast(new_value));
                        }
                    }
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

    fn isMouseOverSliderHandle(self: *GuiRenderer, rect: Rect, slider_value: f32, min_value: f32, max_value: f32) bool {
        const slider_width = rect.width;
        const handle_width: f32 = 10.0; // Width of the slider handle (pixels)

        // Calculate handle position
        const normalized_value = (slider_value - min_value) / (max_value - min_value);
        const handle_x = rect.x + (slider_width - handle_width) * normalized_value;

        const mouse_x = self.mouse_state.x;
        const mouse_y = self.mouse_state.y;

        return mouse_x >= handle_x and
            mouse_x < handle_x + handle_width and
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

    fn drawRoundedRect(self: *Self, rect: Rect, color: [4]f32, radius: f32) void {
        _ = radius;
        // Currently this is a stub. We can approximate it with several rectangles and triangles.
        // A real implementation would involve better geometry.
        self.drawRect(rect, color);
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

// paethPredictor is required for PNG filter type 4.
fn paethPredictor(a: i64, b: i64, C: i64) i64 {
    const p = a + b - C;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - C);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return C;
}

pub const PngParseError = error{
    InvalidSignature,
    ChunkTooShort,
    InvalidIhdr,
    UnsupportedFormat,
    MissingIhdr,
    MissingIdat,
    DecompressionError,
    InvalidFilterType,
    UnsupportedInterlace,
};

pub const PngImage = struct {
    width: u32,
    height: u32,
    bit_depth: u8,
    channels: u8, // 1: Gray, 3: RGB, 4: RGBA
    pixels: []u8,
};

// A helper function to unfilter a single scanline. This is used for both
// non-interlaced and each pass of interlaced images.
fn unfilterScanline(
    output_scanline: []u8,
    filtered_scanline: []const u8,
    prior_scanline: []const u8,
    bytes_per_pixel: u8,
    filter_type: u8,
) PngParseError!void {
    const scanline_len = filtered_scanline.len;
    if (output_scanline.len != scanline_len) @panic("Mismatched scanline lengths");

    for (0..scanline_len) |x| {
        const filt_x = filtered_scanline[x];
        const recon_a = if (x >= bytes_per_pixel) output_scanline[x - bytes_per_pixel] else 0;
        const recon_b = if (prior_scanline.len > 0) prior_scanline[x] else 0;
        const recon_c = if (prior_scanline.len > 0 and x >= bytes_per_pixel) prior_scanline[x - bytes_per_pixel] else 0;

        const predictor = switch (filter_type) {
            0 => 0, // None
            1 => recon_a, // Sub
            2 => recon_b, // Up
            3 => @as(u8, @intCast((@as(u16, recon_a) + @as(u16, recon_b)) / 2)), // Average
            4 => @as(u8, @intCast(paethPredictor(@as(i64, recon_a), @as(i64, recon_b), @as(i64, recon_c)))), // Paeth
            else => return PngParseError.InvalidFilterType,
        };
        output_scanline[x] = @addWithOverflow(filt_x, predictor)[0];
    }
}

// TODO: make return error set
pub fn loadPng(allocator: std.mem.Allocator, png_data: []const u8) !PngImage {
    const png_signature = [_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
    if (png_data.len < 8 or !std.mem.eql(u8, png_data[0..8], &png_signature)) {
        return PngParseError.InvalidSignature;
    }

    var ihdr: ?struct {
        width: u32,
        height: u32,
        bit_depth: u8,
        channels: u8,
        interlace_method: u8,
    } = null;
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

            // For now, only 8-bit depth is supported by the unified unfilter function.
            if (bit_depth != 8) {
                std.log.warn("Unsupported bit depth supported by unified unfilter function: {d}", .{bit_depth});
                // return error.UnsupportedFormat;
            }

            const channels: u8 = switch (color_type) {
                0 => 1, // Grayscale
                2 => 3, // Truecolor (RGB)
                6 => 4, // Truecolor with alpha (RGBA)
                else => return PngParseError.UnsupportedFormat,
            };

            // Check for valid interlace method
            if (interlace_method > 1) return PngParseError.UnsupportedInterlace;

            ihdr = .{
                .width = width,
                .height = height,
                .bit_depth = bit_depth,
                .channels = channels,
                .interlace_method = interlace_method,
            };
        } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
            if (ihdr == null) return PngParseError.MissingIhdr;
            try idat_stream.appendSlice(chunk_data);
        } else if (std.mem.eql(u8, chunk_type, "IEND")) {
            break;
        }
        cursor += data_len + 4; // Skip CRC
    }

    const header = ihdr orelse return PngParseError.MissingIhdr;
    if (idat_stream.items.len == 0) return PngParseError.MissingIdat;
    if (header.width == 0 or header.height == 0) {
        return PngImage{ .width = 0, .height = 0, .bit_depth = header.bit_depth, .channels = header.channels, .pixels = &[_]u8{} };
    }

    var decompressed_buffer = std.ArrayList(u8).init(allocator);
    defer decompressed_buffer.deinit();
    {
        var compressed_reader = std.io.fixedBufferStream(idat_stream.items);
        var decompressor = std.compress.zlib.decompressor(compressed_reader.reader());
        try decompressor.reader().readAllArrayList(&decompressed_buffer, std.math.maxInt(usize));
    }

    const bytes_per_pixel = header.channels * (header.bit_depth / 8);
    const final_pixels = try allocator.alloc(u8, header.width * header.height * bytes_per_pixel);

    if (header.interlace_method == 0) {
        // --- NON-INTERLACED PATH ---
        const scanline_length = header.width * bytes_per_pixel;
        const expected_filtered_size = header.height * (1 + scanline_length);
        if (decompressed_buffer.items.len != expected_filtered_size) return PngParseError.DecompressionError;

        const filtered_scanlines = decompressed_buffer.items;
        var prior_scanline: []const u8 = &[_]u8{};

        for (0..header.height) |y| {
            const filtered_offset = y * (1 + scanline_length);
            const filter_type = filtered_scanlines[filtered_offset];
            const current_filtered = filtered_scanlines[filtered_offset + 1 ..][0..scanline_length];

            const output_offset = y * scanline_length;
            const current_output = final_pixels[output_offset..][0..scanline_length];

            try unfilterScanline(current_output, current_filtered, prior_scanline, bytes_per_pixel, filter_type);

            prior_scanline = current_output;
        }
    } else if (header.interlace_method == 1) {
        // --- INTERLACED (ADAM7) PATH ---
        const adam7_passes = [_]struct { x_start: u32, y_start: u32, x_step: u32, y_step: u32 }{
            .{ .x_start = 0, .y_start = 0, .x_step = 8, .y_step = 8 },
            .{ .x_start = 4, .y_start = 0, .x_step = 8, .y_step = 8 },
            .{ .x_start = 0, .y_start = 4, .x_step = 4, .y_step = 8 },
            .{ .x_start = 2, .y_start = 0, .x_step = 4, .y_step = 4 },
            .{ .x_start = 0, .y_start = 2, .x_step = 2, .y_step = 4 },
            .{ .x_start = 1, .y_start = 0, .x_step = 2, .y_step = 2 },
            .{ .x_start = 0, .y_start = 1, .x_step = 1, .y_step = 2 },
        };

        var data_offset: usize = 0;
        const filtered_data = decompressed_buffer.items;

        for (adam7_passes) |pass| {
            const pass_width = (header.width - pass.x_start + pass.x_step - 1) / pass.x_step;
            const pass_height = (header.height - pass.y_start + pass.y_step - 1) / pass.y_step;

            if (pass_width == 0 or pass_height == 0) continue;

            const pass_scanline_len = pass_width * bytes_per_pixel;
            var prior_pass_scanline: []u8 = &[_]u8{};

            var pass_unfiltered_data = try allocator.alloc(u8, pass_scanline_len * pass_height);
            defer allocator.free(pass_unfiltered_data);

            for (0..pass_height) |pass_y| {
                const filter_type = filtered_data[data_offset];
                data_offset += 1;
                const filtered_pass_scanline = filtered_data[data_offset..][0..pass_scanline_len];
                data_offset += pass_scanline_len;

                const current_pass_output = pass_unfiltered_data[pass_y * pass_scanline_len ..][0..pass_scanline_len];
                try unfilterScanline(current_pass_output, filtered_pass_scanline, prior_pass_scanline, bytes_per_pixel, filter_type);
                prior_pass_scanline = current_pass_output;
            }

            for (0..pass_height) |pass_y| {
                for (0..pass_width) |pass_x| {
                    const final_y = pass.y_start + (pass_y * pass.y_step);
                    const final_x = pass.x_start + (pass_x * pass.x_step);
                    if (final_y >= header.height or final_x >= header.width) continue;

                    const src_offset = (pass_y * pass_scanline_len) + (pass_x * bytes_per_pixel);
                    const dst_offset = (final_y * header.width * bytes_per_pixel) + (final_x * bytes_per_pixel);

                    const src_pixel = pass_unfiltered_data[src_offset..][0..bytes_per_pixel];
                    const dst_pixel = final_pixels[dst_offset..][0..bytes_per_pixel];
                    @memcpy(dst_pixel, src_pixel);
                }
            }
        }
    }

    return PngImage{
        .width = header.width,
        .height = header.height,
        .bit_depth = header.bit_depth,
        .channels = header.channels,
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
