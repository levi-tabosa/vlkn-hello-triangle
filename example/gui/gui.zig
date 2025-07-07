const std = @import("std");
const assert = std.debug.assert;
const vk = @import("../test.zig"); // Reuse structs and helpers from main file
const c = @import("c").c; // Reuse cImport from main file
const font = @import("font");

const gui_vert_shader_bin = @import("spirv").gui_vs;
const gui_frag_shader_bin = @import("spirv").gui_fs;

const OnClickFn = *const fn (app: *anyopaque) void;

const WidgetData = union(enum) {
    button: struct {
        text: []const u8,
        font_size: f32 = 18.0, // Default value
        rect_color: @Vector(4, f32) = .{ 0.8, 0.8, 0.8, 1.0 }, // White rectangle,
        text_color: @Vector(4, f32) = .{ 0.0, 0.0, 0.0, 1.0 }, // Black text,
    },
};

pub const Widget = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    on_click: ?OnClickFn = null,
    data: WidgetData,
};

pub const UI = struct {
    const Self = @This();

    widgets: std.ArrayList(Widget),

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .widgets = std.ArrayList(Widget).init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.widgets.deinit();
    }

    pub fn addButton(self: *Self, widget: Widget) !void {
        assert(widget.data == .button); // Ensure it's a button
        try self.widgets.append(widget);
    }
};

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

const MouseState = struct {
    x: f64 = 0,
    y: f64 = 0,
    left_button_down: bool = false,
};

pub const GuiRenderer = struct {
    const Self = @This();

    vk_ctx: *vk.VulkanContext,
    pipeline: vk.Pipeline = undefined,
    pipeline_layout: vk.PipelineLayout = undefined,

    push_constants: vk.PushConstantRange = .init([16]f32),
    descriptor_set_layout: c.VkDescriptorSetLayout = undefined,
    descriptor_pool: c.VkDescriptorPool = undefined,
    descriptor_set: c.VkDescriptorSet = undefined,

    sampler: c.VkSampler = undefined,
    texture: vk.Image = undefined,
    texture_view: c.VkImageView = undefined,
    dummy_pixel_uv: [2]f32 = undefined, // UV coords for the white pixel in the atlas

    vertex_buffer: vk.Buffer = undefined,
    index_buffer: vk.Buffer = undefined,
    mapped_vertices: [*]GuiVertex = undefined,
    mapped_indices: [*]u32 = undefined,
    vertex_count: u32 = 0,
    index_count: u32 = 0,

    font: Font = undefined,

    mouse_state: MouseState = .{},
    active_id: u32 = 0,
    hot_id: u32 = 0,
    last_id: u32 = 0,
    png_handle: PngImage = undefined,

    const MAX_VERTICES = 8192;
    const MAX_INDICES = MAX_VERTICES * 3 / 2;

    pub fn init(vk_ctx: *vk.VulkanContext, render_pass: vk.RenderPass, swapchain: vk.Swapchain) !Self {
        var self: Self = .{
            .vk_ctx = vk_ctx,
            .font = .init(vk_ctx.allocator),
        };

        try self.font.loadFNT(font.periclesW01_fnt);
        try self.createUnifiedTextureAndSampler(vk_ctx.allocator, font.periclesW01_png);
        try self.createDescriptors();
        try self.createBuffers(vk_ctx);
        try self.createPipeline(render_pass, swapchain);
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroySampler(self.vk_ctx.device.handle, self.sampler, null);
        c.vkDestroyImageView(self.vk_ctx.device.handle, self.texture_view, null);
        self.texture.deinit(self.vk_ctx);
        self.vk_ctx.allocator.free(self.png_handle.pixels);

        c.vkDestroyDescriptorPool(self.vk_ctx.device.handle, self.descriptor_pool, null);
        c.vkDestroyDescriptorSetLayout(self.vk_ctx.device.handle, self.descriptor_set_layout, null);
        self.font.deinit();

        // Deinit unified buffers
        self.vertex_buffer.unmap(self.vk_ctx);
        self.index_buffer.unmap(self.vk_ctx);
        self.vertex_buffer.deinit(self.vk_ctx);
        self.index_buffer.deinit(self.vk_ctx);
    }

    // This function creates a single texture with the font atlas AND a 1x1 white pixel.
    fn createUnifiedTextureAndSampler(self: *Self, allocator: std.mem.Allocator, png_data: []const u8) !void {
        const image = try loadPngGrayscale(allocator, png_data);
        self.png_handle = image;

        const bytes_per_pixel = image.bit_depth / 8;
        // We'll add the white pixel at the bottom, so the new height is original height + 1
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
        // Copy original font data
        @memcpy(data_ptr[0..image_size], image.pixels[0..image_size]);
        // Add the white pixel data at the end. For R8_UNORM, 255 is white.
        @memset(data_ptr[image_size..total_size], 255);
        staging_buffer.unmap(self.vk_ctx);

        // Calculate the UV coordinates for the top-left corner of our 1x1 white pixel
        // We add a half-pixel offset to ensure we sample from the center of the pixel, avoiding bleeding.
        self.dummy_pixel_uv = .{
            (0.5 / @as(f32, @floatFromInt(new_width))),
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
        const sampler_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        const layout_info = c.VkDescriptorSetLayoutCreateInfo{
            .bindingCount = 1,
            .pBindings = &sampler_layout_binding,
        };
        try vk.vkCheck(c.vkCreateDescriptorSetLayout(self.vk_ctx.device.handle, &layout_info, null, &self.descriptor_set_layout));

        const pool_size = c.VkDescriptorPoolSize{
            .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
        };
        const pool_info = c.VkDescriptorPoolCreateInfo{
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
            .maxSets = 1,
        };
        try vk.vkCheck(c.vkCreateDescriptorPool(self.vk_ctx.device.handle, &pool_info, null, &self.descriptor_pool));

        const set_alloc_info = c.VkDescriptorSetAllocateInfo{
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
            .dstSet = self.descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .pImageInfo = &image_info,
        };
        c.vkUpdateDescriptorSets(self.vk_ctx.device.handle, 1, &desc_write, 0, null);
    }

    fn createBuffers(self: *Self, vk_ctx: *vk.VulkanContext) !void {
        const vtx_buffer_size = MAX_VERTICES * @sizeOf(GuiVertex);
        self.vertex_buffer = try vk.Buffer.init(
            vk_ctx,
            vtx_buffer_size,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_vertices = try self.vertex_buffer.map(vk_ctx, GuiVertex);

        const idx_buffer_size = MAX_INDICES * @sizeOf(u32);
        self.index_buffer = try vk.Buffer.init(
            vk_ctx,
            idx_buffer_size,
            c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_indices = try self.index_buffer.map(vk_ctx, u32);
    }

    pub fn createPipeline(self: *Self, render_pass: vk.RenderPass, swapchain: vk.Swapchain) !void {
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
            .srcAlphaBlendFactor = 1,
            .dstAlphaBlendFactor = 0,
            .alphaBlendOp = c.VK_BLEND_OP_ADD,
        };
        const color_blending = c.VkPipelineColorBlendStateCreateInfo{
            .logicOpEnable = c.VK_FALSE,
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
        };
        const depth_stencil = c.VkPipelineDepthStencilStateCreateInfo{
            .depthTestEnable = c.VK_FALSE,
            .depthWriteEnable = c.VK_FALSE,
        };

        const viewport = c.VkViewport{
            .x = 0.0,
            .y = 0.0,
            .width = @floatFromInt(swapchain.extent.width),
            .height = @floatFromInt(swapchain.extent.height),
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };
        const scissor = c.VkRect2D{
            .offset = .{ .x = 0, .y = 0 },
            .extent = swapchain.extent,
        };

        const viewport_state = c.VkPipelineViewportStateCreateInfo{
            .viewportCount = 1,
            .pViewports = &viewport,
            .scissorCount = 1,
            .pScissors = &scissor,
        };

        const pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &color_blending,
            .pDepthStencilState = &depth_stencil,
            .pViewportState = &viewport_state,
            .layout = self.pipeline_layout.handle,
            .renderPass = render_pass.handle,
            .subpass = 0,
        };

        try vk.vkCheck(c.vkCreateGraphicsPipelines(self.vk_ctx.device.handle, null, 1, &pipeline_info, null, &self.pipeline.handle));
    }

    pub fn destroyPipeline(self: *Self) void {
        self.pipeline_layout.deinit(self.vk_ctx);
        self.pipeline.deinit(self.vk_ctx);
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

        const L: f32 = 0;
        const R = window_width;
        const T: f32 = 0;
        const B = window_height;
        const ortho_projection = [_]f32{
            2.0 / (R - L),      0.0,                0.0, 0.0,
            0.0,                2.0 / (B - T),      0.0, 0.0,
            0.0,                0.0,                1.0, 0.0,
            -(R + L) / (R - L), -(B + T) / (B - T), 0.0, 1.0,
        };
        c.vkCmdPushConstants(cmd_buffer, self.pipeline_layout.handle, c.VK_SHADER_STAGE_VERTEX_BIT, 0, @sizeOf([16]f32), &ortho_projection);

        c.vkCmdDrawIndexed(cmd_buffer, self.index_count, 1, 0, 0, 0);

        if (!self.mouse_state.left_button_down) {
            self.active_id = 0;
        }
    }

    pub fn processAndDrawUi(self: *Self, app: *anyopaque, ui_to_draw: *const UI) void {
        for (ui_to_draw.widgets.items, 1..) |widget, i| {
            const id: u32 = @intCast(i);
            var clicked = false;

            const mouse_x = self.mouse_state.x;
            const mouse_y = self.mouse_state.y;

            if (mouse_x >= widget.x and mouse_x <= widget.x + widget.width and
                mouse_y >= widget.y and mouse_y <= widget.y + widget.height)
            {
                self.hot_id = id;
                if (self.mouse_state.left_button_down) {
                    self.active_id = id;
                }
            }

            if (self.hot_id == id and self.active_id == id and !self.mouse_state.left_button_down) {
                clicked = true;
            }

            switch (widget.data) {
                .button => |button_data| {
                    var rect_color = button_data.rect_color;
                    if (self.hot_id == id) {
                        rect_color = .{ rect_color[0] + 0.1, rect_color[1] + 0.1, rect_color[2] + 0.1, rect_color[3] }; // Hover
                        if (self.active_id == id) {
                            rect_color = .{ rect_color[0] - 0.1, rect_color[1] - 0.1, rect_color[2] - 0.1, rect_color[3] }; // Active
                        }
                    }
                    self.drawRect(widget.x, widget.y, widget.width, widget.height, rect_color);

                    const scale = if (button_data.font_size > 0) button_data.font_size / self.font.font_size else 1.0;
                    const text_width = self.measureText(button_data.text, scale);
                    const text_x = widget.x + (widget.width - text_width) / 2.0;
                    const scaled_line_height = self.font.line_height * scale;
                    const text_y = widget.y + (widget.height - scaled_line_height) / 2.0;

                    self.drawText(button_data.text, text_x, text_y, button_data.text_color, scale);
                },
            }

            if (clicked) {
                if (widget.on_click) |callback| {
                    callback(app);
                }
            }
        }
    }

    fn drawRect(self: *Self, x: f32, y: f32, w: f32, h: f32, color: [4]f32) void {
        if (self.vertex_count + 4 > MAX_VERTICES or self.index_count + 6 > MAX_INDICES) return;

        const v_idx = self.vertex_count;
        self.mapped_indices[self.index_count] = v_idx;
        self.mapped_indices[self.index_count + 1] = v_idx + 1;
        self.mapped_indices[self.index_count + 2] = v_idx + 2;
        self.mapped_indices[self.index_count + 3] = v_idx;
        self.mapped_indices[self.index_count + 4] = v_idx + 2;
        self.mapped_indices[self.index_count + 5] = v_idx + 3;

        // Use the pre-calculated UV for the white pixel in our atlas
        const uv = self.dummy_pixel_uv;
        self.mapped_vertices[v_idx] = .{ .pos = .{ x, y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 1] = .{ .pos = .{ x + w, y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 2] = .{ .pos = .{ x + w, y + h }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 3] = .{ .pos = .{ x, y + h }, .uv = uv, .color = color };

        self.vertex_count += 4;
        self.index_count += 6;
    }

    pub fn drawText(self: *Self, text: []const u8, x_start: f32, y_start: f32, color: [4]f32, scale: f32) void {
        var current_x = x_start;
        const scale_w = self.font.scale_w;
        // The texture is 1 pixel taller, but the font data itself is in the same relative position.
        const scale_h = self.font.scale_h;

        for (text) |char_code| {
            if (self.vertex_count + 4 > MAX_VERTICES or self.index_count + 6 > MAX_INDICES) return;

            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;

            const x0 = current_x + glyph.xoffset * scale;
            const y0 = y_start + glyph.yoffset * scale;
            const x1 = x0 + @as(f32, @floatFromInt(glyph.width)) * scale;
            const y1 = y0 + @as(f32, @floatFromInt(glyph.height)) * scale;

            // These UV calculations are correct relative to the original atlas size
            const _u0 = @as(f32, @floatFromInt(glyph.x)) / scale_w;
            const v0 = @as(f32, @floatFromInt(glyph.y)) / scale_h;
            const _u1 = _u0 + (@as(f32, @floatFromInt(glyph.width)) / scale_w);
            const v1 = v0 + (@as(f32, @floatFromInt(glyph.height)) / scale_h);

            const v_idx = self.vertex_count;

            self.mapped_indices[self.index_count + 0] = v_idx;
            self.mapped_indices[self.index_count + 1] = v_idx + 1;
            self.mapped_indices[self.index_count + 2] = v_idx + 2;
            self.mapped_indices[self.index_count + 3] = v_idx;
            self.mapped_indices[self.index_count + 4] = v_idx + 2;
            self.mapped_indices[self.index_count + 5] = v_idx + 3;

            self.mapped_vertices[v_idx + 0] = .{ .pos = .{ x0, y0 }, .uv = .{ _u0, v0 }, .color = color };
            self.mapped_vertices[v_idx + 1] = .{ .pos = .{ x1, y0 }, .uv = .{ _u1, v0 }, .color = color };
            self.mapped_vertices[v_idx + 2] = .{ .pos = .{ x1, y1 }, .uv = .{ _u1, v1 }, .color = color };
            self.mapped_vertices[v_idx + 3] = .{ .pos = .{ x0, y1 }, .uv = .{ _u0, v1 }, .color = color };

            self.vertex_count += 4;
            self.index_count += 6;

            current_x += glyph.xadvance * scale;
        }
    }

    fn measureText(self: *const Self, text: []const u8, scale: f32) f32 {
        var width: f32 = 0;
        for (text) |char_code| {
            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;
            width += glyph.xadvance * scale;
        }
        return width;
    }

    pub fn handleCursorPos(self: *Self, x: f64, y: f64) void {
        self.mouse_state.x = x;
        self.mouse_state.y = y;
    }

    pub fn handleMouseButton(self: *Self, btn: c_int, action: c_int, mods: c_int) void {
        _ = mods;
        if (btn == c.GLFW_MOUSE_BUTTON_LEFT) {
            self.mouse_state.left_button_down = (action == c.GLFW_PRESS);
        }
    }
};

pub const PngImage = struct { width: u32, height: u32, bit_depth: u8, pixels: []u8 };

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
    if (pj <= pk and pj <= pl) return j;
    if (pk <= pl) return k;
    return l;
}

pub fn loadPngGrayscale(
    allocator: std.mem.Allocator,
    png_data: []const u8,
) PngParseError!PngImage {
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
        const chunk_type_slice = png_data[cursor + 4 .. cursor + 8];
        cursor += 8;
        if (cursor + data_len + 4 > png_data.len) return PngParseError.ChunkTooShort;
        const chunk_data = png_data[cursor .. cursor + data_len];
        if (std.mem.eql(u8, chunk_type_slice, "IHDR")) {
            if (data_len != 13) return PngParseError.InvalidIhdr;
            const width = std.mem.readInt(u32, @ptrCast(&chunk_data[0]), .big);
            const height = std.mem.readInt(u32, @ptrCast(&chunk_data[4]), .big);
            const bit_depth = chunk_data[8];
            const color_type = chunk_data[9];
            const interlace_method = chunk_data[12];
            if (color_type != 0 or interlace_method != 0) return PngParseError.UnsupportedFormat;
            if (bit_depth != 8 and bit_depth != 16) return PngParseError.UnsupportedFormat;
            ihdr = .{ .width = width, .height = height, .bit_depth = bit_depth };
        } else if (std.mem.eql(u8, chunk_type_slice, "IDAT")) {
            if (ihdr == null) return PngParseError.MissingIhdr;
            idat_stream.appendSlice(chunk_data) catch @panic("OOM");
        } else if (std.mem.eql(u8, chunk_type_slice, "IEND")) {
            break;
        }
        cursor += data_len;
        cursor += 4; // CRC
    }
    const header = ihdr orelse return PngParseError.MissingIhdr;
    if (idat_stream.items.len == 0) return PngParseError.MissingIdat;
    var decompressed_buffer = std.ArrayList(u8).init(allocator);
    defer decompressed_buffer.deinit();
    {
        var compressed_reader = std.io.fixedBufferStream(idat_stream.items);
        var decompressor = std.compress.zlib.decompressor(compressed_reader.reader());
        decompressor.reader().readAllArrayList(&decompressed_buffer, std.math.maxInt(usize)) catch |err| {
            std.log.err("{}", .{err});
            @panic("Decompressor failed");
        };
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
                    const kv = parseKeyValue(part) orelse {
                        std.log.warn("if ... info\tparseKeyValue returned null for part: {s}", .{part});
                        continue;
                    };
                    if (std.mem.eql(u8, kv.key, "size")) {
                        self.font_size = try std.fmt.parseFloat(f32, kv.value);
                    }
                }
            } else if (std.mem.eql(u8, tag, "common")) {
                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse {
                        std.log.warn("if ... common\tparseKeyValue returned null for part: {s}", .{part});
                        continue;
                    };
                    if (std.mem.eql(u8, kv.key, "lineHeight")) self.line_height = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "base")) self.base = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "scaleW")) self.scale_w = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "scaleH")) self.scale_h = try std.fmt.parseFloat(f32, kv.value);
                }
            } else if (std.mem.eql(u8, tag, "char")) {
                var glyph = Glyph{};
                var id_parsed = false;

                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse {
                        std.log.warn("if ... char\tparseKeyValue returned null for part: {s}", .{part});
                        continue;
                    };
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
