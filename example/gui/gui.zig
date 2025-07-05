// gui.zig
const std = @import("std");
const assert = std.debug.assert;
const vk = @import("../test.zig"); // Reuse structs and helpers from main file
const c = @import("c").c; // Reuse cImport from main file
const font = @import("font.zig");
const gui_vert_shader_code = @import("spirv").gui_vs;
const gui_frag_shader_code = @import("spirv").gui_fs;

const Font = font.Font;
const OnClickFn = *const fn (app: *anyopaque) void;

const WidgetData = union(enum) {
    button: struct {
        text: []const u8,
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

    descriptor_set_layout: c.VkDescriptorSetLayout = undefined,
    descriptor_pool: c.VkDescriptorPool = undefined,
    descriptor_set: c.VkDescriptorSet = undefined,
    push_constants: vk.PushConstantRange = .init([16]f32),

    font: Font = undefined,
    font_texture: vk.Image = undefined,
    font_texture_view: c.VkImageView = undefined,
    font_sampler: c.VkSampler = undefined,
    vertex_buffer: vk.Buffer = undefined,
    index_buffer: vk.Buffer = undefined,

    // Mapped pointers for the current frame's data
    mapped_vertices: [*]GuiVertex = undefined,
    mapped_indices: [*]u32 = undefined,
    vertex_count: u32 = 0,
    index_count: u32 = 0,

    // Simple Input state
    mouse_state: MouseState = .{},
    active_id: u32 = 0, // ID of the widget being interacted with
    hot_id: u32 = 0, // ID of the widget being hovered over

    // We'll use a simple counter for widget IDs each frame
    last_id: u32 = 0,
    image_handle: PngImage = undefined,

    const MAX_VERTICES = 4096;
    const MAX_INDICES = MAX_VERTICES * 3 / 2;

    pub fn init(vk_ctx: *vk.VulkanContext, render_pass: vk.RenderPass, swapchain: vk.Swapchain) !Self {
        var self: Self = .{
            .vk_ctx = vk_ctx,
            .font = .init(vk_ctx.allocator),
        };

        try self.font.loadFNT();
        try self.createFontTextureAndSampler(vk_ctx.allocator, @import("font").mikado_medium_fed68123_png);
        // self.descriptor_pool = try vk.DescriptorPool.init(vk_ctx, .CombinedImageSampler);
        // try self.createDescriptorPool();
        try self.createDescriptors();
        // Create buffers that are permanently mapped for easy writing
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

        try self.createPipeline(render_pass, swapchain);
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroySampler(self.vk_ctx.device.handle, self.font_sampler, null);
        c.vkDestroyImageView(self.vk_ctx.device.handle, self.font_texture_view, null);
        self.font_texture.deinit(self.vk_ctx);
        self.vk_ctx.allocator.free(self.image_handle.pixels);

        c.vkDestroyDescriptorPool(self.vk_ctx.device.handle, self.descriptor_pool, null);
        c.vkDestroyDescriptorSetLayout(self.vk_ctx.device.handle, self.descriptor_set_layout, null);
        self.font.deinit();

        self.vertex_buffer.unmap(self.vk_ctx);
        self.index_buffer.unmap(self.vk_ctx);
        self.vertex_buffer.deinit(self.vk_ctx);
        self.index_buffer.deinit(self.vk_ctx);
    }

    // Function to create font texture, view, and sampler
    fn createFontTextureAndSampler(self: *Self, allocator: std.mem.Allocator, png_data: []const u8) !void {
        const image = try loadPngGrayscale(allocator, png_data);
        self.image_handle = image;
        std.debug.print("{} {} {}\n", .{ image.width, image.height, image.pixels.len });

        const image_size: u64 = @intCast(image.width * image.height);

        var staging_buffer = try vk.Buffer.init(self.vk_ctx, image_size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, u8);
        @memcpy(data_ptr[0..image_size], image.pixels[0..image_size]);
        staging_buffer.unmap(self.vk_ctx);

        self.font_texture = try vk.Image.create(
            self.vk_ctx,
            @intCast(image.width),
            @intCast(image.height),
            c.VK_FORMAT_R8_UNORM,
            c.VK_IMAGE_TILING_OPTIMAL,
            c.VK_IMAGE_USAGE_TRANSFER_DST_BIT | c.VK_IMAGE_USAGE_SAMPLED_BIT,
            c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );

        try self.font_texture.transitionLayout(self.vk_ctx, c.VK_IMAGE_LAYOUT_UNDEFINED, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        try self.font_texture.copyFromBuffer(self.vk_ctx, staging_buffer);
        try self.font_texture.transitionLayout(self.vk_ctx, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        self.font_texture_view = try self.font_texture.createView(self.vk_ctx, c.VK_IMAGE_ASPECT_COLOR_BIT);

        const sampler_info = c.VkSamplerCreateInfo{
            .magFilter = c.VK_FILTER_LINEAR,
            .minFilter = c.VK_FILTER_LINEAR,
        };
        try vk.vkCheck(c.vkCreateSampler(self.vk_ctx.device.handle, &sampler_info, null, &self.font_sampler));
    }

    fn createDescriptorPool(self: *Self) !void {
        const pool_size = c.VkDescriptorPoolSize{
            .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
        };
        const pool_info = c.VkDescriptorPoolCreateInfo{
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        try vk.vkCheck(c.vkCreateDescriptorPool(self.vk_ctx.device.handle, &pool_info, null, &self.descriptor_pool));
    }

    //Function to create descriptor pool, layout, and sets
    fn createDescriptors(self: *Self) !void {
        const sampler_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        // TODO: change to vo
        const layout_info = c.VkDescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &sampler_layout_binding };
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
            .imageView = self.font_texture_view,
            .sampler = self.font_sampler,
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

    pub fn createPipeline(self: *Self, render_pass: vk.RenderPass, swapchain: vk.Swapchain) !void {
        // _ = swapchain;
        // 1. Pipeline Layout
        self.pipeline_layout = try vk.PipelineLayout.init(self.vk_ctx, .{
            .setLayoutCount = 1,
            .pSetLayouts = &self.descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &self.push_constants.handle,
        });

        // 2. Shader Modules
        var vert_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx.device.handle, gui_vert_shader_code);
        defer vert_mod.deinit();
        var frag_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx.device.handle, gui_frag_shader_code);
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

        // --- Blending and No Depth Test ---
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

        // const dynamic_states = [_]c.VkDynamicState{
        //     c.VK_DYNAMIC_STATE_VIEWPORT,
        //     c.VK_DYNAMIC_STATE_SCISSOR,
        // };

        // const dynamic_state_info = c.VkPipelineDynamicStateCreateInfo{
        //     .dynamicStateCount = dynamic_states.len,
        //     .pDynamicStates = &dynamic_states,
        // };

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
            // .pDynamicState = &dynamic_state_info,
            .pViewportState = &viewport_state,
            .layout = self.pipeline_layout.handle,
            .renderPass = render_pass.handle,
            .subpass = 0,
        };

        // self.pipeline = try vk.Pipeline.init(self.vk_ctx, render_pass, self.pipeline_layout);
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
        const vtx_buffers = [_]c.VkBuffer{self.vertex_buffer.handle};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vtx_buffers, &offsets);
        c.vkCmdBindIndexBuffer(cmd_buffer, self.index_buffer.handle, 0, c.VK_INDEX_TYPE_UINT32);

        // const viewport = c.VkViewport{
        //     .x = 0,
        //     .y = 0,
        //     .width = window_width,
        //     .height = window_height,
        //     .minDepth = 0,
        //     .maxDepth = 1,
        // };
        // c.vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);

        // const scissor = c.VkRect2D{
        //     .offset = .{ .x = 0, .y = 0 },
        //     .extent = .{ .width = @intFromFloat(window_width), .height = @intFromFloat(window_height) },
        // };
        // c.vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

        // Set orthographic projection matrix via push constants
        const L: f32 = 0;
        const R = window_width;
        const T: f32 = 0; // Top is at 0
        const B = window_height; // Bottom is at window_height
        const ortho_projection = [_]f32{
            2.0 / (R - L), 0.0, 0.0, 0.0,
            0.0, 2.0 / (B - T), 0.0, 0.0, // Note: No longer flips Y
            0.0,                0.0,                1.0, 0.0, // Can use 1.0 for Z since we don't use depth
            -(R + L) / (R - L), -(B + T) / (B - T), 0.0, 1.0,
        };
        c.vkCmdPushConstants(cmd_buffer, self.pipeline_layout.handle, c.VK_SHADER_STAGE_VERTEX_BIT, 0, @sizeOf([16]f32), &ortho_projection);

        c.vkCmdDrawIndexed(cmd_buffer, self.index_count, 1, 0, 0, 0);

        // De-activate widget if mouse button is released
        if (!self.mouse_state.left_button_down) {
            self.active_id = 0;
        }
    }

    pub fn processAndDrawUi(self: *Self, app: *anyopaque, ui_to_draw: *const UI) void {
        for (ui_to_draw.widgets.items, 1..) |widget, i| {
            const id: u32 = @intCast(i); // Use index + 1 as the ID
            var clicked = false;

            // --- Interaction Logic (same as before, but generic) ---
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

            // A click is registered on mouse-up if this widget was the active one.
            if (self.hot_id == id and self.active_id == id and !self.mouse_state.left_button_down) {
                clicked = true;
            }

            // --- Rendering Logic ---
            switch (widget.data) {
                .button => |button_data| {
                    const text_color = [4]f32{ 1.0, 1.0, 1.0, 1.0 }; // White text
                    // Determine state and color
                    var color: [4]f32 = .{ 0.3, 0.3, 0.8, 1.0 }; // Normal
                    if (self.hot_id == id) {
                        color = .{ 0.4, 0.4, 0.9, 1.0 }; // Hover
                        if (self.active_id == id) {
                            color = .{ 0.2, 0.2, 0.7, 1.0 }; // Active
                        }
                    }
                    self.drawRect(widget.x, widget.y, widget.width, widget.height, color);
                    const text_width = self.measureText(button_data.text);
                    const text_x = widget.x + (widget.width - text_width) / 2.0;
                    const text_y = widget.y + (widget.height - self.font.line_height) / 2.0 + self.font.base;

                    self.drawText(button_data.text, text_x, text_y, text_color);
                },
            }

            // --- Callback Execution ---
            if (clicked) {
                if (widget.on_click) |callback| {
                    // Execute the function pointer!
                    callback(app);
                }
            }
        }
    }

    fn drawRect(self: *Self, x: f32, y: f32, w: f32, h: f32, color: [4]f32) void {
        const v_idx = self.vertex_count;
        self.mapped_indices[self.index_count] = v_idx;
        self.mapped_indices[self.index_count + 1] = v_idx + 1;
        self.mapped_indices[self.index_count + 2] = v_idx + 2;
        self.mapped_indices[self.index_count + 3] = v_idx;
        self.mapped_indices[self.index_count + 4] = v_idx + 2;
        self.mapped_indices[self.index_count + 5] = v_idx + 3;

        // Dummy UVs for solid color rects
        const uv = [2]f32{ 0.0, 0.0 };
        self.mapped_vertices[v_idx] = .{ .pos = .{ x, y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 1] = .{ .pos = .{ x + w, y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 2] = .{ .pos = .{ x + w, y + h }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 3] = .{ .pos = .{ x, y + h }, .uv = uv, .color = color };

        self.vertex_count += 4;
        self.index_count += 6;
    }

    fn drawText(self: *Self, text: []const u8, x_start: f32, y_start: f32, color: [4]f32) void {
        var current_x = x_start;
        const scale_w = self.font.scale_w;
        const scale_h = self.font.scale_h;

        for (text) |char_code| {
            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;

            const x0 = current_x + glyph.xoffset;
            const y0 = y_start + glyph.yoffset;
            const x1 = x0 + @as(f32, @floatFromInt(glyph.width));
            const y1 = y0 + @as(f32, @floatFromInt(glyph.height));

            const _u0 = @as(f32, @floatFromInt(glyph.x)) / scale_w;
            const v0 = @as(f32, @floatFromInt(glyph.y)) / scale_h;
            const _u1 = _u0 + (@as(f32, @floatFromInt(glyph.width)) / scale_w);
            const v1 = v0 + (@as(f32, @floatFromInt(glyph.height)) / scale_h);

            const v_idx = self.vertex_count;
            if (v_idx + 4 > MAX_VERTICES or self.index_count + 6 > MAX_INDICES) return; // Buffer full

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

            current_x += glyph.xadvance;
        }
    }

    fn measureText(self: *const Self, text: []const u8) f32 {
        var width: f32 = 0;
        for (text) |char_code| {
            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;
            width += glyph.xadvance;
        }
        return width;
    }

    // --- Input Handling ---

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

/// Parses a PNG byte slice to extract the image width and height.
/// It only reads the header and does not validate the rest of the file.
/// Returns an error union.
/// Holds the final decoded PNG image data.
/// The caller is responsible for freeing the `pixels` buffer.

// PngImage and PngParseError structs remain the same...

// PngImage, PngParseError, and paethPredictor are unchanged.
// [Copy those from the previous answer]

// PngImage, PngParseError, and paethPredictor are unchanged from the previous version.
// The paethPredictor function itself IS CORRECT, it's the CALL to it that needs fixing.
pub const PngImage = struct { width: u32, height: u32, bit_depth: u8, pixels: []u8 };
pub const PngParseError = error{ FileTooShort, InvalidSignature, ChunkTooShort, MissingIhdr, InvalidIhdr, UnsupportedFormat, MissingIdat, MissingIend, DecompressionError, InvalidFilterType };
fn paethPredictor(a: i64, b: i64, C: i64) i64 {
    const p = a + b - C;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - C);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return C;
}

pub fn loadPngGrayscale(
    allocator: std.mem.Allocator,
    png_data: []const u8,
) !PngImage {
    // [The first part of the function (chunk parsing, decompression) is unchanged]
    // ...
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
            try idat_stream.appendSlice(chunk_data);
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
        try decompressor.reader().readAllArrayList(&decompressed_buffer, std.math.maxInt(usize));
    }
    const filtered_scanlines = decompressed_buffer.items;
    const bytes_per_pixel = header.bit_depth / 8;
    const scanline_length = header.width * bytes_per_pixel;
    const expected_filtered_size = header.height * (1 + scanline_length);
    if (filtered_scanlines.len != expected_filtered_size) return PngParseError.DecompressionError;
    var final_pixels = try allocator.alloc(u8, header.width * header.height * bytes_per_pixel);
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
                        // --- FIXED: Cast the i64 result back to u8 ---
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
                        // --- FIXED: Cast the i64 result back to u16 ---
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
    return PngImage{ .width = header.width, .height = header.height, .bit_depth = header.bit_depth, .pixels = final_pixels };
}
