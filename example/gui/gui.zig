// gui.zig
const std = @import("std");
const assert = std.debug.assert;
const vk = @import("../test.zig"); // Reuse structs and helpers from main file
const App = vk.App;
const c = @import("c").c; // Reuse cImport from main file
const gui_vert_shader_code = @import("spirv").gui_vs;
const gui_frag_shader_code = @import("spirv").gui_fs;

const OnClickFn = *const fn (app: *App) void;

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
    color: [4]f32,

    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(GuiVertex),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    pub fn getAttributeDescriptions() [2]c.VkVertexInputAttributeDescription {
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

    const MAX_VERTICES = 4096;
    const MAX_INDICES = MAX_VERTICES * 3 / 2;

    pub fn init(vk_ctx: *vk.VulkanContext, render_pass: vk.RenderPass, swapchain: vk.Swapchain) !Self {
        var self: Self = .{
            .vk_ctx = vk_ctx,
        };

        // Create buffers that are permanently mapped for easy writing
        const vtx_buffer_size = MAX_VERTICES * @sizeOf(GuiVertex);
        self.vertex_buffer = try vk.Buffer.init(vk_ctx, vtx_buffer_size, c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        self.mapped_vertices = try self.vertex_buffer.map(vk_ctx, GuiVertex);

        const idx_buffer_size = MAX_INDICES * @sizeOf(u32);
        self.index_buffer = try vk.Buffer.init(vk_ctx, idx_buffer_size, c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        self.mapped_indices = try self.index_buffer.map(vk_ctx, u32);

        try self.createPipeline(render_pass, swapchain);
        return self;
    }

    pub fn createPipeline(self: *Self, render_pass: vk.RenderPass, swapchain: vk.Swapchain) !void {
        // 1. Pipeline Layout
        self.pipeline_layout = try vk.PipelineLayout.init(self.vk_ctx, .{
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &vk.PushConstantRange.init([16]f32).handle,
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
        const rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_NONE,
            .frontFace = c.VK_FRONT_FACE_CLOCKWISE,
        };
        const multisampling = c.VkPipelineMultisampleStateCreateInfo{ .sampleShadingEnable = c.VK_FALSE, .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT };

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

        const pipeline_info = c.VkGraphicsPipelineCreateInfo{
            .stageCount = shader_stages.len,
            .pStages = &shader_stages,
            .pVertexInputState = &vertex_input_info,
            .pInputAssemblyState = &input_assembly,
            .pViewportState = &viewport_state,
            .pRasterizationState = &rasterizer,
            .pMultisampleState = &multisampling,
            .pColorBlendState = &color_blending,
            .pDepthStencilState = &depth_stencil,
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

    pub fn deinit(self: *Self) void {
        self.vertex_buffer.unmap(self.vk_ctx);
        self.index_buffer.unmap(self.vk_ctx);
        self.vertex_buffer.deinit(self.vk_ctx);
        self.index_buffer.deinit(self.vk_ctx);
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
        const vtx_buffers = [_]c.VkBuffer{self.vertex_buffer.handle};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &vtx_buffers, &offsets);
        c.vkCmdBindIndexBuffer(cmd_buffer, self.index_buffer.handle, 0, c.VK_INDEX_TYPE_UINT32);

        // Set dynamic viewport/scissor
        const viewport = c.VkViewport{
            .width = window_width,
            .height = window_height,
            .minDepth = 0.0,
            .maxDepth = 1.0,
        };
        const scissor = c.VkRect2D{ .extent = .{
            .width = @intFromFloat(window_width),
            .height = @intFromFloat(window_height),
        } };
        c.vkCmdSetViewport(cmd_buffer, 0, 1, &viewport);
        c.vkCmdSetScissor(cmd_buffer, 0, 1, &scissor);

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

    pub fn processAndDrawUi(self: *Self, app: *App, ui_to_draw: *const UI) void {
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
                    _ = button_data;
                    // Determine state and color
                    var color: [4]f32 = .{ 0.3, 0.3, 0.8, 1.0 }; // Normal
                    if (self.hot_id == id) {
                        color = .{ 0.4, 0.4, 0.9, 1.0 }; // Hover
                        if (self.active_id == id) {
                            color = .{ 0.2, 0.2, 0.7, 1.0 }; // Active
                        }
                    }
                    self.drawRect(widget.x, widget.y, widget.width, widget.height, color);
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

        self.mapped_vertices[v_idx] = .{ .pos = .{ x, y }, .color = color };
        self.mapped_vertices[v_idx + 1] = .{ .pos = .{ x + w, y }, .color = color };
        self.mapped_vertices[v_idx + 2] = .{ .pos = .{ x + w, y + h }, .color = color };
        self.mapped_vertices[v_idx + 3] = .{ .pos = .{ x, y + h }, .color = color };

        self.vertex_count += 4;
        self.index_count += 6;
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
