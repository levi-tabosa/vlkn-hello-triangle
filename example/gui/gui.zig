const std = @import("std");
const vk = @import("../test.zig"); // Vulkan context and helpers
const c = @import("c").imports; // C imports for Vulkan
const font = @import("font");
const png = @import("png");
const Font = font.Font;
// Note: The `util` import for Pool is no longer needed.

const gui_vert_shader_bin = @import("spirv").gui_vert;
const gui_frag_shader_bin = @import("spirv").gui_frag;

// --- Basic UI Data Structures ---

pub const TextBuffer = struct {
    const Self = @This();
    const BUFFER_SIZE = 32;
    buf: [BUFFER_SIZE]u8 = undefined,
    len: u32 = 0,

    pub fn slice(self: *const Self) []const u8 {
        return self.buf[0..self.len];
    }

    pub fn append(self: *Self, char: u8) bool {
        if (self.len >= BUFFER_SIZE) return false;
        self.buf[self.len] = char;
        self.len += 1;
        return true;
    }

    pub fn backspace(self: *Self) void {
        if (self.len > 0) {
            self.len -= 1;
        }
    }
};

// Rectangle structure for widget positioning
const Rect = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
};

// Rectangle structure for widget positioning relative to its parent container.
// For automatic layouts (Vertical/Horizontal), x and y are typically 0.
// The width and height are fractions of the available space in the parent.
const RelativeRect = struct {
    x: f32 = 0,
    y: f32 = 0,
    width: f32 = 1.0,
    height: f32 = 1.0,

    pub fn toAbsolute(self: RelativeRect, parent_rect: Rect) Rect {
        return Rect{
            .x = parent_rect.x + (parent_rect.width * self.x),
            .y = parent_rect.y + (parent_rect.height * self.y),
            .width = parent_rect.width * self.width,
            .height = parent_rect.height * self.height,
        };
    }
};

pub const LayoutType = enum {
    Manual,
    Vertical,
    Horizontal,
    Grid,
};

pub const Layout = union(LayoutType) {
    Manual: void, // Manual layout needs no extra data
    Vertical: struct {
        spacing: f32 = 5.0,
    },
    Horizontal: struct {
        spacing: f32 = 5.0,
    },
    Grid: struct {
        columns: usize,
        // Use a vector for {x, y} spacing between cells
        spacing: @Vector(2, f32) = .{ 5.0, 5.0 },
    },
};

// Widget data variants
const WidgetData = union(enum) {
    button: ButtonData,
    plain_text: PlainTextData,
    slider: SliderData,
    text_field: TextFieldData,
    container: ContainerData,
    tree_node: TreeNodeData,

    const ButtonData = struct {
        text: []const u8,
        font_size: f32 = 20.0,
        on_click: ?*const fn (app: *anyopaque) void = null,
    };

    const PlainTextData = struct {
        text: []const u8,
        font_size: f32 = 20.0,
    };

    const SliderData = struct {
        min_value: f32,
        max_value: f32,
        value: f32,
        on_change: ?*const fn (app: *anyopaque, new_value: f32) void = null,
    };

    const TextFieldData = struct {
        label: []const u8,
        buffer: *TextBuffer,
    };

    const ContainerData = struct {
        // A list of child widgets owned by this container.
        children: std.ArrayList(Widget),
        layout: Layout,
        // Padding: { top, right, bottom, left } in pixels
        padding: @Vector(4, f32) = .{ 5, 5, 5, 5 },
    };

    const TreeNodeData = struct {
        label: []const u8,
        is_expanded: bool = false,
        children: std.ArrayList(Widget),
        layout: Layout,
    };
};

// A node in the UI tree. Can be a basic widget or a container for other widgets.
pub const Widget = struct {
    id: u32,
    // Position and size relative to parent's content area.
    rel_rect: RelativeRect,
    background: @Vector(4, f32) = .{ 0, 0, 0, 0 }, // Transparent by default
    foreground: @Vector(4, f32) = .{ 1, 1, 1, 1 },
    data: WidgetData,
    allocator: std.mem.Allocator,

    /// Recursively deinitializes the widget and all its children.
    pub fn deinit(self: *Widget) void {
        switch (self.data) {
            .button => |*data| self.allocator.free(data.text),
            .plain_text => |*data| self.allocator.free(data.text),
            .slider => {},
            .text_field => |*data| self.allocator.free(data.label),
            .container => |*container| {
                for (container.children.items) |*child| {
                    child.deinit();
                }
                container.children.deinit();
            },
            .tree_node => |*node| {
                for (node.children.items) |*child| {
                    child.deinit();
                }
                node.children.deinit();
            },
        }
    }

    fn copyString(alloc: std.mem.Allocator, text: []const u8) ![]u8 {
        const s = try alloc.alloc(u8, text.len);
        @memcpy(s, text);
        return s;
    }

    /// Adds a new container as a child of this widget.
    /// Panics if this widget is not a container.
    /// Returns a pointer to the new container to allow for nested building.
    pub fn addContainer(
        self: *Widget,
        next_id: *u32,
        rel_rect: RelativeRect,
        layout: Layout,
        bg: @Vector(4, f32),
    ) !*Widget {
        var children: *std.ArrayList(Widget) = switch (self.data) {
            .container => |*ct| &ct.children,
            .tree_node => |*nd| &nd.children,
            else => @panic("Cannot add a tree node to this widget type"),
        };

        const id = next_id.*;
        next_id.* += 1;

        try children.append(.{
            .id = id,
            .rel_rect = rel_rect,
            .background = bg,
            .allocator = self.allocator,
            .data = .{
                .container = .{
                    .children = std.ArrayList(Widget).init(self.allocator),
                    // Set the layout using the new parameter
                    .layout = layout,
                },
            },
        });
        return &children.items[children.items.len - 1];
    }

    pub fn addTreeNode(
        self: *Widget,
        next_id: *u32,
        rel_rect: RelativeRect,
        label: []const u8,
        layout: Layout,
        bg: @Vector(4, f32),
        fg: @Vector(4, f32),
    ) !*Widget {
        var children: *std.ArrayList(Widget) = switch (self.data) {
            .container => |*ct| &ct.children,
            .tree_node => |*nd| &nd.children,
            else => @panic("Cannot add a tree node to this widget type"),
        };

        const id = next_id.*;
        next_id.* += 1;

        try children.append(.{
            .id = id,
            .rel_rect = rel_rect,
            .background = bg,
            .foreground = fg,
            .allocator = self.allocator,
            .data = .{
                .tree_node = .{
                    .label = label,
                    .children = std.ArrayList(Widget).init(self.allocator),
                    .layout = layout,
                },
            },
        });
        return &children.items[children.items.len - 1];
    }

    // TODO: return a pointer to the new widget so the mutable reference can be used
    pub fn addButton(
        self: *Widget,
        next_id: *u32,
        rel_rect: RelativeRect,
        text: []const u8,
        bg: @Vector(4, f32),
        fg: @Vector(4, f32),
        font_size: f32,
        on_click: ?*const fn (app: *anyopaque) void,
    ) !void {
        var children: *std.ArrayList(Widget) = switch (self.data) {
            .container => |*ct| &ct.children,
            .tree_node => |*nd| &nd.children,
            else => @panic("Cannot add a tree node to this widget type"),
        };
        const text_copy = try copyString(self.allocator, text);

        const id = next_id.*;
        next_id.* += 1;
        try children.append(.{
            .id = id,
            .rel_rect = rel_rect,
            .background = bg,
            .foreground = fg,
            .allocator = self.allocator,
            .data = .{
                .button = .{
                    .text = text_copy,
                    .on_click = on_click,
                    .font_size = font_size,
                },
            },
        });
    }

    pub fn addPlainText(
        self: *Widget,
        next_id: *u32,
        rect: RelativeRect,
        text: []const u8,
        foreground: @Vector(4, f32),
        font_size: f32,
    ) ![]u8 {
        var children: *std.ArrayList(Widget) = switch (self.data) {
            .container => |*ct| &ct.children,
            .tree_node => |*nd| &nd.children,
            else => @panic("Cannot add a tree node to this widget type"),
        };
        const text_copy = try copyString(self.allocator, text);

        const id = next_id.*;
        next_id.* += 1;
        try children.append(.{
            .id = id,
            .rel_rect = rect,
            .foreground = foreground,
            .allocator = self.allocator,
            .data = .{
                .plain_text = .{
                    .text = text_copy,
                    .font_size = font_size,
                },
            },
        });

        return text_copy; // Return the copied text for potential use
    }

    pub fn addSlider(
        self: *Widget,
        next_id: *u32,
        rel_rect: RelativeRect,
        min_value: f32,
        max_value: f32,
        initial_value: f32,
        bg: @Vector(4, f32),
        fg: @Vector(4, f32),
        on_change: ?*const fn (app: *anyopaque, new_value: f32) void,
    ) !void {
        var children: *std.ArrayList(Widget) = switch (self.data) {
            .container => |*ct| &ct.children,
            .tree_node => |*nd| &nd.children,
            else => @panic("Cannot add a tree node to this widget type"),
        };

        const id = next_id.*;
        next_id.* += 1;
        try children.append(.{
            .id = id,
            .rel_rect = rel_rect,
            .background = bg,
            .foreground = fg,
            .allocator = self.allocator,
            .data = .{ .slider = .{
                .min_value = min_value,
                .max_value = max_value,
                .value = initial_value,
                .on_change = on_change,
            } },
        });
    }

    pub fn addTextField(
        self: *Widget,
        next_id: *u32,
        rel_rect: RelativeRect,
        label: []const u8,
        buffer: *TextBuffer,
        bg: @Vector(4, f32),
        fg: @Vector(4, f32),
    ) !void {
        var children: *std.ArrayList(Widget) = switch (self.data) {
            .container => |*ct| &ct.children,
            .tree_node => |*nd| &nd.children,
            else => @panic("Cannot add a tree node to this widget type"),
        };
        const label_copy = try copyString(self.allocator, label);

        const id = next_id.*;
        next_id.* += 1;
        try children.append(.{
            .id = id,
            .rel_rect = rel_rect,
            .background = bg,
            .foreground = fg,
            .allocator = self.allocator,
            .data = .{ .text_field = .{ .label = label_copy, .buffer = buffer } },
        });
    }
};

// UI manager which holds the root of the widget tree.
pub const UI = struct {
    const Self = @This();

    root: Widget,
    next_id: u32 = 1,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .root = .{
                .id = 0, // Root is ID 0
                .rel_rect = .{ .x = 0, .y = 0, .width = 1.0, .height = 1.0 },
                .background = .{ 0, 0, 0, 0 }, // Transparent root
                .allocator = allocator,
                .data = .{
                    .container = .{
                        .children = std.ArrayList(Widget).init(allocator),
                        .layout = .Manual, // Change from .Vertical to .Manual
                        .padding = .{ 0, 0, 0, 0 },
                    },
                },
            },
        };
    }

    pub fn deinit(self: *Self) void {
        self.root.deinit();
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
            .{ .binding = 0, .location = 0, .format = c.VK_FORMAT_R32G32_SFLOAT, .offset = @offsetOf(GuiVertex, "pos") },
            .{ .binding = 0, .location = 1, .format = c.VK_FORMAT_R32G32_SFLOAT, .offset = @offsetOf(GuiVertex, "uv") },
            .{ .binding = 0, .location = 2, .format = c.VK_FORMAT_R32G32B32A32_SFLOAT, .offset = @offsetOf(GuiVertex, "color") },
        };
    }
};

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
    focused_id: u32 = 0,
    cursor_pos: u32 = 0,
    png_handle: png.PngImage,

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
            .png_handle = try png.loadPng(vk_ctx.allocator, font.hermit_light_png),
        };

        try self.font.loadFNT(font.hermit_light_fnt);
        try self.createTextureAndSampler();
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

    fn createTextureAndSampler(self: *Self) !void {
        const image = self.png_handle;
        const bytes_per_pixel = image.bit_depth / 8;
        const new_width = image.width;
        const new_height = image.height + 1;
        const image_size: u64 = @intCast(image.width * image.height * bytes_per_pixel);
        const total_size: u64 = @intCast(new_width * new_height * bytes_per_pixel);

        var staging_buffer = try vk.Buffer.init(self.vk_ctx, total_size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, u8);
        @memcpy(data_ptr[0..image_size], image.pixels[0..image_size]);
        @memset(data_ptr[image_size..total_size], 0xFF);
        staging_buffer.unmap(self.vk_ctx);

        self.dummy_pixel_uv = .{ 0.5 / @as(f32, @floatFromInt(new_width)), (@as(f32, @floatFromInt(image.height)) + 0.5) / @as(f32, @floatFromInt(new_height)) };

        self.texture = try vk.Image.create(self.vk_ctx, new_width, new_height, if (image.bit_depth == 8) c.VK_FORMAT_R8_UNORM else c.VK_FORMAT_R16_UNORM, c.VK_IMAGE_TILING_OPTIMAL, c.VK_IMAGE_USAGE_TRANSFER_DST_BIT | c.VK_IMAGE_USAGE_SAMPLED_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        try self.texture.transitionLayout(self.vk_ctx, c.VK_IMAGE_LAYOUT_UNDEFINED, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        try self.texture.copyFromBuffer(self.vk_ctx, staging_buffer);
        try self.texture.transitionLayout(self.vk_ctx, c.VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

        self.texture_view = try self.texture.createView(self.vk_ctx, c.VK_IMAGE_ASPECT_COLOR_BIT);

        const sampler_info = c.VkSamplerCreateInfo{ .magFilter = c.VK_FILTER_LINEAR, .minFilter = c.VK_FILTER_LINEAR, .addressModeU = c.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = c.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeW = c.VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .borderColor = c.VK_BORDER_COLOR_INT_OPAQUE_BLACK, .unnormalizedCoordinates = c.VK_FALSE };
        try vk.vkCheck(c.vkCreateSampler(self.vk_ctx.device.handle, &sampler_info, null, &self.sampler));
    }

    fn createDescriptors(self: *Self) !void {
        self.descriptor_set_layout = try vk.DescriptorSetLayout.init(self.vk_ctx, &.{.{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .stage_flags = c.VK_SHADER_STAGE_FRAGMENT_BIT }});
        self.descriptor_pool = try vk.DescriptorPool.init(self.vk_ctx, 1, &.{.{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .count = 1 }});
        self.descriptor_set = try self.descriptor_pool.allocateSet(self.vk_ctx, self.descriptor_set_layout);
        try self.descriptor_set.update(self.vk_ctx, &.{.{ .binding = 0, .info = .{ .CombinedImageSampler = .{ .imageLayout = c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, .imageView = self.texture_view, .sampler = self.sampler } } }});
    }

    fn createBuffers(self: *Self) !void {
        const vtx_buffer_size = MAX_VERTICES * @sizeOf(GuiVertex);
        self.vertex_buffer = try vk.Buffer.init(self.vk_ctx, vtx_buffer_size, c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        self.mapped_vertices = try self.vertex_buffer.map(self.vk_ctx, GuiVertex);
        const idx_buffer_size = MAX_INDICES * @sizeOf(u32);
        self.index_buffer = try vk.Buffer.init(self.vk_ctx, idx_buffer_size, c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        self.mapped_indices = try self.index_buffer.map(self.vk_ctx, u32);
    }

    pub fn createPipeline(self: *Self, render_pass: vk.RenderPass) !void {
        self.pipeline_layout = try vk.PipelineLayout.init(
            self.vk_ctx,
            .{ .setLayoutCount = 1, .pSetLayouts = &self.descriptor_set_layout.handle, .pushConstantRangeCount = 1, .pPushConstantRanges = &self.push_constants.handle },
        );
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
        const multisampling = c.VkPipelineMultisampleStateCreateInfo{ .sampleShadingEnable = c.VK_FALSE, .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT };
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
        const color_blending = c.VkPipelineColorBlendStateCreateInfo{ .logicOpEnable = c.VK_FALSE, .attachmentCount = 1, .pAttachments = &color_blend_attachment };
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
    }

    fn orthoProjectionMatrix(l: f32, r: f32, t: f32, b: f32) [16]f32 {
        return .{
            2.0 / (r - l),      0.0,                0.0, 0.0,
            0.0,                2.0 / (b - t),      0.0, 0.0,
            0.0,                0.0,                1.0, 0.0,
            -(r + l) / (r - l), -(b + t) / (b - t), 0.0, 1.0,
        };
    }

    /// Public entry point for processing and drawing the entire UI tree.
    pub fn processAndDraw(self: *GuiRenderer, ui: *UI, app: *anyopaque, width: c_int, height: c_int) bool {
        const window_rect = Rect{
            .x = 0,
            .y = 0,
            .width = @floatFromInt(width),
            .height = @floatFromInt(height),
        };

        // Start the recursive drawing process from the root widget.
        self.processAndDrawWidget(&ui.root, window_rect, app);

        // After processing all widgets, if the mouse is not down, clear the active_id.
        if (!self.mouse_state.left_button_down and self.active_id != 0) {
            self.active_id = 0;
        }

        // Return true if the UI is currently being interacted with.
        return self.active_id != 0; // Return the state from before it was cleared
    }

    fn processContainerLayout(
        self: *GuiRenderer,
        children: *std.ArrayList(Widget),
        layout: Layout,
        content_rect: Rect,
        app: *anyopaque,
    ) void {
        switch (layout) {
            .Grid => |grid_layout| {
                const num_children = children.items.len;
                if (num_children == 0) return;

                const columns = grid_layout.columns;
                if (columns == 0) return; // Avoid division by zero
                const rows = std.math.divCeil(usize, num_children, columns) catch 1;

                const total_spacing_x = if (columns > 1) grid_layout.spacing[0] * @as(f32, @floatFromInt(columns - 1)) else 0;
                const cell_width = (content_rect.width - total_spacing_x) / @as(f32, @floatFromInt(columns));

                const total_spacing_y = if (rows > 1) grid_layout.spacing[1] * @as(f32, @floatFromInt(rows - 1)) else 0;
                const cell_height = (content_rect.height - total_spacing_y) / @as(f32, @floatFromInt(rows));

                for (children.items, 0..) |*child, i| {
                    const col = @rem(i, columns);
                    const row = @divFloor(i, columns);

                    const child_abs_rect = Rect{
                        .x = content_rect.x + (@as(f32, @floatFromInt(col)) * (cell_width + grid_layout.spacing[0])),
                        .y = content_rect.y + (@as(f32, @floatFromInt(row)) * (cell_height + grid_layout.spacing[1])),
                        .width = cell_width,
                        .height = cell_height,
                    };
                    self.processAndDrawWidget(child, child_abs_rect, app);
                }
            },
            // The other layouts remain sequential
            else => {
                var cursor: @Vector(2, f32) = .{ content_rect.x, content_rect.y };
                for (children.items) |*child| {
                    var child_abs_rect: Rect = undefined;

                    switch (layout) {
                        .Manual => |_| child_abs_rect = child.rel_rect.toAbsolute(content_rect),
                        .Vertical => |vl| {
                            child_abs_rect = .{
                                .x = cursor[0],
                                .y = cursor[1],
                                .width = content_rect.width, // Takes full available width
                                .height = content_rect.height * child.rel_rect.height,
                            };
                            cursor[1] += child_abs_rect.height + vl.spacing;
                        },
                        .Horizontal => |hl| {
                            child_abs_rect = .{
                                .x = cursor[0],
                                .y = cursor[1],
                                .width = content_rect.width * child.rel_rect.width,
                                .height = content_rect.height, // Takes full available height
                            };
                            cursor[0] += child_abs_rect.width + hl.spacing;
                        },
                        .Grid => unreachable,
                    }
                    self.processAndDrawWidget(child, child_abs_rect, app);
                }
            },
        }
    }

    /// Recursive function to process and draw a widget and its children.
    fn processAndDrawWidget(self: *GuiRenderer, widget: *Widget, abs_rect: Rect, app: *anyopaque) void {
        const id = widget.id;

        self.drawRect(abs_rect, widget.background);
        switch (widget.data) {
            .container => |*container| {
                const c_rect = Rect{
                    .x = abs_rect.x + container.padding[3], // left
                    .y = abs_rect.y + container.padding[0], // top
                    .width = abs_rect.width - (container.padding[1] + container.padding[3]), // right + left
                    .height = abs_rect.height - (container.padding[0] + container.padding[2]), // top + bottom
                };

                self.processContainerLayout(&container.children, container.layout, c_rect, app);
            },
            .tree_node => |*data| {
                const bar_height: f32 = 25.0;
                const bar_rect = Rect{
                    .x = abs_rect.x,
                    .y = abs_rect.y,
                    .width = abs_rect.width,
                    .height = bar_height,
                };
                const has_children = data.children.items.len > 0;

                if (has_children) {
                    const button_size: f32 = 15.0;
                    const button_margin: f32 = (bar_height - button_size);
                    // Draw expand button decoration
                    const expand_button_rect = Rect{
                        .x = bar_rect.x + button_margin,
                        .y = bar_rect.y + button_margin,
                        .width = button_size,
                        .height = button_size,
                    };

                    const expand_button_id = id; // Reuse widget ID for its main interactive part

                    if (self.isMouseOver(expand_button_rect)) {
                        self.hot_id = expand_button_id;
                        if (self.active_id == 0 and self.mouse_state.left_button_down) {
                            self.active_id = expand_button_id;
                        }
                    }

                    // Check for click release event on the expand button
                    if (!self.mouse_state.left_button_down and self.active_id == expand_button_id and self.hot_id == expand_button_id) {
                        data.is_expanded = !data.is_expanded;
                        self.active_id = 0; // Consume the click
                    }

                    var button_color = widget.foreground;
                    if (self.hot_id == expand_button_id) {
                        button_color = .{ 0.8, 0.8, 0.0, 1.0 }; // Highlight yellow
                    }

                    const expand_char = if (data.is_expanded) "-" else "+";
                    const expand_center_x = (expand_button_rect.x + expand_button_rect.width) / 2;
                    const expand_center_y = (expand_button_rect.y + expand_button_rect.height) / 2;
                    self.drawText(expand_char, expand_center_x, expand_center_y, button_color, 1.0);
                }

                const label_x = bar_rect.x + (if (has_children) bar_height else 5.0);
                self.drawText(data.label, label_x, bar_rect.y + 4, widget.foreground, 1.0);

                if (data.is_expanded and has_children) {
                    const child_indent: f32 = 20.0;
                    const children_content_rect = Rect{
                        .x = abs_rect.x + child_indent,
                        .y = abs_rect.y + bar_height,
                        .width = abs_rect.width - child_indent,
                        .height = abs_rect.height - bar_height,
                    };

                    // A TreeNode acts as a container for its children.
                    self.processContainerLayout(&data.children, data.layout, children_content_rect, app);
                }
            },
            .button => |*data| {
                if (self.isMouseOver(abs_rect)) {
                    self.hot_id = id;
                    // If no other widget is active and mouse is pressed, this widget becomes active.
                    if (self.active_id == 0 and self.mouse_state.left_button_down) {
                        self.active_id = id;
                    }
                }

                var bg_color = widget.background;
                if (self.hot_id == id) {
                    if (self.active_id == id) {
                        // Color when button is being pressed
                        bg_color = .{ widget.background[0] * 0.5, widget.background[1] * 0.5, widget.background[2] * 0.5, 1.0 };
                    } else {
                        // Color when mouse is hovering over it
                        bg_color = .{ widget.background[0] * 1.2, widget.background[1] * 1.2, widget.background[2] * 1.2, 1.0 };
                    }
                }
                self.drawRect(abs_rect, bg_color);

                const scale = data.font_size / self.font.font_size;
                const text_width = self.measureText(data.text, scale);
                const text_height = self.font.line_height * scale;
                const text_x = abs_rect.x + (abs_rect.width - text_width) / 2.0;
                const text_y = abs_rect.y + (abs_rect.height - text_height) / 2.0;
                self.drawText(data.text, text_x, text_y, widget.foreground, scale);

                // Check for click
                if (!self.mouse_state.left_button_down and self.active_id == id and self.hot_id == id) {
                    if (data.on_click) |callback| {
                        callback(app);
                    }
                    self.active_id = 0;
                }
            },
            .plain_text => |data| {
                const scale = data.font_size / self.font.font_size;
                const text_height = self.font.line_height * scale;
                const text_y = abs_rect.y + (abs_rect.height - text_height) / 2.0;
                self.drawText(data.text, abs_rect.x, text_y, widget.foreground, scale);
            },
            .slider => |*data| {
                if (self.isMouseOver(abs_rect)) {
                    self.hot_id = id;
                    if (self.active_id == 0 and self.mouse_state.left_button_down) {
                        self.active_id = id;
                    }
                }

                // The track is drawn by the generic background draw at the start of the function.
                // You can add a specific track color here if you want.
                // self.drawRect(abs_rect, track_color);

                const slider_width = abs_rect.width;
                const handle_width: f32 = 10.0;
                const normalized_value = (data.value - data.min_value) / (data.max_value - data.min_value);
                const handle_x = abs_rect.x + (slider_width - handle_width) * normalized_value;
                const handle_rect = Rect{ .x = handle_x, .y = abs_rect.y, .width = handle_width, .height = abs_rect.height };

                var handle_color = widget.foreground;
                if (self.active_id == id) {
                    handle_color = .{ handle_color[0] * 0.5, handle_color[1] * 0.5, handle_color[2] * 0.5, 1.0 };
                } else if (self.hot_id == id) {
                    handle_color = .{ handle_color[0] * 1.2, handle_color[1] * 1.2, handle_color[2] * 1.2, 1.0 };
                }
                self.drawRect(handle_rect, handle_color);

                if (self.active_id == id) {
                    var new_value = data.min_value + ((self.mouse_state.x - abs_rect.x) / slider_width) * (data.max_value - data.min_value);
                    new_value = @max(new_value, data.min_value);
                    new_value = @min(new_value, data.max_value);
                    if (data.value != new_value) {
                        data.value = @floatCast(new_value);
                        if (data.on_change) |callback| {
                            callback(app, @floatCast(new_value));
                        }
                    }
                }
            },
            .text_field => |*data| {
                // *** CHANGE: Interaction logic is now here ***
                if (self.isMouseOver(abs_rect)) {
                    self.hot_id = id;
                    if (self.active_id == 0 and self.mouse_state.left_button_down) {
                        self.active_id = id;
                        // For text fields, becoming active also means becoming focused.
                        self.focused_id = id;
                        self.cursor_pos = data.buffer.len;
                    }
                }

                // If another widget is clicked, the text field should lose focus.
                if (self.mouse_state.left_button_down and self.hot_id != id) {
                    if (self.focused_id == id) {
                        self.focused_id = 0;
                    }
                }

                var bg_color = widget.background;
                if (self.focused_id == id) {
                    // Highlight when focused.
                    bg_color = .{ bg_color[0] * 1.5, bg_color[1] * 1.5, bg_color[2] * 1.5, 1.0 };
                }
                self.drawRect(abs_rect, bg_color);

                const scale = (0.6 * abs_rect.height) / self.font.line_height;
                const label_width = self.measureText(data.label, scale);
                self.drawText(data.label, abs_rect.x - label_width - 5, abs_rect.y, widget.foreground, scale);
                self.drawText(data.buffer.slice(), abs_rect.x + 5, abs_rect.y + (abs_rect.height * 0.2), widget.foreground, scale);

                if (self.focused_id == id) {
                    // Draw blinking cursor
                    const time_ms = std.time.milliTimestamp();
                    if (@mod(@divFloor(time_ms, 500), 2) == 0) {
                        const text_up_to_cursor = data.buffer.buf[0..self.cursor_pos];
                        const cursor_x_offset = self.measureText(text_up_to_cursor, scale);
                        const cursor_rect = Rect{ .x = abs_rect.x + 5 + cursor_x_offset, .y = abs_rect.y + (abs_rect.height * 0.15), .width = 2, .height = abs_rect.height * 0.7 };
                        self.drawRect(cursor_rect, widget.foreground);
                    }
                }
            },
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
        const uv = self.dummy_pixel_uv;
        self.mapped_vertices[v_idx + 0] = .{ .pos = .{ rect.x, rect.y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 1] = .{ .pos = .{ rect.x + rect.width, rect.y }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 2] = .{ .pos = .{ rect.x + rect.width, rect.y + rect.height }, .uv = uv, .color = color };
        self.mapped_vertices[v_idx + 3] = .{ .pos = .{ rect.x, rect.y + rect.height }, .uv = uv, .color = color };
        self.vertex_count += 4;
        self.index_count += 6;
    }

    fn drawText(self: *Self, text: []const u8, x_start: f32, y_start: f32, color: [4]f32, scale: f32) void {
        if (self.vertex_count + 4 * text.len > MAX_VERTICES or self.index_count + 6 * text.len > MAX_INDICES) return;
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

    /// Recursively search the widget tree for a specific ID.
    fn findWidgetById(root: *Widget, id: u32) ?*Widget {
        if (root.id == id) return @constCast(root);
        switch (root.data) {
            .container => |*container| {
                for (container.children.items) |*child| {
                    if (findWidgetById(child, id)) |found| {
                        return found;
                    }
                }
            },
            .tree_node => |*node| {
                for (node.children.items) |*child| {
                    if (findWidgetById(child, id)) |found| {
                        return found;
                    }
                }
            },
            else => {},
        }
        return null;
    }

    pub fn handleKey(self: *Self, key: c_int, action: c_int, ui: *UI) void {
        if (self.focused_id == 0 or (action != c.GLFW_PRESS and action != c.GLFW_REPEAT)) {
            return;
        }

        // Find the focused widget in the tree.
        if (findWidgetById(&ui.root, self.focused_id)) |widget| {
            if (widget.data == .text_field) {
                switch (key) {
                    c.GLFW_KEY_BACKSPACE => widget.data.text_field.buffer.backspace(),
                    c.GLFW_KEY_ENTER, c.GLFW_KEY_ESCAPE => self.focused_id = 0,
                    else => {
                        if (key >= ' ' and key <= '~') {
                            _ = widget.data.text_field.buffer.append(@intCast(key));
                        }
                    },
                }
            }
        }
    }
};
