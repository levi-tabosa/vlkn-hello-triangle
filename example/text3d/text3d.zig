// text3d.zig
const std = @import("std");
const vk = @import("../test.zig");
const c = @import("c").c;
//TODO: move font stuff to another file and import that
const gui = @import("../gui/gui.zig");
const font = @import("font");
const util = @import("util");

const text3d_vert_shader_bin = @import("spirv").text3d_vs;
const text3d_frag_shader_bin = @import("spirv").text3d_fs;

const Text3DVertex = extern struct {
    pos: [3]f32,
    uv: [2]f32,
    color: [4]f32,

    pub fn getBindingDescription() c.VkVertexInputBindingDescription {
        return .{
            .binding = 0,
            .stride = @sizeOf(Text3DVertex),
            .inputRate = c.VK_VERTEX_INPUT_RATE_VERTEX,
        };
    }

    pub fn getAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
        return .{
            .{ .binding = 0, .location = 0, .format = c.VK_FORMAT_R32G32B32_SFLOAT, .offset = @offsetOf(Text3DVertex, "pos") },
            .{ .binding = 0, .location = 1, .format = c.VK_FORMAT_R32G32_SFLOAT, .offset = @offsetOf(Text3DVertex, "uv") },
            .{ .binding = 0, .location = 2, .format = c.VK_FORMAT_R32G32B32A32_SFLOAT, .offset = @offsetOf(Text3DVertex, "color") },
        };
    }
};

const TransformData = union(enum) {
    static: [16]f32,
    billboard: struct {
        position: [3]f32,
    },
};

const DynamicText = struct {
    text: []u8,
    transform: TransformData,
    color: [4]f32,
    font_size: f32,
};

pub const Text3DScene = struct {
    const Self = @This();

    list: std.ArrayList(DynamicText),
    string_pool: util.Pool([256]u8),
    last_cam_matrix: [16]f32 = undefined,

    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .list = .init(allocator),
            .string_pool = .init(allocator),
        };
    }

    pub fn deinit(self: *Self) void {
        self.string_pool.deinit();
        self.list.deinit();
    }

    pub fn addText(self: *Self, text: []const u8, modal: [16]f32, color: ?[4]f32, font_size: ?f32) !void {
        const t = (try self.string_pool.new())[0..text.len];
        @memcpy(t, text);

        try self.list.append(.{
            .text = t,
            .transform = .{ .static = modal },
            .color = color orelse .{ 0.5, 0.5, 0.5, 1.0 },
            .font_size = font_size orelse 1.0,
        });
    }

    pub fn addBillboardText(self: *Self, text: []const u8, position: [3]f32, color: ?[4]f32, font_size: ?f32) !void {
        const t = (try self.string_pool.new())[0..text.len];
        @memcpy(t, text);

        try self.list.append(.{
            .text = t,
            .transform = .{ .billboard = .{ .position = position } },
            .color = color orelse .{ 0.5, 0.5, 0.5, 1.0 },
            .font_size = font_size orelse 1.0,
        });
    }

    pub fn clearText(self: *Self) void {
        for (self.list.items) |*d_text| {
            self.string_pool.delete(d_text.text.ptr);
        }
        self.list.clearAndFree();
    }

    /// Takes a camera matrix to manage the billboard text
    pub fn updateCameraViewMatrix(self: *Self, matrix: [16]f32) void {
        self.last_cam_matrix = matrix;
    }
};

fn createBillboardMatrix(camera_view_matrix: [16]f32, position: [3]f32, scale: f32) [16]f32 {
    var model: [16]f32 = .{0} ** 16;
    model[0] = camera_view_matrix[0];
    model[1] = camera_view_matrix[4];
    model[2] = camera_view_matrix[8];
    model[8] = camera_view_matrix[1];
    model[9] = camera_view_matrix[5];
    model[10] = camera_view_matrix[9];
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        model[i * 4 + 0] *= scale;
        model[i * 4 + 1] *= scale;
        model[i * 4 + 2] *= scale;
    }
    model[12] = position[0];
    model[13] = position[1];
    model[14] = position[2];
    model[15] = 1.0;
    return model;
}

// This struct will be stored in an SSBO. Each instance corresponds to one drawText call.
const DrawData = extern struct {
    model: [16]f32, // mat4
};

const IndirectCommand = c.VkDrawIndexedIndirectCommand;

pub const Text3DRenderer = struct {
    const Self = @This();

    vk_ctx: *vk.VulkanContext,
    pipeline: vk.Pipeline = undefined,
    pipeline_layout: vk.PipelineLayout = undefined,

    // Font Texture Descriptors (Set 1)
    font_descriptor_set_layout: vk.DescriptorSetLayout = undefined,
    font_descriptor_pool: vk.DescriptorPool = undefined, // This pool will now hold two sets
    font_descriptor_set: vk.DescriptorSet = undefined,

    // Draw Data Descriptors (Set 2) for the SSBO
    draw_data_descriptor_set_layout: vk.DescriptorSetLayout = undefined,
    draw_data_descriptor_set: vk.DescriptorSet = undefined,

    sampler: c.VkSampler = undefined,
    texture: vk.Image = undefined,
    texture_view: c.VkImageView = undefined,

    // Buffer for indirect draw commands
    indirect_cmd_buffer: vk.Buffer = undefined,
    mapped_indirect_commands: [*]IndirectCommand = undefined,

    // Buffer for draw data (model matrices) SSBO
    draw_data_buffer: vk.Buffer = undefined,
    mapped_draw_data: [*]DrawData = undefined,

    // Main geometry buffers
    vertex_buffer: vk.Buffer = undefined,
    index_buffer: vk.Buffer = undefined,
    mapped_vertices: [*]Text3DVertex = undefined,
    mapped_indices: [*]u32 = undefined,

    // Frame-specific counters
    vertex_count: u32 = 0,
    index_count: u32 = 0,
    draw_count: u32 = 0, // Counter for indirect draw calls

    font: gui.Font = undefined,
    png_handle: gui.PngImage = undefined,
    last_cam_matrix: [16]f32 = undefined,

    // const MAX_VERTICES = 32768;
    const MAX_VERTICES = 0xF000;
    const MAX_INDICES = MAX_VERTICES * 3 / 2;
    const MAX_DRAW_CALLS = MAX_VERTICES / 2; // Max number of unique strings per frame

    pub fn init(vk_ctx: *vk.VulkanContext, render_pass: vk.RenderPass, main_scene_ds_layout: vk.DescriptorSetLayout) !Self {
        var self: Self = .{
            .vk_ctx = vk_ctx,
            .font = gui.Font.init(vk_ctx.allocator),
        };

        try self.font.loadFNT(font.periclesW01_fnt);
        try self.createFontTextureAndSampler(vk_ctx.allocator, font.periclesW01_png);
        try self.createDescriptors();
        try self.createBuffers(vk_ctx);
        try self.createPipeline(render_pass, main_scene_ds_layout);
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroySampler(self.vk_ctx.device.handle, self.sampler, null);
        c.vkDestroyImageView(self.vk_ctx.device.handle, self.texture_view, null);
        self.texture.deinit(self.vk_ctx);
        self.vk_ctx.allocator.free(self.png_handle.pixels);

        c.vkDestroyDescriptorPool(self.vk_ctx.device.handle, self.font_descriptor_pool.handle, null);
        c.vkDestroyDescriptorSetLayout(self.vk_ctx.device.handle, self.font_descriptor_set_layout.handle, null);
        c.vkDestroyDescriptorSetLayout(self.vk_ctx.device.handle, self.draw_data_descriptor_set_layout.handle, null);

        self.pipeline.deinit(self.vk_ctx);
        self.pipeline_layout.deinit(self.vk_ctx);

        self.font.deinit();
        self.vertex_buffer.unmap(self.vk_ctx);
        self.index_buffer.unmap(self.vk_ctx);
        self.indirect_cmd_buffer.unmap(self.vk_ctx);
        self.draw_data_buffer.unmap(self.vk_ctx);

        self.vertex_buffer.deinit(self.vk_ctx);
        self.index_buffer.deinit(self.vk_ctx);
        self.indirect_cmd_buffer.deinit(self.vk_ctx);
        self.draw_data_buffer.deinit(self.vk_ctx);
    }

    fn createFontTextureAndSampler(self: *Self, allocator: std.mem.Allocator, png_data: []const u8) !void {
        const image = try gui.loadPngGrayscale(allocator, png_data);
        self.png_handle = image;

        const image_size: u64 = @intCast(image.width * image.height * (image.bit_depth / 8));

        var staging_buffer = try vk.Buffer.init(
            self.vk_ctx,
            image_size,
            c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, u8);
        @memcpy(data_ptr[0..image_size], image.pixels[0..image_size]);
        staging_buffer.unmap(self.vk_ctx);

        self.texture = try vk.Image.create(
            self.vk_ctx,
            image.width,
            image.height,
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
    // text.zig

    fn createDescriptors(self: *Self) !void {
        // --- Layouts ---
        self.font_descriptor_set_layout = try vk.DescriptorSetLayout.init(self.vk_ctx, &.{
            .{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .stage_flags = c.VK_SHADER_STAGE_FRAGMENT_BIT },
        });

        self.draw_data_descriptor_set_layout = try vk.DescriptorSetLayout.init(self.vk_ctx, &.{
            .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .stage_flags = c.VK_SHADER_STAGE_VERTEX_BIT },
        });

        // --- Pool ---
        // Create a pool that can hold one of each descriptor type, and specify we need to allocate 2 sets from it.
        self.font_descriptor_pool = try vk.DescriptorPool.init(
            self.vk_ctx,
            2, // max_sets: we will allocate 2 sets
            &.{
                .{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .count = 1 },
                .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .count = 1 },
            },
        );

        // --- Allocation ---
        var sets_to_alloc = [_]vk.DescriptorSet{
            .{}, // font_descriptor_set
            .{}, // draw_data_descriptor_set
        };
        try self.font_descriptor_pool.allocateSets(self.vk_ctx, &.{
            self.font_descriptor_set_layout,
            self.draw_data_descriptor_set_layout,
        }, &sets_to_alloc);
        self.font_descriptor_set = sets_to_alloc[0];
        self.draw_data_descriptor_set = sets_to_alloc[1];

        // --- Update Set 1 (Font Sampler) ---
        try self.font_descriptor_set.update(self.vk_ctx, &.{
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

        // Update Set 2 (Draw Data SSBO) will be done in createBuffers()
    }

    fn createBuffers(self: *Self, vk_ctx: *vk.VulkanContext) !void {
        const vtx_buffer_size = MAX_VERTICES * @sizeOf(Text3DVertex);
        self.vertex_buffer = try vk.Buffer.init(
            vk_ctx,
            vtx_buffer_size,
            c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_vertices = try self.vertex_buffer.map(vk_ctx, Text3DVertex);

        // Index Buffer
        const idx_buffer_size = MAX_INDICES * @sizeOf(u32);
        self.index_buffer = try vk.Buffer.init(
            vk_ctx,
            idx_buffer_size,
            c.VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_indices = try self.index_buffer.map(vk_ctx, u32);

        // Indirect Command Buffer
        const indirect_buffer_size = MAX_DRAW_CALLS * @sizeOf(IndirectCommand);
        self.indirect_cmd_buffer = try vk.Buffer.init(
            vk_ctx,
            indirect_buffer_size,
            c.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_indirect_commands = try self.indirect_cmd_buffer.map(vk_ctx, IndirectCommand);

        // Draw Data SSBO
        const draw_data_buffer_size = MAX_DRAW_CALLS * @sizeOf(DrawData);
        self.draw_data_buffer = try vk.Buffer.init(
            vk_ctx,
            draw_data_buffer_size,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_draw_data = try self.draw_data_buffer.map(vk_ctx, DrawData);

        // --- Update Set 2 (Draw Data SSBO) ---
        try self.draw_data_descriptor_set.update(self.vk_ctx, &.{
            .{
                .binding = 0,
                .info = .{
                    .StorageBuffer = .{
                        .buffer = self.draw_data_buffer.handle,
                        .offset = 0,
                        .range = c.VK_WHOLE_SIZE,
                    },
                },
            },
        });
    }

    pub fn createPipeline(self: *Self, render_pass: vk.RenderPass, main_scene_ds_layout: vk.DescriptorSetLayout) !void {
        const set_layouts = [_]c.VkDescriptorSetLayout{
            main_scene_ds_layout.handle,
            self.font_descriptor_set_layout.handle,
            self.draw_data_descriptor_set_layout.handle,
        };

        self.pipeline_layout = try vk.PipelineLayout.init(self.vk_ctx, .{
            .setLayoutCount = set_layouts.len,
            .pSetLayouts = &set_layouts,
        });

        var vert_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx, text3d_vert_shader_bin);
        defer vert_mod.deinit(self.vk_ctx);
        var frag_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx, text3d_frag_shader_bin);
        defer frag_mod.deinit(self.vk_ctx);

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
            .{ .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vert_mod.handle, .pName = "main" },
            .{ .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_mod.handle, .pName = "main" },
        };

        const binding_desc = Text3DVertex.getBindingDescription();
        const attrib_desc = Text3DVertex.getAttributeDescriptions();
        const vertex_input_info = c.VkPipelineVertexInputStateCreateInfo{
            .vertexBindingDescriptionCount = 1,
            .pVertexBindingDescriptions = &binding_desc,
            .vertexAttributeDescriptionCount = attrib_desc.len,
            .pVertexAttributeDescriptions = &attrib_desc,
        };
        const input_assembly = c.VkPipelineInputAssemblyStateCreateInfo{
            .topology = c.VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
        };

        // We need blending for the transparent parts of the glyphs
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
            .attachmentCount = 1,
            .pAttachments = &color_blend_attachment,
        };

        // IMPORTANT: Enable depth testing so text is occluded correctly
        const depth_stencil = c.VkPipelineDepthStencilStateCreateInfo{
            .depthTestEnable = c.VK_TRUE, // MUST BE TRUE
            .depthWriteEnable = c.VK_TRUE, // Write depth so text occludes other text
            .depthCompareOp = c.VK_COMPARE_OP_GREATER_OR_EQUAL, // MUST MATCH MAIN SCENE
            .depthBoundsTestEnable = c.VK_FALSE,
            .stencilTestEnable = c.VK_FALSE,
        };

        // Other states can be copied/adapted from GuiRenderer's pipeline
        const rasterizer = c.VkPipelineRasterizationStateCreateInfo{
            .depthClampEnable = c.VK_FALSE,
            .rasterizerDiscardEnable = c.VK_FALSE,
            .polygonMode = c.VK_POLYGON_MODE_FILL,
            .lineWidth = 1.0,
            .cullMode = c.VK_CULL_MODE_NONE,
        };
        const multisampling = c.VkPipelineMultisampleStateCreateInfo{
            .sampleShadingEnable = c.VK_FALSE,
            .rasterizationSamples = c.VK_SAMPLE_COUNT_1_BIT,
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
        self.draw_count = 0; // Reset draw counter
    }

    pub fn endFrame(self: *Self, cmd_buffer: c.VkCommandBuffer, main_descriptor_set: vk.DescriptorSet) void {
        if (self.draw_count == 0) return;

        c.vkCmdBindPipeline(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.handle);

        const desc_sets = [_]c.VkDescriptorSet{
            main_descriptor_set.handle, // 0: Scene UBO
            self.font_descriptor_set.handle, // 1: Font Sampler
            self.draw_data_descriptor_set.handle, // 2: Draw Data SSBO
        };
        c.vkCmdBindDescriptorSets(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, desc_sets.len, &desc_sets, 0, null);

        const offset: c.VkDeviceSize = 0;
        c.vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &self.vertex_buffer.handle, &offset);
        c.vkCmdBindIndexBuffer(cmd_buffer, self.index_buffer.handle, 0, c.VK_INDEX_TYPE_UINT32);

        // Calls draw for each `IndirectCommand` in buffer under the hood
        c.vkCmdDrawIndexedIndirect(cmd_buffer, self.indirect_cmd_buffer.handle, 0, self.draw_count, @sizeOf(IndirectCommand));
    }

    pub fn processAndDrawTextScene(self: *Self, scene: *const Text3DScene) void {
        for (scene.list.items) |d_text| {
            const model_matrix = switch (d_text.transform) {
                .static => |mat| mat,
                .billboard => |b| createBillboardMatrix(scene.last_cam_matrix, b.position, d_text.font_size),
            };

            // Delegate the actual buffer population to a private helper.
            self.drawText(d_text.text, model_matrix, d_text.color, d_text.font_size);
        }
    }

    // Populates the indirect and SSBO buffers
    pub fn drawText(self: *Self, text: []const u8, model_matrix: [16]f32, color: [4]f32, size_in_world_units: f32) void {
        const estimated_chars = text.len;

        if (self.draw_count >= MAX_DRAW_CALLS or
            self.vertex_count + (estimated_chars * 4) > MAX_VERTICES or
            self.index_count + (estimated_chars * 6) > MAX_INDICES)
        {
            std.log.warn("IN DRAW TXT3D: limit reached {} {} {}", .{ self.vertex_count, self.index_count, self.draw_count });
            return;
        }

        if (self.font.font_size == 0 or self.font.line_height == 0) return;
        const scale = size_in_world_units / self.font.font_size;

        var local_vertex_count: u32 = 0;
        var local_index_count: u32 = 0;

        const base_vertex_offset = self.vertex_count;
        const base_index_offset = self.index_count;

        var current_x: f32 = 0;
        var current_y: f32 = 0;

        for (text) |char_code| {
            if (char_code == 170) { //End of string
                break;
            }
            if (char_code == '\n') {
                current_x = 0;
                current_y += self.font.line_height;
                continue;
            }

            if (local_vertex_count + 4 > estimated_chars * 4) continue;

            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;

            const x0 = (current_x + glyph.xoffset) * scale;
            const x1 = x0 + (@as(f32, @floatFromInt(glyph.width)) * scale);
            const z0 = -(current_y + glyph.yoffset) * scale;
            const z1 = z0 - (@as(f32, @floatFromInt(glyph.height)) * scale);

            const raw_u0 = @as(f32, @floatFromInt(glyph.x)) / self.font.scale_w;
            const raw_v0 = @as(f32, @floatFromInt(glyph.y)) / self.font.scale_h;
            const raw_u1 = raw_u0 + (@as(f32, @floatFromInt(glyph.width)) / self.font.scale_w);
            const raw_v1 = raw_v0 + (@as(f32, @floatFromInt(glyph.height)) / self.font.scale_h);

            const v_write_idx = base_vertex_offset + local_vertex_count;
            self.mapped_vertices[v_write_idx + 0] = .{ .pos = .{ x0, 0, z0 }, .uv = .{ raw_u0, raw_v0 }, .color = color }; // v0 (TL)
            self.mapped_vertices[v_write_idx + 1] = .{ .pos = .{ x1, 0, z0 }, .uv = .{ raw_u1, raw_v0 }, .color = color }; // v1 (TR)
            self.mapped_vertices[v_write_idx + 2] = .{ .pos = .{ x1, 0, z1 }, .uv = .{ raw_u1, raw_v1 }, .color = color }; // v2 (BR)
            self.mapped_vertices[v_write_idx + 3] = .{ .pos = .{ x0, 0, z1 }, .uv = .{ raw_u0, raw_v1 }, .color = color }; // v3 (BL)

            const i_write_idx = base_index_offset + local_index_count;
            const v_base_idx = local_vertex_count;

            // Triangle 1: (v0, v2, v1)
            self.mapped_indices[i_write_idx + 0] = v_base_idx + 0;
            self.mapped_indices[i_write_idx + 1] = v_base_idx + 2;
            self.mapped_indices[i_write_idx + 2] = v_base_idx + 1;

            // Triangle 2: (v0, v3, v2)
            self.mapped_indices[i_write_idx + 3] = v_base_idx + 0;
            self.mapped_indices[i_write_idx + 4] = v_base_idx + 3;
            self.mapped_indices[i_write_idx + 5] = v_base_idx + 2;

            local_vertex_count += 4;
            local_index_count += 6;

            current_x += glyph.xadvance;
        }

        if (local_index_count > 0) {
            self.mapped_draw_data[self.draw_count].model = model_matrix;

            self.mapped_indirect_commands[self.draw_count] = .{
                .indexCount = local_index_count,
                .instanceCount = 1,
                .firstIndex = base_index_offset,
                .vertexOffset = @intCast(base_vertex_offset),
                .firstInstance = self.draw_count,
            };

            self.vertex_count += local_vertex_count;
            self.index_count += local_index_count;
            self.draw_count += 1;
        }
    }
};
