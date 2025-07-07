const std = @import("std");
const vk = @import("../test.zig");
const c = @import("c").c;
const gui = @import("../gui/gui.zig"); // To reuse Font struct
const font = @import("font");

const text3d_vert_shader_bin = @import("spirv").text3d_vs;
const text3d_frag_shader_bin = @import("spirv").text3d_fs;

// CHANGED: The vertex definition no longer includes the model matrix
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

    // CHANGED: Reduced attribute descriptions
    pub fn getAttributeDescriptions() [3]c.VkVertexInputAttributeDescription {
        return .{
            // location 0: in_pos (vec3)
            .{ .binding = 0, .location = 0, .format = c.VK_FORMAT_R32G32B32_SFLOAT, .offset = @offsetOf(Text3DVertex, "pos") },
            // location 1: in_uv (vec2)
            .{ .binding = 0, .location = 1, .format = c.VK_FORMAT_R32G32_SFLOAT, .offset = @offsetOf(Text3DVertex, "uv") },
            // location 2: in_color (vec4)
            .{ .binding = 0, .location = 2, .format = c.VK_FORMAT_R32G32B32A32_SFLOAT, .offset = @offsetOf(Text3DVertex, "color") },
        };
    }
};

// NEW: This struct will be stored in an SSBO. Each instance corresponds to one drawText call.
const DrawData = extern struct {
    model: [16]f32, // mat4
};

// We will build this buffer on the CPU every frame
const IndirectCommand = c.VkDrawIndexedIndirectCommand;

pub const Text3DRenderer = struct {
    const Self = @This();

    vk_ctx: *vk.VulkanContext,
    pipeline: vk.Pipeline = undefined,
    pipeline_layout: vk.PipelineLayout = undefined,

    // Font Texture Descriptors (Set 1)
    font_descriptor_set_layout: c.VkDescriptorSetLayout = undefined,
    font_descriptor_pool: c.VkDescriptorPool = undefined, // This pool will now hold two sets
    font_descriptor_set: c.VkDescriptorSet = undefined,

    // Draw Data Descriptors (Set 2) for the SSBO
    draw_data_descriptor_set_layout: c.VkDescriptorSetLayout = undefined,
    draw_data_descriptor_set: c.VkDescriptorSet = undefined,

    sampler: c.VkSampler = undefined,
    texture: vk.Image = undefined,
    texture_view: c.VkImageView = undefined,

    // --- Buffers ---
    // NEW: Buffer for indirect draw commands
    indirect_command_buffer: vk.Buffer = undefined,
    mapped_indirect_commands: [*]IndirectCommand = undefined,

    // NEW: Buffer for draw data (model matrices) SSBO
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
    draw_count: u32 = 0, // NEW: Counter for indirect draw calls

    font: gui.Font = undefined,
    png_handle: gui.PngImage = undefined,

    const MAX_VERTICES = 16384;
    const MAX_INDICES = MAX_VERTICES * 3 / 2;
    const MAX_DRAW_CALLS = 256; // This is now the max number of unique strings per frame

    pub fn init(vk_ctx: *vk.VulkanContext, render_pass: vk.RenderPass, main_scene_ds_layout: vk.DescriptorSetLayout, swapchain: vk.Swapchain) !Self {
        var self: Self = .{
            .vk_ctx = vk_ctx,
            .font = gui.Font.init(vk_ctx.allocator),
        };

        try self.font.loadFNT(font.periclesW01_fnt);
        try self.createFontTextureAndSampler(vk_ctx.allocator, font.periclesW01_png);
        // CHANGED: Descriptor creation now handles both font and SSBO
        try self.createDescriptors();
        try self.createBuffers(vk_ctx);
        // CHANGED: Pipeline creation now takes the new SSBO descriptor layout
        try self.createPipeline(render_pass, main_scene_ds_layout, swapchain);
        return self;
    }

    pub fn deinit(self: *Self) void {
        c.vkDestroySampler(self.vk_ctx.device.handle, self.sampler, null);
        c.vkDestroyImageView(self.vk_ctx.device.handle, self.texture_view, null);
        self.texture.deinit(self.vk_ctx);
        self.vk_ctx.allocator.free(self.png_handle.pixels);

        c.vkDestroyDescriptorPool(self.vk_ctx.device.handle, self.font_descriptor_pool, null);
        c.vkDestroyDescriptorSetLayout(self.vk_ctx.device.handle, self.font_descriptor_set_layout, null);
        // NEW: deinit draw data layout
        c.vkDestroyDescriptorSetLayout(self.vk_ctx.device.handle, self.draw_data_descriptor_set_layout, null);

        self.pipeline.deinit(self.vk_ctx);
        self.pipeline_layout.deinit(self.vk_ctx);

        self.font.deinit();
        self.vertex_buffer.unmap(self.vk_ctx);
        self.index_buffer.unmap(self.vk_ctx);
        // NEW: unmap new buffers
        self.indirect_command_buffer.unmap(self.vk_ctx);
        self.draw_data_buffer.unmap(self.vk_ctx);

        self.vertex_buffer.deinit(self.vk_ctx);
        self.index_buffer.deinit(self.vk_ctx);
        // NEW: deinit new buffers
        self.indirect_command_buffer.deinit(self.vk_ctx);
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

    // CHANGED: Renamed and expanded to create all descriptor sets for this renderer
    fn createDescriptors(self: *Self) !void {
        // --- Layouts ---
        // Layout for Font Sampler (Set 1)
        const sampler_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_FRAGMENT_BIT,
        };
        const font_layout_info = c.VkDescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &sampler_layout_binding };
        try vk.vkCheck(c.vkCreateDescriptorSetLayout(self.vk_ctx.device.handle, &font_layout_info, null, &self.font_descriptor_set_layout));

        // NEW: Layout for Draw Data SSBO (Set 2)
        const ssbo_layout_binding = c.VkDescriptorSetLayoutBinding{
            .binding = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = c.VK_SHADER_STAGE_VERTEX_BIT, // Used in vertex shader
        };
        const draw_data_layout_info = c.VkDescriptorSetLayoutCreateInfo{ .bindingCount = 1, .pBindings = &ssbo_layout_binding };
        try vk.vkCheck(c.vkCreateDescriptorSetLayout(self.vk_ctx.device.handle, &draw_data_layout_info, null, &self.draw_data_descriptor_set_layout));

        // --- Pool ---
        // NEW: Pool needs to accommodate both descriptor types
        const pool_sizes = [_]c.VkDescriptorPoolSize{
            .{ .type = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1 },
            .{ .type = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 },
        };
        const pool_info = c.VkDescriptorPoolCreateInfo{
            .poolSizeCount = pool_sizes.len,
            .pPoolSizes = &pool_sizes,
            .maxSets = 2, // We are allocating two sets from this pool
        };
        try vk.vkCheck(c.vkCreateDescriptorPool(self.vk_ctx.device.handle, &pool_info, null, &self.font_descriptor_pool));

        // --- Allocation & Updates ---
        // Allocate Set 1 (Font Sampler)
        const set_layouts = [_]c.VkDescriptorSetLayout{ self.font_descriptor_set_layout, self.draw_data_descriptor_set_layout };
        var allocated_sets: [2]c.VkDescriptorSet = undefined;
        const set_alloc_info = c.VkDescriptorSetAllocateInfo{
            .descriptorPool = self.font_descriptor_pool,
            .descriptorSetCount = set_layouts.len,
            .pSetLayouts = &set_layouts,
        };
        try vk.vkCheck(c.vkAllocateDescriptorSets(self.vk_ctx.device.handle, &set_alloc_info, &allocated_sets));
        self.font_descriptor_set = allocated_sets[0];
        self.draw_data_descriptor_set = allocated_sets[1];

        // Update Set 1 (Font Sampler)
        const image_info = c.VkDescriptorImageInfo{
            .imageLayout = c.VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            .imageView = self.texture_view,
            .sampler = self.sampler,
        };
        const font_desc_write = c.VkWriteDescriptorSet{
            .dstSet = self.font_descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
            .descriptorCount = 1,
            .pImageInfo = &image_info,
        };
        c.vkUpdateDescriptorSets(self.vk_ctx.device.handle, 1, &font_desc_write, 0, null);

        // Update Set 2 (Draw Data SSBO) - We'll do this after creating the buffer in createBuffers()
    }

    fn createBuffers(self: *Self, vk_ctx: *vk.VulkanContext) !void {
        // Vertex Buffer
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

        // NEW: Indirect Command Buffer
        const indirect_buffer_size = MAX_DRAW_CALLS * @sizeOf(IndirectCommand);
        self.indirect_command_buffer = try vk.Buffer.init(
            vk_ctx,
            indirect_buffer_size,
            c.VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_indirect_commands = try self.indirect_command_buffer.map(vk_ctx, IndirectCommand);

        // NEW: Draw Data SSBO
        const draw_data_buffer_size = MAX_DRAW_CALLS * @sizeOf(DrawData);
        self.draw_data_buffer = try vk.Buffer.init(
            vk_ctx,
            draw_data_buffer_size,
            c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        self.mapped_draw_data = try self.draw_data_buffer.map(vk_ctx, DrawData);

        // NEW: Now that the SSBO exists, update its descriptor set
        const buffer_info = c.VkDescriptorBufferInfo{
            .buffer = self.draw_data_buffer.handle,
            .offset = 0,
            .range = c.VK_WHOLE_SIZE,
        };
        const ssbo_desc_write = c.VkWriteDescriptorSet{
            .dstSet = self.draw_data_descriptor_set,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorType = c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .pBufferInfo = &buffer_info,
        };
        c.vkUpdateDescriptorSets(self.vk_ctx.device.handle, 1, &ssbo_desc_write, 0, null);
    }

    pub fn createPipeline(self: *Self, render_pass: vk.RenderPass, main_scene_ds_layout: vk.DescriptorSetLayout, swapchain: vk.Swapchain) !void {
        // CHANGED: Now three descriptor set layouts
        const set_layouts = [_]c.VkDescriptorSetLayout{
            main_scene_ds_layout.handle,
            self.font_descriptor_set_layout,
            self.draw_data_descriptor_set_layout, // NEW
        };

        self.pipeline_layout = try vk.PipelineLayout.init(self.vk_ctx, .{
            .setLayoutCount = set_layouts.len,
            .pSetLayouts = &set_layouts,
        });

        // The rest of the pipeline creation is almost identical, just using the updated vertex definition
        var vert_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx.device.handle, text3d_vert_shader_bin);
        defer vert_mod.deinit();
        var frag_mod = try vk.ShaderModule.init(self.vk_ctx.allocator, self.vk_ctx.device.handle, text3d_frag_shader_bin);
        defer frag_mod.deinit();

        const shader_stages = [_]c.VkPipelineShaderStageCreateInfo{
            .{ .stage = c.VK_SHADER_STAGE_VERTEX_BIT, .module = vert_mod.handle, .pName = "main" },
            .{ .stage = c.VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag_mod.handle, .pName = "main" },
        };

        const binding_desc = Text3DVertex.getBindingDescription();
        const attrib_desc = Text3DVertex.getAttributeDescriptions(); // CHANGED: This is now smaller
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

    pub fn beginFrame(self: *Self) void {
        self.vertex_count = 0;
        self.index_count = 0;
        self.draw_count = 0; // NEW: Reset draw counter
    }

    // CHANGED: endFrame now uses vkCmdDrawIndexedIndirect
    pub fn endFrame(self: *Self, cmd_buffer: c.VkCommandBuffer, main_descriptor_set: vk.DescriptorSet) void {
        if (self.draw_count == 0) return;

        c.vkCmdBindPipeline(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.handle);

        // CHANGED: Bind all three descriptor sets
        const d_sets = [_]c.VkDescriptorSet{
            main_descriptor_set.handle, // Set 0: Scene UBO
            self.font_descriptor_set, // Set 1: Font Sampler
            self.draw_data_descriptor_set, // Set 2: Draw Data SSBO
        };
        c.vkCmdBindDescriptorSets(cmd_buffer, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, d_sets.len, &d_sets, 0, null);

        const offset: c.VkDeviceSize = 0;
        c.vkCmdBindVertexBuffers(cmd_buffer, 0, 1, &self.vertex_buffer.handle, &offset);
        c.vkCmdBindIndexBuffer(cmd_buffer, self.index_buffer.handle, 0, c.VK_INDEX_TYPE_UINT32);

        // CHANGED: The single draw call is now indirect
        c.vkCmdDrawIndexedIndirect(cmd_buffer, self.indirect_command_buffer.handle, // The buffer containing the draw commands
            0, // Offset into the buffer
            self.draw_count, // The number of draws to execute
            @sizeOf(IndirectCommand) // The stride between commands
        );
    }

    // CHANGED: drawText now populates the indirect and SSBO buffers
    // The main drawing function
    pub fn drawText(self: *Self, text: []const u8, model_matrix: [16]f32, color: [4]f32, size_in_world_units: f32) void {
        const char_count = text.len;
        // Check if we have space for the new draw call, its vertices, and its indices
        if (self.draw_count >= MAX_DRAW_CALLS or
            self.vertex_count + (char_count * 4) > MAX_VERTICES or
            self.index_count + (char_count * 6) > MAX_INDICES)
        {
            std.log.info("limit reached\n {} {} {}", .{ MAX_VERTICES, MAX_INDICES, MAX_DRAW_CALLS });
            return;
        }

        if (self.font.font_size == 0 or self.font.line_height == 0) return;
        const scale = size_in_world_units / @as(f32, self.font.font_size);

        // Remember where this string's geometry starts
        const base_vertex_offset = self.vertex_count;
        const base_index_offset = self.index_count;
        var local_index_count: u32 = 0;

        var current_x: f32 = 0;
        var current_y: f32 = 0; // This now represents the Z-axis cursor

        for (text) |char_code| {
            if (char_code == '\n') {
                current_x = 0;
                current_y -= @as(f32, self.font.line_height); // Go "down" for the next line
                continue;
            }

            const glyph = self.font.glyphs.get(char_code) orelse self.font.glyphs.get('?') orelse continue;

            // Local coordinates for the glyph quad.
            // X remains horizontal.
            // Y from the font file now becomes Z in 3D space.
            const x0 = (current_x + glyph.xoffset) * scale;
            const x1 = x0 + (@as(f32, @floatFromInt(glyph.width)) * scale);

            // We subtract here because in many font formats, the Y-axis points downwards from the baseline.
            // This makes the text "stand up" along the positive Z-axis.
            const z0 = -(current_y + glyph.yoffset) * scale;
            const z1 = z0 - (@as(f32, @floatFromInt(glyph.height)) * scale);

            // UV calculations remain the same
            const raw_u0 = @as(f32, @floatFromInt(glyph.x)) / self.font.scale_w;
            const raw_v0 = @as(f32, @floatFromInt(glyph.y)) / self.font.scale_h;
            const raw_u1 = raw_u0 + (@as(f32, @floatFromInt(glyph.width)) / self.font.scale_w);
            const raw_v1 = raw_v0 + (@as(f32, @floatFromInt(glyph.height)) / self.font.scale_h);

            const v_idx = self.vertex_count;

            self.mapped_indices[self.index_count + 0] = v_idx;
            self.mapped_indices[self.index_count + 1] = v_idx + 1;
            self.mapped_indices[self.index_count + 2] = v_idx + 2;
            self.mapped_indices[self.index_count + 3] = v_idx;
            self.mapped_indices[self.index_count + 4] = v_idx + 2;
            self.mapped_indices[self.index_count + 5] = v_idx + 3;

            // --- THE KEY CHANGE IS HERE ---
            // We are mapping the 2D (x, y) coordinates to the 3D (x, 0, z) plane.
            self.mapped_vertices[v_idx + 0] = .{ .pos = .{ x0, 0, z0 }, .uv = .{ raw_u0, raw_v0 }, .color = color };
            self.mapped_vertices[v_idx + 1] = .{ .pos = .{ x1, 0, z0 }, .uv = .{ raw_u1, raw_v0 }, .color = color };
            self.mapped_vertices[v_idx + 2] = .{ .pos = .{ x1, 0, z1 }, .uv = .{ raw_u1, raw_v1 }, .color = color };
            self.mapped_vertices[v_idx + 3] = .{ .pos = .{ x0, 0, z1 }, .uv = .{ raw_u0, raw_v1 }, .color = color };
            // -----------------------------

            self.vertex_count += 4;
            self.index_count += 6;
            local_index_count += 6;

            // Advance horizontal cursor for the next character on the same line
            current_x += glyph.xadvance;
        }

        // std.debug.print("Vertex count: {}\n", .{self.vertex_count});

        // If we actually added any geometry for this string
        if (local_index_count > 0) {
            // Populate the Draw Data SSBO for this draw
            self.mapped_draw_data[self.draw_count].model = model_matrix;

            // Populate the Indirect Command for this draw
            self.mapped_indirect_commands[self.draw_count] = .{
                .indexCount = local_index_count,
                .instanceCount = 1,
                .firstIndex = base_index_offset,
                .vertexOffset = @intCast(base_vertex_offset),
                .firstInstance = self.draw_count,
            };

            // Increment the number of draws for this frame
            self.draw_count += 1;
        }
    }
};
