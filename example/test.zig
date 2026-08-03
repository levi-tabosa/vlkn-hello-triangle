// test.zig
const std = @import("std");
const assert = std.debug.assert;
const spirv = @import("spirv");
const scene = @import("geometry");
const gui = @import("gui");
const text = @import("text3d");
const fps_tracker = @import("fps_tracker");
//TODO: remove
const rand = std.Random;
const Allocator = std.mem.Allocator;

const vk = @import("vulkan");

pub const c = vk.c;

// Re-export Vulkan types for modules that import this file
pub const vkCheck = vk.vkCheck;
pub const checkGlfw = vk.checkGlfw;
pub const VulkanContext = vk.VulkanContext;
pub const Instance = vk.Instance;
pub const Surface = vk.Surface;
pub const PhysicalDevice = vk.PhysicalDevice;
pub const Device = vk.Device;
pub const Queue = vk.Queue;
pub const CommandBuffer = vk.CommandBuffer;
pub const CommandPool = vk.CommandPool;
pub const Swapchain = vk.Swapchain;
pub const Image = vk.Image;
pub const DepthBuffer = vk.DepthBuffer;
pub const RenderPass = vk.RenderPass;
pub const SyncObjects = vk.SyncObjects;
pub const Buffer = vk.Buffer;
pub const PushConstantRange = vk.PushConstantRange;
pub const DescriptorType = vk.DescriptorType;
pub const DescriptorSetLayoutArgs = vk.DescriptorSetLayoutArgs;
pub const DescriptorSetLayout = vk.DescriptorSetLayout;
pub const DescriptorWriteInfo = vk.DescriptorWriteInfo;
pub const DescriptorWrite = vk.DescriptorWrite;
pub const DescriptorSet = vk.DescriptorSet;
pub const DescriptorPoolArgs = vk.DescriptorPoolArgs;
pub const DescriptorPool = vk.DescriptorPool;
pub const ShaderModule = vk.ShaderModule;
pub const PipelineLayout = vk.PipelineLayout;
const Pipeline = vk.Pipeline;

// --- Shader Bytecode ---
const vert_shader_bin = spirv.test_vert;
const frag_shader_bin = spirv.test_frag;

const Vertex = scene.V3;
const Scene = scene.Scene;

// --- Application Constants ---
const WINDOW_WIDTH = 800;
const WINDOW_HEIGHT = 600;

// --- Vertex Definition ---

const UniformBufferObject = extern struct {
    view_matrix: [16]f32,
    perspective_matrix: [16]f32,
    padding: [128]u8 = undefined,
};

// --- Callbacks and Window ---
const Callbacks = struct {
    fn cbCursorPos(wd: ?*c.GLFWwindow, xpos: f64, ypos: f64) callconv(.c) void {
        const app: *App = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(wd) orelse unreachable));
        app.gui_renderer.handleCursorPos(xpos, ypos);
        app.wd_ctx.handleCursorPos(xpos, ypos);
    }

    fn cbMouseButton(wd: ?*c.GLFWwindow, button: c_int, action: c_int, mods: c_int) callconv(.c) void {
        const app: *App = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(wd) orelse unreachable));
        app.gui_renderer.handleMouseButton(button, action, mods);
        app.wd_ctx.handleMouseButton(button, action);
    }

    // in test.zig, inside Callbacks struct
    fn cbKey(wd: ?*c.GLFWwindow, key: c_int, code: c_int, action: c_int, mods: c_int) callconv(.c) void {
        _ = code;
        _ = mods;
        const app: *App = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(wd) orelse unreachable));

        // MODIFIED: Forward key events to the GUI
        app.gui_renderer.handleKey(key, action, &app.main_ui);

        // You can still have app-level keybinds here, just make sure they don't conflict.
        if (key == c.GLFW_KEY_G and action == c.GLFW_PRESS) {}
    }

    fn cbFramebufferResize(wd: ?*c.GLFWwindow, width: c_int, height: c_int) callconv(.c) void {
        const app: *App = @ptrCast(@alignCast(c.glfwGetWindowUserPointer(wd) orelse unreachable));
        app.window.size.x = width;
        app.window.size.y = height;
        app.framebuffer_resized = true;
    }

    fn cbScroll(wd: ?*c.GLFWwindow, xoffset: f64, yoffset: f64) callconv(.c) void {
        const user_ptr = c.glfwGetWindowUserPointer(wd) orelse unreachable;
        const app: *App = @ptrCast(@alignCast(user_ptr));

        app.wd_ctx.handleScroll(xoffset, yoffset);
    }
};

fn addLineCallback(ptr: *anyopaque) void {
    const app: *App = @ptrCast(@alignCast(ptr));
    std.log.info("Button clicked: Adding a new line...", .{});

    const x = std.fmt.parseFloat(f32, app.line_x_buf.slice()) catch blk: {
        std.log.err("Invalid X coordinate: {s}", .{app.line_x_buf.slice()});
        break :blk 0;
    };
    const y = std.fmt.parseFloat(f32, app.line_y_buf.slice()) catch blk: {
        std.log.err("Invalid Y coordinate: {s}", .{app.line_y_buf.slice()});
        break :blk 0;
    };
    const z = std.fmt.parseFloat(f32, app.line_z_buf.slice()) catch blk: {
        std.log.err("Invalid Z coordinate: {s}", .{app.line_z_buf.slice()});
        break :blk 0;
    };

    var end_pos = @Vector(3, f32){ x, y, z };
    if (x == 0 and y == 0 and z == 0) {
        var prng = rand.DefaultPrng.init(42);
        const random = prng.random();

        end_pos = .{
            (random.float(f32) * 2.0 - 1.0) * 5,
            (random.float(f32) * 2.0 - 1.0) * 5,
            (random.float(f32) * 2.0 - 1.0) * 5,
        };
    }
    app.scene.addLine(.{ 0, 0, 0 }, end_pos) catch unreachable;

    const txt = std.fmt.allocPrint(app.allocator, "({d:.1},{d:.1},{d:.1})", .{ end_pos[0], end_pos[1], end_pos[2] }) catch unreachable;
    defer app.allocator.free(txt);
    app.text_scene.addBillboardText(
        txt,
        end_pos,
        .{ 1.0, 1.0, 1.0, 1.0 }, // white color
        0.65, // font scale
    ) catch unreachable;

    app.updateVertexBuffer() catch unreachable;
}

fn clearLinesCallback(ptr: *anyopaque) void {
    const app: *App = @ptrCast(@alignCast(ptr));
    std.log.info("Button clicked: Clearing lines.", .{});
    app.scene.clear();
    app.text_scene.clearText();
}

fn quitCallback(ptr: *anyopaque) void {
    const app: *App = @ptrCast(@alignCast(ptr));
    std.log.info("Button clicked: Quitting.", .{});
    c.glfwSetWindowShouldClose(app.window.handle, 1);
}

fn toggleCameraModeCallback(ptr: *anyopaque) void {
    const app: *App = @ptrCast(@alignCast(ptr));
    std.log.info("Button clicked: Toggling camera mode.", .{});
    app.scene.camera.toggleMode();
}

fn fovSliderCallback(ptr: *anyopaque, new_value: f32) void {
    const app: *App = @ptrCast(@alignCast(ptr));
    const min: f32 = 5.0;
    const max: f32 = 180;

    app.scene.camera.fov_degrees = min + (max - min) * new_value;
}

fn nearPlaneSliderCallback(ptr: *anyopaque, new_value: f32) void {
    const app: *App = @ptrCast(@alignCast(ptr));
    const min: f32 = 0.001;
    const max: f32 = 50.0;

    app.scene.camera.near_plane = min + (max - min) * new_value;
}

fn setGridCallback(ptr: *anyopaque) void {
    const app: *App = @ptrCast(@alignCast(ptr));
    std.debug.print("{any}\n", .{app.grid_res_buff.buf});
    app.scene.setGridResolution(
        std.fmt.parseInt(u32, app.grid_res_buff.slice(), 10) catch blk: {
            std.log.err("failed on parseInt", .{});
            break :blk 10;
        },
    ) catch unreachable;
    app.updateVertexBuffer() catch unreachable;
}

const Window = struct {
    const Self = @This();

    handle: ?*c.GLFWwindow = undefined,
    size: struct {
        x: c_int,
        y: c_int,
    },

    pub fn init(user_ptr: ?*anyopaque, width: c_int, height: c_int, title: [*c]const u8, monitor: ?*c.GLFWmonitor, share: ?*c.GLFWwindow) !Self {
        c.glfwWindowHint(c.GLFW_CLIENT_API, c.GLFW_NO_API);
        c.glfwWindowHint(c.GLFW_RESIZABLE, c.GLFW_TRUE);
        const handle = c.glfwCreateWindow(width, height, title, monitor, share) orelse return error.GlfwCreateWindowFailed;
        c.glfwSetWindowUserPointer(handle, user_ptr);

        _ = c.glfwSetCursorPosCallback(handle, Callbacks.cbCursorPos);
        _ = c.glfwSetMouseButtonCallback(handle, Callbacks.cbMouseButton);
        _ = c.glfwSetKeyCallback(handle, Callbacks.cbKey);
        _ = c.glfwSetFramebufferSizeCallback(handle, Callbacks.cbFramebufferResize);
        _ = c.glfwSetScrollCallback(handle, Callbacks.cbScroll);
        return .{ .handle = handle, .size = .{ .x = width, .y = height } };
    }

    pub fn deinit(self: *Self) void {
        c.glfwDestroyWindow(self.handle);
    }

    pub fn minimized(self: Self) bool {
        if (c.glfwGetWindowAttrib(self.handle, c.GLFW_ICONIFIED) == 1) return true;
        var w: c_int = 0;
        var h: c_int = 0;
        c.glfwGetFramebufferSize(self.handle, &w, &h);
        return w == 0 or h == 0;
    }
};

pub const WindowContext = struct {
    const Self = @This();
    // Raw state from GLFW
    // TODO: Use another struct for windowing back-end
    last_cursor_x: f64 = -1.0,
    last_cursor_y: f64 = -1.0,
    cursor_dx: f64 = 0,
    cursor_dy: f64 = 0,
    scroll_dy: f64 = 0,
    scroll_changed: bool = false,
    left_mouse_down: bool = false,
    ctrl_down: bool = false,

    // current velocity
    fling_dx: f64 = 0,
    fling_dy: f64 = 0, // current vertical   fling velocity
    flinging: bool = false, // are we currently in a fling?
    friction: f64 = 0.98, // per-frame decay factor (0.0–1.0)

    /// Reset per-frame state
    pub fn beginFrame(self: *Self) void {
        self.cursor_dx = 0;
        self.cursor_dy = 0;
        self.scroll_dy = 0;
        self.scroll_changed = false;
    }

    /// Returns true if camera was changed
    pub fn processCameraInput(self: *Self, main_scene: *Scene, text_scene: *text.Text3DScene) bool {
        var updated = false;

        // Use scroll for zoom/fov as before
        if (self.ctrl_down and self.scroll_changed) {
            main_scene.camera.adjustFov(@as(f32, @floatCast(self.scroll_dy)) * 4);
            updated = true;
        } else if (self.scroll_changed) {
            main_scene.camera.adjustRadius(@as(f32, @floatCast(self.scroll_dy)) * 2);
            updated = true;
        }

        // Drag and fling logic
        if (self.left_mouse_down) {
            const pitch_change = @as(f32, @floatCast(self.cursor_dy)) * 0.005;
            const yaw_change = @as(f32, @floatCast(self.cursor_dx)) * 0.005;
            main_scene.camera.adjustPitchYaw(pitch_change, yaw_change);
            text_scene.updateCameraViewMatrix(main_scene.camera.view());
            updated = true;

            self.fling_dx = 0;
            self.fling_dy = 0;
            self.flinging = false;
        } else if (self.flinging) {
            const pitch_change = @as(f32, @floatCast(self.fling_dy)) * 0.005;
            const yaw_change = @as(f32, @floatCast(self.fling_dx)) * 0.005;
            main_scene.camera.adjustPitchYaw(pitch_change, yaw_change);
            text_scene.updateCameraViewMatrix(main_scene.camera.view());
            updated = true;

            self.fling_dx *= self.friction;
            self.fling_dy *= self.friction;

            if (@abs(self.fling_dx) < 0.1 and @abs(self.fling_dy) < 0.1) {
                self.flinging = false;
            }
        }

        return updated;
    }

    pub fn handleCursorPos(self: *Self, x: f64, y: f64) void {
        if (self.last_cursor_x == -1.0) {
            self.last_cursor_x = x;
            self.last_cursor_y = y;
            return;
        }
        self.cursor_dx = x - self.last_cursor_x;
        self.cursor_dy = y - self.last_cursor_y;
        self.last_cursor_x = x;
        self.last_cursor_y = y;
    }

    pub fn handleMouseButton(self: *Self, button: c_int, action: c_int) void {
        if (button == c.GLFW_MOUSE_BUTTON_LEFT) {
            if (action == c.GLFW_RELEASE and self.left_mouse_down) {
                self.fling_dx = self.cursor_dx;
                self.fling_dy = self.cursor_dy;
                self.flinging = true;
            }
            self.left_mouse_down = (action == c.GLFW_PRESS);
        }
    }

    pub fn handleKey(self: *Self, key: c_int, action: c_int, mods: c_int) void {
        if (mods & c.GLFW_MOD_CONTROL != 0) self.ctrl_down = !self.ctrl_down;

        _ = action;
        _ = key;
    }

    pub fn handleScroll(self: *Self, _: f64, yoffset: f64) void {
        self.scroll_changed = true;
        self.scroll_dy = yoffset;
    }
};

// --- MAIN APPLICATION STRUCT ---
pub const App = struct {
    const Self = @This();

    allocator: Allocator,
    window: Window,
    vk_ctx: *VulkanContext,
    scene: Scene,
    text_scene: text.Text3DScene,
    main_ui: gui.UI,

    wd_ctx: WindowContext = undefined,
    gui_renderer: gui.GuiRenderer = undefined,
    text_renderer: text.Text3DRenderer = undefined,

    // Vulkan objects that depend on the swapchain (recreated on window resize)
    depth_buffer: DepthBuffer = undefined,
    swapchain: Swapchain = undefined,
    render_pass: RenderPass = undefined,
    descriptor_layout: DescriptorSetLayout = undefined,
    pipeline_layout: PipelineLayout = undefined,
    pipeline: Pipeline = undefined,

    // Other Vulkan objects
    vertex_buffer: Buffer = undefined,
    uniform_buffer: Buffer = undefined,
    descriptor_pool: DescriptorPool = undefined,
    descriptor_set: DescriptorSet = undefined,
    command_buffer: CommandBuffer = undefined,
    sync: SyncObjects = undefined,

    framebuffer_resized: bool = false,
    perf: fps_tracker.PerformanceTracker,

    line_x_buf: gui.TextBuffer = .{},
    line_y_buf: gui.TextBuffer = .{},
    line_z_buf: gui.TextBuffer = .{},
    grid_res_buff: gui.TextBuffer = .{},

    io: std.Io,

    /// Caller owns memory
    pub fn init(allocator: Allocator, io: std.Io) !*Self {
        var window = try Window.init(null, WINDOW_WIDTH, WINDOW_HEIGHT, "Vulkan Line App", null, null);
        errdefer window.deinit();

        const vk_ctx = try allocator.create(VulkanContext);
        errdefer allocator.destroy(vk_ctx);
        vk_ctx.* = try VulkanContext.init(allocator, window.handle.?);
        errdefer vk_ctx.deinit();

        const app = try allocator.create(App);
        errdefer allocator.destroy(app);

        app.allocator = allocator;
        app.io = io;
        app.window = window;
        app.vk_ctx = vk_ctx;
        app.wd_ctx = .{};

        app.scene = try Scene.init(allocator, 20);
        errdefer app.scene.deinit(allocator);

        app.main_ui = try gui.UI.init(allocator);
        errdefer app.main_ui.deinit();

        app.text_scene = try text.Text3DScene.init(allocator, 20);
        errdefer app.text_scene.deinit();

        app.perf = fps_tracker.PerformanceTracker.init(allocator, io);
        errdefer app.perf.deinit();

        app.line_x_buf = .{};
        app.line_y_buf = .{};
        app.line_z_buf = .{};
        app.grid_res_buff = .{};
        app.framebuffer_resized = false;

        try app.initUi();
        try app.initVulkanResources();

        c.glfwSetWindowUserPointer(app.window.handle, app);
        return app;
    }

    // Initialize Vulkan resources after the context is created
    fn initVulkanResources(self: *Self) !void {
        self.swapchain = try Swapchain.init(self.vk_ctx);
        errdefer self.swapchain.deinit(self.vk_ctx);

        self.depth_buffer = try DepthBuffer.init(self.vk_ctx, self.swapchain.extent.width, self.swapchain.extent.height);
        errdefer self.depth_buffer.deinit(self.vk_ctx);

        self.descriptor_layout = try DescriptorSetLayout.init(
            self.vk_ctx,
            &.{.{ .stage_flags = c.VK_SHADER_STAGE_VERTEX_BIT, .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER }},
        );
        errdefer self.descriptor_layout.deinit(self.vk_ctx);

        self.descriptor_pool = try DescriptorPool.init(self.vk_ctx, 1, &.{
            .{ .type = c.VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .count = 1 },
        });
        errdefer self.descriptor_pool.deinit(self.vk_ctx);

        self.descriptor_set = try self.descriptor_pool.allocateSet(self.vk_ctx, self.descriptor_layout);
        self.command_buffer = try CommandBuffer.allocate(self.vk_ctx, self.vk_ctx.command_pool, true);

        self.sync = try SyncObjects.init(self.vk_ctx);
        errdefer self.sync.deinit(self.vk_ctx);

        try self.initVertexBuffer();
        errdefer self.vertex_buffer.deinit(self.vk_ctx);

        try self.initUniformBuffer();
        errdefer self.uniform_buffer.deinit(self.vk_ctx);

        try self.descriptor_set.update(self.vk_ctx, &.{
            .{
                .binding = 0, // Explicitly state the binding index
                .info = .{
                    // Use the correct union tag for a uniform buffer
                    .UniformBuffer = .{
                        .buffer = self.uniform_buffer.handle,
                        .offset = 0,
                        .range = @sizeOf(UniformBufferObject),
                    },
                },
            },
        });

        self.render_pass = try RenderPass.init(self.vk_ctx, &self.swapchain);
        errdefer self.render_pass.deinit(self.vk_ctx);

        try self.render_pass.initFrameBuffer(self.vk_ctx, &self.swapchain, self.depth_buffer.view);

        self.pipeline_layout = try PipelineLayout.init(self.vk_ctx, .{
            .pSetLayouts = &[_]c.VkDescriptorSetLayout{self.descriptor_layout.handle},
            .setLayoutCount = 1,
        });
        errdefer self.pipeline_layout.deinit(self.vk_ctx);

        self.pipeline = try Pipeline.init(self.vk_ctx, self.render_pass, self.pipeline_layout, Vertex, vert_shader_bin, frag_shader_bin);
        errdefer self.pipeline.deinit(self.vk_ctx);

        // Initialize GUI at the end
        self.gui_renderer = try gui.GuiRenderer.init(self.vk_ctx, self.render_pass, self.io);
        errdefer self.gui_renderer.deinit();

        self.text_renderer = try text.Text3DRenderer.init(self.vk_ctx, self.render_pass, self.descriptor_layout, self.io);
    }
    // In your App struct
    pub fn initUi(self: *Self) !void {
        const root = &self.main_ui.root;
        const next_id = &self.main_ui.next_id;

        var left_panel = try root.addContainer(
            next_id,
            .{ .x = 0.01, .y = 0.01, .width = 0.22, .height = 0.4 }, // Position and size of the whole panel
            .{ .Vertical = .{ .spacing = 8 } },
            .{ 0, 0, 0, 0 },
        );

        left_panel.data.container.padding = .{ 10, 10, 10, 10 };

        try left_panel.addButton(next_id, .{ .height = 0.15 }, "Add Line", .{ 0.2, 0.2, 0.8, 1 }, .{ 1, 1, 1, 1 }, 12, addLineCallback);
        try left_panel.addButton(next_id, .{ .height = 0.15 }, "Clear Lines", .{ 0.2, 0.2, 0.8, 1 }, .{ 1, 1, 1, 1 }, 12, clearLinesCallback);
        try left_panel.addButton(next_id, .{ .height = 0.15 }, "Toggle Camera", .{ 0.2, 0.8, 0.2, 1 }, .{ 1, 1, 1, 1 }, 12, toggleCameraModeCallback);

        var tf_panel = try left_panel.addContainer(
            next_id,
            .{ .height = 0.15 },
            .{ .Horizontal = .{ .spacing = 5 } },
            .{ 0, 0, 0, 0 },
        );

        try tf_panel.addTextField(next_id, .{ .width = 0.7 }, "grid:", &self.grid_res_buff, .{ 0.1, 0.1, 0.1, 1 }, .{ 1, 1, 1, 1 });
        try tf_panel.addButton(next_id, .{ .width = 0.3 }, "Set", .{ 0.8, 0.2, 0.2, 1 }, .{ 1, 1, 1, 1 }, 12, setGridCallback);

        const tree_panel = try left_panel.addTreeNode(
            next_id,
            .{ .height = 1.0 },
            "node",
            .{ .Vertical = .{ .spacing = 18 } },
            .{ 0.2, 0.3, 0, 0.65 },
            .{ 1, 1, 1, 1 },
        );

        try tree_panel.addButton(&self.main_ui.next_id, .{ .width = 0.09, .height = 0.15 }, "Button A", .{ 0.5, 0.2, 0.2, 1 }, .{ 1, 1, 1, 1 }, 20, null);
        _ = try tree_panel.addPlainText(&self.main_ui.next_id, .{ .width = 0.09, .height = 0.15 }, "Some info text 1", .{ 1, 1, 1, 1 }, 18);
        _ = try tree_panel.addPlainText(&self.main_ui.next_id, .{ .width = 0.09, .height = 0.15 }, "Some info text 2", .{ 1, 1, 1, 1 }, 18);
        _ = try tree_panel.addPlainText(&self.main_ui.next_id, .{ .width = 0.09, .height = 0.15 }, "Some info text 3", .{ 1, 1, 1, 1 }, 18);
        _ = try tree_panel.addPlainText(&self.main_ui.next_id, .{ .width = 0.09, .height = 0.15 }, "Some info text 4", .{ 1, 1, 1, 1 }, 18);

        var right_panel = try root.addContainer(
            next_id,
            .{ .x = 0.78, .y = 0.01, .width = 0.2, .height = 0.15 },
            .{ .Vertical = .{ .spacing = 10 } },
            .{ 0, 0, 0, 0 },
        );
        right_panel.data.container.padding = .{ 10, 10, 10, 10 };

        try right_panel.addSlider(next_id, .{ .height = 0.25 }, 0.0, 1.0, 0.5, .{ 0.4, 0.4, 0.4, 1 }, .{ 0.8, 0.8, 0.8, 1 }, fovSliderCallback);
        try right_panel.addSlider(next_id, .{ .height = 0.25 }, 0.0, 1.0, 0.5, .{ 0.4, 0.4, 0.4, 1 }, .{ 0.8, 0.8, 0.8, 1 }, nearPlaneSliderCallback);

        var coord_panel = try root.addContainer(
            next_id,
            .{ .x = 0.01, .y = 0.85, .width = 0.4, .height = 0.1 },
            .{ .Grid = .{ .columns = 2, .spacing = .{ 3, 1 } } },
            .{ 0, 0, 0, 0 },
        );
        coord_panel.data.container.padding = .{ 10, 10, 10, 10 };

        try coord_panel.addTextField(next_id, .{ .width = 0.33 }, "X:", &self.line_x_buf, .{ 0.1, 0.1, 0.1, 1 }, .{ 1, 1, 1, 1 });
        try coord_panel.addTextField(next_id, .{ .width = 0.33 }, "Y:", &self.line_y_buf, .{ 0.1, 0.1, 0.1, 1 }, .{ 1, 1, 1, 1 });
        try coord_panel.addTextField(next_id, .{ .width = 0.33 }, "Z:", &self.line_z_buf, .{ 0.1, 0.1, 0.1, 1 }, .{ 1, 1, 1, 1 });
    }

    pub fn deinit(self: *Self) void {
        // Wait for device to be idle before cleaning up
        _ = c.vkDeviceWaitIdle(self.vk_ctx.device.handle);

        self.perf.deinit();
        self.main_ui.deinit();
        self.text_scene.deinit();
        self.gui_renderer.deinit();
        self.text_renderer.deinit();

        self.pipeline.deinit(self.vk_ctx);
        self.pipeline_layout.deinit(self.vk_ctx);
        self.render_pass.deinit(self.vk_ctx);
        self.depth_buffer.deinit(self.vk_ctx);
        self.swapchain.deinit(self.vk_ctx);

        self.sync.deinit(self.vk_ctx);
        self.vertex_buffer.deinit(self.vk_ctx);
        self.uniform_buffer.deinit(self.vk_ctx);
        self.descriptor_layout.deinit(self.vk_ctx);
        self.descriptor_pool.deinit(self.vk_ctx);

        self.vk_ctx.deinit();
        self.scene.deinit(self.allocator);
        self.window.deinit();
        self.allocator.destroy(self.vk_ctx);
    }

    fn cleanupSwapchain(self: *Self) void {
        self.gui_renderer.destroyPipeline();
        self.text_renderer.destroyPipeline();
        self.pipeline.deinit(self.vk_ctx);
        self.pipeline_layout.deinit(self.vk_ctx);
        self.render_pass.deinit(self.vk_ctx);
        self.depth_buffer.deinit(self.vk_ctx);
        self.swapchain.deinit(self.vk_ctx);
    }

    pub fn run(self: *Self) !void {
        const next_id = &self.main_ui.next_id;
        const fps_label = try self.main_ui.root.addPlainText(
            next_id,
            .{ .x = 0.01, .y = 0.6, .width = 0.4, .height = 0.05 }, // Positioned in root's Manual layout
            "this string will be overwriten " ** 8,
            .{ 1, 1, 0, 1 },
            12,
        );
        self.perf.setPtr(fps_label); //TODO: change this hack
        while (c.glfwWindowShouldClose(self.window.handle) == 0) {
            self.perf.beginFrame();
            self.wd_ctx.beginFrame();

            c.glfwPollEvents();

            if (self.window.minimized()) {
                c.glfwWaitEvents();
                continue;
            }

            try self.perf.beginScope("GUI");
            self.gui_renderer.beginFrame();

            if (self.gui_renderer.processAndDraw(&self.main_ui, self, self.window.size.x, self.window.size.y) or
                self.wd_ctx.processCameraInput(&self.scene, &self.text_scene))
            {
                try self.updateUniformBuffer();
            }

            self.perf.endScope("GUI");

            try self.perf.beginScope("Text");
            self.text_renderer.beginFrame();

            self.text_renderer.processAndDrawTextScene(&self.text_scene);
            self.perf.endScope("Text");

            // This measures the time it takes to build and submit command buffers and present the frame.
            try self.perf.beginScope("Draw");
            try self.draw();
            self.perf.endScope("Draw");
            self.perf.endFrame();
        }
    }

    fn draw(self: *Self) !void {
        try vkCheck(c.vkWaitForFences(self.vk_ctx.device.handle, 1, &self.sync.in_flight_f, c.VK_TRUE, std.math.maxInt(u64)));

        var image_index: u32 = 0;
        const acquire_result = c.vkAcquireNextImageKHR(self.vk_ctx.device.handle, self.swapchain.handle, std.math.maxInt(u64), self.sync.img_available_s, null, &image_index);

        if (acquire_result == c.VK_ERROR_OUT_OF_DATE_KHR) {
            try self.recreateSwapchain();
            return;
        } else if (acquire_result != c.VK_SUCCESS and acquire_result != c.VK_SUBOPTIMAL_KHR) {
            try vkCheck(acquire_result);
        }
        try vkCheck(c.vkResetFences(self.vk_ctx.device.handle, 1, &self.sync.in_flight_f));
        try vkCheck(c.vkResetCommandBuffer(self.command_buffer.handle, 0));
        try self.recordCommandBuffer(image_index);

        const wait_semaphores = [_]c.VkSemaphore{self.sync.img_available_s};
        const wait_stages = [_]c.VkPipelineStageFlags{c.VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
        const signal_semaphores = [_]c.VkSemaphore{self.sync.render_ended_s};
        const submit_info = c.VkSubmitInfo{
            .sType = c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .waitSemaphoreCount = wait_semaphores.len,
            .pWaitSemaphores = &wait_semaphores,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.command_buffer.handle,
            .signalSemaphoreCount = signal_semaphores.len,
            .pSignalSemaphores = &signal_semaphores,
        };
        try vkCheck(c.vkQueueSubmit(self.vk_ctx.graphics_queue.handle, 1, &submit_info, self.sync.in_flight_f));

        const swapchains = [_]c.VkSwapchainKHR{self.swapchain.handle};
        const present_info = c.VkPresentInfoKHR{
            .sType = c.VK_STRUCTURE_TYPE_PRESENT_INFO_KHR,
            .waitSemaphoreCount = signal_semaphores.len,
            .pWaitSemaphores = &signal_semaphores,
            .swapchainCount = swapchains.len,
            .pSwapchains = &swapchains,
            .pImageIndices = &image_index,
        };
        const present_result = c.vkQueuePresentKHR(self.vk_ctx.graphics_queue.handle, &present_info);

        if (present_result == c.VK_ERROR_OUT_OF_DATE_KHR or present_result == c.VK_SUBOPTIMAL_KHR or self.framebuffer_resized) {
            self.framebuffer_resized = false;
            std.log.info("\n\npresent result {}\n VK_SUBOPTIMAL_KHR:{}\n\n", .{ present_result, c.VK_SUBOPTIMAL_KHR });
            try self.recreateSwapchain();
            try self.updateUniformBuffer();
        } else {
            try vkCheck(present_result);
        }
    }

    pub fn recordCommandBuffer(self: *Self, image_index: u32) !void {
        const begin_info = c.VkCommandBufferBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        };
        try vkCheck(c.vkBeginCommandBuffer(self.command_buffer.handle, &begin_info));

        const clear_values = [_]c.VkClearValue{
            // Color attachment clear value
            .{ .color = .{ .float32 = .{ 0.392, 0.584, 0.929, 1.0 } } }, // Cornflower Blue
            // .{ .color = .{ .float32 = .{ 0, 0, 0, 1.0 } } },
            // Depth attachment clear value
            .{ .depthStencil = .{ .depth = 0.0, .stencil = 0 } },
        };

        const render_pass_info = c.VkRenderPassBeginInfo{
            .sType = c.VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
            .renderPass = self.render_pass.handle,
            .framebuffer = self.render_pass.framebuffer[image_index],
            .renderArea = .{
                .offset = .{ .x = 0, .y = 0 },
                .extent = self.swapchain.extent,
            },
            .clearValueCount = clear_values.len,
            .pClearValues = &clear_values,
        };

        c.vkCmdBeginRenderPass(self.command_buffer.handle, &render_pass_info, c.VK_SUBPASS_CONTENTS_INLINE);
        c.vkCmdBindPipeline(self.command_buffer.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline.handle);

        var viewport = c.VkViewport{
            .height = @floatFromInt(self.window.size.y),
            .width = @floatFromInt(self.window.size.x),
        };

        var scissor = c.VkRect2D{
            .extent = .{
                .height = @intCast(self.window.size.y),
                .width = @intCast(self.window.size.x),
            },
            .offset = .{},
        };

        c.vkCmdSetViewport(self.command_buffer.handle, 0, 1, &viewport);
        c.vkCmdSetScissor(self.command_buffer.handle, 0, 1, &scissor);

        const vertex_buffers = [_]c.VkBuffer{self.vertex_buffer.handle};
        const offsets = [_]c.VkDeviceSize{0};
        c.vkCmdBindVertexBuffers(self.command_buffer.handle, 0, 1, &vertex_buffers, &offsets);
        c.vkCmdBindDescriptorSets(self.command_buffer.handle, c.VK_PIPELINE_BIND_POINT_GRAPHICS, self.pipeline_layout.handle, 0, 1, &self.descriptor_set.handle, 0, null);
        c.vkCmdDraw(self.command_buffer.handle, @intCast(self.scene.getTotalVertexCount()), 1, 0, 0);
        self.text_renderer.endFrame(self.command_buffer.handle, self.descriptor_set);
        self.gui_renderer.endFrame(
            self.command_buffer.handle,
            @floatFromInt(self.window.size.x),
            @floatFromInt(self.window.size.y),
        );
        c.vkCmdEndRenderPass(self.command_buffer.handle);
        try vkCheck(c.vkEndCommandBuffer(self.command_buffer.handle));
    }

    fn initVertexBuffer(self: *Self) !void {
        const buffer_size = @sizeOf(Vertex) * self.scene.getTotalVertexCount();
        if (buffer_size == 0) return;

        var staging_buffer = try Buffer.init(self.vk_ctx, buffer_size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, Vertex);
        defer staging_buffer.unmap(self.vk_ctx);

        const mapped_slice = data_ptr[0..self.scene.getTotalVertexCount()];
        const lines_offset = self.scene.axis.len + self.scene.grid.len;

        @memcpy(mapped_slice[0..self.scene.axis.len], &self.scene.axis);
        @memcpy(mapped_slice[self.scene.axis.len..lines_offset], self.scene.grid);
        @memcpy(mapped_slice[lines_offset .. lines_offset + self.scene.lines.items.len], self.scene.lines.items);
        self.vertex_buffer = try Buffer.init(self.vk_ctx, @sizeOf(Vertex) * 1024 * 1024, c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | c.VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

        try staging_buffer.copyTo(self.vk_ctx, self.vertex_buffer);
    }

    fn updateVertexBuffer(self: *Self) !void {
        const buffer_size = @sizeOf(Vertex) * self.scene.getTotalVertexCount();
        if (buffer_size == 0) return;

        var staging_buffer = try Buffer.init(self.vk_ctx, buffer_size, c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        defer staging_buffer.deinit(self.vk_ctx);

        const data_ptr = try staging_buffer.map(self.vk_ctx, Vertex);
        defer staging_buffer.unmap(self.vk_ctx);

        const mapped_slice = data_ptr[0 .. self.scene.axis.len + self.scene.grid.len + self.scene.lines.items.len];
        const lines_offset = self.scene.axis.len + self.scene.grid.len;
        @memcpy(mapped_slice[0..self.scene.axis.len], &self.scene.axis);
        @memcpy(mapped_slice[self.scene.axis.len..lines_offset], self.scene.grid);
        @memcpy(mapped_slice[lines_offset .. lines_offset + self.scene.lines.items.len], self.scene.lines.items);

        try staging_buffer.copyTo(self.vk_ctx, self.vertex_buffer);
    }

    fn initUniformBuffer(self: *Self) !void {
        const ubo_size = @sizeOf(UniformBufferObject);
        self.uniform_buffer = try Buffer.init(self.vk_ctx, ubo_size, c.VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        try self.updateUniformBuffer();
    }

    fn updateUniformBuffer(self: *Self) !void {
        const aspect_ratio = @as(f32, @floatFromInt(self.window.size.x)) / @as(f32, @floatFromInt(self.window.size.y));
        const ubo = UniformBufferObject{
            .view_matrix = self.scene.camera.view(),
            .perspective_matrix = self.scene.camera.projection(aspect_ratio),
        };

        const data_ptr = try self.uniform_buffer.map(self.vk_ctx, UniformBufferObject);
        defer self.uniform_buffer.unmap(self.vk_ctx);
        data_ptr[0] = ubo;
    }

    fn recreateSwapchain(self: *Self) !void {
        var width: c_int = 0;
        var height: c_int = 0;
        c.glfwGetFramebufferSize(self.window.handle, &width, &height);
        while (width == 0 or height == 0) {
            c.glfwGetFramebufferSize(self.window.handle, &width, &height);
            c.glfwWaitEvents();
        }
        self.window.size.x = width;
        self.window.size.y = height;

        try vkCheck(c.vkQueueWaitIdle(self.vk_ctx.graphics_queue.handle));
        self.cleanupSwapchain();

        self.swapchain = try Swapchain.init(self.vk_ctx);
        self.depth_buffer = try DepthBuffer.init(self.vk_ctx, self.swapchain.extent.width, self.swapchain.extent.height);
        self.render_pass = try RenderPass.init(self.vk_ctx, &self.swapchain);
        try self.render_pass.initFrameBuffer(self.vk_ctx, &self.swapchain, self.depth_buffer.view);
        self.pipeline_layout = try PipelineLayout.init(self.vk_ctx, .{
            .pSetLayouts = &self.descriptor_layout.handle,
            .setLayoutCount = 1,
        });
        self.pipeline = try Pipeline.init(self.vk_ctx, self.render_pass, self.pipeline_layout, Vertex, vert_shader_bin, frag_shader_bin);
        try self.gui_renderer.createPipeline(self.render_pass);
        try self.text_renderer.createPipeline(self.render_pass, self.descriptor_layout);
    }
};

fn glfwErrorCallback(err_code: c_int, description: [*c]const u8) callconv(.c) void {
    std.debug.print("[GLFW error {d}]: {s}\n", .{ err_code, description });
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    _ = c.glfwSetErrorCallback(glfwErrorCallback);
    _ = c.glfwInitVulkanLoader(c.vkGetInstanceProcAddr);

    try checkGlfw(c.glfwInit());
    defer c.glfwTerminate();

    var app = try App.init(allocator, io);
    defer allocator.destroy(app);
    defer app.deinit();

    try app.run();
}
