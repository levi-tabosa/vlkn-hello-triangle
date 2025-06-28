const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

pub const V3 = extern struct {
    pos: @Vector(3, f32),
    offset: @Vector(3, f32) = .{ 0, 0, 0 }, // This is an optional offset vector, can be used for transformations.
    color: @Vector(4, f32) = .{ 1, 1, 1, 1 }, // Default color is white.

    pub fn add(a: V3, b: V3) V3 {
        return .{
            .pos = a.pos + b.pos,
        };
    }

    pub fn subtract(a: V3, b: V3) V3 {
        return .{
            .pos = a.pos - b.pos,
        };
    }

    pub fn dot(a: V3, b: V3) f32 {
        return a.pos[0] * b.pos[0] + a.pos[1] * b.pos[1] + a.pos[2] * b.pos[2];
    }

    pub fn cross(a: V3, b: V3) V3 {
        return .{ .pos = .{
            a.pos[1] * b.pos[2] - a.pos[2] * b.pos[1],
            a.pos[2] * b.pos[0] - a.pos[0] * b.pos[2],
            a.pos[0] * b.pos[1] - a.pos[1] * b.pos[0],
        } };
    }

    pub fn normalize(v: V3) V3 {
        const l = [_]f32{std.math.sqrt(
            v.pos[0] * v.pos[0] + v.pos[1] * v.pos[1] + v.pos[2] * v.pos[2],
        )} ** 3;
        return .{
            .pos = v.pos / l,
        };
    }
};

pub const Scene = struct {
    const Self = @This();

    pitch: f32 = 0.5,
    yaw: f32 = 0.2,
    view_matrix: [16]f32,
    camera: Camera,
    axis: [6]V3,
    grid: []V3,

    pub fn init(allocator: std.mem.Allocator, resolution: u32) !Self {
        const res_float: f32 = @floatFromInt(resolution / 2);
        var camera = Camera.init(
            .{ .pos = .{ res_float, res_float, res_float } },
            res_float,
        );
        return .{
            .axis = .{
                // X-axis (Red)
                .{ .pos = .{ -res_float, 0.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } }, .{ .pos = .{ res_float, 0.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } },
                // Y-axis (Green)
                .{ .pos = .{ 0.0, -res_float, 0.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } }, .{ .pos = .{ 0.0, res_float, 0.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } },
                // Z-axis (Blue)
                .{ .pos = .{ 0.0, 0.0, -res_float }, .color = .{ 0.0, 0.0, 1.0, 1.0 } }, .{ .pos = .{ 0.0, 0.0, res_float }, .color = .{ 0.0, 0.0, 1.0, 1.0 } },
            },
            .grid = try createGrid(allocator, resolution),
            .camera = camera,
            .view_matrix = camera.viewMatrix(),
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.grid);
    }

    pub fn setPitchYaw(self: *Self, pitch: f32, yaw: f32) void {
        self.pitch = pitch;
        self.yaw = yaw;
        self.updateViewMatrix();
    }

    pub fn updateViewMatrix(self: *Self) void {
        if (self.camera.radius) |r| {
            self.camera.pos = .{
                .pos = .{
                    r * @cos(self.yaw) * @cos(self.pitch),
                    r * @sin(self.yaw) * @cos(self.pitch),
                    r * @sin(self.pitch),
                },
            };
        } else {
            self.camera.target = V3.add(self.camera.pos, V3{ .pos = .{
                @cos(-self.yaw) * @cos(self.pitch),
                @sin(-self.yaw) * @cos(self.pitch),
                @sin(self.pitch),
            } });
        }

        self.view_matrix = self.camera.viewMatrix();
    }

    fn createGrid(allocator: std.mem.Allocator, resolution: u32) ![]V3 {
        const j: i32 = @intCast(resolution);
        const upperLimit = j;
        var i: i32 = -j;
        var grid = allocator.alloc(V3, resolution * 8) catch unreachable;
        const fixed: f32 = @floatFromInt(j);

        while (i < upperLimit) : (i += 1) {
            const idx: f32 = @as(f32, @floatFromInt(i));
            const index = @as(usize, @intCast((i + j) * 4));
            grid[index] = V3{ .pos = .{ idx, fixed, 0.0 }, .color = .{ idx, fixed, 0.0, 1.0 } };
            grid[index + 1] = V3{ .pos = .{ idx, -fixed, 0.0 }, .color = .{ idx, -fixed, 0.0, 1.0 } };
            grid[index + 2] = V3{ .pos = .{ fixed, idx, 0.0 }, .color = .{ fixed, idx, 0.0, 1.0 } };
            grid[index + 3] = V3{ .pos = .{ -fixed, idx, 0.0 }, .color = .{ -fixed, idx, 0.0, 1.0 } };
        }

        return grid;
    }
};

const Camera = struct {
    const Self = @This();

    pos: V3,
    target: V3 = .{ .pos = .{ 0, 0, 0 } },
    up: V3 = .{ .pos = .{ 0, 0, 1 } }, // Z is up
    radius: ?f32 = null,
    shape: [8]V3,

    pub fn init(pos: V3, radius: ?f32) Self {
        // This value is set based on the near value of the perspective matrix
        // Small value intended to clip the camera lines
        const half_edge_len = 0.05;
        const cube: [8]V3 = .{
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] - half_edge_len } },
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] - half_edge_len, pos.pos[2] - half_edge_len } },
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] - half_edge_len } },
            .{ .pos = .{ pos.pos[0] - half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] + half_edge_len } },
            .{ .pos = .{ pos.pos[0] + half_edge_len, pos.pos[1] + half_edge_len, pos.pos[2] - half_edge_len } },
        };
        return .{
            .pos = pos,
            .radius = radius,
            .shape = cube,
        };
    }

    pub fn viewMatrix(self: Self) [16]f32 {
        const z_axis = V3.normalize(V3.subtract(self.pos, self.target));
        const x_axis = V3.normalize(V3.cross(self.up, z_axis));
        const y_axis = V3.cross(z_axis, x_axis);

        return .{
            x_axis.pos[0],             y_axis.pos[0],             z_axis.pos[0],             0.0,
            x_axis.pos[1],             y_axis.pos[1],             z_axis.pos[1],             0.0,
            x_axis.pos[2],             y_axis.pos[2],             z_axis.pos[2],             0.0,
            -V3.dot(x_axis, self.pos), -V3.dot(y_axis, self.pos), -V3.dot(z_axis, self.pos), 1.0,
        };
    }
};
