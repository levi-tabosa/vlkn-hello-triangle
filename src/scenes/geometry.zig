const std = @import("std");
const math = std.math;
const assert = std.debug.assert;

pub fn mat4Mul(a: [16]f32, b: [16]f32) [16]f32 {
    // Unrolled 4x4 matrix multiplication
    return .{
        a[0] * b[0] + a[1] * b[4] + a[2] * b[8] + a[3] * b[12],
        a[0] * b[1] + a[1] * b[5] + a[2] * b[9] + a[3] * b[13],
        a[0] * b[2] + a[1] * b[6] + a[2] * b[10] + a[3] * b[14],
        a[0] * b[3] + a[1] * b[7] + a[2] * b[11] + a[3] * b[15],

        a[4] * b[0] + a[5] * b[4] + a[6] * b[8] + a[7] * b[12],
        a[4] * b[1] + a[5] * b[5] + a[6] * b[9] + a[7] * b[13],
        a[4] * b[2] + a[5] * b[6] + a[6] * b[10] + a[7] * b[14],
        a[4] * b[3] + a[5] * b[7] + a[6] * b[11] + a[7] * b[15],

        a[8] * b[0] + a[9] * b[4] + a[10] * b[8] + a[11] * b[12],
        a[8] * b[1] + a[9] * b[5] + a[10] * b[9] + a[11] * b[13],
        a[8] * b[2] + a[9] * b[6] + a[10] * b[10] + a[11] * b[14],
        a[8] * b[3] + a[9] * b[7] + a[10] * b[11] + a[11] * b[15],

        a[12] * b[0] + a[13] * b[4] + a[14] * b[8] + a[15] * b[12],
        a[12] * b[1] + a[13] * b[5] + a[14] * b[9] + a[15] * b[13],
        a[12] * b[2] + a[13] * b[6] + a[14] * b[10] + a[15] * b[14],
        a[12] * b[3] + a[13] * b[7] + a[14] * b[11] + a[15] * b[15],
    };
}

pub const Quat = struct {
    x: f32,
    y: f32,
    z: f32,
    w: f32,

    pub fn identity() Quat {
        return .{ .x = 0, .y = 0, .z = 0, .w = 1 };
    }

    pub fn fromAxisAngle(axis: @Vector(3, f32), angle: f32) Quat {
        const half = angle * 0.5;
        const s = std.math.sin(half);
        return .{
            .x = axis[0] * s,
            .y = axis[1] * s,
            .z = axis[2] * s,
            .w = std.math.cos(half),
        };
    }

    pub fn mul(a: Quat, b: Quat) Quat {
        return .{
            .w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
            .x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
            .y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            .z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        };
    }

    pub fn normalize(q: Quat) Quat {
        const len = std.math.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
        return .{
            .x = q.x / len,
            .y = q.y / len,
            .z = q.z / len,
            .w = q.w / len,
        };
    }

    pub fn toMat4(q: Quat) [16]f32 {
        const x2 = q.x + q.x;
        const y2 = q.y + q.y;
        const z2 = q.z + q.z;
        const xx = q.x * x2;
        const yy = q.y * y2;
        const zz = q.z * z2;
        const xy = q.x * y2;
        const xz = q.x * z2;
        const yz = q.y * z2;
        const wx = q.w * x2;
        const wy = q.w * y2;
        const wz = q.w * z2;

        return .{
            1.0 - (yy + zz), xy + wz,         xz - wy,         0.0,
            xy - wz,         1.0 - (xx + zz), yz + wx,         0.0,
            xz + wy,         yz - wx,         1.0 - (xx + yy), 0.0,
            0.0,             0.0,             0.0,             1.0,
        };
    }
};

pub const Transform = struct {
    position: @Vector(3, f32) = .{ 0, 0, 0 },
    rotation: Quat = Quat.identity(),
    scale: @Vector(3, f32) = .{ 1, 1, 1 },

    pub fn new(args: anytype) Transform {
        var t = Transform{};
        if (@hasField(@TypeOf(args), "position")) t.position = args.position;
        if (@hasField(@TypeOf(args), "rotation")) t.rotation = args.rotation;
        if (@hasField(@TypeOf(args), "scale")) t.scale = args.scale;
        return t;
    }

    pub fn toMatrix(self: Transform) [16]f32 {
        // 1. Start with an identity matrix
        const scale_mat: [16]f32 = .{
            self.scale[0], 0,             0,             0,
            0,             self.scale[1], 0,             0,
            0,             0,             self.scale[2], 0,
            0,             0,             0,             1,
        };

        // 2. Get the rotation matrix
        const rot_mat = Quat.toMat4(self.rotation);

        // 3. Create the translation matrix
        const trans_mat: [16]f32 = .{
            1,                0,                0,                0,
            0,                1,                0,                0,
            0,                0,                1,                0,
            self.position[0], self.position[1], self.position[2], 1,
        };

        // 4. Combine them in the correct order: Translate * Rotate * Scale
        // Note: Matrix multiplication is not commutative! The order is crucial.
        return mat4Mul(trans_mat, mat4Mul(rot_mat, scale_mat));
    }
};
pub const V3 = extern struct {
    coor: @Vector(3, f32),
    offset: @Vector(3, f32) = .{ 0, 0, 0 }, // This is an optional offset vector, can be used for transformations.
    color: @Vector(4, f32) = .{ 1, 1, 1, 1 }, // Default color is white.

    pub fn add(a: V3, b: V3) V3 {
        return .{
            .coor = a.coor + b.coor,
        };
    }

    pub fn subtract(a: V3, b: V3) V3 {
        return .{
            .coor = a.coor - b.coor,
        };
    }

    pub fn dot(a: V3, b: V3) f32 {
        return a.coor[0] * b.coor[0] + a.coor[1] * b.coor[1] + a.coor[2] * b.coor[2];
    }

    pub fn cross(a: V3, b: V3) V3 {
        return .{ .coor = .{
            a.coor[1] * b.coor[2] - a.coor[2] * b.coor[1],
            a.coor[2] * b.coor[0] - a.coor[0] * b.coor[2],
            a.coor[0] * b.coor[1] - a.coor[1] * b.coor[0],
        } };
    }

    pub fn len(v: V3) f32 {
        return std.math.sqrt(
            v.coor[0] * v.coor[0] + v.coor[1] * v.coor[1] + v.coor[2] * v.coor[2],
        );
    }

    pub fn normalize(v: V3) V3 {
        const l = [_]f32{v.len()} ** 3;
        return .{
            .coor = v.coor / l,
        };
    }
};

// TODO: move scene logic to scene file
pub const Scene = struct {
    const Self = @This();

    allocator: std.mem.Allocator,
    camera: Camera,
    axis: [6]V3,
    grid: []V3,
    lines: std.ArrayList(V3),

    pub fn init(allocator: std.mem.Allocator, resolution: u32) !Self {
        return .{
            .allocator = allocator,
            .axis = createAxis(resolution),
            .grid = try createGrid(allocator, resolution),
            .lines = std.ArrayList(V3).init(allocator),
            .camera = Camera.init(20.0),
        };
    }

    pub fn deinit(self: *Self, allocator: std.mem.Allocator) void {
        allocator.free(self.grid);
        self.lines.deinit();
    }

    pub fn clear(self: *Self) void {
        self.lines.clearRetainingCapacity();
    }

    pub fn setGridResolution(self: *Self, new_resolution: u32) !void {
        self.allocator.free(self.grid);
        self.grid = try createGrid(self.allocator, new_resolution);
        self.axis = createAxis(new_resolution);
    }

    fn createAxis(resolution: u32) [6]V3 {
        const r: f32 = @floatFromInt(resolution);
        return .{
            .{ .coor = .{ -r, 0.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } }, .{ .coor = .{ r, 0.0, 0.0 }, .color = .{ 1.0, 0.0, 0.0, 1.0 } },
            .{ .coor = .{ 0.0, -r, 0.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } }, .{ .coor = .{ 0.0, r, 0.0 }, .color = .{ 0.0, 1.0, 0.0, 1.0 } },
            .{ .coor = .{ 0.0, 0.0, -r }, .color = .{ 0.0, 0.0, 1.0, 1.0 } }, .{ .coor = .{ 0.0, 0.0, r }, .color = .{ 0.0, 0.0, 1.0, 1.0 } },
        };
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
            grid[index] = V3{ .coor = .{ idx, fixed, 0.0 }, .color = .{ 0.2, 0.2, 0.2, 1.0 } };
            grid[index + 1] = V3{ .coor = .{ idx, -fixed, 0.0 }, .color = .{ 0.2, 0.2, 0.2, 1.0 } };
            grid[index + 2] = V3{ .coor = .{ fixed, idx, 0.0 }, .color = .{ 0.2, 0.2, 0.2, 1.0 } };
            grid[index + 3] = V3{ .coor = .{ -fixed, idx, 0.0 }, .color = .{ 0.2, 0.2, 0.2, 1.0 } };
        }

        return grid;
    }

    pub fn addLine(self: *Self, start: [3]f32, end: [3]f32) !void {
        try self.lines.append(.{ .coor = start, .color = .{ 1.0, 0.0, 1.0, 1.0 } });
        try self.lines.append(.{ .coor = end, .color = .{ 1.0, 0.0, 1.0, 1.0 } });
    }

    pub fn getTotalVertexCount(self: Self) usize {
        return self.axis.len + self.grid.len + self.lines.items.len;
    }
};

const Camera = struct {
    const Self = @This();

    pub const Mode = enum { orbit, fps };

    pos: V3,
    target: V3 = .{ .coor = .{ 0, 0, 0 } },
    up: V3 = .{ .coor = .{ 0, 0, 1 } }, // Z is up
    pitch: f32 = 0.5,
    yaw: f32 = 0.2,
    fov_degrees: f32 = 75.0,
    near_plane: f32 = 0.1,
    mode: Mode = .orbit,
    radius: f32 = 20.0,
    shape: [8]V3,

    pub fn init(radius: f32) Self {
        // This value is set based on the near value of the perspective matrix
        // Small value intended to clip the camera lines
        const half_edge = 0.05;
        const dummy = V3{ .coor = .{ 0, 0, 0 } };
        const cube: [8]V3 = .{
            .{ .coor = .{ dummy.coor[0] - half_edge, dummy.coor[1] - half_edge, dummy.coor[2] - half_edge } },
            .{ .coor = .{ dummy.coor[0] - half_edge, dummy.coor[1] - half_edge, dummy.coor[2] + half_edge } },
            .{ .coor = .{ dummy.coor[0] + half_edge, dummy.coor[1] - half_edge, dummy.coor[2] + half_edge } },
            .{ .coor = .{ dummy.coor[0] + half_edge, dummy.coor[1] - half_edge, dummy.coor[2] - half_edge } },
            .{ .coor = .{ dummy.coor[0] - half_edge, dummy.coor[1] + half_edge, dummy.coor[2] - half_edge } },
            .{ .coor = .{ dummy.coor[0] - half_edge, dummy.coor[1] + half_edge, dummy.coor[2] + half_edge } },
            .{ .coor = .{ dummy.coor[0] + half_edge, dummy.coor[1] + half_edge, dummy.coor[2] + half_edge } },
            .{ .coor = .{ dummy.coor[0] + half_edge, dummy.coor[1] + half_edge, dummy.coor[2] - half_edge } },
        };
        return .{
            .pos = dummy,
            .radius = radius,
            .shape = cube,
        };
    }

    pub fn toggleMode(self: *Self) void {
        switch (self.mode) {
            .orbit => {
                // Switching from Orbit to FPS. The current 'pos' from the last view
                // call is a perfect starting point for FPS mode.
                self.mode = .fps;
            },
            .fps => {
                // Switching from FPS to Orbit. We need to calculate the spherical
                // coordinates (radius, pitch, yaw) from the camera's current
                // cartesian position to ensure a seamless transition.
                self.radius = self.pos.len();
                // Avoid division by zero if camera is at the origin
                if (self.radius < 0.001) {
                    self.radius = 20.0; // Reset to a default radius
                    self.pitch = 0.5;
                    self.yaw = 0.2;
                } else {
                    self.pitch = std.math.asin(self.pos.coor[2] / self.radius);
                    self.yaw = std.math.atan2(self.pos.coor[1], self.pos.coor[0]);
                }
                self.mode = .orbit;
            },
        }
    }

    pub fn adjustPitchYaw(self: *Self, pitch_delta: f32, yaw_delta: f32) void {
        self.pitch += pitch_delta;
        self.yaw += yaw_delta;

        // Clamp pitch to avoid flipping upside down
        const limit = math.pi / 2.0 - 0.01;
        self.pitch = math.clamp(self.pitch, -limit, limit);
    }

    pub fn adjustFov(self: *Self, fov_delta_degrees: f32) void {
        self.fov_degrees -= fov_delta_degrees; // Invert so moving mouse up zooms in
        self.fov_degrees = math.clamp(self.fov_degrees, 15.0, 120.0);
    }

    pub fn adjustRadius(self: *Self, radius_delta: f32) void {
        if (self.mode == .orbit) {
            self.radius += radius_delta;
            self.radius = math.clamp(self.radius, 5.0, 100000.0);
        }
    }

    pub fn view(self: *Self) [16]f32 {
        switch (self.mode) {
            .orbit => {
                // In orbit mode, calculate position from spherical coordinates.
                // The target is always the origin.
                self.pos = .{ .coor = .{
                    self.radius * @cos(self.yaw) * @cos(self.pitch),
                    self.radius * @sin(self.yaw) * @cos(self.pitch),
                    self.radius * @sin(self.pitch),
                } };
                self.target = .{ .coor = .{ 0, 0, 0 } };
            },
            .fps => {
                // In FPS mode, 'pos' is the authority. Calculate a 'target'
                // point in front of it based on the current orientation.
                self.target = V3.add(
                    self.pos,
                    .{
                        .coor = .{
                            @cos(-self.yaw) * @cos(self.pitch),
                            @sin(-self.yaw) * @cos(self.pitch),
                            -@sin(self.pitch), // Correct up-down axis by inverting the Z
                        },
                    },
                );
            },
        }

        // LookAt logic
        const z_axis = V3.normalize(V3.subtract(self.pos, self.target));
        const x_axis = V3.normalize(V3.cross(self.up, z_axis));
        const y_axis = V3.cross(z_axis, x_axis);

        return .{
            x_axis.coor[0],            y_axis.coor[0],            z_axis.coor[0],            0.0,
            x_axis.coor[1],            y_axis.coor[1],            z_axis.coor[1],            0.0,
            x_axis.coor[2],            y_axis.coor[2],            z_axis.coor[2],            0.0,
            -V3.dot(x_axis, self.pos), -V3.dot(y_axis, self.pos), -V3.dot(z_axis, self.pos), 1.0,
        };
    }

    pub fn projection(self: Self, aspect_ratio: f32) [16]f32 {
        const fovy = std.math.degreesToRadians(self.fov_degrees);
        const f = 1.0 / std.math.tan(fovy / 2.0);
        const near = self.near_plane; // Use the dynamic near plane

        // Reversed-Z projection matrix
        return .{
            f / aspect_ratio, 0.0, 0.0, 0.0,
            0.0, -f, 0.0, 0.0, // Flip Y for Vulkan
            0.0, 0.0, 0.0, -1.0, // Maps far plane to depth 0
            0.0, 0.0, near, 0.0, // Maps near plane to depth 1
        };
    }
};
