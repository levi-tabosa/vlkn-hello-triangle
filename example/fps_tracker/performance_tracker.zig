const std = @import("std");

pub const PerformanceTracker = struct {
    const Self = @This();

    // Configuration
    const SAMPLES: usize = 120; // Number of frames to average over

    // Timers
    last_frame_time: i128,
    frame_times: [SAMPLES]f32 = .{0} ** SAMPLES,
    frame_index: usize = 0,
    frames_recorded: usize = 0,

    // Calculated stats
    delta_time_ms: f32 = 0,
    avg_fps: f32 = 0,
    min_fps: f32 = 0,
    max_fps: f32 = 0,
    mapped_string: ?[]u8 = null,

    pub fn init() Self {
        return .{
            .last_frame_time = std.time.nanoTimestamp(),
        };
    }

    pub fn setPtr(self: *Self, ptr: []u8) void {
        self.mapped_string = ptr;
    }

    pub fn beginFrame(self: *Self) void {
        const current_time = std.time.nanoTimestamp();
        const elapsed_nanos = current_time - self.last_frame_time;
        self.last_frame_time = current_time;

        // Convert nanoseconds to milliseconds
        self.delta_time_ms = @as(f32, @floatFromInt(elapsed_nanos)) / 1_000_000.0;

        // Store the frame time for averaging
        self.frame_times[self.frame_index] = self.delta_time_ms;
        self.frame_index = (self.frame_index + 1) % SAMPLES;

        if (self.frames_recorded < SAMPLES) {
            self.frames_recorded += 1;
        }
    }

    pub fn endFrame(self: *Self) void {
        if (self.frames_recorded == 0) return;

        var total_time_ms: f32 = 0;
        var min_time_ms: f32 = 1000.0;
        var max_time_ms: f32 = 0.0;

        // Calculate stats over the recorded samples
        const sample_count = self.frames_recorded;
        for (self.frame_times[0..sample_count]) |frame_time| {
            total_time_ms += frame_time;
            min_time_ms = @min(min_time_ms, frame_time);
            max_time_ms = @max(max_time_ms, frame_time);
        }

        const avg_time_ms = total_time_ms / @as(f32, @floatFromInt(sample_count));

        // Convert times to FPS
        self.avg_fps = 1000.0 / avg_time_ms;
        self.min_fps = 1000.0 / max_time_ms;
        self.max_fps = 1000.0 / min_time_ms;

        _ = std.fmt.bufPrint(self.mapped_string.?, "fps:{d:.2}", .{self.avg_fps}) catch unreachable;
    }
};
