const std = @import("std");

pub const PerformanceTracker = struct {
    const Self = @This();

    // Configuration
    const SAMPLES: usize = 120; // Number of frames to average over

    // --- Internal struct for scoped timer data ---
    const ScopeData = struct {
        times: [SAMPLES]f32 = .{0} ** SAMPLES,
        avg_ms: f32 = 0.0,
    };

    allocator: std.mem.Allocator,

    // Main frame timers
    last_frame_time: i128,
    frame_times: [SAMPLES]f32 = .{0} ** SAMPLES,
    frame_index: usize = 0,
    frames_recorded: usize = 0,

    // Calculated frame stats
    delta_time_ms: f32 = 0,
    avg_fps: f32 = 0,
    min_fps: f32 = 0,
    max_fps: f32 = 0,
    mapped_string: ?[]u8 = null,

    // --- Data structures for scoped timers ---
    active_scopes: std.StringHashMap(i128), // Stores start times for currently running scopes
    scope_data: std.StringHashMap(ScopeData), // Stores historical data and averages for each scope

    // --- MODIFIED: init now requires an allocator ---
    pub fn init(allocator: std.mem.Allocator) Self {
        return .{
            .allocator = allocator,
            .last_frame_time = std.time.nanoTimestamp(),
            .active_scopes = std.StringHashMap(i128).init(allocator),
            .scope_data = std.StringHashMap(ScopeData).init(allocator),
        };
    }

    // --- deinit is required to free HashMap memory ---
    pub fn deinit(self: *Self) void {
        // Deinit the historical data map
        var it = self.scope_data.iterator();
        while (it.next()) |_| {
            // Keys are string literals, so they don't need to be freed.
            // The value (ScopeData) is a struct, not a pointer, so no free needed.
        }
        self.scope_data.deinit();

        // Active scopes should ideally be empty, but deinit just in case.
        self.active_scopes.deinit();
    }

    pub fn setPtr(self: *Self, ptr: []u8) void {
        self.mapped_string = ptr;
    }

    pub fn beginFrame(self: *Self) void {
        const current_time = std.time.nanoTimestamp();
        const elapsed_nanos = current_time - self.last_frame_time;
        self.last_frame_time = current_time;

        self.delta_time_ms = @as(f32, @floatFromInt(elapsed_nanos)) / 1_000_000.0;
        self.frame_times[self.frame_index] = self.delta_time_ms;
    }

    // --- Start timing a named scope ---
    pub fn beginScope(self: *Self, name: []const u8) void {
        // Put the current time into the map, overwriting any previous entry for safety.
        self.active_scopes.put(name, std.time.nanoTimestamp()) catch |err| {
            std.log.err("Failed to begin scope '{s}': {any}", .{ name, err });
        };
    }

    // --- End timing a named scope and record the duration ---
    pub fn endScope(self: *Self, name: []const u8) void {
        const end_time = std.time.nanoTimestamp();
        const start_time = self.active_scopes.fetchRemove(name) orelse return;

        const duration_ms = @as(f32, @floatFromInt(end_time - start_time.value)) / 1_000_000.0;

        // Get or create an entry for this scope name in our historical data
        var gop = self.scope_data.getOrPut(name) catch |err| {
            std.log.err("Failed to record scope '{s}': {any}", .{ name, err });
            return;
        };

        // If this is the first time we've seen this scope, initialize its data
        if (!gop.found_existing) {
            gop.value_ptr.* = .{};
        }

        // Record the timing in the correct slot for this frame
        gop.value_ptr.times[self.frame_index] = duration_ms;
    }

    // --- MODIFIED: endFrame now calculates scope averages and updates the string ---
    pub fn endFrame(self: *Self) void {
        // --- Part 1: Advance frame counters ---
        self.frame_index = (self.frame_index + 1) % SAMPLES;
        if (self.frames_recorded < SAMPLES) {
            self.frames_recorded += 1;
        }
        if (self.frames_recorded == 0) return;

        // --- Part 2: Calculate FPS stats (same as before) ---
        var total_frame_time_ms: f32 = 0;
        const sample_count = self.frames_recorded;
        for (self.frame_times[0..sample_count]) |frame_time| {
            total_frame_time_ms += frame_time;
        }
        const avg_time_ms = total_frame_time_ms / @as(f32, @floatFromInt(sample_count));
        self.avg_fps = 1000.0 / avg_time_ms;

        // --- Part 3: Calculate averages for all tracked scopes ---
        var it = self.scope_data.iterator();
        while (it.next()) |entry| {
            var total_scope_time: f32 = 0;
            for (entry.value_ptr.times[0..sample_count]) |t| {
                total_scope_time += t;
            }
            entry.value_ptr.avg_ms = total_scope_time / @as(f32, @floatFromInt(sample_count));
        }

        // --- Part 4: Format the output string ---
        if (self.mapped_string) |buffer| {
            var fbs = std.io.fixedBufferStream(buffer);
            const writer = fbs.writer();

            writer.print("fps:{d:.1}\n", .{self.avg_fps}) catch {};

            // Iterate again to print the now-calculated averages
            it = self.scope_data.iterator();
            while (it.next()) |entry| {
                writer.print("{s}:{d:.2}ms\n", .{ entry.key_ptr.*, entry.value_ptr.avg_ms }) catch {};
            }
            writer.print(&.{170}, .{}) catch {};
            // Fill remaining space with null terminators/spaces for cleanup
            const written_len = try fbs.getPos();
            @memset(buffer[written_len..], 0);
        }
    }
};
