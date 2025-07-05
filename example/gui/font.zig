// gui/font.zig
const std = @import("std");

const Allocator = std.mem.Allocator;

// MODIFIED: Make sure the types match what the FNT file provides.
// x, y, width, height are integers. offsets and advance are floats.
pub const Glyph = struct {
    id: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    xoffset: f32,
    yoffset: f32,
    xadvance: f32,
};

pub const Font = struct {
    allocator: Allocator,
    glyphs: std.AutoHashMap(u32, Glyph),
    line_height: f32 = 0,
    base: f32 = 0,
    scale_w: f32 = 0,
    scale_h: f32 = 0,

    pub fn init(allocator: Allocator) Font {
        return .{
            .allocator = allocator,
            .glyphs = std.AutoHashMap(u32, Glyph).init(allocator),
        };
    }

    pub fn deinit(self: *Font) void {
        self.glyphs.deinit();
    }

    pub fn loadFNT(self: *Font) !void {
        const file_contents = @import("font").mikado_medium_fed68123_fnt; // Adjust path if needed

        var lines = std.mem.splitScalar(u8, file_contents, '\n');
        while (lines.next()) |line| {
            // Trim whitespace from the line, just in case
            const trimmed_line = std.mem.trim(u8, line, " \r");
            if (trimmed_line.len == 0) continue;

            // --- USE TOKENIZE INSTEAD OF SPLIT ---
            var parts = std.mem.tokenizeScalar(u8, trimmed_line, ' ');
            const tag = parts.next() orelse continue;

            if (std.mem.eql(u8, tag, "common")) {
                // This part is likely fine, but let's make it robust too
                while (parts.next()) |part| {
                    var kv = std.mem.splitScalar(u8, part, '=');
                    const key = kv.next() orelse continue;
                    const value = kv.next() orelse continue; // If key exists, value should too
                    if (std.mem.eql(u8, key, "lineHeight")) self.line_height = try std.fmt.parseFloat(f32, value);
                    if (std.mem.eql(u8, key, "base")) self.base = try std.fmt.parseFloat(f32, value);
                    if (std.mem.eql(u8, key, "scaleW")) self.scale_w = try std.fmt.parseFloat(f32, value);
                    if (std.mem.eql(u8, key, "scaleH")) self.scale_h = try std.fmt.parseFloat(f32, value);
                }
            } else if (std.mem.eql(u8, tag, "char")) {
                var glyph = Glyph{ .id = 0, .x = 0, .y = 0, .width = 0, .height = 0, .xoffset = 0, .yoffset = 0, .xadvance = 0 };

                while (parts.next()) |part| {
                    // This check is now redundant with tokenize, but good practice
                    if (part.len == 0) continue;

                    var kv = std.mem.splitScalar(u8, part, '=');
                    const key = kv.next() orelse continue;
                    // --- CRITICAL FIX: Check that `value` is not null ---
                    const value = kv.next() orelse {
                        std.log.warn("Malformed FNT part: '{s}', skipping.", .{key});
                        continue;
                    };

                    // Using a switch is cleaner here
                    if (std.mem.eql(u8, key, "id")) {
                        glyph.id = try std.fmt.parseInt(u32, value, 10);
                    } else if (std.mem.eql(u8, key, "x")) {
                        glyph.x = try std.fmt.parseInt(u32, value, 10);
                    } else if (std.mem.eql(u8, key, "y")) {
                        glyph.y = try std.fmt.parseInt(u32, value, 10);
                    } else if (std.mem.eql(u8, key, "width")) {
                        glyph.width = try std.fmt.parseInt(u32, value, 10);
                    } else if (std.mem.eql(u8, key, "height")) {
                        glyph.height = try std.fmt.parseInt(u32, value, 10);
                    } else if (std.mem.eql(u8, key, "xoffset")) {
                        glyph.xoffset = try std.fmt.parseFloat(f32, value);
                    } else if (std.mem.eql(u8, key, "yoffset")) {
                        glyph.yoffset = try std.fmt.parseFloat(f32, value);
                    } else if (std.mem.eql(u8, key, "xadvance")) {
                        glyph.xadvance = try std.fmt.parseFloat(f32, value);
                    }
                    // Ignore other fields like `page` and `chnl`
                }
                // Only try to put if the ID is valid
                if (glyph.id != 0) {
                    try self.glyphs.put(glyph.id, glyph);
                }
            }
        }
    }
};
