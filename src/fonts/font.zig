const std = @import("std");

pub const periclesW01_fnt = @embedFile("./assets/pericles-W01-regular-fed68123.fnt");
pub const periclesW01_png = @embedFile("./assets/pericles-W01-regular-fed68123.png");
pub const consolas_regular_fnt = @embedFile("./assets/consolas-regular-fed68123.fnt");
pub const consolas_regular_png = @embedFile("./assets/consolas-regular-fed68123.png");
// pub const inter_24pt_bold_fnt = @embedFile("./assets/inter_24pt-bold-6d0e72dd.fnt");
// pub const inter_24pt_bold_png = @embedFile("./assets/inter_24pt-bold-6d0e72dd.png");
pub const report_regular_fnt = @embedFile("./assets/report-regular-fed68123.fnt");
pub const report_regular_png = @embedFile("./assets/report-regular-fed68123.png");
pub const swanseabold_fnt = @embedFile("./assets/swanseabold-d0ox.fnt");
pub const swanseabold_png = @embedFile("./assets/swanseabold-d0ox.png");
pub const notosanstc_variablefont_wght_fnt = @embedFile("./assets/notosanstc-variablefont_wght-fed68123.fnt");
pub const notosanstc_variablefont_wght_png = @embedFile("./assets/notosanstc-variablefont_wght-fed68123.png");
pub const hermit_light_fnt = @embedFile("./assets/hermit_light-fed68123.fnt");
pub const hermit_light_png = @embedFile("./assets/hermit_light-fed68123.png");
pub const exo2_0_regular_fnt = @embedFile("./assets/exo2_0_regular-fed68123.fnt");
pub const exo2_0_regular_png = @embedFile("./assets/exo2_0_regular-fed68123.png");
pub const helvetica_fnt = @embedFile("./assets/helvetica-fed68123.fnt");
pub const helvetica_png = @embedFile("./assets/helvetica-fed68123.png");
// pub const _fnt = @embedFile("./assets/swanseabold-d0ox.fnt");
// pub const _png = @embedFile("./assets/swanseabold-d0ox.png");

// Font glyph structure
pub const Glyph = struct {
    id: u32 = 0,
    x: u32 = 0,
    y: u32 = 0,
    width: u32 = 0,
    height: u32 = 0,
    xoffset: f32 = 0,
    yoffset: f32 = 0,
    xadvance: f32 = 0,
};

// Font data and glyph management
pub const Font = struct {
    glyphs: std.AutoHashMap(u32, Glyph),
    line_height: f32 = 0,
    base: f32 = 0,
    scale_w: f32 = 0,
    scale_h: f32 = 0,
    font_size: f32 = 0,

    const Entry = struct { key: []const u8, value: []const u8 };

    pub fn init(allocator: std.mem.Allocator) Font {
        return .{
            .glyphs = std.AutoHashMap(u32, Glyph).init(allocator),
        };
    }

    pub fn deinit(self: *Font) void {
        self.glyphs.deinit();
    }

    fn parseKeyValue(part: []const u8) ?Entry {
        const eq_idx = std.mem.indexOfScalar(u8, part, '=') orelse return null;
        const key = part[0..eq_idx];
        var value = part[eq_idx + 1 ..];
        if (value.len >= 2 and value[0] == '"' and value[value.len - 1] == '"') {
            value = value[1 .. value.len - 1];
        }
        return .{ .key = key, .value = value };
    }

    pub fn loadFNT(self: *Font, data: []const u8) !void {
        var lines = std.mem.splitScalar(u8, data, '\n');
        while (lines.next()) |line| {
            const trimmed_line = std.mem.trim(u8, line, " \r");
            if (trimmed_line.len == 0) continue;

            var parts = std.mem.tokenizeScalar(u8, trimmed_line, ' ');
            const tag = parts.next() orelse continue;

            if (std.mem.eql(u8, tag, "info")) {
                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse continue;
                    if (std.mem.eql(u8, kv.key, "size")) {
                        self.font_size = try std.fmt.parseFloat(f32, kv.value);
                    }
                }
            } else if (std.mem.eql(u8, tag, "common")) {
                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse continue;
                    if (std.mem.eql(u8, kv.key, "lineHeight")) self.line_height = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "base")) self.base = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "scaleW")) self.scale_w = try std.fmt.parseFloat(f32, kv.value);
                    if (std.mem.eql(u8, kv.key, "scaleH")) self.scale_h = try std.fmt.parseFloat(f32, kv.value);
                }
            } else if (std.mem.eql(u8, tag, "char")) {
                var glyph = Glyph{};
                var id_parsed = false;

                while (parts.next()) |part| {
                    const kv = parseKeyValue(part) orelse continue;
                    if (std.mem.eql(u8, kv.key, "id")) {
                        glyph.id = try std.fmt.parseInt(u32, kv.value, 10);
                        id_parsed = true;
                    } else if (std.mem.eql(u8, kv.key, "x")) {
                        glyph.x = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "y")) {
                        glyph.y = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "width")) {
                        glyph.width = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "height")) {
                        glyph.height = try std.fmt.parseInt(u32, kv.value, 10);
                    } else if (std.mem.eql(u8, kv.key, "xoffset")) {
                        glyph.xoffset = try std.fmt.parseFloat(f32, kv.value);
                    } else if (std.mem.eql(u8, kv.key, "yoffset")) {
                        glyph.yoffset = try std.fmt.parseFloat(f32, kv.value);
                    } else if (std.mem.eql(u8, kv.key, "xadvance")) {
                        glyph.xadvance = try std.fmt.parseFloat(f32, kv.value);
                    }
                }

                if (id_parsed) {
                    try self.glyphs.put(glyph.id, glyph);
                }
            }
        }
    }
};
