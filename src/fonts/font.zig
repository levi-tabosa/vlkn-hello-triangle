const std = @import("std");

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

/// Substitui os antigos @embedFile: cada variante conhece seu caminho base
/// em disco (sem extensão) e deriva ".fnt" / ".png" sob demanda.
pub const FontAsset = enum {
    pericles_w01,
    consolas_regular,
    report_regular,
    swanseabold,
    notosanstc_variablefont_wght,
    hermit_light,
    exo2_0_regular,
    helvetica,

    fn basePath(self: FontAsset) []const u8 {
        return switch (self) {
            .pericles_w01 => "src/fonts/assets/pericles-W01-regular-fed68123",
            .consolas_regular => "src/fonts/assets/consolas-regular-fed68123",
            .report_regular => "src/fonts/assets/report-regular-fed68123",
            .swanseabold => "src/fonts/assets/swanseabold-d0ox",
            .notosanstc_variablefont_wght => "src/fonts/assets/notosanstc-variablefont_wght-fed68123",
            .hermit_light => "src/fonts/assets/hermit_light-fed68123",
            .exo2_0_regular => "src/fonts/assets/exo2_0_regular-fed68123",
            .helvetica => "src/fonts/assets/helvetica-fed68123",
        };
    }

    pub fn fntPath(self: FontAsset, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}.fnt", .{self.basePath()}) catch unreachable;
    }

    pub fn pngPath(self: FontAsset, buf: []u8) []const u8 {
        return std.fmt.bufPrint(buf, "{s}.png", .{self.basePath()}) catch unreachable;
    }
};

pub const FontLoader = struct {
    const Self = @This();

    glyphs: std.AutoHashMap(u32, Glyph),
    line_height: f32 = 0,
    base: f32 = 0,
    scale_w: f32 = 0,
    scale_h: f32 = 0,
    font_size: f32 = 0,

    /// Lê o .fnt do `asset` do disco via `io` e faz o parse.
    /// Não carrega o PNG do atlas — use png.loadPngFile(allocator, io, asset.pngPath(buf))
    /// pra isso, como no GuiRenderer.init abaixo.
    pub fn init(allocator: std.mem.Allocator, asset: FontAsset, io: std.Io) !Self {
        var path_buf: [128]u8 = undefined;
        const fnt_path = asset.fntPath(&path_buf);
        std.debug.print("{s}\n", .{fnt_path});

        var file = try std.Io.Dir.cwd().openFile(io, fnt_path, .{ .mode = .read_only });
        defer file.close(io);

        const size = try file.length(io);
        const data = try allocator.alloc(u8, size);
        defer allocator.free(data);

        var reader = file.reader(io, data);
        reader.interface.readSliceAll(data) catch |err| switch (err) {
            error.ReadFailed => return reader.err orelse err,
            else => return err,
        };

        var self = Self{
            .glyphs = std.AutoHashMap(u32, Glyph).init(allocator),
        };
        try self.loadFNT(data);
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.glyphs.deinit();
    }

    fn parseKeyValue(part: []const u8) ?struct { key: []const u8, value: []const u8 } {
        const eq_idx = std.mem.indexOfScalar(u8, part, '=') orelse return null;
        const key = part[0..eq_idx];
        var value = part[eq_idx + 1 ..];
        if (value.len >= 2 and value[0] == '"' and value[value.len - 1] == '"') {
            value = value[1 .. value.len - 1];
        }
        return .{ .key = key, .value = value };
    }

    pub fn loadFNT(self: *Self, data: []const u8) !void {
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
