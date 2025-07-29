const std = @import("std");

// paethPredictor is required for PNG filter type 4.
fn paethPredictor(a: i64, b: i64, C: i64) i64 {
    const p = a + b - C;
    const pa = @abs(p - a);
    const pb = @abs(p - b);
    const pc = @abs(p - C);
    if (pa <= pb and pa <= pc) return a;
    if (pb <= pc) return b;
    return C;
}

pub const PngParseError = error{
    InvalidSignature,
    ChunkTooShort,
    InvalidIhdr,
    UnsupportedFormat,
    MissingIhdr,
    MissingIdat,
    DecompressionError,
    InvalidFilterType,
    UnsupportedInterlace,
};

pub const PngImage = struct {
    width: u32,
    height: u32,
    bit_depth: u8,
    channels: u8, // 1: Gray, 3: RGB, 4: RGBA
    pixels: []u8,
};

// A helper function to unfilter a single scanline. This is used for both
// non-interlaced and each pass of interlaced images.
fn unfilterScanline(
    output_scanline: []u8,
    filtered_scanline: []const u8,
    prior_scanline: []const u8,
    bytes_per_pixel: u8,
    filter_type: u8,
) PngParseError!void {
    const scanline_len = filtered_scanline.len;
    if (output_scanline.len != scanline_len) @panic("Mismatched scanline lengths");

    for (0..scanline_len) |x| {
        const filt_x = filtered_scanline[x];
        const recon_a = if (x >= bytes_per_pixel) output_scanline[x - bytes_per_pixel] else 0;
        const recon_b = if (prior_scanline.len > 0) prior_scanline[x] else 0;
        const recon_c = if (prior_scanline.len > 0 and x >= bytes_per_pixel) prior_scanline[x - bytes_per_pixel] else 0;

        const predictor = switch (filter_type) {
            0 => 0, // None
            1 => recon_a, // Sub
            2 => recon_b, // Up
            3 => @as(u8, @intCast((@as(u16, recon_a) + @as(u16, recon_b)) / 2)), // Average
            4 => @as(u8, @intCast(paethPredictor(@as(i64, recon_a), @as(i64, recon_b), @as(i64, recon_c)))), // Paeth
            else => return PngParseError.InvalidFilterType,
        };
        output_scanline[x] = @addWithOverflow(filt_x, predictor)[0];
    }
}

// TODO: make return error set
pub fn loadPng(allocator: std.mem.Allocator, png_data: []const u8) !PngImage {
    const png_signature = [_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
    if (png_data.len < 8 or !std.mem.eql(u8, png_data[0..8], &png_signature)) {
        return PngParseError.InvalidSignature;
    }

    var ihdr: ?struct {
        width: u32,
        height: u32,
        bit_depth: u8,
        channels: u8,
        interlace_method: u8,
    } = null;
    var idat_stream = std.ArrayList(u8).init(allocator);
    defer idat_stream.deinit();

    var cursor: usize = 8;
    while (cursor < png_data.len) {
        if (cursor + 8 > png_data.len) return PngParseError.ChunkTooShort;
        const data_len = std.mem.readInt(u32, @ptrCast(&png_data[cursor]), .big);
        const chunk_type = png_data[cursor + 4 .. cursor + 8];
        cursor += 8;
        if (cursor + data_len + 4 > png_data.len) return PngParseError.ChunkTooShort;
        const chunk_data = png_data[cursor .. cursor + data_len];

        if (std.mem.eql(u8, chunk_type, "IHDR")) {
            if (data_len != 13) return PngParseError.InvalidIhdr;
            const width = std.mem.readInt(u32, @ptrCast(&chunk_data[0]), .big);
            const height = std.mem.readInt(u32, @ptrCast(&chunk_data[4]), .big);
            const bit_depth = chunk_data[8];
            const color_type = chunk_data[9];
            const interlace_method = chunk_data[12];

            // For now, only 8-bit depth is supported by the unified unfilter function.
            if (bit_depth != 8) {
                std.log.warn("Unsupported bit depth supported by unified unfilter function: {d}", .{bit_depth});
                // return error.UnsupportedFormat;
            }

            const channels: u8 = switch (color_type) {
                0 => 1, // Grayscale
                2 => 3, // Truecolor (RGB)
                6 => 4, // Truecolor with alpha (RGBA)
                else => return PngParseError.UnsupportedFormat,
            };

            // Check for valid interlace method
            if (interlace_method > 1) return PngParseError.UnsupportedInterlace;

            ihdr = .{
                .width = width,
                .height = height,
                .bit_depth = bit_depth,
                .channels = channels,
                .interlace_method = interlace_method,
            };
        } else if (std.mem.eql(u8, chunk_type, "IDAT")) {
            if (ihdr == null) return PngParseError.MissingIhdr;
            try idat_stream.appendSlice(chunk_data);
        } else if (std.mem.eql(u8, chunk_type, "IEND")) {
            break;
        }
        cursor += data_len + 4; // Skip CRC
    }

    const header = ihdr orelse return PngParseError.MissingIhdr;
    if (idat_stream.items.len == 0) return PngParseError.MissingIdat;
    if (header.width == 0 or header.height == 0) {
        return PngImage{ .width = 0, .height = 0, .bit_depth = header.bit_depth, .channels = header.channels, .pixels = &[_]u8{} };
    }

    var decompressed_buffer = std.ArrayList(u8).init(allocator);
    defer decompressed_buffer.deinit();
    {
        var compressed_reader = std.io.fixedBufferStream(idat_stream.items);
        var decompressor = std.compress.zlib.decompressor(compressed_reader.reader());
        try decompressor.reader().readAllArrayList(&decompressed_buffer, std.math.maxInt(usize));
    }

    const bytes_per_pixel = header.channels * (header.bit_depth / 8);
    const final_pixels = try allocator.alloc(u8, header.width * header.height * bytes_per_pixel);

    if (header.interlace_method == 0) {
        // --- NON-INTERLACED PATH ---
        const scanline_length = header.width * bytes_per_pixel;
        const expected_filtered_size = header.height * (1 + scanline_length);
        if (decompressed_buffer.items.len != expected_filtered_size) return PngParseError.DecompressionError;

        const filtered_scanlines = decompressed_buffer.items;
        var prior_scanline: []const u8 = &[_]u8{};

        for (0..header.height) |y| {
            const filtered_offset = y * (1 + scanline_length);
            const filter_type = filtered_scanlines[filtered_offset];
            const current_filtered = filtered_scanlines[filtered_offset + 1 ..][0..scanline_length];

            const output_offset = y * scanline_length;
            const current_output = final_pixels[output_offset..][0..scanline_length];

            try unfilterScanline(current_output, current_filtered, prior_scanline, bytes_per_pixel, filter_type);

            prior_scanline = current_output;
        }
    } else if (header.interlace_method == 1) {
        // --- INTERLACED (ADAM7) PATH ---
        const adam7_passes = [_]struct { x_start: u32, y_start: u32, x_step: u32, y_step: u32 }{
            .{ .x_start = 0, .y_start = 0, .x_step = 8, .y_step = 8 },
            .{ .x_start = 4, .y_start = 0, .x_step = 8, .y_step = 8 },
            .{ .x_start = 0, .y_start = 4, .x_step = 4, .y_step = 8 },
            .{ .x_start = 2, .y_start = 0, .x_step = 4, .y_step = 4 },
            .{ .x_start = 0, .y_start = 2, .x_step = 2, .y_step = 4 },
            .{ .x_start = 1, .y_start = 0, .x_step = 2, .y_step = 2 },
            .{ .x_start = 0, .y_start = 1, .x_step = 1, .y_step = 2 },
        };

        var data_offset: usize = 0;
        const filtered_data = decompressed_buffer.items;

        for (adam7_passes) |pass| {
            const pass_width = (header.width - pass.x_start + pass.x_step - 1) / pass.x_step;
            const pass_height = (header.height - pass.y_start + pass.y_step - 1) / pass.y_step;

            if (pass_width == 0 or pass_height == 0) continue;

            const pass_scanline_len = pass_width * bytes_per_pixel;
            var prior_pass_scanline: []u8 = &[_]u8{};

            var pass_unfiltered_data = try allocator.alloc(u8, pass_scanline_len * pass_height);
            defer allocator.free(pass_unfiltered_data);

            for (0..pass_height) |pass_y| {
                const filter_type = filtered_data[data_offset];
                data_offset += 1;
                const filtered_pass_scanline = filtered_data[data_offset..][0..pass_scanline_len];
                data_offset += pass_scanline_len;

                const current_pass_output = pass_unfiltered_data[pass_y * pass_scanline_len ..][0..pass_scanline_len];
                try unfilterScanline(current_pass_output, filtered_pass_scanline, prior_pass_scanline, bytes_per_pixel, filter_type);
                prior_pass_scanline = current_pass_output;
            }

            for (0..pass_height) |pass_y| {
                for (0..pass_width) |pass_x| {
                    const final_y = pass.y_start + (pass_y * pass.y_step);
                    const final_x = pass.x_start + (pass_x * pass.x_step);
                    if (final_y >= header.height or final_x >= header.width) continue;

                    const src_offset = (pass_y * pass_scanline_len) + (pass_x * bytes_per_pixel);
                    const dst_offset = (final_y * header.width * bytes_per_pixel) + (final_x * bytes_per_pixel);

                    const src_pixel = pass_unfiltered_data[src_offset..][0..bytes_per_pixel];
                    const dst_pixel = final_pixels[dst_offset..][0..bytes_per_pixel];
                    @memcpy(dst_pixel, src_pixel);
                }
            }
        }
    }

    return PngImage{
        .width = header.width,
        .height = header.height,
        .bit_depth = header.bit_depth,
        .channels = header.channels,
        .pixels = final_pixels,
    };
}
