const std = @import("std");

// PngImage, PngParseError, and paethPredictor are unchanged.
// [Copy those from the previous answer]
pub const PngImage = struct { width: u32, height: u32, bit_depth: u8, pixels: []u8 };
pub const PngParseError = error{ FileTooShort, InvalidSignature, ChunkTooShort, MissingIhdr, InvalidIhdr, UnsupportedFormat, MissingIdat, MissingIend, DecompressionError, InvalidFilterType };
fn paethPredictor(a: u32, b: u32, c: u32) u32 {
    const p = a + b - c;
    const pa = std.math.absInt(@as(i64, p) - @as(i64, a)) catch unreachable;
    const pb = std.math.absInt(@as(i64, p) - @as(i64, b)) catch unreachable;
    const pc = std.math.absInt(@as(i64, p) - @as(i64, c)) catch unreachable;
    if (pa <= pb and pa <= pc) {
        return a;
    } else if (pb <= pc) {
        return b;
    } else {
        return c;
    }
}

pub fn loadPngGrayscale(
    allocator: std.mem.Allocator,
    png_data: []const u8,
) PngParseError!PngImage {
    const png_signature = [_]u8{ 137, 80, 78, 71, 13, 10, 26, 10 };
    if (png_data.len < 8 or !std.mem.eql(u8, png_data[0..8], &png_signature)) {
        return PngParseError.InvalidSignature;
    }
    var ihdr: ?struct { width: u32, height: u32, bit_depth: u8 } = null;
    var idat_stream = std.ArrayList(u8).init(allocator);
    defer idat_stream.deinit();
    var cursor: usize = 8;
    while (cursor < png_data.len) {
        if (cursor + 8 > png_data.len) return PngParseError.ChunkTooShort;

        // --- FIXED ---
        const data_len = std.mem.readInt(u32, @ptrCast(&png_data[cursor]), .big);
        // -----------
        const chunk_type_slice = png_data[cursor + 4 .. cursor + 8];
        cursor += 8;

        if (cursor + data_len + 4 > png_data.len) return PngParseError.ChunkTooShort;
        const chunk_data = png_data[cursor .. cursor + data_len];

        if (std.mem.eql(u8, chunk_type_slice, "IHDR")) {
            if (data_len != 13) return PngParseError.InvalidIhdr;

            // --- FIXED ---
            const width = std.mem.readInt(u32, @ptrCast(&chunk_data[0]), .big);
            const height = std.mem.readInt(u32, @ptrCast(&chunk_data[4]), .big);
            // -----------
            const bit_depth = chunk_data[8];
            const color_type = chunk_data[9];
            const interlace_method = chunk_data[12];

            if (color_type != 0 or interlace_method != 0) return PngParseError.UnsupportedFormat;
            if (bit_depth != 8 and bit_depth != 16) return PngParseError.UnsupportedFormat;
            ihdr = .{ .width = width, .height = height, .bit_depth = bit_depth };
        } else if (std.mem.eql(u8, chunk_type_slice, "IDAT")) {
            if (ihdr == null) return PngParseError.MissingIhdr;
            try idat_stream.appendSlice(chunk_data);
        } else if (std.mem.eql(u8, chunk_type_slice, "IEND")) {
            break;
        }
        cursor += data_len;
        cursor += 4; // CRC
    }
    const header = ihdr orelse return PngParseError.MissingIhdr;
    if (idat_stream.items.len == 0) return PngParseError.MissingIdat;
    var decompressed_buffer = std.ArrayList(u8).init(allocator);
    defer decompressed_buffer.deinit();
    {
        var compressed_reader = std.io.fixedBufferStream(idat_stream.items);
        var decompressor = std.compress.zlib.decompressor(compressed_reader.reader());
        try decompressor.reader().readAllArrayList(&decompressed_buffer, std.math.maxInt(usize));
    }
    const filtered_scanlines = decompressed_buffer.items;
    const bytes_per_pixel = header.bit_depth / 8;
    const scanline_length = header.width * bytes_per_pixel;
    const expected_filtered_size = header.height * (1 + scanline_length);
    if (filtered_scanlines.len != expected_filtered_size) return PngParseError.DecompressionError;
    var final_pixels = try allocator.alloc(u8, header.width * header.height * bytes_per_pixel);
    var prior_scanline: []const u8 = &[_]u8{};

    for (0..header.height) |y| {
        const filtered_offset = y * (1 + scanline_length);
        const filter_type = filtered_scanlines[filtered_offset];
        const current_filtered = filtered_scanlines[filtered_offset + 1 ..][0..scanline_length];
        const output_offset = y * scanline_length;
        const current_output = final_pixels[output_offset..][0..scanline_length];
        switch (header.bit_depth) {
            8 => {
                for (0..scanline_length) |x| {
                    const filt_x = current_filtered[x];
                    const recon_a = if (x > 0) current_output[x - 1] else 0;
                    const recon_b = if (y > 0) prior_scanline[x] else 0;
                    const recon_c = if (x > 0 and y > 0) prior_scanline[x - 1] else 0;
                    current_output[x] = @as(u8, @truncate(@addWithOverflow(filt_x, switch (filter_type) {
                        0 => 0,
                        1 => recon_a,
                        2 => recon_b,
                        3 => (@as(u16, recon_a) + @as(u16, recon_b)) / 2,
                        4 => @as(u8, @truncate(paethPredictor(recon_a, recon_b, recon_c))),
                        else => return PngParseError.InvalidFilterType,
                    })[0]));
                }
            },
            16 => {
                for (0..header.width) |px_idx| {
                    const x = px_idx * 2;
                    // --- FIXED ---
                    const filt_x = std.mem.readInt(u16, @ptrCast(current_filtered[x]), .big);
                    const recon_a = if (x > 0) std.mem.readInt(u16, @ptrCast(current_output[x - 2]), .big) else 0;
                    const recon_b = if (y > 0) std.mem.readInt(u16, @ptrCast(&prior_scanline[x]), .big) else 0;
                    const recon_c = if (x > 0 and y > 0) std.mem.readInt(u16, @ptrCast(&prior_scanline[x - 2]), .big) else 0;
                    // -----------
                    const recon_pixel = @as(u16, @truncate(@addWithOverflow(filt_x, switch (filter_type) {
                        0 => 0,
                        1 => recon_a,
                        2 => recon_b,
                        3 => (@as(u32, recon_a) + @as(u32, recon_b)) / 2,
                        4 => @as(u16, @truncate(paethPredictor(recon_a, recon_b, recon_c))),
                        else => return PngParseError.InvalidFilterType,
                    })[0]));
                    // --- FIXED ---
                    std.mem.writeInt(u16, @ptrCast(current_output[x]), recon_pixel, .big);
                    // -----------
                }
            },
            else => unreachable,
        }
        prior_scanline = current_output;
    }
    return PngImage{ .width = header.width, .height = header.height, .bit_depth = header.bit_depth, .pixels = final_pixels };
}
