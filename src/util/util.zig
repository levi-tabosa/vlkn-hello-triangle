const std = @import("std");

pub fn Pool(T: type) type {
    return struct {
        const Self = @This();
        const List = std.SinglyLinkedList(T);

        arena: std.heap.ArenaAllocator,
        free: List = .{},

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .arena = .init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
        }

        pub fn new(self: *Self) !*T {
            const node = if (self.free.popFirst()) |pop|
                pop
            else
                try self.arena.allocator().create(List.Node);

            return &node.data;
        }

        pub fn delete(self: *Self, obj: *anyopaque) void {
            // Print raw pointer
            std.log.info("Deleting object at address: {p}", .{obj});

            const casted: *[256]u8 = @alignCast(@ptrCast(obj));
            const node: *List.Node = @alignCast(@fieldParentPtr("data", casted));

            // Optionally print part of the data for verification
            const data_ptr = &node.data;
            const data_bytes = @as([*]const u8, @ptrCast(data_ptr));
            std.log.info("First few data bytes: {d}, {d}, {d}, {d}", .{
                data_bytes[0],
                data_bytes[1],
                data_bytes[2],
                data_bytes[3],
            });

            self.free.prepend(node);
        }
    };
}
