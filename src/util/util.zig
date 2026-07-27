const std = @import("std");

pub fn Pool(comptime T: type) type {
    return struct {
        const Self = @This();

        /// Estrutura que contém o nó da lista e o dado do usuário.
        const Item = struct {
            node: std.SinglyLinkedList.Node = .{},
            data: T,
        };

        arena: std.heap.ArenaAllocator,
        free: std.SinglyLinkedList = .{},

        pub fn init(allocator: std.mem.Allocator) Self {
            return .{
                .arena = .init(allocator),
            };
        }

        pub fn deinit(self: *Self) void {
            self.arena.deinit();
            self.* = undefined;
        }

        /// Aloca (ou reutiliza) um item e retorna um ponteiro para o dado.
        pub fn new(self: *Self) !*T {
            // Tenta reutilizar um nó da lista livre
            if (self.free.popFirst()) |node| {
                const item: *Item = @alignCast(@fieldParentPtr("node", node));
                return &item.data;
            }
            // Caso contrário, aloca um novo Item no arena
            const item = try self.arena.allocator().create(Item);
            item.* = .{ .data = undefined };
            return &item.data;
        }

        /// Libera um item previamente obtido com `new` e o coloca na lista livre.
        /// O ponteiro `obj` deve ser exatamente aquele retornado por `new`.
        pub fn delete(self: *Self, obj: *anyopaque) void {
            // Converte o ponteiro opaco para um ponteiro alinhado a T
            const data_ptr: *T = @ptrCast(@alignCast(obj));
            const item: *Item = @alignCast(@fieldParentPtr("data", data_ptr));

            // (Opcional) Log para debug
            // std.log.info("Deleting object at address: {*}", .{item});
            // const data_bytes = @as([*]const u8, @ptrCast(data_ptr));
            // std.log.info("First few data bytes: {d}, {d}, {d}, {d}", .{
            //     data_bytes[0], data_bytes[1], data_bytes[2], data_bytes[3],
            // });

            // Devolve o nó para a lista livre
            self.free.prepend(&item.node);
        }
    };
}
