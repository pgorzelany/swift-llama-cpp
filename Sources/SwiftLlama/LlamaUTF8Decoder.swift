/// Keeps incomplete UTF-8 scalars between token pieces; invalid bytes are replaced only when complete or flushed.
struct LlamaUTF8Decoder {
    private var pending: [UInt8] = []

    mutating func append(_ bytes: [UInt8]) -> String {
        pending.append(contentsOf: bytes)
        var boundary = pending.count
        if let last = pending.indices.last {
            var lead = last
            while lead > 0 && pending[lead] & 0xC0 == 0x80 { lead -= 1 }
            let byte = pending[lead]
            let width = byte >= 0xC2 && byte <= 0xDF ? 2 : byte >= 0xE0 && byte <= 0xEF ? 3 : byte >= 0xF0 && byte <= 0xF4 ? 4 : 1
            if pending.count - lead < width { boundary = lead }
        }
        let text = String(decoding: pending.prefix(boundary), as: UTF8.self)
        pending.removeFirst(boundary)
        return text
    }

    mutating func finish() -> String {
        defer { pending.removeAll(keepingCapacity: true) }
        return String(decoding: pending, as: UTF8.self)
    }

    mutating func reset() { pending.removeAll(keepingCapacity: true) }
}
