public struct LlamaContextUsage: Equatable, Sendable {
    public let usedTokens: Int
    /// Largest formatted input accepted by inference after its fixed four-token decode reserve.
    public let effectiveCapacity: Int

    public init(usedTokens: Int, effectiveCapacity: Int) {
        self.usedTokens = usedTokens
        self.effectiveCapacity = effectiveCapacity
    }
}
