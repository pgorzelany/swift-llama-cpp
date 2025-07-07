//
//  LlamaConfig.swift
//  LlamaSwift
//
//  Created by Piotr Gorzelany on 05/11/2024.
//

public struct LlamaConfig: Equatable, Sendable {
    public let batchSize: UInt32
    public let maxTokenCount: UInt32
    public let useGPU: Bool

    public init(
        batchSize: UInt32,
        maxTokenCount: UInt32,
        useGPU: Bool = true
    ) {
        self.batchSize = batchSize
        self.maxTokenCount = maxTokenCount
        self.useGPU = useGPU
    }
}
