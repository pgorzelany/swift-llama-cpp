//
//  LlamaChatMessage.swift
//  PrivateAI
//
//  Created by Piotr Gorzelany on 13/06/2024.
//

import Foundation

public struct LlamaChatMessage: Sendable {
    public enum Role: String, Sendable {
        case system
        case user
        case assistant
        case tool
    }
    
    public let role: Role
    public let content: String
    var usesLFMToolTemplate = false

    public init(role: LlamaChatMessage.Role, content: String) {
        self.role = role
        self.content = content
    }
}
