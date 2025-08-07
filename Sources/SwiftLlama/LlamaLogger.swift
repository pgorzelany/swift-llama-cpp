//
//  LlamaLoger.swift
//  swift-llama-cpp
//
//  Created by Piotr Gorzelany on 22/07/2025.
//

import OSLog
import llama

extension Logger {
    static func llama(category: String) -> Logger {
        return Logger(subsystem: "swift-llama-ccp", category: category)
    }
}

public enum LlamaLog {
    /// Set a global log callback that bridges to Swift. If `logger` is nil, logging is disabled.
    public static func setLogger(_ logger: Logger?) {
        if let logger {
            let box = Unmanaged.passRetained(LoggerBox(logger: logger))
            llama_log_set({ level, message, userData in
                guard let userData else { return }
                let box = Unmanaged<LoggerBox>.fromOpaque(userData).takeUnretainedValue()
                let msg = message != nil ? String(cString: message!) : ""
                switch level.rawValue {
                case 0: box.logger.debug("\(msg)")
                case 1: box.logger.info("\(msg)")
                case 2: box.logger.error("\(msg)")
                default: box.logger.log("\(msg)")
                }
            }, box.toOpaque())
        } else {
            llama_log_set(nil, nil)
        }
    }
}

private final class LoggerBox {
    let logger: Logger
    init(logger: Logger) { self.logger = logger }
    func toOpaque() -> UnsafeMutableRawPointer { UnsafeMutableRawPointer(Unmanaged.passUnretained(self).toOpaque()) }
}
