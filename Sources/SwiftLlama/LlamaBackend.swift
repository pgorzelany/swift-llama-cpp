import Foundation
import llama

public enum LlamaBackend {
    /// Initialize the llama + ggml backend. Call once at program start.
    private final class State: @unchecked Sendable {
        let lock = NSLock()
        var initialized = false
        var models = 0
        var shutdownRequested = false
    }
    private static let state = State()

    public static func initialize() {
        state.lock.lock()
        defer { state.lock.unlock() }
        initializeLocked()
    }

    private static func initializeLocked() {
        if !state.initialized {
            llama_backend_init()
            state.initialized = true
        }
        state.shutdownRequested = false
    }

    static func acquireModel() {
        state.lock.lock()
        defer { state.lock.unlock() }
        initializeLocked()
        state.models += 1
    }

    static func releaseModel() {
        state.lock.lock()
        defer { state.lock.unlock() }
        state.models -= 1
        if state.models == 0 && state.shutdownRequested { shutdownLocked() }
    }

    private static func shutdownLocked() {
        if state.initialized { llama_backend_free() }
        state.initialized = false
        state.shutdownRequested = false
    }
    /// Requests shutdown; active models and their contexts keep the backend alive.
    public static func shutdown() {
        state.lock.lock()
        defer { state.lock.unlock() }
        state.shutdownRequested = true
        if state.models == 0 { shutdownLocked() }
    }
    /// Whether mmap/mlock/gpu offload/rpc are supported by the compiled library.
    public static var supportsMmap: Bool { llama_supports_mmap() }
    public static var supportsMlock: Bool { llama_supports_mlock() }
    public static var supportsGpuOffload: Bool { llama_supports_gpu_offload() }
    public static var supportsRpc: Bool { llama_supports_rpc() }
    /// Maximum devices and parallel sequences
    public static var maxDevices: Int { Int(llama_max_devices()) }
    public static var maxParallelSequences: Int { Int(llama_max_parallel_sequences()) }

    /// Initialize NUMA with a given strategy.
    public static func numaInit(_ strategy: ggml_numa_strategy) { llama_numa_init(strategy) }

    /// Microsecond timer from llama.cpp
    public static func timeMicros() -> Int64 { llama_time_us() }

    /// Return system info string provided by llama.cpp
    public static func systemInfo() -> String {
        guard let c = llama_print_system_info() else { return "" }
        return String(cString: c)
    }

    /// Uses the ggml fallback threadpool by detaching explicit pools.
    public static func attachAutoThreadpool(to context: LlamaContext) {
        llama_attach_threadpool(context.contextPointer, nil, nil)
    }

    /// Detach any threadpools from the context.
    public static func detachThreadpool(from context: LlamaContext) {
        llama_detach_threadpool(context.contextPointer)
    }
}

