import llama

public enum LlamaBackend {
    /// Initialize the llama + ggml backend. Call once at program start.
    public static func initialize() { llama_backend_init() }
    /// Free the backend. Call once at program end.
    public static func shutdown() { llama_backend_free() }
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

    /// Attach the library-managed auto threadpool to a context.
    public static func attachAutoThreadpool(to context: LlamaContext) {
        llama_attach_threadpool(context.contextPointer, nil, nil)
    }

    /// Detach any threadpools from the context.
    public static func detachThreadpool(from context: LlamaContext) {
        llama_detach_threadpool(context.contextPointer)
    }
}

