#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using clock_type = std::chrono::steady_clock;
static double seconds(clock_type::time_point start) {
    return std::chrono::duration<double>(clock_type::now() - start).count();
}
static void quiet(enum ggml_log_level, const char *, void *) {}
static void check(int status) { if (status) throw std::runtime_error("decode status=" + std::to_string(status)); }

// Same pinned framework as SwiftLlama; compare C API scheduling without Swift/session overhead.
int main(int argc, char ** argv) {
    if (argc < 2 || argc > 3) { fprintf(stderr, "usage: probe model.gguf [--metal-only]\n"); return 2; }
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    puts("mode,threads,batch,ubatch,sync,fa,kv,sampling,offload_ops,repeat,prompt_tokens,output_tokens,pp_tps,tg_tps,pp_seconds,tg_seconds");
    const auto modes = argc == 3 && std::string(argv[2]) == "--metal-only"
        ? std::vector<bool>{true} : std::vector<bool>{false, true};
    for (bool gpu : modes) {
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = gpu ? -1 : 0;
        auto * model = llama_model_load_from_file(argv[1], mp);
        if (!model) return 3;
        auto * vocab = llama_model_get_vocab(model);
        std::string text;
        for (int i = 0; i < 400; ++i) text += "The quick brown fox walks past the quiet river. ";
        int needed = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
        std::vector<llama_token> prompt(needed);
        check(llama_tokenize(vocab, text.data(), text.size(), prompt.data(), prompt.size(), true, true) < 0 ? -1 : 0);
        prompt.resize(gpu ? 2048 : 64);
        int output_tokens = gpu ? 128 : 16;
        struct config { int threads, batch, ubatch; bool sync; llama_flash_attn_type fa; ggml_type kv; int sampling = 0; bool offload = true; };
        std::vector<config> configs = gpu ? std::vector<config>{
            {1, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {4, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {1, 1024, 1024, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {1, 1024, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {1, 256, 256, false, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {1, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_DISABLED, GGML_TYPE_F16},
            {1, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_ENABLED, GGML_TYPE_Q8_0},
            {1, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16, 1},
            {1, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16, 2},
        } : std::vector<config>{
            {1, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {4, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {8, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16},
            {1, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16, 0, false},
            {8, 256, 256, true, LLAMA_FLASH_ATTN_TYPE_AUTO, GGML_TYPE_F16, 0, false},
        };
        for (auto c : configs) {
            auto cp = llama_context_default_params();
            cp.n_ctx = gpu ? 4096 : 2048; cp.n_batch = c.batch; cp.n_ubatch = c.ubatch;
            cp.n_threads = c.threads; cp.n_threads_batch = c.threads;
            cp.flash_attn_type = c.fa; cp.type_k = cp.type_v = c.kv; cp.no_perf = false;
            cp.offload_kqv = cp.op_offload = c.offload;
            auto * ctx = llama_init_from_model(model, cp);
            if (!ctx) { fprintf(stderr, "unsupported configuration\n"); continue; }
            auto batch = llama_batch_init(c.batch, 0, 1);
            for (int repeat = -1; repeat < 3; ++repeat) {
                llama_memory_clear(llama_get_memory(ctx), false);
                auto * sampler = llama_sampler_init_greedy();
                if (c.sampling != 0) {
                    llama_sampler_free(sampler);
                    sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
                    if (c.sampling == 2) llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
                    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
                    llama_sampler_chain_add(sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.1f, 0, 0));
                    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.5f));
                    llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));
                }
                auto start = clock_type::now();
                for (int offset = 0; offset < (int)prompt.size(); offset += c.batch) {
                    batch.n_tokens = std::min(c.batch, (int)prompt.size() - offset);
                    for (int i = 0; i < batch.n_tokens; ++i) {
                        batch.token[i] = prompt[offset+i]; batch.pos[i] = offset+i;
                        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
                        batch.logits[i] = offset+i+1 == (int)prompt.size();
                    }
                    check(llama_decode(ctx, batch));
                    if (c.sync) llama_synchronize(ctx);
                }
                llama_synchronize(ctx);
                double pp = seconds(start);
                start = clock_type::now();
                for (int i = 0; i < output_tokens; ++i) {
                    auto token = llama_sampler_sample(sampler, ctx, -1);
                    batch.n_tokens = 1; batch.token[0] = token; batch.pos[0] = prompt.size()+i;
                    batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
                    check(llama_decode(ctx, batch));
                    if (c.sync) llama_synchronize(ctx);
                }
                llama_synchronize(ctx);
                double tg = seconds(start);
                if (repeat >= 0) {
                    printf("%s,%d,%d,%d,%d,%d,%s,%d,%d,%d,%zu,%d,%.3f,%.3f,%.6f,%.6f\n",
                           gpu ? "metal" : "cpu", c.threads, c.batch, c.ubatch, c.sync, int(c.fa),
                           c.kv == GGML_TYPE_F16 ? "f16" : "q8_0", c.sampling, c.offload, repeat, prompt.size(), output_tokens,
                           prompt.size()/pp, output_tokens/tg, pp, tg);
                    fflush(stdout);
                }
                llama_sampler_free(sampler);
            }
            llama_batch_free(batch); llama_free(ctx);
        }
        llama_model_free(model);
    }
    llama_backend_free();
}
