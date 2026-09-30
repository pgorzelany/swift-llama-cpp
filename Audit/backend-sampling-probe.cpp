#include "llama.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

static void quiet(enum ggml_log_level, const char *, void *) {}
static void check(int status) { if (status) throw std::runtime_error("decode status=" + std::to_string(status)); }
using audit_clock = std::chrono::steady_clock;
static double seconds(audit_clock::time_point start) {
    return std::chrono::duration<double>(audit_clock::now() - start).count();
}
int main(int argc, char ** argv) {
    if (argc != 2) return 2;
    llama_log_set(quiet, nullptr);
    llama_backend_init();
    auto * model = llama_model_load_from_file(argv[1], llama_model_default_params());
    if (!model) return 3;
    auto * vocab = llama_model_get_vocab(model);
    std::string text;
    for (int i = 0; i < 400; ++i) text += "The quick brown fox walks past the quiet river. ";
    int count = -llama_tokenize(vocab, text.data(), text.size(), nullptr, 0, true, true);
    std::vector<llama_token> prompt(count);
    check(llama_tokenize(vocab, text.data(), text.size(), prompt.data(), prompt.size(), true, true) < 0 ? -1 : 0);
    prompt.resize(2048);
    puts("backend_requested,attach_ok,chain,repeat,sampled_token_available,sampled_logits_count,sampled_probs_count,pp_tps,tg_tps");
    for (bool backend : {false, true}) {
        auto cp = llama_context_default_params();
        cp.n_ctx = 4096; cp.n_batch = cp.n_ubatch = 256; cp.n_threads = cp.n_threads_batch = 1;
        auto * ctx = llama_init_from_model(model, cp);
        if (!ctx) return 4;
        auto * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.1f, 0, 0));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.5f));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(42));
        bool attached = backend && llama_set_sampler(ctx, 0, smpl);
        auto batch = llama_batch_init(256, 0, 1);
        for (int repeat = -1; repeat < 3; ++repeat) {
            llama_memory_clear(llama_get_memory(ctx), false);
            llama_sampler_reset(smpl);
            auto start = audit_clock::now();
            for (int offset = 0; offset < 2048; offset += 256) {
                batch.n_tokens = 256;
                for (int i = 0; i < 256; ++i) {
                    batch.token[i] = prompt[offset+i]; batch.pos[i] = offset+i;
                    batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = offset+i == 2047;
                }
                check(llama_decode(ctx, batch)); llama_synchronize(ctx);
            }
            double pp = seconds(start);
            bool selected = llama_get_sampled_token_ith(ctx, -1) != LLAMA_TOKEN_NULL;
            auto logits_count = llama_get_sampled_logits_count_ith(ctx, -1);
            auto probs_count = llama_get_sampled_probs_count_ith(ctx, -1);
            start = audit_clock::now();
            for (int i = 0; i < 128; ++i) {
                batch.n_tokens = 1; batch.token[0] = llama_sampler_sample(smpl, ctx, -1);
                batch.pos[0] = 2048+i; batch.n_seq_id[0] = 1; batch.seq_id[0][0] = 0; batch.logits[0] = true;
                check(llama_decode(ctx, batch)); llama_synchronize(ctx);
            }
            double tg = seconds(start);
            if (repeat >= 0) {
                printf("%d,%d,\"%s\",%d,%d,%u,%u,%.3f,%.3f\n", backend, attached, llama_sampler_name(smpl), repeat, selected, logits_count, probs_count, 2048/pp, 128/tg);
                fflush(stdout);
            }
        }
        llama_set_sampler(ctx, 0, nullptr);
        llama_batch_free(batch); llama_free(ctx); llama_sampler_free(smpl);
    }
    llama_model_free(model); llama_backend_free();
}
