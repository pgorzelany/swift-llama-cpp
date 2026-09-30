zero-rank1-llama-1b.gguf is a synthetic test adapter, not a trained model.
It contains two all-zero F32 tensors of rank 1 for blk.0.attn_q.weight
(2048 input/output dimensions) and the four llama.cpp adapter metadata keys.
Its purpose is to exercise real adapter loading, application and ownership.
It has no downloaded weights. Creation used the pinned upstream gguf-py writer:

    writer = gguf.GGUFWriter(path, "llama")
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", 1)
    writer.add_tensor("blk.0.attn_q.weight.lora_a", np.zeros((1, 2048), dtype=np.float32))
    writer.add_tensor("blk.0.attn_q.weight.lora_b", np.zeros((2048, 1), dtype=np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
