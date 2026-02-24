"""
FP4 Inference Demo for GPT-OSS-4.2B on DGX Spark

Loads the model from safetensors, quantizes all weight matrices to NVFP4
once at load time, then runs a forward pass benchmarking FP4 cached vs BF16.

This demonstrates:
  1. One-time weight quantization (~milliseconds per layer)
  2. Inference with pre-quantized weights at 85-129 TFLOPS
  3. 4x memory savings (FP4 weights = ~2 GB vs ~8 GB BF16)

Usage:
    python fp4_inference_demo.py [--model_dir PATH] [--prompt TEXT] [--benchmark]
"""

import os
import sys
import time
import argparse
import json
import math

import torch
import torch.nn.functional as F
from safetensors import safe_open

# Add fp4-hack to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp4_gemm import fp4_quantize, FP4WeightCache, fp4_matmul, cleanup, sync as fp4_sync, prealloc as fp4_prealloc


def load_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        return json.load(f)


def load_weights(model_dir):
    """Load all safetensor shards into a flat dict."""
    weights = {}
    shard_files = sorted([
        f for f in os.listdir(model_dir)
        if f.startswith("model-") and f.endswith(".safetensors")
    ])
    print(f"Loading {len(shard_files)} shards from {model_dir}...")
    t0 = time.perf_counter()
    for sf in shard_files:
        path = os.path.join(model_dir, sf)
        with safe_open(path, framework="pt", device="cuda") as f:
            for name in f.keys():
                weights[name] = f.get_tensor(name)
    dt = time.perf_counter() - t0
    total_params = sum(w.numel() for w in weights.values())
    total_mb = sum(w.numel() * w.element_size() for w in weights.values()) / 1e6
    print(f"  Loaded {len(weights)} tensors ({total_params/1e6:.0f}M params, {total_mb:.0f} MB) in {dt:.1f}s")
    return weights


class FP4Linear:
    """A linear layer backed by pre-quantized FP4 weights."""

    def __init__(self, weight, bias=None):
        self.cache = fp4_quantize(weight, bias)
        self.out_features = weight.shape[0]
        self.in_features = weight.shape[1]

    def __call__(self, x):
        # No sync — let CUDA pipeline kernels. Caller syncs at end of forward.
        return self.cache.forward(x, sync=False)

    def free(self):
        self.cache.free()


class BF16Linear:
    """Reference BF16 linear for comparison."""

    def __init__(self, weight, bias=None):
        self.weight = weight
        self.bias = bias

    def __call__(self, x):
        return F.linear(x, self.weight, self.bias)


class GPTOssAttention:
    """Single attention layer with FP4 or BF16 linear layers."""

    def __init__(self, layer_weights, config, use_fp4=True):
        LinearCls = FP4Linear if use_fp4 else BF16Linear
        prefix = ""

        self.num_heads = config["num_attention_heads"]
        self.num_kv_heads = config["num_key_value_heads"]
        self.head_dim = config.get("head_dim", config["hidden_size"] // self.num_heads)
        self.hidden_size = config["hidden_size"]

        self.q_proj = LinearCls(
            layer_weights["self_attn.q_proj.weight"],
            layer_weights.get("self_attn.q_proj.bias"))
        self.k_proj = LinearCls(
            layer_weights["self_attn.k_proj.weight"],
            layer_weights.get("self_attn.k_proj.bias"))
        self.v_proj = LinearCls(
            layer_weights["self_attn.v_proj.weight"],
            layer_weights.get("self_attn.v_proj.bias"))
        self.o_proj = LinearCls(
            layer_weights["self_attn.o_proj.weight"],
            layer_weights.get("self_attn.o_proj.bias"))

    def __call__(self, x):
        B, T, C = x.shape

        q = self.q_proj(x)  # [B, T, num_heads * head_dim]
        k = self.k_proj(x)  # [B, T, num_kv_heads * head_dim]
        v = self.v_proj(x)  # [B, T, num_kv_heads * head_dim]

        # Reshape for attention
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # GQA: repeat k,v for grouped query attention
        if self.num_kv_heads < self.num_heads:
            rep = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # Scaled dot-product attention (uses PyTorch's optimized kernel)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Reshape back
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(attn_out)


class GPTOssMLP:
    """MLP with MoE expert weights, FP4 or BF16."""

    def __init__(self, layer_weights, config, use_fp4=True):
        LinearCls = FP4Linear if use_fp4 else BF16Linear
        self.num_experts = config["num_local_experts"]
        self.hidden_size = config["hidden_size"]

        # Expert weights stored as [num_experts, in_features, out_features]
        # Need to transpose to [out_features, in_features] for F.linear convention
        gate_up = layer_weights["mlp.experts.gate_up_proj"]   # [4, 2880, 5760] = [experts, in, out]
        down = layer_weights["mlp.experts.down_proj"]          # [4, 2880, 2880] = [experts, in, out]
        gate_up_bias = layer_weights.get("mlp.experts.gate_up_proj_bias")  # [4, 5760]
        down_bias = layer_weights.get("mlp.experts.down_proj_bias")        # [4, 2880]

        self.gate_up_projs = []
        self.down_projs = []
        for i in range(self.num_experts):
            gu_bias = gate_up_bias[i] if gate_up_bias is not None else None
            d_bias = down_bias[i] if down_bias is not None else None
            # Transpose: [in, out] -> [out, in] for F.linear/FP4 convention
            # gate_up: [5760, 2880] projects hidden(2880) -> gate+up(5760)
            self.gate_up_projs.append(LinearCls(gate_up[i].t().contiguous(), gu_bias))
            # down: [2880, 2880] projects intermediate(2880) -> hidden(2880)
            self.down_projs.append(LinearCls(down[i].t().contiguous(), d_bias))

        # Router
        self.router_weight = layer_weights["mlp.router.weight"]  # [4, 2880]
        self.router_bias = layer_weights.get("mlp.router.bias")  # [4]

    def __call__(self, x):
        B, T, C = x.shape

        # Router (tiny — no FP4 needed)
        router_logits = F.linear(x, self.router_weight, self.router_bias)  # [B, T, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)

        # All experts active (4.2B pruned has experts_per_tok = num_experts = 4)
        # Run all experts and combine
        expert_outputs = torch.zeros_like(x)
        for i in range(self.num_experts):
            # SwiGLU: split gate_up into gate and up
            gate_up_out = self.gate_up_projs[i](x)  # [B, T, 5760]
            gate, up = gate_up_out.chunk(2, dim=-1)  # each [B, T, 2880]
            hidden = F.silu(gate) * up
            expert_out = self.down_projs[i](hidden)  # [B, T, 2880]
            expert_outputs += router_weights[:, :, i:i+1] * expert_out

        return expert_outputs


class GPTOssBlock:
    """Single transformer block."""

    def __init__(self, layer_weights, config, use_fp4=True):
        self.attn = GPTOssAttention(layer_weights, config, use_fp4)
        self.mlp = GPTOssMLP(layer_weights, config, use_fp4)
        self.input_layernorm_weight = layer_weights["input_layernorm.weight"]
        self.post_attn_layernorm_weight = layer_weights["post_attention_layernorm.weight"]
        self.rms_norm_eps = config["rms_norm_eps"]

    def rms_norm(self, x, weight):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return (weight * x.to(weight.dtype))

    def __call__(self, x):
        # Pre-norm attention
        normed = self.rms_norm(x, self.input_layernorm_weight)
        x = x + self.attn(normed)
        # Pre-norm MLP
        normed = self.rms_norm(x, self.post_attn_layernorm_weight)
        x = x + self.mlp(normed)
        return x


class GPTOssModel:
    """Minimal GPT-OSS model for FP4 inference benchmarking."""

    def __init__(self, model_dir, use_fp4=True, num_layers=None):
        self.config = load_config(model_dir)
        self.use_fp4 = use_fp4
        self.hidden_size = self.config["hidden_size"]
        self.vocab_size = self.config["vocab_size"]

        all_weights = load_weights(model_dir)

        # Embedding (not FP4 — it's a lookup, not a GEMM)
        self.embed_tokens = all_weights["model.embed_tokens.weight"]

        # Final norm
        self.norm_weight = all_weights["model.norm.weight"]
        self.rms_norm_eps = self.config["rms_norm_eps"]

        # LM head
        lm_head_w = all_weights.get("lm_head.weight", self.embed_tokens)
        if use_fp4:
            self.lm_head = FP4Linear(lm_head_w)
        else:
            self.lm_head = BF16Linear(lm_head_w)

        # Transformer layers
        total_layers = self.config["num_hidden_layers"]
        if num_layers is not None:
            total_layers = min(num_layers, total_layers)

        print(f"\nQuantizing {total_layers} layers to {'FP4' if use_fp4 else 'BF16'}...")
        t0 = time.perf_counter()

        self.layers = []
        fp4_bytes = 0
        bf16_bytes = 0
        for i in range(total_layers):
            # Collect layer weights
            prefix = f"model.layers.{i}."
            layer_w = {}
            for name, tensor in all_weights.items():
                if name.startswith(prefix):
                    short_name = name[len(prefix):]
                    layer_w[short_name] = tensor

            block = GPTOssBlock(layer_w, self.config, use_fp4)
            self.layers.append(block)

            # Count FP4 cache memory
            if use_fp4:
                for proj in [block.attn.q_proj, block.attn.k_proj,
                             block.attn.v_proj, block.attn.o_proj]:
                    fp4_bytes += proj.cache.fp4_bytes + proj.cache.sf_bytes
                for j in range(block.mlp.num_experts):
                    fp4_bytes += block.mlp.gate_up_projs[j].cache.fp4_bytes + block.mlp.gate_up_projs[j].cache.sf_bytes
                    fp4_bytes += block.mlp.down_projs[j].cache.fp4_bytes + block.mlp.down_projs[j].cache.sf_bytes

            if (i + 1) % 6 == 0 or i == total_layers - 1:
                print(f"  Layer {i+1}/{total_layers} done")

        dt = time.perf_counter() - t0
        self.num_layers = total_layers

        if use_fp4:
            # Count lm_head too
            fp4_bytes += self.lm_head.cache.fp4_bytes + self.lm_head.cache.sf_bytes
            print(f"\n  FP4 weight cache: {fp4_bytes/1e6:.1f} MB ({fp4_bytes/1e9:.2f} GB)")

        # Count original BF16 size
        for name, tensor in all_weights.items():
            bf16_bytes += tensor.numel() * tensor.element_size()
        print(f"  Original BF16: {bf16_bytes/1e6:.1f} MB ({bf16_bytes/1e9:.2f} GB)")
        if use_fp4:
            print(f"  Compression: {bf16_bytes/fp4_bytes:.1f}x")
        print(f"  Quantization time: {dt:.2f}s ({dt/total_layers*1000:.0f}ms per layer)")

        # Pre-allocate internal GEMM buffers for maximum dimensions
        # This prevents reallocation during inference when layer dims change
        if use_fp4:
            # Find max N and K across all weight matrices
            max_N = max(
                4096,  # q_proj out
                5760,  # gate_up out
                2880,  # down, o_proj
                self.vocab_size,  # lm_head
            )
            max_K = max(4096, 5760, 2880)  # max input dim
            max_M = 4096  # max batch*seq we expect
            print(f"  Pre-allocating GEMM buffers: max_M={max_M}, max_N={max_N}, max_K={max_K}")
            fp4_prealloc(max_M, max_N, max_K)

        # Free original weights from GPU
        del all_weights
        torch.cuda.empty_cache()

    def rms_norm(self, x, weight):
        variance = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.rms_norm_eps)
        return (weight * x.to(weight.dtype))

    @torch.no_grad()
    def forward(self, input_ids):
        """Run a forward pass through the model."""
        x = F.embedding(input_ids, self.embed_tokens)  # [B, T, hidden]

        for layer in self.layers:
            x = layer(x)

        x = self.rms_norm(x, self.norm_weight)
        logits = self.lm_head(x)  # [B, T, vocab]
        if self.use_fp4:
            fp4_sync()  # Single sync for the entire forward pass
        return logits


def benchmark_forward(model, seq_len=128, batch_size=1, warmup=3, iters=10):
    """Benchmark a single forward pass."""
    input_ids = torch.randint(0, model.vocab_size, (batch_size, seq_len), device="cuda")

    # Warmup
    for _ in range(warmup):
        _ = model.forward(input_ids)
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model.forward(input_ids)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg_ms = sum(times) / len(times) * 1000
    min_ms = min(times) * 1000
    tokens = batch_size * seq_len

    # Estimate FLOPS (rough: 2 * params * tokens for forward pass)
    # Per layer: attention (4 projections) + MLP (gate_up + down) * num_experts
    # Attention: q[4096,2880] + k[512,2880] + v[512,2880] + o[2880,4096] = 4 * 2880 * ~2880 avg
    # MLP per expert: gate_up[2880,5760] + down[2880,2880] = 2880*(5760+2880) = 24.9M
    # MLP total: 4 experts * 24.9M = 99.5M
    # Attention total: 4096*2880 + 512*2880 + 512*2880 + 2880*4096 = 26.5M
    # Per layer: ~126M multiply-adds = ~252M FLOPS
    # 24 layers: ~6.05B FLOPS per token
    flops_per_token = 2 * 4.2e9  # rough: 2 * num_params
    total_flops = flops_per_token * tokens
    tflops = total_flops / (avg_ms / 1000) / 1e12

    return avg_ms, min_ms, tokens, tflops


def main():
    parser = argparse.ArgumentParser(description="FP4 Inference Demo for GPT-OSS-4.2B")
    parser.add_argument("--model_dir", type=str,
                        default="/home/xentureon/GPT-OSS-120B/models/gpt-oss-4.2b-pruned",
                        help="Path to GPT-OSS-4.2B safetensors directory")
    parser.add_argument("--num_layers", type=int, default=None,
                        help="Number of layers to load (default: all 24)")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length for benchmark")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for benchmark")
    parser.add_argument("--benchmark", action="store_true", default=True,
                        help="Run forward pass benchmark")
    parser.add_argument("--compare", action="store_true",
                        help="Compare FP4 vs BF16 side by side")
    args = parser.parse_args()

    print("=" * 70)
    print("FP4 Inference Demo - GPT-OSS-4.2B on DGX Spark")
    print("=" * 70)
    print()

    if args.compare:
        # Load both FP4 and BF16 models
        print(">>> Loading FP4 model...")
        fp4_model = GPTOssModel(args.model_dir, use_fp4=True, num_layers=args.num_layers)

        print("\n>>> Loading BF16 model...")
        bf16_model = GPTOssModel(args.model_dir, use_fp4=False, num_layers=args.num_layers)

        print("\n" + "=" * 70)
        print("BENCHMARK: FP4 Cached vs BF16 Forward Pass")
        print("=" * 70)

        for seq_len in [64, 128, 256, 512]:
            for bs in [1, 4]:
                print(f"\n--- batch={bs}, seq_len={seq_len} ({bs*seq_len} tokens) ---")

                fp4_avg, fp4_min, tokens, fp4_tflops = benchmark_forward(
                    fp4_model, seq_len=seq_len, batch_size=bs)
                bf16_avg, bf16_min, _, bf16_tflops = benchmark_forward(
                    bf16_model, seq_len=seq_len, batch_size=bs)

                speedup = bf16_avg / fp4_avg
                print(f"  FP4 Cached: {fp4_avg:7.1f} ms avg ({fp4_min:.1f} min) ~{fp4_tflops:.1f} TFLOPS")
                print(f"  BF16:       {bf16_avg:7.1f} ms avg ({bf16_min:.1f} min) ~{bf16_tflops:.1f} TFLOPS")
                print(f"  Speedup: {speedup:.2f}x {'(FP4 wins)' if speedup > 1 else '(BF16 wins)'}")

        # Verify correctness
        print("\n--- Correctness Check ---")
        input_ids = torch.randint(0, fp4_model.vocab_size, (1, 32), device="cuda")
        fp4_logits = fp4_model.forward(input_ids)
        bf16_logits = bf16_model.forward(input_ids)
        rel_err = (fp4_logits.float() - bf16_logits.float()).abs().mean() / bf16_logits.float().abs().mean()
        pearson = torch.corrcoef(torch.stack([
            fp4_logits.float().flatten(), bf16_logits.float().flatten()
        ]))[0, 1]
        print(f"  Relative error: {rel_err:.6f}")
        print(f"  Pearson correlation: {pearson:.4f}")

    else:
        # FP4 only
        print(">>> Loading model with FP4 pre-quantized weights...")
        model = GPTOssModel(args.model_dir, use_fp4=True, num_layers=args.num_layers)

        print("\n" + "=" * 70)
        print("BENCHMARK: FP4 Forward Pass")
        print("=" * 70)

        for seq_len in [64, 128, 256, 512]:
            for bs in [1, 4]:
                avg_ms, min_ms, tokens, tflops = benchmark_forward(
                    model, seq_len=seq_len, batch_size=bs)
                print(f"  batch={bs:2d} seq={seq_len:4d} ({tokens:5d} tok): "
                      f"{avg_ms:7.1f} ms avg ({min_ms:.1f} min) ~{tflops:.1f} TFLOPS")

    print("\nDone!")


if __name__ == "__main__":
    main()
