"""patches.py — corrections to upstream code, applied explicitly and verifiably.

Nothing here is clever. Each patch documents the exact upstream defect, why our configuration
triggers it, and what the fix changes. Call `apply_all()` once, before the model is built.

Run `python training/patches.py` to test the patches on CPU without downloading any weights.
"""
from __future__ import annotations

APPLIED: list[str] = []


def patch_deepseek_v3_moe_dtype() -> str:
    """transformers 4.57.x — DeepseekV3MoE.moe builds its accumulator in the wrong dtype.

    UPSTREAM (modeling_deepseek_v3.py, ~line 180):

        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        ...
        weighted_output = expert_output * expert_weights.unsqueeze(-1)
        final_hidden_states.index_add_(0, token_indices, weighted_output)
        ...
        return final_hidden_states.type(hidden_states.dtype)

    The accumulator takes `topk_weights.dtype`, but what is added to it follows
    `hidden_states.dtype` — bitsandbytes Linear4bit returns its INPUT dtype. The final line
    shows the author considered hidden_states.dtype canonical, so line 180 is simply
    inconsistent with line 193.

    In most setups both are fp16 and nothing goes wrong. QLoRA on a pre-Ampere card diverges:

      * prepare_model_for_kbit_training upcasts every non-quantised parameter to fp32,
        layernorms included
      * autocast keeps layer_norm in fp32     -> hidden_states  = Float
      * autocast casts the router's linear    -> topk_weights   = Half
      * Linear4bit preserves input dtype      -> expert_output  = Float

        RuntimeError: index_add_(): self (Half) and source (Float)
                      must have the same scalar type

    FIX: accumulate in promote_types(hidden_states, topk_weights) — the dtype the product
    actually has — and cast the addend to match. The return cast is untouched, so the module's
    output dtype is exactly what it was before. Accumulating in the wider dtype is also the
    numerically better choice across 64 routed experts, and costs one [tokens, hidden] fp32
    buffer: about 8MB at seq 1024.
    """
    import torch
    from transformers.models.deepseek_v3 import modeling_deepseek_v3 as m

    def moe(self, hidden_states, topk_indices, topk_weights):
        acc_dtype = torch.promote_types(hidden_states.dtype, topk_weights.dtype)
        final_hidden_states = torch.zeros_like(hidden_states, dtype=acc_dtype)
        expert_mask = torch.nn.functional.one_hot(topk_indices, num_classes=len(self.experts))
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(len(self.experts)):
            expert = self.experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output.to(acc_dtype))

        return final_hidden_states.type(hidden_states.dtype)

    m.DeepseekV3MoE.moe = moe
    return "deepseek_v3_moe_dtype"


def apply_all(verbose: bool = True) -> list[str]:
    """Apply every patch. Safe to call more than once."""
    if APPLIED:
        return APPLIED
    APPLIED.append(patch_deepseek_v3_moe_dtype())
    if verbose:
        import transformers
        print(f"patches applied to transformers {transformers.__version__}: {APPLIED}")
        print("  deepseek_v3_moe_dtype — upstream builds the MoE accumulator in the router's")
        print("  dtype but adds expert outputs in the hidden-state dtype; they differ under")
        print("  QLoRA + fp16 autocast. See training/patches.py for the full derivation.")
    return APPLIED


# -------------------------------------------------------------------------------------------
# TEST — reproduces the failure on CPU, then shows the patch fixes it. No download, no GPU.
# -------------------------------------------------------------------------------------------
def _test() -> int:
    import torch
    from transformers.models.deepseek_v3 import modeling_deepseek_v3 as m
    from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

    print("=" * 72)
    print("PATCH TEST — dtype mismatch in DeepseekV3MoE.moe")
    print("=" * 72)

    cfg = DeepseekV3Config(
        hidden_size=64, intermediate_size=128, moe_intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2,
        first_k_dense_replace=0, vocab_size=128,
    )
    torch.manual_seed(0)
    moe_mod = m.DeepseekV3MoE(cfg)

    # The exact condition QLoRA produces: fp32 hidden states, fp16 router weights.
    hidden = torch.randn(8, 64, dtype=torch.float32)
    idx = torch.randint(0, 4, (8, 2))
    w = torch.rand(8, 2, dtype=torch.float16)

    original = m.DeepseekV3MoE.moe
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # 1. Unpatched: must fail, otherwise the patch is unnecessary and should be deleted.
    try:
        original(moe_mod, hidden, idx, w)
        check("unpatched code fails as expected", False,
              "it did NOT fail — upstream may be fixed; re-check whether this patch is needed")
    except RuntimeError as e:
        ok = "same scalar type" in str(e)
        check("unpatched code fails as expected", ok, str(e)[:72])

    # 2. Patched: must succeed and preserve the output dtype/shape.
    apply_all(verbose=False)
    try:
        out = m.DeepseekV3MoE.moe(moe_mod, hidden, idx, w)
        check("patched code runs", True)
        check("output dtype follows hidden_states", out.dtype == hidden.dtype, str(out.dtype))
        check("output shape unchanged", out.shape == hidden.shape, str(tuple(out.shape)))
        check("output is finite", bool(torch.isfinite(out).all()))
    except Exception as e:                                       # noqa: BLE001
        check("patched code runs", False, f"{type(e).__name__}: {e}")

    # 3. The uniform-dtype case must be untouched — the patch must not change normal behaviour.
    #    Compare in fp32 on both sides: the module's own weights are fp32, and feeding it fp16
    #    activations would fail inside the expert matmul for reasons unrelated to this patch.
    w32 = w.float()
    try:
        a = original(moe_mod, hidden, idx, w32)
        b = m.DeepseekV3MoE.moe(moe_mod, hidden, idx, w32)
        same = torch.equal(a, b)
        check("identical to upstream when dtypes already agree", same,
              "patch is a no-op in the normal case" if same else "PATCH CHANGED BEHAVIOUR")
    except Exception as e:                                       # noqa: BLE001
        check("identical to upstream when dtypes already agree", False, str(e)[:72])

    print("\n" + "=" * 72)
    print("PATCH TEST FAILED: " + str(fails) if fails else "PATCH TEST PASSED")
    return 1 if fails else 0


if __name__ == "__main__":
    import sys
    sys.exit(_test())
