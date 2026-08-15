"""
Derive a models.json entry for any MoE repo WITHOUT downloading weights.

Reads two small files from the Hub:
  config.json                  -> architecture shapes
  model.safetensors.index.json -> metadata.total_size, the exact byte count

Routed experts, shared experts, embeddings, lm_head and routers all have exact
closed forms from config. Attention does not -- it varies per architecture (MLA,
GQA, linear/hybrid). So attention is recovered as the RESIDUAL:

    attn = total_params - routed_pool - shared - embed - lm_head - router - dense_ffn

which is exact whenever total_params is known, and needs no per-arch formula.

  python scripts/derive_spec.py deepseek-ai/DeepSeek-V3
  python scripts/derive_spec.py openai/gpt-oss-120b --total-params 116800000000
  python scripts/derive_spec.py Qwen/Qwen3.8-2.4T-A95B --write qwen3.8-2.4t-a95b
"""

import argparse
import json
import os
import sys

DTYPE_BYTES = {"float32": 4, "bfloat16": 2, "float16": 2, "float8_e4m3fn": 1, "int8": 1}
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _hub_json(repo, path):
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id=repo, filename=path)
    except Exception as e:
        return None, str(e)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f), None


def _unwrap(cfg):
    """Multimodal configs bury the LLM under text_config (e.g. Qwen3.6-35B-A3B)."""
    if "text_config" in cfg and isinstance(cfg["text_config"], dict):
        merged = dict(cfg)
        merged.update(cfg["text_config"])
        return merged, True
    return cfg, False


def _get(cfg, *names, default=None):
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return default


def derive(repo, total_params=None, active_params=None):
    cfg, err = _hub_json(repo, "config.json")
    if cfg is None:
        return {"repo": repo, "error": f"config.json: {err}"}
    cfg, nested = _unwrap(cfg)

    H = _get(cfg, "hidden_size", "n_embd")
    L = _get(cfg, "num_hidden_layers", "n_layer")
    V = _get(cfg, "vocab_size")
    E = _get(cfg, "num_experts", "num_local_experts", "n_routed_experts", default=0)
    K = _get(cfg, "num_experts_per_tok", "experts_per_token", "num_experts_per_token", default=0)
    Mi = _get(cfg, "moe_intermediate_size", default=None)
    if Mi is None:
        # gpt_oss stores the per-expert width in plain intermediate_size
        Mi = _get(cfg, "intermediate_size", default=0)
    Si = _get(cfg, "shared_expert_intermediate_size", default=0)
    n_shared = _get(cfg, "n_shared_experts", default=1 if Si else 0)
    dense_ffn = _get(cfg, "intermediate_size", default=0)
    n_dense = _get(cfg, "first_k_dense_replace", "num_dense_layers", default=0)
    tie = bool(_get(cfg, "tie_word_embeddings", default=False))
    mtp = _get(cfg, "mtp_num_hidden_layers", "num_nextn_predict_layers", default=0)
    dtype = _get(cfg, "dtype", "torch_dtype", default="bfloat16")
    quant = _get(cfg, "quantization_config", default=None)

    if not E or not K:
        return {"repo": repo, "error": "not an MoE (no expert fields in config)"}

    # num_hidden_layers is NOT the MoE layer count on hybrid stacks. Nemotron-3.5
    # has 52 layers of which only 23 are MoE (23 mamba + 6 attention + 23 moe),
    # and no config field carries that count -- assuming L - n_dense overstates
    # the expert pool by 2.1x there. Prefer counting layer_types when present.
    layer_types = _get(cfg, "layer_types", default=None)
    moe_layers, layer_src = L - n_dense, "num_hidden_layers - dense"
    if isinstance(layer_types, list) and layer_types:
        moe_tagged = sum(1 for t in layer_types if "moe" in str(t).lower())
        if moe_tagged:
            moe_layers, layer_src = moe_tagged, "counted from layer_types"
        elif len(layer_types) != L:
            layer_src += f" (WARNING: layer_types has {len(layer_types)}, not {L})"
    per_expert = 3 * H * Mi                      # SwiGLU: gate, up, down
    pool = moe_layers * E * per_expert
    active_routed = moe_layers * K * per_expert
    shared = moe_layers * n_shared * 3 * H * Si if Si else 0
    dense = n_dense * 3 * H * dense_ffn if n_dense else 0
    embed = V * H
    lm_head = 0 if tie else V * H
    router = moe_layers * H * E

    # total_size is bytes. Only convertible to params when the checkpoint is a
    # single uniform dtype -- a quantized repo mixes widths and cannot be divided.
    src = "supplied"
    if total_params is None:
        idx, ierr = _hub_json(repo, "model.safetensors.index.json")
        if idx and not quant:
            tot_bytes = idx.get("metadata", {}).get("total_size")
            if tot_bytes:
                bpp = DTYPE_BYTES.get(str(dtype).replace("torch.", ""), 2)
                total_params = int(tot_bytes / bpp)
                src = f"index total_size / {bpp}B ({dtype})"
        elif quant:
            src = "UNKNOWN - repo is quantized, total_size is not params*const"

    known = pool + shared + dense + embed + lm_head + router
    attn_residual = (total_params - known) if total_params else None

    # The residual over-attributes whenever the checkpoint carries weights that
    # are NOT used in ordinary decoding -- most importantly an MTP layer, which
    # on Qwen models ships its own full expert block (512 x 50.3M = 25.8B on the
    # 2.4T). Those bytes are real on disk but never touched per token.
    #
    # The clean identity avoids the problem entirely:
    #
    #     hot = active_total - active_routed
    #
    # since "active" is by definition everything touched per token, and routed
    # experts are the only part of it that streams. Use the published active
    # count when available and keep the residual purely as a cross-check.
    attn = attn_residual

    hot_from_active = (active_params - active_routed) if active_params else None
    hot_from_residual = (attn + lm_head + router + shared + dense) if attn is not None else None
    hot = hot_from_active if hot_from_active is not None else hot_from_residual

    # attn is whatever is left of hot after the exactly-known pieces.
    if hot_from_active is not None:
        attn = hot_from_active - (lm_head + router + shared + dense)

    # A routed pool larger than the model is impossible; a pool at >99% of total
    # leaves no room for attention and means the layer count is wrong.
    warnings = []
    if total_params:
        ratio = pool / total_params
        if ratio > 1.0:
            warnings.append(f"IMPOSSIBLE: routed pool is {ratio:.2f}x total params -- "
                            f"moe_layers={moe_layers} ({layer_src}) is almost certainly wrong")
        elif ratio > 0.99:
            warnings.append(f"SUSPECT: routed pool is {ratio:.1%} of total, leaving "
                            f"{(total_params-pool)/1e9:.2f}B for everything else")
    if hot is not None and hot <= 0:
        warnings.append(f"IMPOSSIBLE: hot resolved to {hot/1e9:.2f}B")

    check = None
    if hot_from_active is not None and hot_from_residual is not None:
        gap = hot_from_residual - hot_from_active
        check = {
            "residual_hot": hot_from_residual,
            "active_hot": hot_from_active,
            "gap": gap,
            "gap_explained_by_mtp": bool(mtp) and gap > 0.5 * E * per_expert,
        }

    return {
        "repo": repo, "nested_text_config": nested, "error": None,
        "hidden_size": H, "num_layers": L, "num_moe_layers": moe_layers,
        "num_experts": E, "experts_per_tok": K, "moe_intermediate_size": Mi,
        "expert_matrices": 3, "mtp_layers": mtp, "quantized": bool(quant),
        "total_params": total_params, "total_params_source": src,
        "active_params": active_params,
        "params": {
            "attn": attn, "lm_head": lm_head, "router": router,
            "shared_expert": shared + dense, "embed_gather_only": embed,
            "routed_expert_pool": pool,
        },
        "active_routed": active_routed,
        "hot": hot,
        "hot_source": "active - active_routed" if hot_from_active is not None else "residual",
        "cross_check": check,
        "moe_layer_source": layer_src,
        "warnings": warnings,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--total-params", type=int, default=None)
    ap.add_argument("--active-params", type=int, default=None,
                    help="published activated-parameter count; makes hot exact")
    ap.add_argument("--write", default=None, help="key to upsert into minillm/models.json")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", r"D:\minillm\hf" if sys.platform == "win32"
                          else os.path.expanduser("~/minillm/hf"))
    single = len(args.repos) == 1
    out = {}
    for repo in args.repos:
        d = derive(repo,
                   args.total_params if single else None,
                   args.active_params if single else None)
        out[repo] = d
        if d.get("error"):
            print(f"\n{repo}\n  ERROR: {d['error']}")
            continue
        b = lambda x: f"{x/1e9:9.2f}B" if x is not None else "        ?"
        print(f"\n{repo}")
        for w in d.get("warnings", []):
            print(f"  !! {w}")
        print(f"  layers {d['num_layers']} ({d['num_moe_layers']} MoE, {d['moe_layer_source']}) "
              f"| hidden {d['hidden_size']} "
              f"| {d['num_experts']} experts, top-{d['experts_per_tok']} | moe_inter {d['moe_intermediate_size']}")
        print(f"  total          {b(d['total_params'])}   [{d['total_params_source']}]")
        print(f"  routed pool    {b(d['params']['routed_expert_pool'])}")
        print(f"  active routed  {b(d['active_routed'])}  <- streams per token")
        print(f"  attn(residual) {b(d['params']['attn'])}")
        print(f"  shared+dense   {b(d['params']['shared_expert'])}")
        print(f"  lm_head        {b(d['params']['lm_head'])}")
        print(f"  HOT (resident) {b(d['hot'])}   [{d['hot_source']}]")
        if d["mtp_layers"]:
            print(f"  MTP head       yes ({d['mtp_layers']} layer) -> native speculative decoding")
        c = d.get("cross_check")
        if c:
            print(f"  cross-check    residual {b(c['residual_hot'])} vs active-derived "
                  f"{b(c['active_hot'])}, gap {b(c['gap'])}")
            if c["gap_explained_by_mtp"]:
                print(f"                 gap ~= one MTP layer's expert block -> "
                      f"checkpoint carries weights never touched per token")

    if args.write and len(args.repos) == 1:
        d = out[args.repos[0]]
        if d.get("error") or d["hot"] is None:
            print("\nnot written: incomplete derivation")
            return 1
        path = os.path.join(HERE, "minillm", "models.json")
        with open(path, "r", encoding="utf-8") as f:
            models = json.load(f)
        models[args.write] = {
            "repo": d["repo"], "source": "derive_spec.py",
            "num_layers": d["num_layers"], "num_moe_layers": d["num_moe_layers"],
            "hidden_size": d["hidden_size"], "num_experts": d["num_experts"],
            "experts_per_tok": d["experts_per_tok"],
            "moe_intermediate_size": d["moe_intermediate_size"],
            "expert_matrices": 3, "mtp_layers": d["mtp_layers"],
            "params": {k: int(v) for k, v in d["params"].items()},
            "notes": f"derived; total from {d['total_params_source']}",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(models, f, indent=2)
        print(f"\nwrote '{args.write}' to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
