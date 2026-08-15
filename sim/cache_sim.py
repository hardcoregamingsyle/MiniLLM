"""
Expert-cache policy simulator.

The whole project rests on one empirical claim: a routing-aware cache beats the
OS page cache. This simulates that claim before any runtime exists.

The decisive number is not what any clever policy achieves -- it is what BELADY
achieves. Belady is the optimal offline policy (evict whatever is needed
furthest in the future); no online policy can beat it. So:

    if Belady at this residency < required hit rate  ->  the target is
    UNREACHABLE by any caching strategy, and no amount of engineering fixes it.

That makes Belady the cheapest possible kill-shot for the thesis, and it runs
in seconds on a trace.

Trace format: .npz with array `experts` of shape [n_tokens, n_layers, top_k],
dtype int16/int32, holding routed expert ids. Produced by trace/ (real) or
--synthetic (parameterized locality).

  python sim/cache_sim.py --synthetic --model gpt-oss-120b --machine laptop-i5-8350u
  python sim/cache_sim.py --trace traces/gpt-oss-120b.npz --model gpt-oss-120b
  python sim/cache_sim.py --synthetic --sweep-locality
"""

import argparse
import json
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from minillm.capacity import BPW, Machine, Model, load, GB  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic traces: tunable locality so we can ask "how much would we NEED?"
# ---------------------------------------------------------------------------

def synth_trace(n_tokens, n_layers, n_experts, top_k, zipf_s=1.1,
                stickiness=0.7, seed=0):
    """Generate routing with controllable skew and temporal reuse.

    zipf_s     : popularity skew across experts. Higher = a few hot experts.
    stickiness : P(an expert from the previous token's set is reused). This is
                 the parameter that actually drives cache hit rate, because the
                 cache only holds ~1 token's working set.
    """
    rng = np.random.default_rng(seed)
    ranks = np.arange(1, n_experts + 1)
    p = 1.0 / ranks ** zipf_s
    p /= p.sum()

    out = np.zeros((n_tokens, n_layers, top_k), dtype=np.int32)
    # Each layer gets its own permutation, so "expert 0" is not hot everywhere.
    perms = [rng.permutation(n_experts) for _ in range(n_layers)]

    prev = [None] * n_layers
    for t in range(n_tokens):
        for l in range(n_layers):
            if prev[l] is None:
                pick = rng.choice(n_experts, size=top_k, replace=False, p=p)
            else:
                n_keep = rng.binomial(top_k, stickiness)
                keep = rng.choice(prev[l], size=min(n_keep, top_k), replace=False)
                need = top_k - len(keep)
                fresh = []
                while len(fresh) < need:
                    c = rng.choice(n_experts, p=p)
                    if c not in keep and c not in fresh:
                        fresh.append(c)
                pick = np.concatenate([keep, np.array(fresh, dtype=np.int64)])
            prev[l] = pick
            out[t, l] = perms[l][pick.astype(int)]
    return out


# ---------------------------------------------------------------------------
# Policies. Each returns the number of MISSES over the trace.
# Cache entries are keyed (layer, expert); every entry is one expert's bytes.
# ---------------------------------------------------------------------------

def policy_lru(trace, capacity):
    cache = OrderedDict()
    misses = hits = 0
    for t in range(trace.shape[0]):
        for l in range(trace.shape[1]):
            for e in trace[t, l]:
                k = (l, int(e))
                if k in cache:
                    cache.move_to_end(k)
                    hits += 1
                else:
                    misses += 1
                    cache[k] = True
                    if len(cache) > capacity:
                        cache.popitem(last=False)
    return hits, misses


def policy_lfu(trace, capacity):
    cache, freq = set(), defaultdict(int)
    misses = hits = 0
    for t in range(trace.shape[0]):
        for l in range(trace.shape[1]):
            for e in trace[t, l]:
                k = (l, int(e))
                freq[k] += 1
                if k in cache:
                    hits += 1
                else:
                    misses += 1
                    if len(cache) >= capacity:
                        victim = min(cache, key=lambda x: freq[x])
                        cache.discard(victim)
                    cache.add(k)
    return hits, misses


def policy_belady(trace, capacity):
    """Optimal offline. Upper bound for ANY policy, online or not."""
    T, L, K = trace.shape
    flat = [(l, int(e)) for t in range(T) for l in range(L) for e in trace[t, l]]

    # next_use[i] = index of the next reference to the same key after i
    nxt = [len(flat)] * len(flat)
    last = {}
    for i in range(len(flat) - 1, -1, -1):
        k = flat[i]
        nxt[i] = last.get(k, len(flat))
        last[k] = i

    cache, cache_next = set(), {}
    misses = hits = 0
    for i, k in enumerate(flat):
        if k in cache:
            hits += 1
        else:
            misses += 1
            if len(cache) >= capacity:
                victim = max(cache, key=lambda x: cache_next.get(x, len(flat)))
                cache.discard(victim)
                cache_next.pop(victim, None)
            cache.add(k)
        cache_next[k] = nxt[i]
    return hits, misses


def policy_layer_aware(trace, capacity):
    """Partition the cache evenly per layer.

    Rationale: a global LRU lets a burst in layer 3 evict layer 30's working
    set, but every layer is visited exactly once per token, so no layer should
    ever starve another. Same total bytes, different allocation.
    """
    L = trace.shape[1]
    per = max(1, capacity // L)
    caches = [OrderedDict() for _ in range(L)]
    misses = hits = 0
    for t in range(trace.shape[0]):
        for l in range(L):
            c = caches[l]
            for e in trace[t, l]:
                k = int(e)
                if k in c:
                    c.move_to_end(k)
                    hits += 1
                else:
                    misses += 1
                    c[k] = True
                    if len(c) > per:
                        c.popitem(last=False)
    return hits, misses


POLICIES = {
    "LRU (page-cache analogue)": policy_lru,
    "LFU": policy_lfu,
    "layer-partitioned LRU": policy_layer_aware,
    "Belady (optimal, ceiling)": policy_belady,
}


# ---------------------------------------------------------------------------

def evaluate(trace, model, machine, bits_hot, bits_exp):
    per_expert_bytes = model.params_per_expert * BPW[bits_exp] / 8
    hot_gb = model.hot * BPW[bits_hot] / 8 / GB
    cache_gb = machine.weight_budget_gb - hot_gb
    capacity = int(cache_gb * GB / per_expert_bytes)

    T, L, K = trace.shape
    refs_per_token = L * K
    active_gb = refs_per_token * per_expert_bytes / GB

    print(f"  expert size      : {per_expert_bytes/1e6:.2f} MB at {bits_exp}")
    print(f"  hot set          : {hot_gb:.2f} GB at {bits_hot}")
    print(f"  cache            : {cache_gb:.2f} GB = {capacity} experts")
    print(f"  per-token demand : {refs_per_token} experts = {active_gb:.2f} GB")
    print(f"  cache holds      : {capacity/refs_per_token:.2f} tokens of working set")
    if capacity < refs_per_token:
        print(f"  !! cache is SMALLER than one token's working set -- thrashing floor")
    print()

    tgt = machine.targets.get("aim_tok_s", 1.0)
    budget_s = 1.0 / tgt

    print(f"  {'policy':<28}{'hit rate':>10}{'miss GB/tok':>13}{'s/token':>10}{'tok/s':>9}")
    print("  " + "-" * 70)
    results = {}
    for name, fn in POLICIES.items():
        hits, misses = fn(trace, capacity)
        total = hits + misses
        hr = hits / total if total else 0.0
        miss_gb = (misses / T) * per_expert_bytes / GB
        hit_gb = (hits / T) * per_expert_bytes / GB
        t_ram = (hot_gb + hit_gb) / machine.dram
        t_disk = miss_gb / machine.disk
        t_kern = (hot_gb + active_gb) / machine.kernel
        t = max(t_ram, t_disk, t_kern)
        results[name] = {"hit_rate": hr, "miss_gb": miss_gb, "s_per_token": t,
                         "tok_s": 1 / t}
        flag = "  <-- ceiling" if "Belady" in name else ""
        print(f"  {name:<28}{hr:>9.1%}{miss_gb:>13.3f}{t:>10.2f}{1/t:>9.2f}{flag}")

    ceil = results["Belady (optimal, ceiling)"]
    print()
    if ceil["tok_s"] < tgt:
        print(f"  VERDICT: even OPTIMAL caching gives {ceil['tok_s']:.2f} tok/s, below the")
        print(f"           {tgt} tok/s aim. No online policy can fix this -- the")
        print(f"           constraint is capacity or bandwidth, not policy.")
    else:
        best_online = max((v["tok_s"], k) for k, v in results.items() if "Belady" not in k)
        gap = ceil["tok_s"] - best_online[0]
        print(f"  VERDICT: optimal reaches {ceil['tok_s']:.2f} tok/s; best online "
              f"({best_online[1]}) reaches {best_online[0]:.2f}.")
        print(f"           Headroom for a smarter policy: {gap:.2f} tok/s.")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--machine", default="laptop-i5-8350u")
    ap.add_argument("--bits-hot", default="q8_0", choices=list(BPW))
    ap.add_argument("--bits-expert", default="mxfp4", choices=list(BPW))
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--stickiness", type=float, default=0.7)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--sweep-locality", action="store_true")
    args = ap.parse_args()

    models, machines = load("models.json"), load("machines.json")
    model = Model(args.model, models[args.model])
    machine = Machine(args.machine, machines[args.machine])

    if args.trace:
        trace = np.load(args.trace)["experts"]
        src = f"real trace {args.trace}"
    elif args.synthetic or args.sweep_locality:
        trace = synth_trace(args.tokens, model.moe_layers, model.n_experts,
                            model.top_k, args.zipf, args.stickiness)
        src = (f"SYNTHETIC (zipf={args.zipf}, stickiness={args.stickiness}) "
               f"-- calibrate against a real trace before trusting")
    else:
        print("need --trace or --synthetic")
        return 1

    print("=" * 74)
    print(f"{args.model} on {machines[args.machine]['label']}")
    print(f"trace: {src}")
    print(f"shape: {trace.shape[0]} tokens x {trace.shape[1]} layers x {trace.shape[2]} experts")
    print("=" * 74 + "\n")

    if args.sweep_locality:
        print("How much consecutive-token expert reuse would be REQUIRED?\n")
        print(f"  {'stickiness':>11}{'LRU hit':>10}{'Belady':>9}{'LRU tok/s':>11}{'Belady tok/s':>14}")
        print("  " + "-" * 56)
        for s in (0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            tr = synth_trace(args.tokens, model.moe_layers, model.n_experts,
                             model.top_k, args.zipf, s)
            pe = model.params_per_expert * BPW[args.bits_expert] / 8
            hot_gb = model.hot * BPW[args.bits_hot] / 8 / GB
            cap = int((machine.weight_budget_gb - hot_gb) * GB / pe)
            T = tr.shape[0]
            out = {}
            for nm, fn in (("lru", policy_lru), ("bel", policy_belady)):
                h, m = fn(tr, cap)
                hr = h / (h + m)
                miss_gb = (m / T) * pe / GB
                hit_gb = (h / T) * pe / GB
                act_gb = tr.shape[1] * tr.shape[2] * pe / GB
                t = max((hot_gb + hit_gb) / machine.dram, miss_gb / machine.disk,
                        (hot_gb + act_gb) / machine.kernel)
                out[nm] = (hr, 1 / t)
            print(f"  {s:>11.2f}{out['lru'][0]:>10.1%}{out['bel'][0]:>9.1%}"
                  f"{out['lru'][1]:>11.2f}{out['bel'][1]:>14.2f}")
        print("\n  Read this as: what expert reuse rate must real routing exhibit")
        print("  for the target to be reachable. Then measure the real number.")
        return 0

    evaluate(trace, model, machine, args.bits_hot, args.bits_expert)
    return 0


if __name__ == "__main__":
    sys.exit(main())
