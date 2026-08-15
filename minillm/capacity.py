"""
Capacity planner for MoE inference on machines smaller than the model.

Answers three questions for any (machine, model, quantization) triple:
  - does it fit on disk at all?
  - what expert-cache hit rate is required to hit a tokens/sec target?
  - which resource is actually binding -- DRAM, disk, or the SIMD kernel?

Adding a machine means adding an entry to machines.json. Nothing here is
specific to one box, which is the point: the laptop and the server are the
same calculation with different constants.

  python -m minillm.capacity report
  python -m minillm.capacity report --model gpt-oss-120b --machine laptop-i5-8350u
  python -m minillm.capacity solve --model gpt-oss-120b --machine laptop-i5-8350u
  python -m minillm.capacity frontier --machine server-32gb-nvme
"""

import argparse
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
GB = 1024.0 ** 3

# Effective bits per weight INCLUDING block scales and zero-points. These are
# what you actually get on disk, not the nominal name. Q4_K_M is not 4.0.
BPW = {
    "iq1_s": 1.56,
    "iq2_xxs": 2.06,
    "q2_k": 2.63,
    "iq3_xxs": 3.06,
    "q3_k_m": 3.44,
    "mxfp4": 4.25,
    "q4_k_m": 4.53,
    "q5_k_m": 5.53,
    "q6_k": 6.56,
    "q8_0": 8.50,
    "fp8": 8.00,
    "bf16": 16.0,
}


def load(name):
    with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
        d = json.load(f)
    return {k: v for k, v in d.items() if not k.startswith("_")}


class Model:
    def __init__(self, key, d):
        self.key = key
        self.d = d
        p = d["params"]
        self.attn = p["attn"]
        self.lm_head = p["lm_head"]
        self.router = p["router"]
        self.shared = p["shared_expert"]
        self.embed = p["embed_gather_only"]
        self.pool = p["routed_expert_pool"]
        self.n_experts = d["num_experts"]
        self.top_k = d["experts_per_tok"]
        self.moe_layers = d["num_moe_layers"]
        self.mtp = d.get("mtp_layers", 0)

    @property
    def params_per_expert(self):
        return self.d["expert_matrices"] * self.d["hidden_size"] * self.d["moe_intermediate_size"]

    @property
    def hot(self):
        """Touched every token, never evictable. Embeddings excluded -- gather."""
        return self.attn + self.lm_head + self.router + self.shared

    @property
    def active_routed(self):
        return self.moe_layers * self.top_k * self.params_per_expert

    @property
    def total(self):
        return self.hot + self.embed + self.pool

    def union_factor(self, k, correlation=0.5):
        """Distinct experts touched over k tokens, as a multiple of one token's.

        Independent-uniform upper bound, discounted by routing correlation --
        consecutive tokens reuse experts, so the true union is smaller.
        correlation=0 means independent, 1 means every token routes identically.
        """
        if k <= 1:
            return 1.0
        e, E = self.top_k, self.n_experts
        indep = E * (1.0 - (1.0 - e / E) ** k) / e
        return 1.0 + (indep - 1.0) * (1.0 - correlation)


class Machine:
    def __init__(self, key, d):
        self.key = key
        self.d = d
        self.dram = d["dram_gbps"]
        self.disk = d["disk_gbps"]
        self.kernel = d["kernel_gbps"]
        self.free_disk = d["free_disk_gb"]
        self.targets = d.get("targets", {})

    @property
    def weight_budget_gb(self):
        return self.d["ram_total_gb"] - self.d["os_reserve_gb"] - self.d["runtime_reserve_gb"]


def plan(machine, model, bits_hot="q8_0", bits_expert="mxfp4",
         hit_rate=0.70, k=1, accept=0.0, correlation=0.5):
    """Predict seconds/token. I/O is assumed overlapped with compute."""
    bh, be = BPW[bits_hot], BPW[bits_expert]

    hot_gb = model.hot * bh / 8 / GB
    embed_gb = model.embed * be / 8 / GB
    pool_gb = model.pool * be / 8 / GB
    storage_gb = hot_gb + embed_gb + pool_gb

    cache_gb = machine.weight_budget_gb - hot_gb
    active_exp_gb = model.active_routed * be / 8 / GB

    uf = model.union_factor(k, correlation)
    sweep_exp_gb = active_exp_gb * uf

    # The cache serves at most what it holds, and at most what locality allows.
    hit_gb = max(0.0, min(cache_gb, sweep_exp_gb * hit_rate))
    miss_gb = max(0.0, sweep_exp_gb - hit_gb)

    t_ram = (hot_gb + hit_gb) / machine.dram
    t_disk = miss_gb / machine.disk
    t_kernel = (hot_gb + sweep_exp_gb) / machine.kernel
    t_sweep = max(t_ram, t_disk, t_kernel)

    tokens_per_sweep = 1.0 + (k - 1) * accept
    t_token = t_sweep / tokens_per_sweep

    binding = max(
        (("dram", t_ram), ("disk", t_disk), ("kernel", t_kernel)),
        key=lambda x: x[1],
    )[0]

    return {
        "fits_disk": storage_gb <= machine.free_disk,
        "storage_gb": storage_gb,
        "hot_gb": hot_gb,
        "pool_gb": pool_gb,
        "cache_gb": cache_gb,
        "cache_ok": cache_gb > 0,
        "active_exp_gb": active_exp_gb,
        "sweep_exp_gb": sweep_exp_gb,
        "union_factor": uf,
        "hit_gb": hit_gb,
        "miss_gb": miss_gb,
        "capacity_ceiling": min(1.0, cache_gb / sweep_exp_gb) if sweep_exp_gb > 0 else 1.0,
        "t_ram": t_ram, "t_disk": t_disk, "t_kernel": t_kernel,
        "t_token": t_token, "tok_s": 1.0 / t_token if t_token > 0 else float("inf"),
        "binding": binding,
        "tokens_per_sweep": tokens_per_sweep,
    }


def required_hit_rate(machine, model, target_tok_s, bits_hot, bits_expert,
                      k=1, accept=0.0, correlation=0.5):
    """Smallest hit rate reaching target_tok_s. None if unreachable at any rate."""
    lo, hi = 0.0, 1.0
    best = plan(machine, model, bits_hot, bits_expert, 1.0, k, accept, correlation)
    if best["tok_s"] < target_tok_s or not best["cache_ok"]:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2
        if plan(machine, model, bits_hot, bits_expert, mid, k, accept, correlation)["tok_s"] >= target_tok_s:
            hi = mid
        else:
            lo = mid
    return hi


PLANS = [
    ("bf16", "bf16"), ("q8_0", "q8_0"), ("q8_0", "q4_k_m"), ("q8_0", "mxfp4"),
    ("q8_0", "q3_k_m"), ("q8_0", "q2_k"), ("q6_k", "q2_k"), ("q4_k_m", "q2_k"),
    ("q4_k_m", "iq2_xxs"), ("q4_k_m", "iq1_s"),
    # Sub-4-bit hot sets. Quantizing attention this hard is quality-destructive
    # and is a last resort, but for models whose hot half alone exceeds RAM it is
    # the only thing that opens any headroom for an expert cache at all.
    ("q3_k_m", "q2_k"), ("q3_k_m", "iq2_xxs"),
    ("q2_k", "q2_k"), ("q2_k", "iq2_xxs"), ("q2_k", "iq1_s"),
]


def cmd_report(args):
    machines, models = load("machines.json"), load("models.json")
    mkeys = [args.machine] if args.machine else list(machines)
    okeys = [args.model] if args.model else list(models)

    for mk in mkeys:
        M = Machine(mk, machines[mk])
        print("=" * 96)
        print(f"{machines[mk]['label']}")
        print(f"  DRAM {M.dram:.1f} GB/s | disk {M.disk:.2f} GB/s | kernel {M.kernel:.0f} GB/s "
              f"| weights {M.weight_budget_gb:.1f} GB | disk free {M.free_disk:.0f} GB")
        t = M.targets
        print(f"  targets: aim {t.get('aim_tok_s','?')} tok/s, floor {t.get('floor_tok_s','?')} tok/s")
        print("=" * 96)

        for ok in okeys:
            mo = Model(ok, models[ok])
            print(f"\n  {ok}  --  hot {mo.hot/1e9:.2f}B | pool {mo.pool/1e9:.1f}B | "
                  f"active-routed {mo.active_routed/1e9:.2f}B | total {mo.total/1e9:.1f}B")
            print(f"  {'hot':>7} {'expert':>9} {'disk GB':>9} {'hot GB':>8} {'cache':>7} "
                  f"{'need':>7} {'miss':>7} {'tok/s':>7} {'bind':>7}")
            print("  " + "-" * 78)
            for bh, be in PLANS:
                r = plan(M, mo, bh, be, args.hit)
                fit = " " if r["fits_disk"] and r["cache_ok"] else "X"
                print(f"{fit} {bh:>7} {be:>9} {r['storage_gb']:9.1f} {r['hot_gb']:8.2f} "
                      f"{r['cache_gb']:7.2f} {r['active_exp_gb']:7.2f} {r['miss_gb']:7.2f} "
                      f"{r['tok_s']:7.2f} {r['binding']:>7}")
            print(f"  (X = does not fit disk or hot set exceeds RAM; hit rate assumed {args.hit:.0%})")


def cmd_solve(args):
    machines, models = load("machines.json"), load("models.json")
    M = Machine(args.machine, machines[args.machine])
    mo = Model(args.model, models[args.model])
    tgt = args.target or M.targets.get("aim_tok_s", 1.0)

    print(f"{machines[args.machine]['label']}")
    print(f"model {args.model}, target {tgt} tok/s\n")
    print(f"  {'hot':>7} {'expert':>9} {'spec-k':>7} {'need hit':>9} {'ceiling':>8} {'verdict'}")
    print("  " + "-" * 62)

    for bh, be in PLANS:
        base = plan(M, mo, bh, be, 1.0)
        if not base["fits_disk"]:
            print(f"  {bh:>7} {be:>9} {'-':>7} {'-':>9} {'-':>8} disk: needs "
                  f"{base['storage_gb']:.0f} GB > {M.free_disk:.0f} GB")
            continue
        if not base["cache_ok"]:
            print(f"  {bh:>7} {be:>9} {'-':>7} {'-':>9} {'-':>8} hot set "
                  f"{base['hot_gb']:.1f} GB exceeds {M.weight_budget_gb:.1f} GB RAM")
            continue
        for k, acc in ((1, 0.0), (3, 0.6)):
            if k > 1 and mo.mtp == 0 and not args.assume_draft:
                continue
            need = required_hit_rate(M, mo, tgt, bh, be, k, acc)
            ceil = plan(M, mo, bh, be, 1.0, k, acc)["capacity_ceiling"]
            if need is None:
                v = f"unreachable (best {plan(M, mo, bh, be, 1.0, k, acc)['tok_s']:.2f} tok/s)"
                ns = "-"
            else:
                ns = f"{need:.1%}"
                v = "plausible" if need <= 0.85 else "needs exceptional locality"
            print(f"  {bh:>7} {be:>9} {k:>7} {ns:>9} {ceil:>7.0%}  {v}")


def cmd_frontier(args):
    """Largest total-parameter model reachable, as a function of sparsity."""
    machines = load("machines.json")
    M = Machine(args.machine, machines[args.machine])
    tgt = args.target or M.targets.get("floor_tok_s", 0.5)
    be = args.expert_bits
    bpw = BPW[be]

    print(f"{machines[args.machine]['label']}")
    print(f"experts at {be} ({bpw} bpw), target {tgt} tok/s, hot set assumed 1.5 GB\n")
    print(f"  {'active-routed B':>16} {'active GB':>10} {'cacheable':>10} "
          f"{'miss GB':>9} {'best tok/s':>11}  verdict")
    print("  " + "-" * 76)

    hot_gb = 1.5
    cache_gb = M.weight_budget_gb - hot_gb
    budget_s = 1.0 / tgt

    for active_b in (1, 2, 3.58, 5, 8, 12, 17, 25, 46.3, 95):
        act_gb = active_b * 1e9 * bpw / 8 / GB
        hit_gb = min(cache_gb, act_gb)          # perfect-locality upper bound
        miss_gb = max(0.0, act_gb - hit_gb)
        t = max((hot_gb + hit_gb) / M.dram, miss_gb / M.disk, (hot_gb + act_gb) / M.kernel)
        best = 1.0 / t
        ok = best >= tgt
        print(f"{' ' if ok else 'X'} {active_b:>16.2f} {act_gb:>10.2f} "
              f"{hit_gb/act_gb:>9.0%} {miss_gb:>9.2f} {best:>11.2f}  "
              f"{'ok' if ok else 'cannot reach target'}")

    print(f"\n  Rows assume a PERFECT cache, so these are ceilings -- a real policy lands below.")
    print(f"  X = cannot reach {tgt} tok/s even with perfect locality.")
    print(f"  Disk ceiling: {M.free_disk:.0f} GB at {bpw} bpw holds "
          f"{M.free_disk*GB*8/bpw/1e9:.0f}B total parameters.")
    print(f"  So the model must satisfy BOTH: total <= "
          f"{M.free_disk*GB*8/bpw/1e9:.0f}B AND active-routed small enough above.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("report")
    r.add_argument("--machine"); r.add_argument("--model")
    r.add_argument("--hit", type=float, default=0.70)
    r.set_defaults(fn=cmd_report)

    s = sub.add_parser("solve")
    s.add_argument("--machine", required=True); s.add_argument("--model", required=True)
    s.add_argument("--target", type=float); s.add_argument("--assume-draft", action="store_true")
    s.set_defaults(fn=cmd_solve)

    f = sub.add_parser("frontier")
    f.add_argument("--machine", required=True); f.add_argument("--target", type=float)
    f.add_argument("--expert-bits", default="q2_k", choices=list(BPW))
    f.set_defaults(fn=cmd_frontier)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
