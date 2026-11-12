"""Shape-derived pi0.5 inference graph; timing/traffic only, not model execution."""

from dataclasses import dataclass, asdict

from .workload import Region, positive

OPENPI_REVISION = "215abfb217dbac7d5f1273282331b9b1866c0479"


@dataclass(frozen=True)
class Shape:
    images: int = 3
    image_tokens: int = 256
    text_tokens: int = 200
    vocab_size: int = 257152
    patch_features: int = 588
    vision_width: int = 1152
    vision_mlp: int = 4304
    vision_layers: int = 27
    vision_heads: int = 16
    prefix_width: int = 2048
    prefix_mlp: int = 16384
    layers: int = 18
    expert_width: int = 1024
    expert_mlp: int = 4096
    heads: int = 8
    head_dim: int = 256
    action_tokens: int = 10
    action_dim: int = 32
    steps: int = 10

    @property
    def prefix_tokens(self):
        return self.images * self.image_tokens + self.text_tokens

    @classmethod
    def parse(cls, overrides):
        shape = cls(**overrides)
        for name, value in asdict(shape).items():
            positive(value, f"shape.{name}")
        if shape.vision_width % shape.vision_heads:
            raise ValueError("vision_width must be divisible by vision_heads")
        return shape


@dataclass(eq=False)
class Tensor:
    name: str
    rows: int
    cols: int
    element: int = 2
    base: int | None = None
    weight: bool = False

    @property
    def size(self):
        return self.rows * self.cols * self.element

    def view(self, row=0, col=0, rows=None, cols=None):
        return View(self, row, col, self.rows - row if rows is None else rows,
                    self.cols - col if cols is None else cols)


@dataclass(frozen=True)
class View:
    tensor: Tensor
    row: int
    col: int
    rows: int
    cols: int
    transposed: bool = False

    def transpose(self):
        return View(self.tensor, self.row, self.col, self.cols, self.rows, not self.transposed)

    def regions(self, row, col, rows, cols):
        if not (0 <= row < row + rows <= self.rows and 0 <= col < col + cols <= self.cols):
            raise ValueError(f"Out-of-bounds view: {self.tensor.name}")
        if self.tensor.base is None:
            raise ValueError(f"Unallocated tensor: {self.tensor.name}")
        if self.transposed:
            row, col, rows, cols = col, row, cols, rows
        t = self.tensor
        return (Region(t.base + ((self.row + row) * t.cols + self.col + col) * t.element,
                       rows, cols, t.cols * t.element, t.element),)

    @property
    def tensors(self):
        return (self.tensor,)


@dataclass(frozen=True)
class Rows:
    """Logical concatenation, including per-request KV, without a free copy."""
    parts: tuple

    @property
    def rows(self):
        return sum(p.rows for p in self.parts)

    @property
    def cols(self):
        return self.parts[0].cols

    @property
    def tensors(self):
        return tuple(t for p in self.parts for t in p.tensors)

    def regions(self, row, col, rows, cols):
        if not (0 <= row < row + rows <= self.rows and 0 <= col < col + cols <= self.cols):
            raise ValueError("Out-of-bounds concatenation")
        result, offset = [], 0
        for part in self.parts:
            lo, hi = max(row, offset), min(row + rows, offset + part.rows)
            if lo < hi:
                result.extend(part.regions(lo - offset, col, hi - lo, cols))
            offset += part.rows
        return tuple(result)

    def transpose(self):
        return Transpose(self)


@dataclass(frozen=True)
class Transpose:
    source: Rows

    @property
    def rows(self):
        return self.source.cols

    @property
    def cols(self):
        return self.source.rows

    @property
    def tensors(self):
        return self.source.tensors

    def regions(self, row, col, rows, cols):
        return self.source.regions(col, row, cols, rows)


@dataclass
class Op:
    name: str
    kind: str
    inputs: tuple
    output: View
    vector_ops: int = 0
    category: str = "activation"

    @property
    def flops(self):
        return (2 * self.inputs[0].rows * self.inputs[0].cols * self.inputs[1].cols
                if self.kind == "gemm" else 0)


class Allocator:
    def __init__(self, base=0):
        self.end = base

    def allocate(self, size):
        address = self.end
        self.end += (size + 4095) // 4096 * 4096
        return address


def place_scratch(ops, base):
    """Operator barriers make last-use reclamation legal across the whole stage."""
    last, tensors = {}, {}
    for index, op in enumerate(ops):
        for view in (*op.inputs, op.output):
            for tensor in view.tensors:
                tensors[id(tensor)] = tensor
                last[id(tensor)] = index
    scratch = {key for key, t in tensors.items() if t.base is None}
    free, live, end = [], {}, base
    for index, op in enumerate(ops):
        for view in (*op.inputs, op.output):
            for t in view.tensors:
                key = id(t)
                if key not in scratch or key in live:
                    continue
                size = (t.size + 4095) // 4096 * 4096
                match = next((i for i, (_, n) in enumerate(free) if n >= size), None)
                if match is None:
                    t.base, end = end, end + size
                else:
                    address, available = free.pop(match)
                    t.base = address
                    if available > size:
                        free.append((address + size, available - size))
                live[key] = (t.base, size)
        for key in [k for k in live if last[k] == index]:
            free.append(live.pop(key))
        free.sort()
        merged = []
        for address, size in free:
            if merged and sum(merged[-1]) == address:
                a, n = merged.pop()
                merged.append((a, n + size))
            else:
                merged.append((address, size))
        free = merged
    return end


class Pi05:
    def __init__(self, shape, slots):
        self.shape = shape
        self.allocator = Allocator()
        self.weights = {}
        self.slots = []
        s = shape
        for slot in range(slots):
            def external(name, rows, cols):
                t = Tensor(f"slot{slot}/{name}", rows, cols)
                t.base = self.allocator.allocate(t.size)
                return t
            self.slots.append({
                "image": external("image", s.images * s.image_tokens, s.patch_features),
                "text": external("embedded_text", s.text_tokens, s.prefix_width),
                "action": external("action", s.action_tokens, s.action_dim),
                "k": [external(f"kv{l}/k", s.prefix_tokens, s.head_dim) for l in range(s.layers)],
                "v": [external(f"kv{l}/v", s.prefix_tokens, s.head_dim) for l in range(s.layers)],
            })

    def weight(self, name, k, n):
        if name not in self.weights:
            t = Tensor(name, k, n, weight=True)
            t.base = self.allocator.allocate(t.size)
            self.weights[name] = t
        t = self.weights[name]
        if (t.rows, t.cols) != (k, n):
            raise ValueError(f"Weight shape mismatch: {name}")
        return t.view()

    def graph(self, slots, prefix=False, first_step=0, steps=1):
        s, ops = self.shape, []

        def tensor(name, rows, cols, element=2):
            return Tensor(name, rows, cols, element).view()

        def gemm(name, a, b, out=None, category="activation"):
            if a.cols != b.rows:
                raise ValueError(f"GEMM shape mismatch: {name}")
            out = out or tensor(name, a.rows, b.cols)
            ops.append(Op(name, "gemm", (a, b), out, category=category))
            return out

        def linear(name, x, n, out=None):
            return gemm(name, x, self.weight(name, x.cols, n), out, "weight")

        def vector(name, inputs, cost, out=None):
            out = out or tensor(name, inputs[0].rows, inputs[0].cols)
            ops.append(Op(name, "vector", tuple(inputs), out, cost))
            return out

        def attention(name, q, k, v, heads, hd):
            # Dense materialized attention, not FlashAttention. Each request/head
            # has its own scores; grouped-query KV is shared across query heads.
            outputs = []
            for h in range(heads):
                qh = View(q.tensor, q.row, q.col + h * hd, q.rows, hd)
                logits = gemm(f"{name}/h{h}/qk", qh, k.transpose(),
                              tensor(f"{name}/scores{h}", q.rows, k.rows, 4), "kv")
                probs = vector(f"{name}/h{h}/softmax", [logits], 8,
                               tensor(f"{name}/probs{h}", q.rows, k.rows))
                outputs.append(gemm(f"{name}/h{h}/av", probs, v, category="kv"))
            joined = tensor(name + "/joined", q.rows, heads * hd)
            for h, part in enumerate(outputs):
                target = View(joined.tensor, 0, h * hd, q.rows, hd)
                vector(f"{name}/h{h}/join", [part], 1, target)
            return joined

        def block(name, x, width, mlp, heads, hd, saved_k=None, saved_v=None, cond=None):
            def norm(label, value):
                if cond is not None:
                    modulation = linear(name + "/" + label + "/modulation", cond, 3 * width)
                    # Modulation is broadcast over each request's action tokens.
                    return vector(name + "/" + label, [value, modulation], 12)
                return vector(name + "/" + label, [value], 8)
            z = norm("norm1", x)
            q = linear(name + "/q", z, heads * hd)
            if prefix:
                k = linear(name + "/k", z, hd, saved_k)
                v = linear(name + "/v", z, hd, saved_v)
            else:
                k = linear(name + "/k", z, hd)
                v = linear(name + "/v", z, hd)
            q = vector(name + "/q_rope", [q], 6)
            # Persist the position-encoded prefix keys, not their raw projection.
            k = vector(name + "/k_rope", [k], 6, saved_k if prefix else None)
            if saved_k is not None and not prefix:
                k = Rows((saved_k, k))
                v = Rows((saved_v, v))
            a = attention(name + "/attn", q, k, v, heads, hd)
            a = linear(name + "/out", a, width)
            x = vector(name + "/residual1", [x, a], 3 if cond is not None else 1)
            z = norm("norm2", x)
            gate = linear(name + "/gate", z, mlp)
            up = linear(name + "/up", z, mlp)
            z = vector(name + "/gelu_mul", [gate, up], 10)
            z = linear(name + "/down", z, width)
            return vector(name + "/residual2", [x, z], 3 if cond is not None else 1)

        if prefix:
            if len(slots) != 1:
                raise ValueError("Prefix batching is not supported by this frontend")
            slot = self.slots[slots[0]]
            embedding = self.weight("prefix/token_embedding", s.vocab_size, s.prefix_width)
            token_rows = Rows(tuple(View(embedding.tensor, (slots[0] * s.text_tokens + i) % s.vocab_size,
                                        0, 1, s.prefix_width) for i in range(s.text_tokens)))
            vector("prefix/token_lookup", [token_rows], 1, slot["text"].view())
            x = linear("vision/patch", slot["image"].view(), s.vision_width)
            x = vector("vision/position", [x], 1)
            # Independent images cannot attend to one another.
            for layer in range(s.vision_layers):
                normed = vector(f"vision/{layer}/norm1", [x], 10)
                q = linear(f"vision/{layer}/q", normed, s.vision_width)
                k = linear(f"vision/{layer}/k", normed, s.vision_width)
                v = linear(f"vision/{layer}/v", normed, s.vision_width)
                joined = tensor(f"vision/{layer}/joined", x.rows, x.cols)
                hd = s.vision_width // s.vision_heads
                for image in range(s.images):
                    row = image * s.image_tokens
                    # Vision uses multi-head KV rather than Gemma's single KV head.
                    for head in range(s.vision_heads):
                        qh, kh, vh = [View(t.tensor, row, head * hd, s.image_tokens, hd)
                                      for t in (q, k, v)]
                        tag = f"vision/{layer}/img{image}/h{head}"
                        scores = gemm(tag + "/qk", qh, kh.transpose(),
                                      tensor(tag + "/scores", s.image_tokens, s.image_tokens, 4))
                        p = vector(tag + "/softmax", [scores], 8,
                                   tensor(tag + "/p", s.image_tokens, s.image_tokens))
                        gemm(tag + "/av", p, vh,
                             View(joined.tensor, row, head * hd, s.image_tokens, hd))
                a = linear(f"vision/{layer}/out", joined, s.vision_width)
                x = vector(f"vision/{layer}/residual1", [x, a], 1)
                z = vector(f"vision/{layer}/norm2", [x], 10)
                z = linear(f"vision/{layer}/up", z, s.vision_mlp)
                z = vector(f"vision/{layer}/gelu", [z], 8)
                z = linear(f"vision/{layer}/down", z, s.vision_width)
                x = vector(f"vision/{layer}/residual2", [x, z], 1)
            x = vector("vision/final_norm", [x], 10)
            x = linear("vision/project", x, s.prefix_width)
            x = Rows((x, slot["text"].view()))
            for layer in range(s.layers):
                x = block(f"prefix/{layer}", x, s.prefix_width, s.prefix_mlp,
                          s.heads, s.head_dim, slot["k"][layer].view(), slot["v"][layer].view())
            vector("prefix/final_norm", [x], 8)
        else:
            for step in range(first_step, first_step + steps):
                # Linear operators merge M across requests; attention remains per
                # request. This gives real tile-level batch reuse, not bytes/b.
                actions = Rows(tuple(self.slots[i]["action"].view() for i in slots))
                x = linear("expert/action_in", actions, s.expert_width)
                time = tensor(f"step{step}/time", len(slots), s.expert_width)
                # Sine/cos is generated locally, with explicit vector cost.
                ops.append(Op(f"step{step}/time_embed", "generate", (), time, 16))
                cond = linear("expert/time_in", time, s.expert_width)
                cond = vector("expert/time_in/swish", [cond], 8)
                cond = linear("expert/time_out", cond, s.expert_width)
                cond = vector("expert/time_out/swish", [cond], 8)
                for layer in range(s.layers):
                    name = f"expert/{layer}"
                    norm_mod = linear(name + "/norm1/modulation", cond, 3 * s.expert_width)
                    z = vector(name + "/norm1", [x, norm_mod], 12)
                    q = linear(name + "/q", z, s.heads * s.head_dim)
                    k = linear(name + "/k", z, s.head_dim)
                    v = linear(name + "/v", z, s.head_dim)
                    q = vector(name + "/q_rope", [q], 6)
                    k = vector(name + "/k_rope", [k], 6)
                    joined = tensor(name + "/joined", x.rows, s.heads * s.head_dim)
                    for batch, slot_id in enumerate(slots):
                        row = batch * s.action_tokens
                        kv = self.slots[slot_id]
                        qi = View(q.tensor, row, 0, s.action_tokens, q.cols)
                        ki = Rows((kv["k"][layer].view(), View(k.tensor, row, 0, s.action_tokens, k.cols)))
                        vi = Rows((kv["v"][layer].view(), View(v.tensor, row, 0, s.action_tokens, v.cols)))
                        a = attention(f"{name}/r{batch}/attn", qi, ki, vi, s.heads, s.head_dim)
                        vector(name + f"/r{batch}/join", [a], 1,
                               View(joined.tensor, row, 0, s.action_tokens, joined.cols))
                    a = linear(name + "/out", joined, s.expert_width)
                    x = vector(name + "/residual1", [x, a, norm_mod], 3)
                    mod = linear(name + "/norm2/modulation", cond, 3 * s.expert_width)
                    z = vector(name + "/norm2", [x, mod], 12)
                    gate = linear(name + "/gate", z, s.expert_mlp)
                    up = linear(name + "/up", z, s.expert_mlp)
                    z = vector(name + "/gelu_mul", [gate, up], 10)
                    z = linear(name + "/down", z, s.expert_width)
                    x = vector(name + "/residual2", [x, z, mod], 3)
                mod = linear("expert/final_norm/modulation", cond, 3 * s.expert_width)
                x = vector("expert/final_norm", [x, mod], 12)
                velocity = linear("expert/action_out", x, s.action_dim)
                for batch, slot_id in enumerate(slots):
                    action = self.slots[slot_id]["action"].view()
                    vel = View(velocity.tensor, batch * s.action_tokens, 0,
                               s.action_tokens, s.action_dim)
                    vector(f"step{step}/r{batch}/euler", [action, vel], 2, action)
        return ops

    def describe(self):
        prefix = self.graph([0], prefix=True)
        expert = self.graph([0])
        return {
            "openpi_revision": OPENPI_REVISION, "shape": asdict(self.shape),
            "prefix_tokens": self.shape.prefix_tokens,
            "prefix_gemm_flops": sum(op.flops for op in prefix),
            "expert_gemm_flops_per_step": sum(op.flops for op in expert),
            "weight_bytes": sum(t.size for t in self.weights.values()),
            "prefix_weight_bytes": sum(t.size for name, t in self.weights.items()
                                       if not name.startswith("expert/")),
            "prefix_dense_weight_bytes": sum(t.size for name, t in self.weights.items()
                if not name.startswith("expert/") and name != "prefix/token_embedding"),
            "token_embedding_table_bytes": self.weights["prefix/token_embedding"].size,
            "expert_weight_bytes": sum(t.size for name, t in self.weights.items()
                                       if name.startswith("expert/")),
            "kv_bytes_per_request": 2 * self.shape.layers * self.shape.prefix_tokens * self.shape.head_dim * 2,
            "prefix_ops": len(prefix), "expert_ops_per_step": len(expert),
            "limitations": ["dense materialized attention; no FlashAttention fusion",
                            "timing/traffic only; no numerical execution",
                            "vector work uses explicit configurable element-cost estimates",
                            "token IDs are deterministic synthetic indices; preprocessing is outside measured latency",
                            "biases and small normalization/position parameters are omitted"],
        }
