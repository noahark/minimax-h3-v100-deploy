"""Rewrite an H3 Turbo LoRA's keys into a candidate sd.cpp naming scheme.

usage: python convert_h3_lora.py SRC DST MODE
  MODE = bare : lora.<base>.weight.lora_down / .lora_up
  MODE = mm   : lora.model.diffusion_model.<base>.weight.lora_down / .lora_up
  MODE = raw  : <base>.lora_A.weight / .lora_B.weight   (leave file as shipped)

Only the JSON header is rewritten; the tensor data blob is copied verbatim so
bf16 payloads (which numpy cannot express) survive exactly.
"""
import json
import struct
import sys

src, dst, mode = sys.argv[1], sys.argv[2], sys.argv[3]

with open(src, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    data = f.read()

meta = hdr.pop("__metadata__", None)
out = {}
for k, v in hdr.items():
    if k.endswith(".lora_A.weight"):
        base, suffix = k[: -len(".lora_A.weight")], "lora_down"
    elif k.endswith(".lora_B.weight"):
        base, suffix = k[: -len(".lora_B.weight")], "lora_up"
    else:
        raise SystemExit(f"unexpected key: {k}")
    if mode == "bare":
        new = f"lora.{base}.weight.{suffix}"
    elif mode == "mm":
        new = f"lora.model.diffusion_model.{base}.weight.{suffix}"
    elif mode == "raw":
        new = k
    else:
        raise SystemExit(f"bad mode {mode}")
    out[new] = v

new_hdr = {}
if meta is not None:
    new_hdr["__metadata__"] = meta
new_hdr.update(out)
blob = json.dumps(new_hdr, separators=(",", ":")).encode("utf-8")

with open(dst, "wb") as f:
    f.write(struct.pack("<Q", len(blob)))
    f.write(blob)
    f.write(data)

print(f"wrote {dst} [mode={mode}]: {len(out)} tensors")
for k in list(out)[:2]:
    print("  ", k, out[k]["shape"])
