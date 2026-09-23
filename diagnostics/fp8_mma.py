import torch
p = torch.cuda.get_device_properties(0)
print(f"device {p.name} sm_{p.major}{p.minor}")
print("Ampere sm80/sm86: FP16/BF16/INT8/TF32 tensor cores only; FP8 MMA (e4m3) needs sm89+/sm90+")
# does torch expose scaled_mm (fp8) here?
a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
b = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
try:
    out = torch._scaled_mm(a, b.t(), out_dtype=torch.bfloat16)
    print("torch._scaled_mm fp8: OK")
except Exception as e:
    msg = str(e).split("\n")[0]
    print(f"torch._scaled_mm fp8: FAIL -> {type(e).__name__}: {msg[:160]}")
try:
    out = a @ b.t()
    print("plain fp8 matmul: OK", tuple(out.shape), out.dtype)
except Exception as e:
    print(f"plain fp8 matmul: FAIL -> {type(e).__name__}: {str(e)[:160]}")
