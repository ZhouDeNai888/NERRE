import torch
import transformers

print("-" * 30)
print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")
print("-" * 30)

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"✅ CUDA is available! Found {device_count} GPU(s).")
    for i in range(device_count):
        print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        # เช็คว่าเป็น A100 จริงไหม และรองรับ BF16 (Bfloat16) ไหม
        if torch.cuda.is_bf16_supported():
            print("   🚀 BFloat16 is supported! (Great for A100)")
else:
    print("❌ CUDA is NOT available. You are running on CPU.")