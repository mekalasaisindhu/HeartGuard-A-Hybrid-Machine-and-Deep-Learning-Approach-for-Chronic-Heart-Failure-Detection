import os

print("Checking folders")

paths = [
    "data/spectrogram",
    "data/spectrogram/normal",
    "data/spectrogram/abnormal"
]

for p in paths:
    print(p, "exists?" , os.path.exists(p))

print("\nCounting PNG files:")
for p in ["data/spectrogram/normal", "data/spectrogram/abnormal"]:
    if os.path.exists(p):
        print(p, "PNG count:", len([f for f in os.listdir(p) if f.endswith('.png')]))
