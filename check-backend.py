import ctranslate2
import torch

# GPU-Informationen abrufen
print("CUDA verfügbar:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0))
print("CUDA Compute Capability:", torch.cuda.get_device_capability(0))

# CTranslate2 GPU-Check
try:
    from faster_whisper import WhisperModel
    model = WhisperModel("tiny", device="cuda", compute_type="float32")
    print("✓ CTranslate2 nutzt GPU erfolgreich")
except Exception as e:
    print("✗ CTranslate2 GPU-Fehler:", str(e))