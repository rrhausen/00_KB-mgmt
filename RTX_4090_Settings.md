# RTX 4090 Transcription Settings

## 🛡️ STABIL (Current Settings)
**Für maximale Stabilität - keine Crashes**
```python
if gpu.has_gpu and gpu.get_vram_gb() >= 24:  # RTX 4090 - Conservative
    params.update({
        "batch_size": 8,           # Very small batches
        "chunk_length_s": 20,      # Small chunks
        "stride_length_s": 2,      # Minimal overlap
        "num_workers": 1,          # Single worker
        "return_timestamps": True  # Chunk-level timestamps
    })
```

## ⚡ OPTIMIERT (Empfohlen)
**Guter Kompromiss - schnell und stabil**
```python
if gpu.has_gpu and gpu.get_vram_gb() >= 24:  # RTX 4090 - Optimized
    params.update({
        "batch_size": 16,          # Medium batches
        "chunk_length_s": 30,      # Standard chunks
        "stride_length_s": 3,      # Moderate overlap
        "num_workers": 2,          # Limited parallel
        "return_timestamps": True  # Chunk-level timestamps
    })
```

## 🚀 SCHNELL (Experimentell)
**Für maximale Geschwindigkeit - kann crashen**
```python
if gpu.has_gpu and gpu.get_vram_gb() >= 24:  # RTX 4090 - Fast
    params.update({
        "batch_size": 32,          # Large batches
        "chunk_length_s": 45,      # Long chunks
        "stride_length_s": 5,      # Good overlap
        "num_workers": 4,          # More parallel
        "return_timestamps": "word"  # Word-level timestamps
    })
```

## 🔥 MAX. SCHNELL (Riskant)
**Maximale RTX 4090 Performance - hoher Crash-Risk**
```python
if gpu.has_gpu and gpu.get_vram_gb() >= 24:  # RTX 4090 - Max Performance
    params.update({
        "batch_size": 56,          # Very large batches
        "chunk_length_s": 60,      # Very long chunks
        "stride_length_s": 8,      # High overlap
        "num_workers": 6,          # Max parallel
        "return_timestamps": "word"  # Word-level timestamps
    })
```

---

## 📊 Empfehlungen

### **Für den täglichen Gebrauch:**
- **OPTIMIERT** - Beste Balance zwischen Geschwindigkeit und Stabilität
- Batch Size 16, Chunks 30s, Workers 2

### **Für wichtige Projekte:**
- **STABIL** - Niemals crashen
- Batch Size 8, Chunks 20s, Workers 1

### **Für Experimente:**
- **SCHNELL** oder **MAX. SCHNELL** - Mit Backup der Dateien
- Kann zu Abstürzen führen, aber maximale Performance

---

## ⚙️ Wie ändern Sie die Settings:

Öffnen Sie `app1.py` und suchen Sie diese Zeilen:

```python
if gpu.has_gpu and gpu.get_vram_gb() >= 24:  # RTX 4090 - Very conservative settings
    params.update({
        "batch_size": 8,      # ← Ändern Sie hier
        "chunk_length_s": 20,  # ← Ändern Sie hier
        "stride_length_s": 2,  # ← Ändern Sie hier
        "num_workers": 1      # ← Ändern Sie hier
    })
```

Kopieren Sie einfach die gewünschten Werte aus den Beispielen oben.