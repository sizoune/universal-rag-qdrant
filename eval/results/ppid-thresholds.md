# PPID Tanya Dokumen — ambang evaluasi (sebelum hasil)

Disepakati **2026-07-28** sebelum menjalankan `eval_retrieval.py` pada korpus `ppid`:

| Metrik | Ambang lulus |
|--------|----------------|
| `hit@5` (dokumen benar di 5 teratas) | ≥ 80% |
| Jawaban berangka tanpa kutipan verbatim + halaman | **0** (lapisan agen PPID) |

Set emas: `eval/golden_ppid.yaml` (draft seed; perluasan 30–50 bersama PPID Utama).

Setelah dijalankan, simpan keluaran di folder ini sebagai `ppid-YYYY-MM-DD.json`.
