#!/usr/bin/env python3
"""
generate_audio.py
Persistent Zonos TTS workers that keep their mouths shut about euros.

Changes since last fiasco
• SENTINEL is now a boring string that survives pickling.
• Worker exits on `job == SENTINEL`, so no dict‑access meltdowns.
• GPU detection bumps worker count if you own an NVIDIA L40 (48 GB VRAM).
• VRAM per worker can be tuned with env var ZONOS_VRAM_GB.
• Re‑sanitises Unicode ghosts and clears model KV cache per call.
"""

import os, sys, re, json, math, gc, traceback, multiprocessing as mp
from datetime import datetime

import torch, torchaudio

# ----- Zonos imports -----
try:
    from zonos.model import Zonos
    from zonos.conditioning import make_cond_dict
    from zonos.utils import DEFAULT_DEVICE as _DEVNAME
except ImportError as e:
    sys.stderr.write(f"Zonos import failed: {e}\n")
    sys.exit(1)

MODEL_NAME = "Zyphra/Zonos-v0.1-transformer"
DEFAULT_LANG = "en-us"
VRAM_PER_WORKER_GB = float(os.getenv("ZONOS_VRAM_GB", 4.5))   # override if you know better
SENTINEL = "__STOP__"                                         # pickle‑safe
CUDA_OK = torch.cuda.is_available()

# ---- Unicode sanitation -----
CTL = re.compile(r"[\u0000-\u001F\u007F]")                     # control chars
FMT = re.compile(r"[\u2000-\u206F\uFEFF]")                     # zero‑width varmints
def clean(txt: str) -> str:
    txt = txt.replace("...", ".").replace("—", "").replace("–", "")
    txt = CTL.sub("", txt)
    txt = FMT.sub("", txt)
    return txt.strip()

# ---- VRAM helpers -----
def free_vram() -> int:
    if not CUDA_OK:
        return 0
    free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
    return max(0, free - (100 << 20))                          # keep 100 MB headroom

def gpu_name() -> str:
    return torch.cuda.get_device_name(0) if CUDA_OK else "CPU"

# ---- single TTS call -----
@torch.inference_mode()
def run_tts(model, speaker_embed, text: str):
    safe_embed = speaker_embed.detach().clone()
    cond = make_cond_dict(text=text, speaker=safe_embed, language=DEFAULT_LANG)
    conditioning = model.prepare_conditioning(cond)
    if hasattr(model, "clear_kv_cache"):
        model.clear_kv_cache()
    codes = model.generate(conditioning, disable_torch_compile=True)
    codes = codes.to(model.autoencoder.dac.device)
    wav = model.autoencoder.decode(codes)[0].cpu()
    del conditioning, codes, safe_embed
    return wav

# ---- worker -----
def worker(task_q, result_q, base_dir):
    pid = os.getpid()
    dev = torch.device(_DEVNAME if CUDA_OK else "cpu")
    sys.stderr.write(f"[{pid}] online on {dev} ({gpu_name()})\n")

    try:
        model = Zonos.from_pretrained(MODEL_NAME, device=dev).eval()
    except Exception as e:
        result_q.put({"fatal": True, "msg": str(e)})
        return

    cache = {}  # speaker path -> embedding

    while True:
        job = task_q.get()
        if job == SENTINEL:
            break

        try:
            rid   = job["id"]
            text  = clean(job["text"])
            out   = os.path.abspath(job["out"])
            spk_p = os.path.join(base_dir, job["speaker"])
            seed  = job.get("seed")

            if seed is not None:
                torch.manual_seed(seed)

            if spk_p not in cache:
                wav, sr = torchaudio.load(spk_p)
                cache[spk_p] = model.make_speaker_embedding(wav.to(dev), sr)

            audio = run_tts(model, cache[spk_p], text)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            torchaudio.save(out, audio, model.autoencoder.sampling_rate)

            result_q.put({"id": rid, "ok": True, "file": out})

        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            result_q.put({"id": job.get('id', 'unk'), "ok": False, "err": str(e)})

        finally:
            torch.cuda.empty_cache()
            gc.collect()

    del model, cache
    torch.cuda.empty_cache()
    sys.stderr.write(f"[{pid}] bye\n")

# ---- main -----
if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    base_dir = os.path.dirname(os.path.realpath(__file__))
    requests = []

    # ---- read stdin JSONL -----
    for ln, raw in enumerate(sys.stdin, 1):
        if not raw.strip():
            continue
        try:
            d = json.loads(raw)
            spk = d["speaker_path"]
            if "texts" in d:
                for idx, (t, p) in enumerate(zip(d["texts"], d["output_paths"])):
                    requests.append({"id": f"{ln}-{idx}", "text": t,
                                     "out": p, "speaker": spk})
            else:
                requests.append({"id": f"{ln}", "text": d["text"],
                                 "out": d["output_path"], "speaker": spk})
        except Exception as e:
            sys.stderr.write(f"Bad line {ln}: {e}\n")

    if not requests:
        sys.stderr.write("No jobs found. Bye.\n")
        sys.exit(0)

    # ---- worker count decision -----
    total_vram = free_vram()
    name = gpu_name()
    if "L40" in name.upper():
        VRAM_PER_WORKER_GB = 8.0                          # play it safe; tweak if brave
    wrk_by_vram = max(1, math.floor(total_vram / (VRAM_PER_WORKER_GB * (1 << 30))))
    wrk_by_cpu  = max(1, os.cpu_count() // 2 or 1)
    workers = min(wrk_by_vram, wrk_by_cpu)
    sys.stderr.write(f"GPU: {name}  free VRAM: {total_vram/1e9:.1f} GB  workers: {workers}\n")

    task_q, res_q = mp.Queue(), mp.Queue()
    procs = [mp.Process(target=worker, args=(task_q, res_q, base_dir)) for _ in range(workers)]
    for p in procs:
        p.start()

    for req in requests:
        task_q.put(req)
    for _ in procs:
        task_q.put(SENTINEL)

    done = 0
    while done < len(requests):
        try:
            res = res_q.get(timeout=600)          # 10 min per job
            done += 1
            if res.get("ok"):
                print(res["file"], flush=True)
            else:
                sys.stderr.write(f"Fail {res['id']}: {res.get('err')}\n")
        except mp.queues.Empty:
            sys.stderr.write("Timeout waiting for results\n")
            break

    for p in procs:
        p.join(10)
        if p.is_alive():
            p.terminate()
