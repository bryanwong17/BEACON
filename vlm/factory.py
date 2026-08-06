def build_vlm_encoder(backend: str, ckpt_path: str):
    if backend == "plip":
        from vlm.plip_encoder import PLIPEncoder
        return PLIPEncoder(ckpt_path)
    if backend == "conch":
        from vlm.conch_encoder import CONCHEncoder
        return CONCHEncoder(ckpt_path)
    raise ValueError(f"Unknown VLM backend: {backend!r} (expected 'plip' or 'conch')")
