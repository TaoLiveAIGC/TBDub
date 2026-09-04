import torch


def TBDubDiTStateDictConverter(state_dict):
    """Adapt a legacy 2048-channel audio projection to TBDub's HuBERT input.

    Reinitializes audio_embedding.proj_in weights with xavier_uniform when
    the input dimension doesn't match (old whisper+wav2vec=2048 vs new hubert=4096).
    """
    target_input_dim = 4096
    key_w = "audio_embedding.proj_in.weight"

    state_dict_ = {name: state_dict[name] for name in state_dict}

    if key_w in state_dict_:
        old_w = state_dict_[key_w]
        if old_w.shape[1] != target_input_dim:
            out_dim = old_w.shape[0]
            new_w = torch.empty(out_dim, target_input_dim, dtype=old_w.dtype, device=old_w.device)
            torch.nn.init.xavier_uniform_(new_w)
            state_dict_[key_w] = new_w
            print(f"[TBDubDiTStateDictConverter] Reinitialized {key_w}: {list(old_w.shape)} -> {list(new_w.shape)}")

    return state_dict_
