"""Smoke tests for psub_solver (CPU, no network).

Uses a small randomly initialized OPT model and the same flatten ->
objective -> unflatten path as LOZOtrainer. Run: python test_psub_local.py
"""
import sys
import numpy as np
import tracemalloc

FAILED = []


def check(name, cond, detail=''):
    tag = '[PASS]' if cond else '[FAIL]'
    print('%s %s %s' % (tag, name, detail))
    if not cond:
        FAILED.append(name)


def build_tiny_model():
    import torch
    from transformers import OPTConfig
    from transformers.models.opt.modeling_opt import OPTForCausalLM
    torch.manual_seed(0)
    cfg = OPTConfig(vocab_size=256, hidden_size=32, num_hidden_layers=2,
                    ffn_dim=64, num_attention_heads=2, max_position_embeddings=64,
                    word_embed_proj_dim=32)
    model = OPTForCausalLM(cfg)
    model.eval()
    return model, torch


def make_objective(model, torch, n_batch=4, seq=16, vocab=256):
    torch.manual_seed(1)
    ids = torch.randint(0, vocab, (n_batch, seq))
    params = [(nm, p) for nm, p in model.named_parameters() if p.requires_grad]

    def flatten():
        return np.concatenate([p.data.detach().cpu().numpy().ravel()
                               for _, p in params])

    def unflatten(flat):
        i = 0
        for _, p in params:
            k = p.numel()
            p.data.copy_(torch.tensor(flat[i:i + k], dtype=p.dtype).view(p.shape))
            i += k

    calls = {'n': 0}

    def objective(x):
        unflatten(x)
        calls['n'] += 1
        with torch.inference_mode():
            return float(model(input_ids=ids, labels=ids).loss.item())

    return flatten, unflatten, objective, calls, params


def main():
    from psub_solver import psolve, psolve_stream, PSubState, _dir_from_seed

    P = 8

    model, torch = build_tiny_model()
    flatten, unflatten, objective, calls, params = make_objective(model, torch)
    x0 = flatten()
    n = x0.size
    print('trainable n = %d (%d tensors), p = %d' % (n, len(params), P))

    f0 = objective(x0)
    unflatten(x0)
    check('flatten/unflatten round trip', np.allclose(flatten(), x0, atol=0),
          '| f(x0)=%.6f' % f0)

    a = _dir_from_seed(0, 2, 5, n, np.float32)
    b = _dir_from_seed(0, 2, 5, n, np.float32)
    c = _dir_from_seed(0, 2, 6, n, np.float32)
    check('direction regeneration is deterministic', np.array_equal(a, b),
          '| |v|=%.6f' % np.linalg.norm(a))
    check('distinct i gives distinct direction', not np.array_equal(a, c),
          '| cos=%.5f' % float(a @ c))

    st = PSubState(delta0=1e-2)
    tracemalloc.start()
    xb, fb, st = psolve(objective, x0, st, p=P, n_steps=10, seed=0)
    peak_std = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    check('psolve decreases the loss', fb < f0,
          '| %.6f -> %.6f (delta=%.2e)' % (f0, fb, st.delta))

    st2 = PSubState(delta0=1e-2)
    tracemalloc.start()
    xs, fs, st2 = psolve_stream(objective, x0, st2, p=P, n_steps=10, seed=0)
    peak_stream = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    check('psolve_stream decreases the loss', fs < f0,
          '| %.6f -> %.6f (delta=%.2e)' % (f0, fs, st2.delta))
    check('streaming peak memory below psolve peak', peak_stream < peak_std,
          '| stream=%.2fMB vs psolve=%.2fMB (p*n*8=%.2fMB)'
          % (peak_stream / 1e6, peak_std / 1e6, P * n * 8 / 1e6))

    calls['n'] = 0
    st3 = PSubState(delta0=1e-2)
    psolve_stream(objective, x0, st3, p=P, n_steps=3, seed=0)
    expect = 3 * (2 * P + 1) + 1
    check('evaluation count equals n_steps*(2p+1)+1', calls['n'] == expect,
          '| got=%d want=%d' % (calls['n'], expect))

    st4 = PSubState(delta0=1e-2)
    d_hist = []
    for _ in range(3):
        _, _, st4 = psolve_stream(objective, x0, st4, p=4, n_steps=2, seed=0)
        d_hist.append(st4.delta)
    check('delta persists across calls', len(set(d_hist)) > 1 or st4.delta != 1e-2,
          '| delta trajectory=%s' % ['%.2e' % d for d in d_hist])

    unflatten(xs)
    moved = float(np.linalg.norm(flatten() - x0))
    check('weights actually moved', moved > 0, '| ||dw||=%.6e' % moved)

    f_reload = objective(xs)
    check('best weights reload to the same loss', abs(f_reload - fs) < 1e-4,
          '| reload=%.6f vs best=%.6f' % (f_reload, fs))

    from psub_solver import fp16_safety_report
    N125 = 1.25e8
    bad, msg_bad = fp16_safety_report(1e-2, N125, 'torch.float16')
    good, _ = fp16_safety_report(1e-1, N125, 'torch.float16')
    f32, _ = fp16_safety_report(1e-2, N125, 'torch.float32')
    check('fp16 guard flags delta=1e-2 as underflow', not bad,
          '| ' + msg_bad.split('-> ')[-1])
    check('fp16 guard accepts delta=1e-1', good)
    check('fp32 always safe at these scales', f32)

    if FAILED:
        print('RESULT: %d failed -> %s' % (len(FAILED), ', '.join(FAILED)))
        return 1
    print('RESULT: all passed')
    return 0


if __name__ == '__main__':
    sys.exit(main())
