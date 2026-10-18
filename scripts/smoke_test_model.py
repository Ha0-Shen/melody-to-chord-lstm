"""Test the actual notebook model classes using random weights and synthetic tokens."""
import json
import torch
import torch.nn as nn
from notebook_tools import BASELINE_NOTEBOOK, WEIGHTED_NOTEBOOK, load_definitions


def check_model(path):
    namespace = load_definitions(path, ['MelodyToChordCausalLSTM'], {'torch': torch, 'nn': nn})
    torch.manual_seed(42)
    model = namespace['MelodyToChordCausalLSTM'](
        input_vocab_size=64, chord_vocab_size=12, pad_token_id=0,
        embedding_dim=16, hidden_size=16, num_layers=2, dropout=0.30,
    ).eval()
    tokens = torch.randint(1, 64, (2, 73))
    changed = tokens.clone()
    changed[:, 29:] = (changed[:, 29:] + 7) % 63 + 1
    with torch.inference_mode():
        full = model(tokens)
        perturbed = model(changed)
        torch.testing.assert_close(full[:, :29], perturbed[:, :29], atol=1e-6, rtol=1e-5)
        chunks, state = [], None
        for chunk in tokens.split(11, dim=1):
            logits, state = model(chunk, state=state, return_state=True)
            chunks.append(logits)
        streamed = torch.cat(chunks, dim=1)
        torch.testing.assert_close(full, streamed, atol=1e-6, rtol=1e-5)
        steps, state = [], None
        for position in range(tokens.shape[1]):
            logits, state = model.stream_step(tokens[:, position], state=state)
            steps.append(logits.unsqueeze(1))
        stepped = torch.cat(steps, dim=1)
        torch.testing.assert_close(full, stepped, atol=1e-6, rtol=1e-5)
        assert tuple(state[0].shape) == (2, 2, 16)
        scalar_logits, _ = model.stream_step(int(tokens[0, 0]))
        assert tuple(scalar_logits.shape) == (1, 12)
        future_diff = float((full[:, :29] - perturbed[:, :29]).abs().max())
        chunk_diff = float((full - streamed).abs().max())
        step_diff = float((full - stepped).abs().max())
    model.train()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.classifier.weight.detach().clone()
    logits = model(tokens)
    target = torch.randint(1, 12, tokens.shape)
    loss = nn.functional.cross_entropy(logits.transpose(1, 2), target)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimiser.step()
    assert not torch.equal(before, model.classifier.weight.detach())
    return {'notebook': path.name, 'future_max_abs_diff': future_diff,
            'chunk_max_abs_diff': chunk_diff, 'step_max_abs_diff': step_diff,
            'finite_training_update': True, 'status': 'PASS'}


if __name__ == '__main__':
    torch.set_num_threads(1)
    print(json.dumps({'torch_version': torch.__version__, 'device': 'cpu',
                      'scope': 'synthetic inputs and random weights; no corpus training',
                      'checks': [check_model(path) for path in
                                 (BASELINE_NOTEBOOK, WEIGHTED_NOTEBOOK)]}, indent=2))
