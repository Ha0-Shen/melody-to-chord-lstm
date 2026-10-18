"""Synthetic behavioural checks for the actual functions in the research notebooks."""
import hashlib
import math
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from notebook_tools import (PREPROCESS_NOTEBOOK, BASELINE_NOTEBOOK,
                            WEIGHTED_NOTEBOOK, load_definitions)


class FakeTokenizer:
    tokens = {0: 'PAD_None', 1: 'BOS_None', 2: 'EOS_None',
              3: 'TimeShift_0.1.24', 4: 'Pitch_60', 5: 'Duration_0.2.24',
              6: 'Pitch_64', 7: 'Duration_0.1.24', 8: 'Velocity_64'}

    def __getitem__(self, token_id):
        return self.tokens[token_id]

    def token_id_type(self, token_id):
        return self.tokens[token_id].split('_', 1)[0]


def adapter_namespace(path):
    return load_definitions(path, ['duration_token_to_steps', 'build_causal_song_arrays',
                                   'starts_for_song'], {
        'np': np, 'Counter': Counter, 'SELECTED_PPQ': 24,
        'miditok_tokenizer': FakeTokenizer(), 'MIDITOK_CLOCK_ID': 3,
        'MIDITOK_BOS_ID': 1, 'MIDITOK_EOS_ID': 2, 'NOTE_OFF_ID': 9,
        'CHORD_PAD_ID': 0, 'DURATION_TOKEN_IDS': np.array([5, 7]),
        'WINDOW_TOKENS': 512, 'MIN_WINDOW_TOKENS': 128,
    })


def fixture():
    # Pitch 60 spans [0,2), pitch 64 spans [2,3), with release at song end.
    return [np.array(x) for x in (
        [1, 4, 5, 3, 3, 6, 7, 3, 2],
        [0, 0, 0, 0, 1, 2, 2, 2, 3],
        [0, 0, 0, 10, 11, 0, 0, 11, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0],
    )]


class DataContractTests(unittest.TestCase):
    def test_release_order_and_song_end(self):
        for path in (BASELINE_NOTEBOOK, WEIGHTED_NOTEBOOK):
            with self.subTest(notebook=path.name):
                actual = adapter_namespace(path)['build_causal_song_arrays'](*fixture())
                np.testing.assert_array_equal(actual['input_ids'], [1, 4, 3, 3, 9, 6, 3, 9, 2])
                self.assertFalse(np.isin(actual['input_ids'], [5, 7]).any())
                self.assertEqual(int(actual['grid_mask'].sum()), 3)

    def test_gold_chords_cannot_change_input_stream(self):
        ns = adapter_namespace(BASELINE_NOTEBOOK)
        a, b = fixture(), fixture()
        b[2][b[3].astype(bool)] = [20, 21, 22]
        b[4][b[3].astype(bool)] = [1, 1, 1]
        left, right = ns['build_causal_song_arrays'](*a), ns['build_causal_song_arrays'](*b)
        np.testing.assert_array_equal(left['input_ids'], right['input_ids'])
        self.assertFalse(np.array_equal(left['chord_target_ids'], right['chord_target_ids']))
        self.assertTrue((left['chord_target_ids'][left['grid_mask'] == 0] == 0).all())

    def test_invalid_clock_alignment_is_rejected(self):
        arrays = fixture()
        arrays[1][4] = 0
        with self.assertRaisesRegex(ValueError, 'Clock event_steps'):
            adapter_namespace(BASELINE_NOTEBOOK)['build_causal_song_arrays'](*arrays)

    def test_unmatched_pitch_and_duration_is_rejected(self):
        arrays = fixture()
        arrays[0][2] = 8
        with self.assertRaisesRegex(ValueError, 'Unexpected token type'):
            adapter_namespace(BASELINE_NOTEBOOK)['build_causal_song_arrays'](*arrays)

    def test_evaluation_covers_every_position_once_including_short_tail(self):
        starts = adapter_namespace(BASELINE_NOTEBOOK)['starts_for_song']
        for length in [1, 127, 128, 512, 513, 1025]:
            windows = starts(length, 512, keep_short_tail=True)
            covered = [i for start, end in windows for i in range(start, end)]
            self.assertEqual(covered, list(range(length)))

    def test_training_windows_overlap_without_adding_short_tail(self):
        starts = adapter_namespace(BASELINE_NOTEBOOK)['starts_for_song']
        self.assertEqual(starts(513, 256), [(0, 512), (256, 513)])
        self.assertEqual(starts(100, 256), [])

    def test_group_assignment_is_deterministic_and_order_independent(self):
        split = load_definitions(PREPROCESS_NOTEBOOK, ['deterministic_split'],
                                 {'hashlib': hashlib})['deterministic_split']
        names = ['song-a', 'song-b', 'song-c', 'song-a']
        first = [split(n) for n in names]
        self.assertEqual(first, list(reversed([split(n) for n in reversed(names)])))
        self.assertEqual(first[0], first[-1])
        self.assertTrue(set(first) <= {'train', 'validation', 'test'})

    def test_vocabulary_reserves_no_chord_and_uses_only_training_ranks(self):
        ns = load_definitions(PREPROCESS_NOTEBOOK, ['select_top_n_labels'], {
            'math': math, 'ALWAYS_RETAIN_CHORDS': ('N',),
            'ranked_training_labels': ['C:maj', 'G:7', 'N'],
            'ranking_position': {'C:maj': 0, 'G:7': 1, 'N': 2},
        })
        retained = ns['select_top_n_labels'](2)
        self.assertEqual(set(retained), {'N', 'C:maj'})
        self.assertNotIn('TEST_ONLY_LABEL', retained)
        with self.assertRaises(ValueError):
            ns['select_top_n_labels'](0)


if __name__ == '__main__':
    unittest.main()
