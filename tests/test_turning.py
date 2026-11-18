"""FOTOMATES_TURNING: a wheel caught between two numbers is one of the two, the higher, and the
TURNING line names it (document of 17 Nov 2026).

    python -m unittest discover tests

The prompt tests import extract_readings in a separate Python with the environment set, so the
module the other tests use is never reloaded (audit_readings keeps references into it).
"""
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import audit_readings  # noqa: E402
import extract_readings  # noqa: E402
from extract_readings import _cells_digits, _line_digits, _reading_digits, extract_turning  # noqa: E402

PROBE = r'''
import json, extract_readings as e
print(json.dumps(dict(system=e._SYSTEM_PROMPT, user=e._USER_PROMPT, box=e._BOX_SYSTEM_PROMPT,
                      note=e.settings_note(), turning=e.TURNING)))
'''


def prompts(prompt, turning):
    """The prompts extract_readings builds with FOTOMATES_PROMPT and FOTOMATES_TURNING set; the
    error text when the import fails."""
    env = dict(os.environ, FOTOMATES_PROMPT=prompt, FOTOMATES_TURNING='1' if turning else '0',
               FOTOMATES_OCR_URL='http://127.0.0.1:1/v1', FOTOMATES_OCR_MODEL='test-model',
               FOTOMATES_FEWSHOT='0', FOTOMATES_REPEAT_HINT='0', FOTOMATES_CROP_FIRST='1',
               FOTOMATES_REREAD_MODE='crop', PYTHONIOENCODING='utf-8')
    done = subprocess.run([sys.executable, '-c', PROBE], cwd=ROOT, env=env, capture_output=True, text=True,
                          encoding='utf-8', timeout=120)
    if done.returncode:
        return done.stderr
    return json.loads(done.stdout.strip().splitlines()[-1])


ANSWER = 'TYPE: MECHANICAL\nCELLS: 7 0 6 9 1 9\nREADING: 706919\nDECIMALS: 1\nQUALITY: 8\n'


class PromptTest(unittest.TestCase):
    def test_off_is_as_before(self):
        for prompt in ('v2', 'v3', 'v4'):
            with self.subTest(prompt=prompt):
                built = prompts(prompt, False)
                self.assertIsInstance(built, dict, built)
                self.assertIn('write the lower one', built['user'])
                for text in (built['system'], built['user'], built['box']):
                    self.assertNotIn('TURNING', text)
                self.assertNotIn('turning', built['note'])

    def test_on(self):
        for prompt in ('v2', 'v3', 'v4'):
            with self.subTest(prompt=prompt):
                built = prompts(prompt, True)
                self.assertIsInstance(built, dict, built)
                self.assertNotIn('write the lower one', built['user'])
                self.assertIn('never a third', built['user'])
                self.assertTrue(built['system'].endswith('TURNING: 5 0'))
                self.assertIn('turning-wheel rule and TURNING line', built['note'])
                # the line count of the answer is not changed: TURNING is an optional line
                self.assertEqual(built['system'].count('exactly these'), 1)

    def test_the_box_prompt_has_one_box_line_and_the_turning_text(self):
        built = prompts('v2', True)
        self.assertEqual(built['box'].count('\nBOX:'), 1)
        self.assertIn('TURNING: the position of that cell', built['box'])
        self.assertTrue(built['box'].rstrip().endswith('TURNING: 5 0'))

    def test_compact_is_refused(self):
        built = prompts('compact', True)
        self.assertIsInstance(built, str)
        self.assertIn('FOTOMATES_TURNING needs FOTOMATES_PROMPT', built)


class ExtractTurningTest(unittest.TestCase):
    def test_pairs(self):
        cases = [(ANSWER + 'TURNING: 5 0', '2:0'),
                 (ANSWER + 'TURNING: 5:0', '2:0'),
                 (ANSWER + 'TURNING: 5 0, 6 8', '2:0,1:8'),
                 (ANSWER + 'TURNING: 5 4', None),      # 4 is not the number before 1
                 (ANSWER + 'TURNING: 9 0', None),      # no ninth cell
                 (ANSWER + 'TURNING: NONE', None),
                 (ANSWER, None),
                 ('TYPE: MECHANICAL\nCELLS: 4 9 7 8 0\nREADING: 49780\nTURNING: 5 9', '1:9'),
                 ('TYPE: MECHANICAL\nCELLS: NONE\nREADING: UNREADABLE\nTURNING: 1 0', None)]
        for text, expected in cases:
            with self.subTest(text=text.splitlines()[-1]):
                self.assertEqual(extract_turning(text), expected)

    def test_the_turning_digits_are_not_the_reading(self):
        text = ANSWER + 'TURNING: 5 0'
        self.assertEqual(_cells_digits(text), '706919')
        self.assertEqual(_reading_digits(text), '706919')

    def test_the_turning_digits_get_no_probability(self):
        pieces = ['TYPE', ':', ' MECH', 'ANICAL', '\n', 'CELLS', ':', ' ', '7', ' ', '0', ' ', '6', ' ', '9', ' ', '1',
                  ' ', '9', '\n', 'READING', ':', ' ', '7', '0', '6', '9', '1', '9', '\n', 'TURNING', ':', ' ', '5', ' ',
                  '0']
        tokens = [NS(token=piece, logprob=-0.01, top_logprobs=[]) for piece in pieces]
        probabilities = _line_digits(tokens, 'CELLS:')[0]
        self.assertEqual(len(probabilities), 6)


class ColumnTest(unittest.TestCase):
    def test_the_column_is_saved(self):
        name = 'AL0060991_20261031125226_READER_70691_57-5701_ExpAktiv_41-33_19-82.jpg'
        text = ANSWER + 'TURNING: 5 0'
        result = NS(text=text, digit_confidence=[0.99] * 6, digit_alternatives=[], digit_margin=None,
                    digit_runner_up=None, image_report=dict(width=240, height=400), evidence_source='crop(2.7x)',
                    device_type='mechanical', code_confidence=[], clear=None, decimal_digits=1,
                    reading_seconds=1.0)
        one = dict(result=result, cells='706919', whole='70691', decimals=1, match='yes', parts=None, score=0.9,
                   scale='crop(2.7x)', role='crop')
        read = dict(one, reread=False, seconds=1.0, box_answer='100,300,700,450', box_reason=None, reads=[one])
        with mock.patch.object(audit_readings, 'read_with_retries', return_value=read):
            row = audit_readings.read_one(Path(name), None)
        self.assertEqual(row['turning'], '2:0')
        self.assertEqual((row['status'], row['match'], row['score']), ('done', 'yes', 0.9))
        db = audit_readings.connect(':memory:')
        audit_readings.save(db, row)
        self.assertEqual(db.execute('SELECT turning FROM readings').fetchone(), ('2:0',))
        self.assertIn('turning', [name for name, _ in audit_readings.DETAIL_COLUMNS])

    def test_switch_is_off_by_default(self):
        if 'FOTOMATES_TURNING' not in os.environ:
            self.assertFalse(extract_readings.TURNING)


if __name__ == '__main__':
    unittest.main()
