"""The three rules of 19 November 2026: FOTOMATES_KEEP_LONGER, FOTOMATES_TYPE_FROM_READS,
FOTOMATES_ACCEPT_AGREED. All off by default.

    python -m unittest discover tests
"""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest import mock

os.environ.setdefault('FOTOMATES_OCR_URL', 'http://127.0.0.1:1/v1')
os.environ.setdefault('FOTOMATES_OCR_MODEL', 'test-model')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import audit_readings  # noqa: E402
from audit_readings import read_best, read_details, reads_settled, rules_note, type_from_reads  # noqa: E402

PHOTO = Path('AL0010556_20261023092021_JURGENDU_1462__ExpAktiv_41-33_19-82.jpg')
BOX = dict(box=(100, 100, 600, 300), why=None, answer='100,100,600,300', seconds=1.0)


def read(cells, score, sent='crop(2.7x)', kind='digital', source='LCD'):
    text = f"TYPE: {'M' if kind == 'mechanical' else 'D'}\nCELLS: {cells or 'UNREADABLE'}\nSOURCE: {source}"
    result = NS(text=text, reading_seconds=1.0, digit_confidence=[0.9] * len(cells), digit_alternatives=[],
                image_report={}, device_type=kind, code_confidence=[], evidence_source=sent, clear=None,
                decimal_digits=None)
    return dict(result=result, cells=cells, score=score, scale=1.0, whole=cells, decimals=0, match='no', parts=None)


def run(reads, **settings):
    """read_best under FOTOMATES_KEEP=agree with the reads given in order: crop, whole, third."""
    made = []

    def scored(path, saved, model=None, scale=1.0, crop=False, box=None, margin=None):
        made.append((scale, crop))
        return dict(reads[len(made) - 1])

    values = dict(CROP_FIRST=True, KEEP='agree', REREAD_EMPTY=True, REREAD_SCALE=3.0, REREAD_MODE='enlarge',
                  KEEP_LONGER=False)
    values.update(settings)
    with mock.patch.multiple(audit_readings, read_scored=scored, find_box=lambda path, model=None: dict(BOX),
                             _black_frame=lambda path: None, **values):
        return read_best(PHOTO, '1462')


class KeepLonger(unittest.TestCase):
    CUT = [read('0011', 0.9), read('00117317', 0.8, 'enlarged(2x)'), read('0011', 0.9, 'enlarged(3x)')]

    def test_off_the_two_reads_that_agree_are_kept(self):
        kept = run(self.CUT)
        self.assertEqual((kept['cells'], kept['score']), ('0011', 0.9))

    def test_on_the_whole_photo_read_that_extends_them_is_kept_with_a_capped_score(self):
        kept = run(self.CUT, KEEP_LONGER=True)
        self.assertEqual(kept['cells'], '00117317')
        self.assertEqual(kept['score'], audit_readings.DISAGREE_SCORE)
        self.assertEqual(kept['role'], 'whole')
        self.assertEqual(len(kept['others']), 2)

    def test_on_a_whole_photo_read_that_only_differs_changes_nothing(self):
        reads = [read('001173', 0.9), read('00217317', 0.8, 'enlarged(2x)'), read('001173', 0.9, 'enlarged(3x)')]
        self.assertEqual(run(reads, KEEP_LONGER=True)['cells'], '001173')

    def test_on_the_same_digits_in_both_reads_stay(self):
        kept = run([read('001462', 0.9), read('001462', 0.8, 'enlarged(2x)')], KEEP_LONGER=True)
        self.assertEqual((kept['cells'], kept['score'], kept['role']), ('001462', 0.9, 'crop'))

    def test_on_no_two_alike_and_the_whole_extends_the_best(self):
        reads = [read('0011', 0.6), read('00117317', 0.5, 'enlarged(2x)'), read('0019', 0.4, 'enlarged(3x)')]
        self.assertEqual(run(reads)['cells'], '0011')
        self.assertEqual(run(reads, KEEP_LONGER=True)['cells'], '00117317')


class TypeFromReads(unittest.TestCase):
    def test_wheels_seen_twice_outvote_one_display(self):
        reads = [read('1', 0.9, kind='mechanical', source='WHEELS'), read('1', 0.9, kind='mechanical', source='WHEELS'),
                 read('1', 0.9, kind='digital', source='LCD')]
        self.assertEqual(type_from_reads(reads, 'digital'), 'mechanical')

    def test_a_read_that_saw_digits_outvotes_a_bare_type(self):
        reads = [read('', None, kind='mechanical', source='NONE'), read('1', 0.9, kind='digital', source='LCD')]
        self.assertEqual(type_from_reads(reads, 'mechanical'), 'digital')

    def test_no_type_when_no_read_saw_digits_on_a_register(self):
        reads = [read('', None, kind='mechanical', source='NONE'), read('', None, kind='mechanical', source='NONE')]
        self.assertIsNone(type_from_reads(reads, 'mechanical'))
        self.assertIsNone(type_from_reads([read('12', 0.9, kind='mechanical', source='HANDWRITTEN')], 'mechanical'))

    def test_a_tie_keeps_the_kept_reads_type(self):
        reads = [read('1', 0.9, kind='mechanical', source='WHEELS'), read('1', 0.9, kind='digital', source='LCD')]
        self.assertEqual(type_from_reads(reads, 'digital'), 'digital')
        self.assertEqual(type_from_reads(reads, 'mechanical'), 'mechanical')

    def test_the_details_use_it_only_when_the_setting_is_on(self):
        reads = [dict(read('', None, kind='mechanical', source='NONE'), role='crop'),
                 dict(read('', None, kind='mechanical', source='NONE'), role='whole')]
        result = reads[0]['result']
        with mock.patch.object(audit_readings, 'TYPE_FROM_READS', False):
            self.assertEqual(read_details(result, None, True, reads=reads)['device_type'], 'mechanical')
        with mock.patch.object(audit_readings, 'TYPE_FROM_READS', True):
            self.assertIsNone(read_details(result, None, True, reads=reads)['device_type'])


class AcceptAgreed(unittest.TestCase):
    def test_one_read_made_is_outside_the_rule(self):
        self.assertTrue(reads_settled([read('001462', 0.3)], 0.3))
        self.assertTrue(reads_settled(None, None))

    def test_the_same_digits_settle_it_whatever_the_score(self):
        self.assertTrue(reads_settled([read('001462', 0.3), read('001462', 0.4)], 0.3))

    def test_the_same_length_needs_the_score(self):
        reads = [read('001462', 0.9), read('001468', 0.9)]
        self.assertTrue(reads_settled(reads, 0.7))
        self.assertFalse(reads_settled(reads, 0.69))
        self.assertFalse(reads_settled(reads, None))

    def test_different_lengths_or_a_single_read_with_digits_do_not(self):
        self.assertFalse(reads_settled([read('0014', 0.99), read('001462', 0.99)], 0.99))
        self.assertFalse(reads_settled([read('', None), read('001462', 0.99)], 0.99))


class Notes(unittest.TestCase):
    def test_nothing_is_said_while_all_are_off(self):
        with mock.patch.multiple(audit_readings, KEEP_LONGER=False, TYPE_FROM_READS=False, ACCEPT_AGREED=False):
            self.assertEqual(rules_note(), '')

    def test_each_rule_that_is_on_is_named(self):
        with mock.patch.multiple(audit_readings, KEEP_LONGER=True, TYPE_FROM_READS=True, ACCEPT_AGREED=True):
            note = rules_note()
        for words in ('a longer whole-photo read kept', 'meter type from all reads', 'accepted only when the reads agree'):
            self.assertIn(words, note)

    def test_off_by_default(self):
        for name in ('FOTOMATES_KEEP_LONGER', 'FOTOMATES_TYPE_FROM_READS', 'FOTOMATES_ACCEPT_AGREED'):
            if name not in os.environ:
                self.assertFalse(getattr(audit_readings, name.replace('FOTOMATES_', '')))


if __name__ == '__main__':
    unittest.main()
