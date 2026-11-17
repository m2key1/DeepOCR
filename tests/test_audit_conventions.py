import unittest

from scripts.audit_conventions import count_marks

LINE = '‘Haus-Tür’ - 1870-80, x-y – ”s'


class AuditCounts(unittest.TestCase):
    def test_only_a_dash_counts_against_the_typographic_dash(self):
        self.assertEqual(count_marks('typographic_punctuation', LINE), (4, 2))

    def test_every_hyphen_counts_against_the_oblique_hyphen(self):
        self.assertEqual(count_marks('oblique_hyphen', LINE + ' Waſſer⸗'), (1, 4))


if __name__ == '__main__':
    unittest.main()
