import unicodedata
import unittest

from utils.tokenizer import canonicalize


def nfd(text: str) -> str:
    return unicodedata.normalize('NFD', text)


class CyrillicHomoglyphs(unittest.TestCase):
    def test_russian_words_keep_their_letters(self):
        self.assertEqual(canonicalize('Фамилія какъ? Очень хорошо.'), 'Фамилія какъ? Очень хорошо.')

    def test_cyrillic_lookalike_inside_a_latin_word_becomes_latin(self):
        self.assertEqual(canonicalize('Gеschichtе dеs Hauses'), 'Geschichte des Hauses')

    def test_lookalike_beside_latin_diacritics_is_folded(self):
        self.assertEqual(canonicalize('Mäкler'), nfd('Mäkler'))

    def test_each_word_is_judged_on_its_own(self):
        self.assertEqual(canonicalize('Casus obliquus, лошадкѣ, oder Gеld'), 'Casus obliquus, лошадкѣ, oder Geld')

    def test_word_made_only_of_lookalikes_stays_cyrillic(self):
        self.assertEqual(canonicalize('кара мера'), 'кара мера')

    def test_word_with_other_cyrillic_letters_is_left_alone(self):
        self.assertEqual(canonicalize('Мeтер'), 'Мeтер')

    def test_folding_is_idempotent(self):
        text = 'Gеschichtе, Фамилія, Mäкler'
        self.assertEqual(canonicalize(canonicalize(text)), canonicalize(text))


if __name__ == '__main__':
    unittest.main()
