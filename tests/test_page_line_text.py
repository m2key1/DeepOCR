import unittest

from utils.prepare_data import extract_page_line_regions, extract_xml_textlines

PAGE_NS = 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'
COORDS = '<Coords points="10,10 200,10 200,40 10,40"/>'


def word(text: str) -> str:
    return f'<Word><Coords points="10,10 50,10 50,40 10,40"/><TextEquiv><Unicode>{text}</Unicode></TextEquiv></Word>'


def page(*lines: str) -> str:
    return (f'<PcGts xmlns="{PAGE_NS}"><Page imageFilename="page.jpg"><TextRegion id="r1">'
            + ''.join(lines) + '</TextRegion></Page></PcGts>')


def line(line_id: str, inner: str) -> str:
    return f'<TextLine id="{line_id}">{COORDS}{inner}</TextLine>'


class PageLineText(unittest.TestCase):
    def test_line_text_wins_over_the_words_written_before_it(self):
        xml = page(line('l1', word('Hänen') + word('elämänsä') + '<TextEquiv><Unicode>Hänen elämänsä</Unicode></TextEquiv>'))
        _, regions = extract_page_line_regions(xml)
        self.assertEqual(regions, [('l1', [(10, 10), (200, 10), (200, 40), (10, 40)], 'Hänen elämänsä')])

    def test_words_are_joined_when_the_line_has_no_text_of_its_own(self):
        xml = page(line('l1', word('ambos') + word('asinaraõ')))
        _, regions = extract_page_line_regions(xml)
        self.assertEqual([text for _, _, text in regions], ['ambos asinaraõ'])

    def test_line_without_words_keeps_its_text(self):
        xml = page(line('l1', '<TextEquiv><Unicode>manoel De ceita o escreui</Unicode></TextEquiv>'))
        _, regions = extract_page_line_regions(xml)
        self.assertEqual([text for _, _, text in regions], ['manoel De ceita o escreui'])

    def test_line_text_file_reads_the_line_not_its_first_word(self):
        xml = page(line('l1', word('i') + word('en') + '<TextEquiv><Unicode>i en beholder</Unicode></TextEquiv>'))
        self.assertEqual(extract_xml_textlines(xml), ['i en beholder'])


if __name__ == '__main__':
    unittest.main()
