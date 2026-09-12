import unittest
from i18n import MESSAGES,LANGUAGES,translate,DEFAULT_LANGUAGE
from gpu_bench import MODES,describe

class TranslationTests(unittest.TestCase):
    def test_catalog_roundtrip(self):
        for row in MESSAGES:
            values={'gpu':'RX 9070 XT','count':'17','total':'26','mode':'INT2_MATRIX','format':'BF16'}
            for source in row:
                for language,index in [('pt-BR',0),('en',1),('zh-CN',2)]:
                    self.assertEqual(translate(source.format(**values),language),row[index].format(**values))

    def test_all_mode_labels(self):
        for mode in MODES:
            for lang in ('en','zh-CN'):
                self.assertNotEqual(translate(describe(mode),lang),describe(mode),mode)

    def test_default_and_technical_identifiers(self):
        self.assertEqual(DEFAULT_LANGUAGE,'en')
        self.assertEqual(translate('Iniciar benchmark'),'Start benchmark')
        for lang in LANGUAGES.values():
            self.assertEqual(translate('gfx1201 / FP32_VECTOR / HIP error 5',lang),'gfx1201 / FP32_VECTOR / HIP error 5')

if __name__=='__main__': unittest.main()
