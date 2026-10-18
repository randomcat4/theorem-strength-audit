import unittest
from static_tex_context import extract_context,project
from strength_signals import analyze_pairs
from test_strength_signals import side,pair

class StaticTexContext(unittest.TestCase):
    def test_literal_mathbb_alias(self):
        text=r'\def\Q{{\mathbb Q}} Let $x\in\Q$.'
        context=extract_context(text,text.index('Let'))
        self.assertIn(r'\Q',context['macros'])
        self.assertIn(r'\mathbb Q',project(r'$x\in\Q$',context)[0])

    def test_color_identity_wrapper(self):
        text=r'\newcommand{\blue}[1]{\textcolor{blue}{#1}} Let X.'
        context=extract_context(text,text.index('Let'))
        self.assertEqual(project(r'a\blue{b}c',context)[0],'abc')

    def test_nested_color_wrapper(self):
        text=r'\newcommand{\blu}[1]{{\color{black}{#1}}} Let X.'
        context=extract_context(text,text.index('Let'))
        self.assertEqual(project(r'\blu{x^{\blu{2}}}',context)[0],'x^{2}')

    def test_color_macro_with_extra_brackets_not_identity(self):
        text=r'\newcommand{\blue}[1]{{\color{blue}[#1]}} Let X.'
        self.assertFalse(extract_context(text,text.index('Let'))['macros'])

    def test_redefinition_rejected(self):
        text=r'\def\Q{\mathbb Q}\def\Q{\mathbb R} Let X.'
        self.assertNotIn(r'\Q',extract_context(text,text.index('Let'))['macros'])

    def test_after_statement_not_used(self):
        text=r'Let X. \def\Q{\mathbb Q}'
        self.assertFalse(extract_context(text,0)['macros'])

    def test_conditional_definition_rejected(self):
        text=r'\iftrue\def\Q{\mathbb Q}\fi Let X.'
        self.assertFalse(extract_context(text,text.index('Let'))['macros'])

    def test_parameterized_math_not_literal_alias(self):
        text=r'\newcommand{\Q}[1]{\mathbb{Q}^{#1}} Let X.'
        self.assertFalse(extract_context(text,text.index('Let'))['macros'])

    def test_malicious_supplied_map_cannot_expand_arbitrary_body(self):
        self.assertEqual(project(r'\a',{'macros':{r'\a':{'body':r'\input{elsewhere}', 'arity':0}}})[0],r'\a')

    def test_known_macro_color_only_difference_filtered(self):
        text=r'\newcommand{\blue}[1]{\textcolor{blue}{#1}} Let X.'
        a=side('p',r'Let $x>0$. Then $f(x)=2$.',version=1)
        b=side('p',r'Let $x>0$. Then $f(x)=\blue{2}$.',version=2)
        b['static_tex_context']=extract_context(text,text.index('Let'))
        result=analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertTrue(result['rewrite_equivalence']['suppresses_strength_alert'])

    def test_unknown_color_named_macro_not_filtered(self):
        a=side('p',r'Let $x>0$. Then $f(x)=2$.',version=1)
        b=side('p',r'Let $x>0$. Then $f(x)=\blue{2}$.',version=2)
        result=analyze_pairs([pair(a,b)])[0]['algorithm']
        self.assertFalse(result['rewrite_equivalence']['suppresses_strength_alert'])

    def test_color_wrapper_cannot_hide_actual_exponent_change(self):
        text=r'\newcommand{\blue}[1]{\textcolor{blue}{#1}} Let X.'
        a=side('p',r'Let $x>0$. Then $f(x)=2$.',version=1)
        b=side('p',r'Let $x>0$. Then $f(x)=\blue{2^2}$.',version=2)
        b['static_tex_context']=extract_context(text,text.index('Let'))
        self.assertFalse(analyze_pairs([pair(a,b)])[0]['algorithm']['rewrite_equivalence']['suppresses_strength_alert'])

if __name__=='__main__':unittest.main()
