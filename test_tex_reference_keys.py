import unittest
from tex_reference_keys import statement_reference


class StaticReferences(unittest.TestCase):
    prefix = r'\newtheorem{thm}{Theorem}[section]\newtheorem{lem}[thm]{Lemma}\begin{document}'

    def test_shared_counters_reset_at_section(self):
        source = self.prefix + r'\section{A}\begin{thm}A\end{thm}\begin{lem}B\end{lem}\section{B}\begin{lem}\label{x}C\end{lem}'
        self.assertEqual(statement_reference(source, source.index(r'\label{x}'))['keys'], ['Lemma 2.1'])

    def test_comments_are_not_counter_events(self):
        source = self.prefix + '\n%\\section{fake}\n' + r'\section{A}\begin{lem}\label{x}C\end{lem}'
        self.assertEqual(statement_reference(source, source.index(r'\label{x}'))['keys'], ['Lemma 1.1'])

    def test_dynamic_counter_override_unknown(self):
        source = self.prefix + r'\setcounter{thm}{7}\begin{lem}\label{x}C\end{lem}'
        self.assertEqual(statement_reference(source, source.index(r'\label{x}'))['keys'], [])

    def test_body_conditional_unknown(self):
        source = self.prefix + r'\ifdraft\begin{lem}\label{x}C\end{lem}\fi'
        self.assertEqual(statement_reference(source, source.index(r'\label{x}'))['keys'], [])

    def test_not_inside_statement_unknown(self):
        source = self.prefix + r'\section{A}\begin{lem}C\end{lem} Outside.'
        self.assertEqual(statement_reference(source, source.index('Outside'))['keys'], [])

    def test_unrelated_counter_format_does_not_poison_target(self):
        source = (r'\newtheorem{intro}{Theorem}\renewcommand{\theintro}{\Alph{intro}}' + self.prefix +
                  r'\section{A}\begin{lem}\label{x}C\end{lem}')
        self.assertEqual(statement_reference(source, source.index(r'\label{x}'))['keys'], ['Lemma 1.1'])


if __name__ == '__main__':
    unittest.main()
