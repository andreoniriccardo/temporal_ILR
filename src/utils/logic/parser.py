from pathlib import Path
import os
from lark import Lark, Transformer
from .ltlf_to_pl import (
    LTLfProposition,
    LTLfNegation,
    LTLfAnd,
    LTLfOr,
    LTLfUntil,
    LTLfRelease,
    LTLfNext,
    LTLfAlways,
    LTLfEventually,
    LTLfImplication,
    LTLfEquivalence,
    LTLfWeakUntil,
    LTLfWeakNext
)
from .formula import Predicate

CURRENT_SCRIPT_DIR = Path(os.path.abspath(__file__)).parent

PL_GRAMMAR_FILE = CURRENT_SCRIPT_DIR / "parser/pl.lark"
LTLF_GRAMMAR_FILE = CURRENT_SCRIPT_DIR / "parser/ltlf.lark"

class LTLfTransformer(Transformer):
    """LTLf Transformer."""

    def __init__(self, sequence_len, activities, predicates):
        """Initialize."""
        super().__init__()
        self.sequence_len = sequence_len
        self.predicates = predicates


    def start(self, args):
        """Entry point."""
        # check_(len(args) == 1)
        return args[0]
    
    def ltlf_not(self, args):
        """Parse LTLf Not."""
        if len(args) == 1:
            return args[0]
        f = args[-1]
        for _ in args[:-1]:
            f = LTLfNegation(f)
        return f
    
    def ltlf_and(self, args):
        """Parse LTLf And."""
        if len(args) == 1:
            return args[0]
        if (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfAnd(subformulas)
        raise ValueError("Parsing error.")
    
    def ltlf_or(self, args):
        """Parse LTLf Or."""
        if len(args) == 1:
            return args[0]
        if (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfOr(subformulas)
        raise ValueError("Parsing error.")
    
    def ltlf_implication(self, args):
        """Parse LTLf Implication."""
        if len(args) == 1:
            return args[0]
        if (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfImplication(subformulas)
        raise ValueError("Parsing error.")
    
    def ltlf_equivalence(self, args):
        """Parse LTLf Equivalence."""
        if len(args) == 1:
            return args[0]
        if (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfEquivalence(subformulas)
        raise ValueError("Parsing error.")
    
    def ltlf_always(self, args, current_time=0):
        """Parse LTLf Always."""
        if len(args) == 1:
            return args[0]
        f = args[-1]
        for _ in args[:-1]:
            f = LTLfAlways(f)
        return f
    
    def ltlf_eventually(self, args):
        """Parse LTLf Eventually."""
        if len(args) == 1:
            return args[0]
        f = args[-1]
        for _ in args[:-1]:
            f = LTLfEventually(f)
        return f
    
    def ltlf_until(self, args):
        """Parse LTLf Until."""
        if len(args) == 1:
            return args[0]
        if (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfUntil(subformulas)
        raise ValueError("Parsing error.")
    
    def ltlf_release(self, args):
        """Parse LTLf Release."""
        if len(args) == 1:
            return args[0]
        if (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            return LTLfRelease(subformulas)
        raise ValueError("Parsing error.")
    
    def ltlf_next(self, args):
        """Parse LTLf Next."""
        if len(args) == 1:
            return args[0]
        f = args[-1]
        for _ in args[:-1]:
            f = LTLfNext(f)
        return f
    
    def ltlf_weak_until(self, args):
        """Parse LTLf Until."""

        if len(args) == 1:
            return args[0]
        if (len(args) - 1) % 2 == 0:
            subformulas = args[::2]
            # print(subformulas)
            return LTLfWeakUntil(subformulas)
        raise ValueError("Parsing error.")
    
    def ltlf_weak_next(self, args):
        """Parse LTLf Weak Next."""
        if len(args) == 1:
            return args[0]
        f = args[-1]
        for _ in args[:-1]:
            f = LTLfWeakNext(f)
        return f

    def ltlf_symbol(self, args):
        """Parse LTLf Symbol."""
        token = args[0]
        symbol = str(token)
        return LTLfProposition(symbol)    
    
    def ltlf_wrapped(self, args):
        """Parse LTLf wrapped formula."""
        if len(args) == 1:
            return args[0]
        if len(args) == 3:
            _, formula, _ = args
            return formula
        raise ValueError("Parsing error.")
    

_ltlf_parser_lark = LTLF_GRAMMAR_FILE.read_text()
class LTLfParser:
    """LTLf Parser class."""

    def __init__(self, sequence_len:int, activities:list):
        """Initialize."""
        self.sequence_len = sequence_len
        self.activities = activities
        self.n_activities = len(self.activities)

        self.predicates = {}

        i = 0
        for t in range(self.sequence_len):
            for s in self.activities:
                p = f'{s}_{t}'
                self.predicates[p] = Predicate(p, i)
                i+=1
        
        self._transformer = LTLfTransformer(self.sequence_len, self.activities, self.predicates)
        self._parser = Lark(
            _ltlf_parser_lark, parser="lalr", import_paths=[str(CURRENT_SCRIPT_DIR / "parser")]
        )

    def __call__(self, text):
        """Call."""
        tree = self._parser.parse(text)
        formula = self._transformer.transform(tree)
        return formula