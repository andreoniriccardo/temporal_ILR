formulas = []

formulas_names = []

# init(A)
formulas.append("a")
formulas_names.append("Init(_a)")
#responded existence(A, B)
formulas.append("F a -> F b")
formulas_names.append("Responded existence(_a,_b)")
#response(A,B)
formulas.append("G( a -> F b)")
formulas_names.append("Response(_a,_b)")
#precedence(A,B)
formulas.append("(! b U a) | G (! b)")
formulas_names.append("Precedence(_a,_b)")
#succession(A,B)
# formulas.append('G( a -> F b) & (! b U a) | G (! b)') # old
formulas.append('G( a -> F b) & ((! b U a) | G (! b))') # correct
formulas_names.append("Succession(_a,_b)")
#alternate response(A,B)
formulas.append('G( a -> X (! a U b))')
formulas_names.append("Alternate response(_a,_b)")
#alternate precedence(A,B)'
formulas.append('(! b U a) | G(! b) & G (b -> X ((!b U a) | G (!b)))')
formulas_names.append("Alternate precedence(_a,_b)")
#alternate succession
formulas.append('G( a -> X (! a U b)) & (! b U a) | G (! b)')
formulas_names.append("Alternate succesion(_a, _b)")
#chain response
formulas.append('G(a -> X b)')
formulas_names.append("Chain response(_a, _b)")
#chain precedence
formulas.append('G((X b) -> a)')
formulas_names.append("Chain precedence(_a,_b)")
#not co-existence
formulas.append('!(F a & F b)')
formulas_names.append("Not co-existence(_a,_b)")
#not succession
formulas.append('G (a -> ! (F b))')
formulas_names.append("Not succession(_a,_b)")
#not chain succession
formulas.append('G(a -> ! (X b))')
formulas_names.append("Not chain succession(_a,_b)")
#choice
formulas.append('F a | F b')
formulas_names.append("Choice(_a,_b)")
#existence(a,2)
formulas.append("F(a & X ( F(a)))")
formulas_names.append("Existence(_a,2)")
#absence(a,2)
formulas.append("!(F(a & X ( F(a))))")
formulas_names.append("Absence(_a,2)")
#exactly(a,2)
formulas.append("(F(a & X ( F(a)))) & !(  F(a & X ( F ( F (a & X( F(a)))))  )  )")
formulas_names.append("Exactly(_a,2)")
#co-existence(a,b)
formulas.append("(F(b)) -> (F(a))")
formulas_names.append("Co-existence(_a,_b)")
#chain succession(a,b)
formulas.append("G((a <-> X(b)))")
formulas_names.append("Chain succession(_a,_b)")
#exclusive choice
formulas.append('(F a | F b) & ! (F a & F b)')
formulas_names.append("Exclusive choice(_a,_b)")
# RA: response(a, b) AND precedence(a, c2)
# formulas.append('G( a -> F b) & ((! c2 U a) | G (! c2))')
# formulas_names.append("Response(a,b) AND Precedence(a,c2)")