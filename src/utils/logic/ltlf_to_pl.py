from .formula import Formula, Predicate, AND, OR, NOT, TRUE, FALSE, IMPLIES

class LTLfNode():
    def to_propositional(self, sequence_len, current_time):
        raise NotImplementedError("Subclasses must implement this method.")

class LTLfProposition(LTLfNode):
    def __init__(self, name):
        self.name = name

    def to_propositional(self, predicates, sequence_len, current_time):
        return predicates[f"{self.name}_{current_time}"]

class LTLfNegation(LTLfNode):
    def __init__(self, formula):
        self.formula = formula
    def to_propositional(self, predicates, sequence_len, current_time):
        converted = self.formula.to_propositional(predicates, sequence_len, current_time)
        return NOT(converted)

class LTLfAnd(LTLfNode):
    def __init__(self, subformulas):
        self.subformulas = subformulas            
    def to_propositional(self, predicates, sequence_len, current_time):
        converted = [f.to_propositional(predicates, sequence_len, current_time) for f in self.subformulas]
        return AND(converted)
    
class LTLfOr(LTLfNode):
    def __init__(self, subformulas):
        self.subformulas = subformulas 
    def to_propositional(self, predicates, sequence_len, current_time):
        converted = [f.to_propositional(predicates, sequence_len, current_time) for f in self.subformulas]
        return OR(converted)

# material implication A -> B = ¬A v B
class LTLfImplication(LTLfNode):
    def __init__(self, subformulas):
        self.left = subformulas[0]
        self.right = subformulas[1]
    def to_propositional(self, predicates, sequence_len, current_time):
        converted_left = self.left.to_propositional(predicates, sequence_len, current_time)
        converted_right = self.right.to_propositional(predicates, sequence_len, current_time)
        return OR([NOT(converted_left), converted_right])


# class LTLfImplication(LTLfNode):
#     def __init__(self, subformulas):
#         self.left = subformulas[0]
#         self.right = subformulas[1]
#     def to_propositional(self, predicates, sequence_len, current_time):
#         converted_left = self.left.to_propositional(predicates, sequence_len, current_time)
#         converted_right = self.right.to_propositional(predicates, sequence_len, current_time)
#         return IMPLIES([converted_left, converted_right])

class LTLfEquivalence(LTLfNode):
    def __init__(self, subformulas):
        self.left = subformulas[0]
        self.right = subformulas[1]
    def to_propositional(self, predicates, sequence_len, current_time):
        implication_1 = LTLfImplication([self.left, self.right]).to_propositional(predicates, sequence_len, current_time)
        implication_2 = LTLfImplication([self.right, self.left]).to_propositional(predicates, sequence_len, current_time)
        return AND([implication_1, implication_2])

class LTLfAlways(LTLfNode):
    def __init__(self, formula):
        self.formula = formula
    def to_propositional(self, predicates, sequence_len, current_time):
        subformulas = []
        for t in range(current_time, sequence_len):
            subformula = self.formula.to_propositional(predicates, sequence_len, t)
            subformulas.append(subformula)
        if subformulas:
            return AND(subformulas)
        else:
            raise ValueError("No argument give.")
        
class LTLfEventually(LTLfNode):
    def __init__(self, formula):
        self.formula = formula
    def to_propositional(self, predicates, sequence_len, current_time):
        subformulas = []
        for t in range(current_time, sequence_len):
            subformula = self.formula.to_propositional(predicates, sequence_len, t)
            subformulas.append(subformula)
        if subformulas:
            return OR(subformulas)
        else:
            raise ValueError("No argument give.")

class LTLfUntil(LTLfNode):
    def __init__(self, subformulas):
        self.left = subformulas[0]
        self.right = subformulas[1]
    def to_propositional(self, predicates, sequence_len, current_time):
        if current_time >= sequence_len:
            return FALSE()
        
        converted_right = self.right.to_propositional(predicates, sequence_len, current_time)

        if current_time == sequence_len - 1:
            # return converted_right
            return OR([converted_right, FALSE()]) 
        
        converted_left = self.left.to_propositional(predicates, sequence_len, current_time)
        until_next = LTLfUntil([self.left, self.right]).to_propositional(predicates, sequence_len, current_time + 1)
            
        return OR([converted_right, AND([converted_left, until_next])])

class LTLfRelease(LTLfNode):
    def __init__(self, subformulas):
        self.left = subformulas[0]
        self.right = subformulas[1]
    def to_propositional(self, predicates, sequence_len, current_time):
        neg_left = LTLfNegation(self.left)
        neg_right = LTLfNegation(self.right)
        until_formula = LTLfUntil([neg_left, neg_right])
        neg_until = LTLfNegation(until_formula)
        return neg_until.to_propositional(predicates, sequence_len, current_time)


class LTLfNext(LTLfNode):
    def __init__(self, formula):
        self.formula = formula
    def to_propositional(self, predicates, sequence_len, current_time):
        if current_time >= sequence_len-1:
            return FALSE()
        return self.formula.to_propositional(predicates, sequence_len, current_time + 1)
    
class LTLfWeakUntil(LTLfNode):
    def __init__(self, subformulas):
        self.left = subformulas[0]
        self.right = subformulas[1]
    def to_propositional(self, predicates, sequence_len, current_time):
        if current_time >= sequence_len - 1:
            return TRUE()
        
        converted_right = self.right.to_propositional(predicates, sequence_len, current_time)
        
        converted_left = self.left.to_propositional(predicates, sequence_len, current_time)
        until_next = LTLfWeakUntil([self.left, self.right]).to_propositional(predicates, sequence_len, current_time + 1)
            
        return OR([converted_right, AND([converted_left, until_next])])

class LTLfWeakNext(LTLfNode):
    def __init__(self, formula):
        self.formula = formula
    def to_propositional(self, predicates, sequence_len, current_time):
        if current_time >= sequence_len-1:
            return TRUE()
        return self.formula.to_propositional(predicates, sequence_len, current_time + 1)


