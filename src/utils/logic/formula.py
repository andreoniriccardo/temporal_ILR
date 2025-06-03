import torch

class Formula(torch.nn.Module):
    def __init__(self, sub_formulas):
        '''
        Iterates through the subformulas and, for each subformula iterates through its predicates (i.e. atomic propositions).
        Adds all the atomic propositons to the predicates attribute.
        '''
        super().__init__()
        if sub_formulas is not None:
            self.sub_formulas = sub_formulas
            self.predicates = list(set([p for sf in self.sub_formulas for p in sf.predicates]))

        self.input_tensor = None

    def function(self, truth_values):
        '''
        function() method is defined by each subclass of Formula (AND, OR, NOT)
        '''
        pass

    def boost_function(self, truth_values, delta):
        '''
        boost_function() method is defined by each subclass of Formula (AND, OR, NOT)
        '''
        pass

    def get_name(self, parenthesis=False):
        '''
        get_name() method is defined by each subclass of Formula (AND, OR, NOT)
        '''
        pass

    def forward(self, truth_values):
        '''
        The forward pass of a Formula instance is evaluated by iterating the forward() method
        of each subformula, and finally the function() method of the Formula instance itself.
        '''
        inputs = []
        for sf in self.sub_formulas:
            inputs.append(sf.forward(truth_values))

        if len(inputs) > 1:
            self.input_tensor = torch.concat(inputs, 1)
        else:
            self.input_tensor = inputs[0]
        return self.function(self.input_tensor)

    def backward(self, delta, randomized=False):
        '''
        to be defined
        '''
        deltas = self.boost_function(self.input_tensor, delta)
        if randomized:
            deltas = deltas * torch.rand(deltas.shape)

        for sf, d in zip(self.sub_formulas, deltas.t()):
            sf.backward(torch.unsqueeze(d, 0).t())


    def get_delta_tensor(self, truth_values, method='max'):
        '''
        Aggregate the deltas of each atomic proposition in a tensor.
        The tensor is then used to update the truth values.
        '''
        indices = []
        deltas = []
        for p in self.predicates:
            i, d = p.aggregate_deltas(method)
            p.reset_deltas()
            indices.append(i)
            deltas.append(d)

        delta_tensor = torch.zeros_like(truth_values)
        delta_tensor[..., indices] = torch.concat(deltas, 1).type(torch.float)
        return delta_tensor

class Predicate(Formula):
    '''
    Class for atomic propositions.
    '''
    def __init__(self, name, index):
        '''
        name: name of the atomic proposition.
        index: index of the atomic proposition inside the truth_values tensor.
        deltas: list of deltas accumulated by the backward pass.
        predicates: list of atomic propositions. For atomic propositions predicates is just the atomic proposition itself.
        '''
        super().__init__(None)
        self.name = name
        self.index = index
        self.deltas = []
        self.predicates = [self]

    def forward(self, truth_values):
        '''
        The forward pass, for an atomic proposition, returns the truth value of the atomic proposition itself.
        '''
        return torch.unsqueeze(truth_values[:, self.index], 1)
    
    def backward(self, delta, randomized=False):  # TODO: implement the usage of randomized
        '''
        The backward pass, for an atomic proposition, appends the delta to the deltas list.
        '''
        self.deltas.append(delta)
        # print(self.name, 'deltas', self.deltas)

    def get_name(self, parenthesis=False):
        '''
        Returns the name of the atomic proposition.
        '''
        return self.name
    
    def aggregate_deltas(self, method='max'):

        if method == 'mean':
            deltas = torch.concat(self.deltas, 1)

            return self.index, torch.nan_to_num(
                torch.sum(deltas, 1, keepdim=True) / torch.sum(deltas != 0.0, 1, keepdim=True))
        if method == 'max':
            deltas = torch.concat(self.deltas, 1)
            abs_deltas = deltas.abs()

            i = torch.argmax(abs_deltas, 1, keepdim=True)

            return self.index, torch.gather(deltas, 1, i)
        
    def reset_deltas(self):
        self.deltas = []

    
class NOT(Formula):
    def __init__(self, sub_formula):
        '''
        The instance of NOT class is instantiated as the argument (i.e. subformula)
        of the NOT operator.
        '''
        super().__init__([sub_formula])
    
    def get_name(self, parenthesis=False):
        '''
        TO BE DESCRIBED
        '''
        return 'NOT(' + self.sub_formulas[0].get_name() + ')'
    
    def function(self, truth_values):
        '''
        The forward pass in the ILR for the NOT operator is defined as 1 - the truth values of the subformula.
        '''

        # New with padding, TRUE, FALSE
        pad_mask = (truth_values == -1.0)
        false_mask = (truth_values == -2.0)
        true_mask = (truth_values == 2.0)

        # Compute NOT for normal values (0.0–1.0)
        not_values = 1 - truth_values

        # not(false) -> 1, not(true) -> 0
        not_values = torch.where(false_mask, 1.0, not_values)
        not_values = torch.where(true_mask, -1.0, not_values)

        # not(pad) -> 1
        not_values = torch.where(pad_mask, 1.0, not_values)

        self.forward_output = not_values
        return self.forward_output
    
    def boost_function(self, truth_values, delta):
        '''
        TO BE DESCRIBED
        '''
        return - delta
    
    
    
class AND(Formula):
    
    def get_name(self, parenthesis=False):
        '''
        TO BE DESCRIBED
        '''
        s = ''
        for sf in self.sub_formulas[:-1]:
            s += sf.get_name(parenthesis=True) + ' AND '

        s += self.sub_formulas[-1].get_name(parenthesis=True)

        if parenthesis:
            return '(' + s + ')'
        else:
            return s
        
    def function(self, truth_values):
        '''
        The forward pass in the ILR for the AND operator is defined as the minimum of the truth values of the subformulas.
        '''

        # Replace padding (-1.0) with 0, FALSE (-2.0) with 0, TRUE (2.0) with 1
        adjusted = torch.where(
            truth_values == -1.0, 0.0,            # Replace padding with 0
            torch.where(
                truth_values == -2.0, 0.0,        # Replace FALSE with 0
                torch.where(
                    truth_values == 2.0, 1.0,     # Replace TRUE with 1
                    truth_values                  # Keep normal values [0.0, 1.0]
                )
            )
        )

        # Compute minimum along sequence dimension
        min_vals, _ = torch.min(adjusted, dim=1)
        
        # Check if ALL elements were padding (-1.0)
        all_pad_mask = torch.all(truth_values == -1.0, dim=1)
        
        # Replace output with -1.0 for fully padded sequences
        final_output = torch.where(all_pad_mask, -1.0, min_vals)
        
        self.forward_output = final_output.unsqueeze(1)
        return self.forward_output


    def boost_function(self, truth_values, delta):
        '''
        TO BE DESCRIBED
        '''

        # Nuovo metodo - with padding
        # truth_values shape: [batch_size, n] (padded con -1)
        target = self.forward_output + delta # [batch_size, 1]
        # Identifico le posizioni valide (non padding)
        valid_positions = truth_values != -1  # [batch_size, n]

        # Per ogni data point del batch controllo la condizione target >= self.forward_output
        cond_case1 = (target >= self.forward_output) # [batch_size, 1]

        # Sostituisco il valore di padding con un valore alto (2.0) per calcolare il minimo in sicurezza
        t_valid = torch.where(valid_positions, truth_values, torch.full_like(truth_values, 2.0))
        min_vals = t_valid.min(dim=1, keepdim=True)[0]  # [batch_size, 1]

        # Mask caso 1: elementi dove truth_value < target E validi (non padding)
        mask_case1 = (truth_values < target) & valid_positions

        # Mask caso 2: elementi uguali al valore minimo E validi (non padding)
        mask_case2 = (truth_values == min_vals) & valid_positions

        # Combino le due maschere in base alla condizione iniziale
        combined_mask = torch.where(cond_case1, mask_case1, mask_case2)

        # Aggiorno i valori in base alla maschera
        t_new = torch.where(combined_mask, target, truth_values)
        # Mantenendo il padding
        t_new = torch.where(valid_positions, t_new, torch.full_like(truth_values, -1.0))

        return t_new - truth_values




    
        
class OR(Formula):
    def get_name(self, parenthesis=False):
        '''
        TO BE DESCRIBED
        '''
        s = ''
        for sf in self.sub_formulas[:-1]:
            s += sf.get_name(parenthesis=True) + ' OR '

        s += self.sub_formulas[-1].get_name(parenthesis=True)

        if parenthesis:
            return '(' + s + ')'
        else:
            return s
        
    def function(self, truth_values):
        '''
        The forward pass in the ILR for the OR operator is defined as the maximum of the truth values of the subformulas.
        '''

        # Replace padding (-1.0) with 0, FALSE (-2.0) with 0, TRUE (2.0) with 1
        adjusted = torch.where(
            truth_values == -1.0, 0.0,            # Replace padding with 0
            torch.where(
                truth_values == -2.0, 0.0,        # Replace FALSE with 0
                torch.where(
                    truth_values == 2.0, 1.0,     # Replace TRUE with 1
                    truth_values                  # Keep normal values [0.0, 1.0]
                )
            )
        )

        # Compute maximum along sequence dimension
        max_vals, _ = torch.max(adjusted, dim=1)
        
        # Check if ALL elements were padding (-1.0)
        all_pad_mask = torch.all(truth_values == -1.0, dim=1)
        
        # Replace output with -1.0 for fully padded sequences
        final_output = torch.where(all_pad_mask, -1.0, max_vals)
        
        self.forward_output = final_output.unsqueeze(1)
        return self.forward_output

    def boost_function(self, truth_values, delta):
        '''
        TO BE DESCRIBED
        '''
        # Nuovo metodo - with padding
        # truth_values shape: [batch_size, n] (padded con -1)
        target = self.forward_output + delta # [batch_size, 1]
        # Identifico le posizioni valide (non padding)
        valid_positions = truth_values != -1  # [batch_size, n]

        # Per ogni data point del batch controllo la condizione target < self.forward_output
        cond_case1 = (target < self.forward_output)

        # Sostituisco il valore di padding con un valore alto (-2.0) per calcolare il massimo in sicurezza
        t_valid = torch.where(valid_positions, truth_values, torch.full_like(truth_values, -2.0))
        max_vals = t_valid.max(dim=1, keepdim=True)[0]  # [batch_size, 1]

        # Mask caso 1: elementi dove truth_value >= target E validi (non padding)
        mask_case1 = (truth_values >= target) & valid_positions

        # Mask caso 2: elementi uguali al valore massimo E validi (non padding)
        mask_case2 = (truth_values == max_vals) & valid_positions

        # Combino le due maschere in base alla condizione iniziale
        combined_mask = torch.where(cond_case1, mask_case1, mask_case2)

        # Aggiorno i valori in base alla maschera
        t_new = torch.where(combined_mask, target, truth_values)
        # Mantenendo il padding
        t_new = torch.where(valid_positions, t_new, torch.full_like(truth_values, -1.0))

        return t_new - truth_values



    
class FALSE(Formula):
    def __init__(self):
        self.predicates = []
    def get_name(self, parenthesis=False):
        if parenthesis:
            return '(FALSE)'
        else:
            return 'FALSE'
    
    def forward(self, truth_values):
        self.input_tensor = truth_values
        return -2. * torch.ones(truth_values.shape[0], 1, 
                           device=truth_values.device,
                           dtype=truth_values.dtype,
                           requires_grad=truth_values.requires_grad)
    def backward(self, delta, randomized=False):
        return torch.zeros_like(self.input_tensor)

class TRUE(Formula):
    def __init__(self):
        self.predicates = []
    def get_name(self, parenthesis=False):
        if parenthesis:
            return '(TRUE)'
        else:
            return 'TRUE'  
    def forward(self, truth_values):
        self.input_tensor = truth_values
        return 2. * torch.ones(truth_values.shape[0], 1, 
                           device=truth_values.device,
                           dtype=truth_values.dtype,
                           requires_grad=truth_values.requires_grad)
    def backward(self, delta, randomized=False):
        return torch.zeros_like(self.input_tensor)

class IMPLIES(Formula):
    def function(self, truth_values):
        return torch.where(truth_values[:, 0:1] > truth_values[:, 1:2], truth_values[:, 1:2].double(), 1.)


    def boost_function(self, truth_values, delta):
        return torch.where(
            truth_values[:, 0:1] > truth_values[:, 1:2],
            torch.concat([torch.zeros_like(truth_values[:, 0:1]),
                          torch.minimum(truth_values[:, 0:1] - truth_values[:, 1:2], delta)], 1).double(),
            0.)


    def get_name(self, parenthesis=False):
        s = self.sub_formulas[0].get_name() + ' -> ' + self.sub_formulas[1].get_name()

        if parenthesis:
            return '(' + s + ')'
        else:
            return s