import torch

class Formula(torch.nn.Module):
    def __init__(self, sub_formulas):
        super().__init__()
        if sub_formulas is not None:
            self.sub_formulas = sub_formulas
            self.predicates = list(set([p for sf in self.sub_formulas for p in sf.predicates]))

        self.input_tensor = None

    def function(self, truth_values):
        pass

    def boost_function(self, truth_values, delta):
        pass

    def get_name(self, parenthesis=False):
        pass

    def forward(self, truth_values):
        inputs = []
        for sf in self.sub_formulas:
            inputs.append(sf.forward(truth_values))

        if len(inputs) > 1:
            self.input_tensor = torch.concat(inputs, 1)
        else:
            self.input_tensor = inputs[0]
        return self.function(self.input_tensor)

    def backward(self, delta, randomized=False):
        deltas = self.boost_function(self.input_tensor, delta)
        if randomized:
            deltas = deltas * torch.rand(deltas.shape)

        for sf, d in zip(self.sub_formulas, deltas.t()):
            sf.backward(torch.unsqueeze(d, 0).t())


    def get_delta_tensor(self, truth_values, method='max'):
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
    def __init__(self, name, index):
        super().__init__(None)
        self.name = name
        self.index = index
        self.deltas = []
        self.predicates = [self]

    def forward(self, truth_values):
        return torch.unsqueeze(truth_values[:, self.index], 1)

    def backward(self, delta, randomized=False):  # TODO: implement the usage of randomized

        self.deltas.append(delta)

    def get_name(self, parenthesis=False):
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

class PadZero(Formula):
    """
    """
    def __init__(self, sub_formula):
        super().__init__([sub_formula])
        
    def get_name(self, parenthesis=False):
        return self.sub_formulas[0].get_name(parenthesis)

    def function(self, truth_values):
        return torch.where(truth_values == -1.0, 0.0, truth_values)
    def boost_function(self, truth_values, delta):
        # Refinement logic:
        # If input is valid, propagate delta (Identity).
        # If input is padding (-1), we cannot refine it. Return 0.
        return torch.where(truth_values == -1.0, 0.0, delta)

class PadOne(Formula):
    """
    """
    def __init__(self, sub_formula):
        super().__init__([sub_formula])
        
    def get_name(self, parenthesis=False):
        return self.sub_formulas[0].get_name(parenthesis)

    def function(self, truth_values):
        return torch.where(truth_values == -1.0, 1.0, truth_values)
    def boost_function(self, truth_values, delta):
        # Refinement logic:
        # If input is valid, propagate delta (Identity).
        # If input is padding (-1), we cannot refine it. Return 0.
        return torch.where(truth_values == -1.0, 0.0, delta)
class NOT(Formula):
    def __init__(self, sub_formula):
        super().__init__([sub_formula])
    
    def get_name(self, parenthesis=False):
        return 'NOT(' + self.sub_formulas[0].get_name() + ')'
    
    def function(self, truth_values):

        pad_mask = (truth_values == -1.0)
        false_mask = (truth_values == -2.0)
        true_mask = (truth_values == 2.0)

        not_values = 1 - truth_values

        not_values = torch.where(false_mask, 1.0, not_values)
        not_values = torch.where(true_mask, -1.0, not_values)

        not_values = torch.where(pad_mask, -1.0, not_values) # pad is propagated

        self.forward_output = not_values
        return self.forward_output
    
    def boost_function(self, truth_values, delta):
        '''
        TO BE DESCRIBED
        '''
        return - delta
    
    
    
class AND(Formula):
    
    def get_name(self, parenthesis=False):
        s = ''
        for sf in self.sub_formulas[:-1]:
            s += sf.get_name(parenthesis=True) + ' AND '

        s += self.sub_formulas[-1].get_name(parenthesis=True)

        if parenthesis:
            return '(' + s + ')'
        else:
            return s
        
    def function(self, truth_values):
        any_pad_mask = torch.any(truth_values == -1.0, dim=1, keepdim=True)
        adjusted = torch.where(
            truth_values == -1.0, 1.0, # pad diventa 1 temporaneamente per non interferire, poi viene propagato
            torch.where(
                truth_values == -2.0, 0.0,   
                torch.where(
                    truth_values == 2.0, 1.0,    
                    truth_values           
                )
            )
        )

        min_vals, _ = torch.min(adjusted, dim=1, keepdim=True)
        
        final_output = torch.where(any_pad_mask, -1.0, min_vals)
        
        self.forward_output = final_output#.unsqueeze(1)
        return self.forward_output


    def boost_function(self, truth_values, delta):
        target = self.forward_output + delta 
        valid_positions = truth_values != -1  

        cond_case1 = (target >= self.forward_output)

        t_valid = torch.where(valid_positions, truth_values, torch.full_like(truth_values, 2.0))
        min_vals = t_valid.min(dim=1, keepdim=True)[0]  
        mask_case1 = (truth_values < target) & valid_positions
        mask_case2 = (truth_values == min_vals) & valid_positions

        combined_mask = torch.where(cond_case1, mask_case1, mask_case2)

        t_new = torch.where(combined_mask, target, truth_values)
        t_new = torch.where(valid_positions, t_new, torch.full_like(truth_values, -1.0))

        return t_new - truth_values




    
        
class OR(Formula):
    def get_name(self, parenthesis=False):
        s = ''
        for sf in self.sub_formulas[:-1]:
            s += sf.get_name(parenthesis=True) + ' OR '

        s += self.sub_formulas[-1].get_name(parenthesis=True)

        if parenthesis:
            return '(' + s + ')'
        else:
            return s
        
    def function(self, truth_values):
        any_pad_mask = torch.any(truth_values == -1.0, dim=1, keepdim=True)
        adjusted = torch.where(
            truth_values == -1.0, 0.0,       # pad diventa 0 temporaneamente per non interferire, poi viene propagato   
            torch.where(
                truth_values == -2.0, 0.0,       
                torch.where(
                    truth_values == 2.0, 1.0,     
                    truth_values                  
                )
            )
        )

        max_vals, _ = torch.max(adjusted, dim=1, keepdim=True)
        
        final_output = torch.where(any_pad_mask, -1.0, max_vals)
        
        self.forward_output = final_output#.unsqueeze(1)
        return self.forward_output

    def boost_function(self, truth_values, delta):
        target = self.forward_output + delta
        valid_positions = truth_values != -1 

        cond_case1 = (target < self.forward_output)

        t_valid = torch.where(valid_positions, truth_values, torch.full_like(truth_values, -2.0))
        max_vals = t_valid.max(dim=1, keepdim=True)[0]
        mask_case1 = (truth_values >= target) & valid_positions
        mask_case2 = (truth_values == max_vals) & valid_positions
        combined_mask = torch.where(cond_case1, mask_case1, mask_case2)

        t_new = torch.where(combined_mask, target, truth_values)
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