import torch

from DeepAutoma import LSTMAutoma, FuzzyAutoma, FuzzyAutoma_non_mutex, recurrent_write_guard
# from Classifier import CNN
from losses import final_states_loss, not_final_states_loss
import itertools
import math
from utils import eval_acceptance, eval_image_classification_from_traces_ME, eval_image_classification_from_traces_NME
import time
import os
import sys

# Get absolute path to src directory
current_dir = os.path.dirname(os.path.abspath("__file__"))
parent_dir = os.path.dirname(current_dir)
src_path = os.path.join(parent_dir, 'src')
utils_path = os.path.join(src_path, 'utils')

# Add to Python path
sys.path.insert(0, src_path)
sys.path.insert(0, utils_path)
from classifier import CNN_ME, CNN_NME

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class LTL_grounding:
    def __init__(self, ltl_formula, dfa, mutex, symbolic_dataset, image_seq_dataset, C, N, T, dataset='MNIST', train_with_accepted_only= True, automa_implementation = 'logic_circuit', lstm_output= "acceptance", num_exp=0,log_dir="Results/", cnn_initialization = "", lr=0.001, batch_size=64):
        self.log_dir = log_dir
        self.exp_num=num_exp
        self.ltl_formula_string = ltl_formula
        self.dfa = dfa
        self.mutually_exclusive = mutex
        #save the dfa image
        # self.dfa.to_graphviz().render("Automas/"+self.ltl_formula_string)

        self.numb_of_symbols = C
        self.length_traces = T
        self.numb_of_states = self.dfa._state_counter

        self.alphabet = ["c"+str(i) for i in range(C) ]
        self.final_states = list(self.dfa._final_states)

        #reduced dfa for single label image classification
        if self.mutually_exclusive:
            self.reduced_dfa = self.reduce_dfa()
        else:
            self.reduced_dfa = self.reduce_dfa_non_mutex()

        print("DFA: ",self.dfa._transition_function)

        #################### networks
        self.hidden_dim =6
        self.automa_implementation = automa_implementation


        if self.automa_implementation == 'lstm':
            if lstm_output== "states":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, self.numb_of_states)
            elif lstm_output == "acceptance":
                self.deepAutoma = LSTMAutoma(self.hidden_dim, self.numb_of_symbols, 2)
            else:
                print("INVALID LSTM OUTPUT. Choose between 'states' and 'acceptance'")
        elif self.automa_implementation == 'logic_circuit':
            if self.mutually_exclusive:
                self.deepAutoma = FuzzyAutoma(self.numb_of_symbols, self.numb_of_states, self.reduced_dfa)
            else:
                self.deepAutoma = FuzzyAutoma_non_mutex(self.numb_of_symbols, self.numb_of_states, self.reduced_dfa)
        else:
            print("INVALID AUTOMA IMPLEMENTATION. Choose between 'lstm' and 'logic_circuit'")

        if dataset == 'MNIST':
            self.num_classes = self.numb_of_symbols
            self.num_channels = 1
            nodes_linear = 54

        if self.mutually_exclusive:
            self.classifier = CNN_ME(self.num_channels, self.num_classes, nodes_linear, self.mutually_exclusive)
        elif self.mutually_exclusive == False:
            self.classifier = CNN_NME(self.num_channels, self.num_classes, nodes_linear, self.mutually_exclusive)
        self.classifier.load_state_dict(torch.load(cnn_initialization, weights_only=True))
        #dataset
        self.train_traces, self.test_traces, train_acceptance_tr, test_acceptance_tr = symbolic_dataset
        
        if self.mutually_exclusive:
            self.train_img_seq, self.train_acceptance_img, self.test_img_seq_hard, self.test_acceptance_img_hard = image_seq_dataset  
        else:
            self.train_img_seq,self.train_symbolic_sequences,self.train_acceptance_img, self.test_img_seq_hard, self.test_symbolic_sequences, self.test_acceptance_img_hard = image_seq_dataset  

        self.lr = lr
        self.batch_size = batch_size
        
    def reduce_dfa(self):
        dfa = self.dfa

        admissible_transitions = []
        for true_sym in self.alphabet:
            trans = {}
            for i,sym in enumerate(self.alphabet):
                trans[sym] = False
            trans[true_sym] = True
            admissible_transitions.append(trans)
        red_trans_funct = {}
        for s0 in self.dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = self.dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                for sym, at in enumerate(admissible_transitions):
                    if label.subs(at):
                        red_trans_funct[s0][sym] = key

        return red_trans_funct

    def reduce_dfa_non_mutex(self):

        red_trans_funct = {}
        for s0 in self.dfa._states:
            red_trans_funct[s0] = {}
            transitions_from_s0 = self.dfa._transition_function[s0]
            for key in transitions_from_s0:
                label = transitions_from_s0[key]
                label = recurrent_write_guard(label)

                red_trans_funct[s0][label] = key

        return red_trans_funct


    def eval_image_classification(self):
        if self.mutually_exclusive:
            train_acc = eval_image_classification_from_traces_ME(self.train_img_seq, self.train_traces, self.classifier, self.mutually_exclusive)
            test_acc = eval_image_classification_from_traces_ME(self.test_img_seq_hard, self.test_traces, self.classifier, self.mutually_exclusive)
        else:
            train_acc = eval_image_classification_from_traces_NME(self.train_img_seq, self.train_symbolic_sequences, self.classifier, mutually_exclusive=self.mutually_exclusive, device=device)
            test_acc = eval_image_classification_from_traces_NME(self.test_img_seq_hard, self.test_symbolic_sequences, self.classifier, mutually_exclusive=self.mutually_exclusive, device=device)
        return train_acc, test_acc


    def train_classifier(self, num_of_epochs):

        self.classifier.to(device)
        self.deepAutoma.to(device)

        print("_____________training the classifier_____________")
        loss_final = final_states_loss
        self.classifier.train()
        if self.automa_implementation == 'lstm':
            params = [self.classifier.parameters(), self.deepAutoma.parameters()]
            params = itertools.chain(*params)
        else:
            params = self.classifier.parameters()
        
        optimizer = torch.optim.Adam(params=params, lr=self.lr)

        batch_size = self.batch_size
        tot_size = len(self.train_img_seq)

        loss_list = []
        train_image_classification_accuracy_list = []
        test_image_classification_accuracy_list = []
        time_list = []
        time_logical_list = []
        time_perception_list = []

        start_time = time.time()

        for epoch in range(num_of_epochs):
            print("epoch: ", epoch)
            elapsed_time_logical = 0.
            elapsed_time_perception = 0.
            for b in range(math.floor(tot_size/batch_size)):
                start = batch_size*b
                end = min(batch_size*(b+1), tot_size)
                batch_image_dataset = self.train_img_seq[start:end]
                batch_acceptance = self.train_acceptance_img[start:end]
                optimizer.zero_grad()
                losses_f = torch.zeros(0 ).to(device)
                losses_c = torch.zeros(0 ).to(device)

                
                for i in range(len(batch_image_dataset)):   
                    start_time_perception = time.time()    
                    img_sequence =batch_image_dataset[i].to(device)
                    target = batch_acceptance[i]
                    sym_sequence = self.classifier(img_sequence)
                    elapsed_time_perception += time.time() - start_time_perception

                    start_time_logical = time.time()
                    if self.automa_implementation == 'lstm':
                        states_sequence = self.deepAutoma.predict(sym_sequence)
                        final_state = states_sequence[-1]
                    else:
                        final_state = self.deepAutoma(sym_sequence)
                    if target == 0:
                        loss_f = not_final_states_loss(self.final_states, final_state)
                    
                    else:
                        loss_f = loss_final(self.final_states, final_state)
                    
                    losses_f = torch.cat((losses_f, loss_f.unsqueeze(dim=0)), 0)
                    elapsed_time_logical += time.time() - start_time_logical

                loss = losses_f.mean()


                if self.automa_implementation == 'lstm':
                    loss += losses_c.mean()
                
                loss.backward()
                optimizer.step()

            print("loss: ", loss)


            train_image_classification_accuracy, test_image_classification_accuracy = self.eval_image_classification()
            print("IMAGE CLASSIFICATION: train accuracy : {}\ttest accuracy : {}".format(train_image_classification_accuracy,test_image_classification_accuracy))
            # Time
            elapsed_time = time.time() - start_time
            print(f"Elapsed time: {elapsed_time}. Logical/Perception layer time per epoch: {elapsed_time_logical}/{elapsed_time_perception}")

            loss_list.append(loss)
            train_image_classification_accuracy_list.append(train_image_classification_accuracy)
            test_image_classification_accuracy_list.append(test_image_classification_accuracy)
            time_list.append(elapsed_time)
            time_logical_list.append(elapsed_time_logical)
            time_perception_list.append(elapsed_time_perception)

           
        return loss_list, train_image_classification_accuracy_list, test_image_classification_accuracy_list, {'time': time_list, 'time_logical': time_logical_list, 'time_perception': time_perception_list}
    



