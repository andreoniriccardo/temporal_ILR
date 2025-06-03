import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def pad_sequence(sym_sequences, max_len):
    padded_sequences = []
    for seq in sym_sequences:
        seq_len = seq.shape[0]
        pad_len = max_len - seq_len
        padded = torch.cat([seq, torch.full((pad_len, seq.shape[1]), -1.0)], dim=0)  # Pad with -1
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)], dim=0)
        padded_sequences.append(padded)
    return torch.stack(padded_sequences)


def eval_image_classification_from_traces_ME(traces_images, traces_labels, classifier, mutually_exclusive):
    classifier.eval()
    with torch.no_grad():
        # Take the first len(traces_labels) elements from traces_images to align with labels
        images_list = traces_images[:len(traces_labels)]
        labels_list = traces_labels

        # Concatenate all images and labels into single tensors
        images = torch.cat(images_list, dim=0).to(device)
        labels = torch.cat(labels_list, dim=0).to(device)

        # Get predictions from the classifier
        preds = classifier(images)

         # Since mutually_exclusive is always True, compute argmax for predictions and labels
        pred_labels = preds.argmax(dim=1, keepdim=True)
        true_labels = labels.argmax(dim=1, keepdim=True)

        # Calculate correct predictions and total number
        correct = (pred_labels == true_labels).sum().item()
        total = true_labels.size(0)
        
    accuracy = 100.0 * correct / total
    return accuracy

def eval_image_classification_from_traces_NME(
    traces_images, 
    traces_labels, 
    classifier, 
    mutually_exclusive, 
    device
):
    classifier.eval()
    
    if not mutually_exclusive:
        
        # --- Logic for mutually_exclusive=False ---
        total_symbol_predictions = 0
        correct_symbol_predictions = 0

        num_symbols = traces_labels[0].shape[1]
        with torch.no_grad():
            for i in range(len(traces_labels)):
                t_sym_truth_multi_hot = traces_labels[i].to(device) 
                t_img_sequence_pair = traces_images[i].to(device)

                if t_sym_truth_multi_hot.numel() == 0 or t_sym_truth_multi_hot.shape[1] != num_symbols:
                    if t_sym_truth_multi_hot.numel() > 0 and t_sym_truth_multi_hot.shape[1] != num_symbols:
                        print(f"Warning: Trace {i} has {t_sym_truth_multi_hot.shape[1]} symbols, expected {num_symbols}. Skipping.")
                    continue

                pred_sym_probabilities = classifier(t_img_sequence_pair)

                y1 = torch.ones_like(pred_sym_probabilities)
                y2 = torch.zeros_like(pred_sym_probabilities)
                output_sym_binarized = torch.where(pred_sym_probabilities > 0.5, y1, y2)


                # Overall micro-averaged accuracy calculation
                correct_symbol_predictions += torch.sum(output_sym_binarized == t_sym_truth_multi_hot).item()
                total_symbol_predictions += t_sym_truth_multi_hot.numel() 



        if total_symbol_predictions == 0:
            overall_accuracy = 0.0
        else:
            overall_accuracy = 100. * correct_symbol_predictions / float(total_symbol_predictions)


    return overall_accuracy

class LogicalNetwork(nn.Module):
    def __init__(self, nn_layer, formula, data, symbolic_dataset, lr, batch_size, seq_max_len, mutex, w=1.0, schedule=1.0, max_iterations_lrl=10):
        super(LogicalNetwork, self).__init__()
        self.nn_layer = nn_layer # Perception layer: CNN
        self.formula = formula.to(device) # Logical formula unraveled for the maximum sequence length
        self.seq_max_len = seq_max_len

        # self.w = w # Probabilmente non mi serve più perché uso il label al suo posto
        self.convergence_condition = 1e-4
        self.schedule = schedule
        self.max_iterations_lrl = max_iterations_lrl
        self.mutex = mutex

        # Data
        if mutex:
            self.train_img_seq, self.train_acceptance_img, self.test_img_seq_hard, self.test_acceptance_img_hard = data    
        else:    
            self.train_img_seq, self.train_symbolic_sequences, self.train_acceptance_img, self.test_img_seq_hard, self.test_symbolic_sequences, self.test_acceptance_img_hard = data
        
        self.train_traces, self.test_traces, train_acceptance_tr, test_acceptance_tr = symbolic_dataset

        # Training hyperparameters
        self.lr = lr
        self.batch_size = batch_size

    def eval_image_classification(self):
        if self.mutex:
            train_acc = eval_image_classification_from_traces_ME(self.train_img_seq, self.train_traces, self.nn_layer, mutually_exclusive=self.mutex)
            test_acc = eval_image_classification_from_traces_ME(self.test_img_seq_hard, self.test_traces, self.nn_layer, mutually_exclusive=self.mutex)
        else:
            train_acc = eval_image_classification_from_traces_NME(self.train_img_seq, self.train_symbolic_sequences, self.nn_layer, mutually_exclusive=self.mutex, device=device)
            test_acc = eval_image_classification_from_traces_NME(self.test_img_seq_hard, self.test_symbolic_sequences, self.nn_layer, mutually_exclusive=self.mutex, device=device)
        return train_acc, test_acc
        
    def train_classifier(self, num_of_epochs, max_grad_norm=1.):

        self.nn_layer.to(device)
        print("_____________training the classifier_____________")
        self.nn_layer.train()
        optimizer = torch.optim.Adam(params=self.nn_layer.parameters(), lr=self.lr) # originally lr=0.001
        batch_size = self.batch_size # originally 64
        tot_size = len(self.train_img_seq)

        # Metriche da salvare
        loss_list = []
        train_image_classification_accuracy_list = []
        test_image_classification_accuracy_list = []
        time_list = []
        time_logical_list = [] # Just the time of the logical layer, excluding the perception layer
        time_perception_list = []

        start_time = time.time()
        

        for epoch in range(num_of_epochs):
            print("epoch: ", epoch)
            for b in range(math.floor(tot_size/batch_size)):
                # print("batch: ", b)
                start = batch_size*b
                end = min(batch_size*(b+1), tot_size)
                batch_image_dataset = self.train_img_seq[start:end]
                batch_acceptance = self.train_acceptance_img[start:end]
                target = torch.tensor(batch_acceptance, dtype=torch.float).unsqueeze(1).to(device)

                optimizer.zero_grad()

                # Passo alla CNN ogni sequenza individualmente
                # Effettuo il padding alle sequenze simboliche (output della CNN) e concateno
                elapsed_time_perception = 0.
                batch_symbolic = []
                for i in range(len(batch_image_dataset)):
                    start_time_perception = time.time() 
                    sym_sequence = self.nn_layer(batch_image_dataset[i].to(device)) # [sequence_len, num_symbols]
                    elapsed_time_perception += time.time() - start_time_perception
                    sym_sequence_padded = pad_sequence(sym_sequence.unsqueeze(0), self.seq_max_len) # [1, max_sequence_len, num_symbols]
                    batch_symbolic.append(sym_sequence_padded)
                batch_symbolic_tensor = torch.cat(batch_symbolic, dim=0).to(device) # [batch_size, max_sequence_len, num_symbols]
                original_batch_symbolic_tensor_shape = batch_symbolic_tensor.shape
                # Appiattisco le sequenze
                batch_symbolic_tensor = batch_symbolic_tensor.flatten(start_dim=1) # [batch_size, max_sequence_len * num_symbols]

                start_time_logical = time.time()
                satisfaction = self.formula.forward(batch_symbolic_tensor)
                
                loss = nn.BCELoss()(satisfaction.float(), target)

                elapsed_time_logical = time.time() - start_time_logical
                loss.backward()
                if not self.mutex:
                    params_to_clip = [p for p in self.nn_layer.parameters() if p.grad is not None]
                    if params_to_clip:
                        total_norm_before_clip = torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=max_grad_norm)
                    else:
                        print(f"WARNING: No gradients found for clipping at Epoch {epoch}, Batch {b}.")
                        total_norm_before_clip = torch.tensor(0.0)
                optimizer.step()
                
            print("loss: ", loss)
            # Accuracy
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

