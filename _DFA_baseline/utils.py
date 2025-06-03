import torch
import random
from numpy.random import RandomState
import os
import numpy as np
if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def set_seed(seed: int) -> RandomState:
    """ Method to set seed across runs to ensure reproducibility.
    It fixes seed for single-gpu machines.
    Args:
        seed (int): Seed to fix reproducibility. It should different for
            each run
    Returns:
        RandomState: fixed random state to initialize dataset iterators
    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # set to false for reproducibility, True to boost performance
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    random_state = random.getstate()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    return random_state

def eval_acceptance(classifier, automa, final_states, dfa, alphabet, dataset, automa_implementation='dfa', mutually_exc_sym=True):
    #automa implementation =
    #   - 'dfa' use the perfect dfa given
    #   - 'lstm' use the lstm model
    #   - 'logic_circuit' use the fuzzy automaton
    total = 0
    correct = 0
    test_loss = 0
    classifier.eval()

    with torch.no_grad():
        for i in range(len(dataset[0])):
            images = dataset[0][i].to(device)
            label = dataset[1][i]
            # primo modo usando la lstm o l'automa continuo
            if automa_implementation == 'lstm':
                accepted = automa(classifier(images))
                accepted = accepted[-1]

                output = torch.argmax(accepted).item()


            #secondo modo usando l'automa
            elif automa_implementation == 'dfa':
                pred_labels = classifier(images)
                if mutually_exc_sym:
                    pred_labels = pred_labels.data.max(1, keepdim=False)[1]

                    trace = []
                    for p_l in pred_labels:
                        truth_v = {}
                        for symbol in alphabet:
                            truth_v[symbol] = False

                        truth_v[alphabet[p_l.item()]] = True
                        trace.append(truth_v)
                else:
                    trace = []

                    for pred in pred_labels:
                        truth_v = {}
                        for i, symbol in enumerate(alphabet):
                            if pred[i] > 0.5:
                                truth_v[symbol] = True
                            else:
                                truth_v[symbol] = False
                        trace.append(truth_v)

                output = int(dfa.accepts(trace))

            #terzo modo: usando il circuito logico continuo
            elif automa_implementation == 'logic_circuit':
                sym = classifier(images)

                last_state = automa(sym)
                last_state = torch.argmax(last_state).item()

                output = int(last_state in final_states)


            else:
                print("INVALID AUTOMA IMPLEMENTATION: ", automa_implementation)

            total += 1


            correct += int(output==label)


        test_accuracy = 100. * correct/(float)(total)

    return test_accuracy

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