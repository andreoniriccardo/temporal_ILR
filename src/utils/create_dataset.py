import random
from itertools import product
import torch
import numpy as np

# random.seed(42)
########################################################################################################################################
def pad_sequence(sym_sequences, max_len):
    padded_sequences = []
    for seq in sym_sequences:
        seq_len = seq.shape[0]
        pad_len = max_len - seq_len
        padded = torch.cat([seq, torch.full((pad_len, seq.shape[1]), -1.0)], dim=0)  # Pad with -1
        mask = torch.cat([torch.ones(seq_len), torch.zeros(pad_len)], dim=0)
        padded_sequences.append(padded)
    return torch.stack(padded_sequences)

def generate_sample_traces_sym_ME(
        max_length_traces,
        alphabet,
        formula,
        num_samples,
        train_size, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sequences = []


    length_weights = np.exp(-0.5 * np.arange(2, max_length_traces+1))
    length_weights /= length_weights.sum()

    for _ in range(num_samples):
        seq_len = np.random.choice(
            np.arange(2, max_length_traces+1),
            # p=length_weights # UNCOMMENT THIS TO WEIGHT LENGTHS
        )
        
        sequence = torch.zeros((seq_len, len(alphabet)), dtype=torch.float32)
        active_symbols = torch.randint(0, len(alphabet), (seq_len,))
        sequence[torch.arange(seq_len), active_symbols] = 1.0

        sequences.append(sequence)
    
    # Shuffle
    random.shuffle(sequences)

    # Check formula satisfaction
    padded_sequences = []
    for i in range(len(sequences)):
        padded_sequence = pad_sequence(sequences[i].unsqueeze(0), max_length_traces)
        padded_sequences.append(padded_sequence)
    padded_sequences_tensor = torch.cat(padded_sequences, dim=0)
    padded_sequences_tensor = padded_sequences_tensor.flatten(start_dim=1)
    satisfaction_scores = formula.forward(padded_sequences_tensor)
    satisfaction = satisfaction_scores.squeeze().int().tolist()

    # Split
    split_idx = int(len(sequences) * train_size)

    # Generate train and test sets
    train_traces, test_traces = sequences[:split_idx], sequences[split_idx:]
    accepted_train, accepted_test = satisfaction[:split_idx], satisfaction[split_idx:]



    return train_traces, test_traces, accepted_train, accepted_test

def create_complete_traces_sym_NME(max_length_generated,
                                       max_length_formula,
                                        alphabet, 
                                        formula, 
                                        seed,
                                        train_size=0.5, 
                                        max_concurrent_symbols=2):
    random.seed(seed)

    traces_t = []
    accepted = []
    n_symbols = len(alphabet)

    for length_traces in range(2, max_length_generated + 1):
        possible_values = list(range(2 ** n_symbols))
        prod = product(possible_values, repeat=length_traces)

        for trace in list(prod):
            t_t = torch.zeros((len(trace), n_symbols))

            for step, true_literal in enumerate(trace):
                for i in range(n_symbols):
                    if (true_literal >> i) & 1 == 0:
                        t_t[step, i] = 1.0

            if t_t.sum(dim=1).max().item() <= max_concurrent_symbols:
                traces_t.append(t_t)
                accepted.append(1 if formula.forward(pad_sequence(t_t.unsqueeze(0), max_length_formula).flatten(start_dim=1)).int().item() else 0)

    # Shuffle the dataset
    dataset = list(zip(traces_t, accepted))
    random.shuffle(dataset)
    traces_t, accepted = zip(*dataset)

    # Split the dataset
    split_index = round(len(traces_t) * train_size)

    traces_t_train = traces_t[:split_index]
    traces_t_test = traces_t[split_index:]
    accepted_train = accepted[:split_index]
    accepted_test = accepted[split_index:]

    return traces_t_train, traces_t_test, accepted_train, accepted_test

def generate_sample_traces_sym_NME(
        min_length_traces,
        max_length_traces,
        alphabet,
        formula,
        num_samples,
        train_size, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sequences = []

    length_weights = np.exp(-0.5 * np.arange(2, max_length_traces+1))
    length_weights /= length_weights.sum()

    for _ in range(num_samples):
        seq_len = np.random.choice(
            np.arange(min_length_traces, max_length_traces+1),
            # p=length_weights # UNCOMMENT THIS TO WEIGHT LENGTHS
        )
        sequence = torch.zeros((seq_len, len(alphabet)), dtype=torch.float32)

        for i in range(seq_len): 
            num_ones_in_row = np.random.choice([0, 1, 2]) 

            if num_ones_in_row == 1:
                active_symbol_idx = torch.randint(0, len(alphabet), (1,))
                sequence[i, active_symbol_idx] = 1.0
            elif num_ones_in_row == 2:
                indices = torch.randperm(len(alphabet))[:2]
                sequence[i, indices] = 1.0


        sequences.append(sequence)
    
    # Shuffle
    random.shuffle(sequences)

    # Check formula satisfaction
    padded_sequences = []
    for i in range(len(sequences)):
        padded_sequence = pad_sequence(sequences[i].unsqueeze(0), max_length_traces)
        padded_sequences.append(padded_sequence)
    padded_sequences_tensor = torch.cat(padded_sequences, dim=0)
    padded_sequences_tensor = padded_sequences_tensor.flatten(start_dim=1)
    satisfaction_scores = formula.forward(padded_sequences_tensor)
    satisfaction = satisfaction_scores.squeeze().int().tolist()

    # Split
    split_idx = int(len(sequences) * train_size)

    # Generate train and test sets
    train_traces, test_traces = sequences[:split_idx], sequences[split_idx:]
    accepted_train, accepted_test = satisfaction[:split_idx], satisfaction[split_idx:]

    return train_traces, test_traces, accepted_train, accepted_test

def create_image_sequence_dataset_sampling_ME(image_data, 
                                           numb_of_classes,
                                           traces,
                                           acceptance,
                                           num_passes=5,
                                           seed=42,
                                           shuffle=True):
    random.seed(seed)
    torch.manual_seed(seed)

    channels = 1
    pixels_h, pixels_v = image_data.data[0].shape

    class_images = []
    for label in range(numb_of_classes):
        indices = image_data.targets == label
        class_images.append(image_data.data[indices])
    for idx, imgs in enumerate(class_images):
        if len(imgs) == 0:
            raise ValueError(f"No images found for class {idx}")
    
    image_sequences = []
    acceptance_labels = []

    for _ in range(num_passes):
        if shuffle:
            combined = list(zip(traces, acceptance))
            random.shuffle(combined)
            shuffled_traces, shuffled_acceptance = zip(*combined)
        else:
            shuffled_traces = traces
            shuffled_acceptance = acceptance

        for trace, accept in zip(shuffled_traces, shuffled_acceptance):
            seq_len = trace.shape[0]
            sequence = torch.zeros((seq_len, channels, pixels_h, pixels_v), dtype=torch.float32)

            for step in range(seq_len):
                symbol_idx = torch.argmax(trace[step]).item()

                class_pool = class_images[symbol_idx]
                random_idx = random.randint(0, len(class_pool) - 1)
                sequence[step] = class_pool[random_idx]

            image_sequences.append(sequence)
            acceptance_labels.append(accept)
    
    print(f"Created image dataset with {len(image_sequences)} sequences")

    return image_sequences, acceptance_labels

def create_image_sequence_dataset_sampling_NME(image_data, numb_of_classes, traces, acceptance, num_passes=5,
                                           seed=42,
                                           shuffle=True,print_size=False):
    random.seed(seed)
    torch.manual_seed(seed)

    channels = 1
    pixels_h, pixels_v = image_data.data[0].size()

    data_for_classes = []
    for label in range(numb_of_classes):
        indices_i = image_data.targets == label
        data_i, target_i = image_data.data[indices_i], image_data.targets[indices_i]
        data_for_classes.append(data_i)


    image_sequences = []
    symbolic_sequences = []
    acceptance_labels = []

    for _ in range(num_passes):
        if shuffle:
            combined = list(zip(traces, acceptance))
            random.shuffle(combined)
            shuffled_traces, shuffled_acceptance = zip(*combined)
        else:
            shuffled_traces = traces
            shuffled_acceptance = acceptance

        for trace, accept in zip(shuffled_traces, shuffled_acceptance):
            seq_len = trace.shape[0]
            sequence = torch.zeros((seq_len, channels, pixels_h, pixels_v), dtype=torch.float32)

            for step in range(seq_len):
                for digit in range(numb_of_classes):
                    if trace[step][digit] > 0.5:
                        sequence[step] += data_for_classes[digit][random.randint(0, len(data_for_classes[digit]) - 1)]
            image_sequences.append(sequence)
            symbolic_sequences.append(trace)
            acceptance_labels.append(accept)
            

    print(f"Created image dataset with {len(image_sequences)} sequences")

    return image_sequences, symbolic_sequences, acceptance_labels