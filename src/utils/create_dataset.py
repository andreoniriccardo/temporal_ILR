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

def create_complete_set_traces_one_true_literal(max_length_traces, alphabet, dfa, train_size,train_with_accepted_only, verbose=False): #<----------------------------------
    random.seed(42)
    traces = []
    traces_t = []
    accepted = []

    for length_traces in range(1, max_length_traces+1):
        prod = product(alphabet, repeat=length_traces)

        for trace in list(prod):
            t = []
            t_t = torch.zeros((len(trace), len(alphabet)))

            for step, true_literal in enumerate(trace):
                truth_v = {}
                for s, symbol in enumerate(alphabet):
                    if symbol == true_literal:
                        truth_v[symbol] = True
                        t_t[step, s] = 1.0
                    else:
                        truth_v[symbol] = False

                t.append(truth_v)
            traces.append(t)
            traces_t.append(t_t)
            if dfa.accepts(t):
                accepted.append(1)
            else:
                accepted.append(0)

    #shuffle
    dataset = list(zip(traces, traces_t, accepted))
    random.shuffle(dataset)
    traces, traces_t, accepted = zip(*dataset)

    if verbose:
        print("----TRACES:----")
        for i in range(len(traces)):
            print(traces[i])
            print(traces_t[i])
            if accepted[i] == 1:
                print("YES")
            else:
                print("NO")
        print("------------------------")

    #split
    split_index = round(len(traces) * train_size)

    if not train_with_accepted_only:
        traces_train = traces[:split_index]
        traces_test = traces[split_index:]

        traces_t_train = traces_t[:split_index]
        traces_t_test = traces_t[split_index:]

        accepted_train = accepted[:split_index]
        accepted_test = accepted[split_index:]
    else:
        traces_train = []
        traces_test = []
        traces_t_train = []
        traces_t_test = []
        accepted_train = []
        accepted_test = []

        index = 0
        for i in range(len(traces)):
            if index < split_index and accepted[i] == 1:
                traces_train.append(traces[i])
                traces_t_train.append(traces_t[i])
                accepted_train.append(accepted[i])
            else:
                traces_test.append(traces[i])
                traces_t_test.append(traces_t[i])
                accepted_test.append(accepted[i])


    print("created symbolic dataset with all the {} traces of maximum length {}; {} train, {} test".format(len(traces), max_length_traces, len(traces_train), len(traces_test)))

    return traces_train, traces_test, traces_t_train, traces_t_test, accepted_train, accepted_test

def create_image_sequence_dataset(image_data, numb_of_classes, traces, acceptance, print_size=False):
    channels = 1
    pixels_h, pixels_v = image_data.data[0].size()
    how_many = []
    data_for_classes = []
    for label in range(numb_of_classes):
        indices_i = image_data.targets == label
        data_i, target_i = image_data.data[indices_i], image_data.targets[indices_i]
        how_many.append(len(data_i))
        data_for_classes.append(data_i)

    num_of_images = sum(how_many)

    img_seq_train = []
    acceptance_train = []
    img_seq_test = []
    acceptance_test = []

    i_i = [0 for _ in range(len(how_many)) ]
    seen_images = sum(i_i)


    while True:
        for j in range(len(traces)):
            x = traces[j]
            a = acceptance[j]
            num_img = len(x)
            x_i_img = torch.zeros(num_img, channels,pixels_h, pixels_v)

            for step in range(num_img):
                if x[step][0] > 0.5:

                    x_i_img[step] = data_for_classes[0][i_i[0]]
                    i_i[0] += 1
                    if i_i[0] >= how_many[0]:
                        break
                else:
                    x_i_img[step] = data_for_classes[1][i_i[1]]
                    i_i[1] += 1
                    if i_i[1] >= how_many[1]:
                        break
            if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
                break
            img_seq_train.append(x_i_img)
            acceptance_train.append(a)

            seen_images +=num_img
        if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
            break
    if print_size:
        print("Created image dataset with {} sequences".format(len(img_seq_train) ))

    return img_seq_train, acceptance_train

def create_complete_set_traces(max_length_traces, alphabet, dfa, train_size,train_with_accepted_only, verbose=False):
    traces = []
    traces_t = []
    accepted = []

    for length_traces in range(1, max_length_traces+1):
        #AD HOC FOR 2 SYMBOLS
        prod = product([0,1,2,3], repeat=length_traces)

        for trace in list(prod):
            t = []
            t_t = torch.zeros((len(trace), len(alphabet)))

            for step, true_literal in enumerate(trace):
                truth_v = {}
                if true_literal % 2 == 0:
                    truth_v[alphabet[0]] = True
                    t_t[step, 0] = 1.0
                else:
                    truth_v[alphabet[0]] = False

                if true_literal < 2:
                    truth_v[alphabet[1]] = True
                    t_t[step, 1] = 1.0
                else:
                    truth_v[alphabet[1]] = False

                t.append(truth_v)

            traces.append(t)
            traces_t.append(t_t)
            if dfa.accepts(t):
                accepted.append(1)
            else:
                accepted.append(0)

    #shuffle
    dataset = list(zip(traces, traces_t, accepted))
    random.shuffle(dataset)
    traces, traces_t, accepted = zip(*dataset)

    if verbose:
        print("----TRACES:----")
        for i in range(len(traces)):
            print(traces[i])
            print(traces_t[i])
            if accepted[i] == 1:
                print("YES")
            else:
                print("NO")
        print("------------------------")

    #split
    split_index = round(len(traces) * train_size)

    if not train_with_accepted_only:
        traces_train = traces[:split_index]
        traces_test = traces[split_index:]

        traces_t_train = traces_t[:split_index]
        traces_t_test = traces_t[split_index:]

        accepted_train = accepted[:split_index]
        accepted_test = accepted[split_index:]
    else:
        traces_train = []
        traces_test = []
        traces_t_train = []
        traces_t_test = []
        accepted_train = []
        accepted_test = []

        index = 0
        for i in range(len(traces)):
            if index < split_index and accepted[i] == 1:
                traces_train.append(traces[i])
                traces_t_train.append(traces_t[i])
                accepted_train.append(accepted[i])
            else:
                traces_test.append(traces[i])
                traces_t_test.append(traces_t[i])
                accepted_test.append(accepted[i])


    print(
        "created symbolic dataset with all the {} traces of maximum length {}; {} train, {} test".format(len(traces), max_length_traces, len(traces_train), len(traces_test)))

    return traces_train, traces_test, traces_t_train, traces_t_test, accepted_train, accepted_test

def create_image_sequence_dataset_non_mut_ex(image_data, numb_of_classes, traces, acceptance, print_size=False):
    channels = 1
    pixels_h, pixels_v = image_data.data[0].size()
    how_many = []
    data_for_classes = []
    for label in range(numb_of_classes):
        indices_i = image_data.targets == label
        data_i, target_i = image_data.data[indices_i], image_data.targets[indices_i]
        how_many.append(len(data_i))
        data_for_classes.append(data_i)

    num_of_images = sum(how_many)

    img_seq_train = []
    acceptance_train = []


    i_i = [0 for _ in range(len(how_many)) ]
    seen_images = sum(i_i)


    while True:
        for j in range(len(traces)):
            x = traces[j]
            a = acceptance[j]
            num_img = len(x)
            x_i_img = torch.zeros(num_img, channels,pixels_h, pixels_v)

            for step in range(num_img):
                if x[step][0] > 0.5:

                    x_i_img[step] += data_for_classes[0][i_i[0]]
                    i_i[0] += 1
                    if i_i[0] >= how_many[0]:
                        break
                if x[step][1] > 0.5:
                    x_i_img[step] += data_for_classes[1][i_i[1]]
                    i_i[1] += 1
                    if i_i[1] >= how_many[1]:
                        break
            if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
                break
            img_seq_train.append(x_i_img)
            acceptance_train.append(a)

            seen_images +=num_img
        if i_i[0] >= how_many[0] or i_i[1] >= how_many[1]:
            break
    if print_size:
        print("Created image dataset with {} sequences ".format( len(img_seq_train)))

    return img_seq_train, acceptance_train
