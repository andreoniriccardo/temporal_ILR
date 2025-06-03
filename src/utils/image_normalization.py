import torch

def normalize_image_sequences_pytorch(list_of_seq_tensors):
    """
    Normalizes each image in a list of sequence tensors to the range [0., 255.].

    Args:
        list_of_seq_tensors (list): A list of PyTorch tensors.
            Each tensor has shape (sequence_length, 1, 28, 28).
            sequence_length can vary.

    Returns:
        list: A new list of PyTorch tensors with normalized images.
    """
    normalized_sequences = []
    for seq_tensor in list_of_seq_tensors:
        # seq_tensor shape: (sequence_length, 1, 28, 28)
        normalized_images_in_seq = []
        
        # Iterate through each image in the sequence
        for i in range(seq_tensor.shape[0]):
            image_tensor = seq_tensor[i]  # Shape: (1, 28, 28)
            
            min_val = torch.min(image_tensor)
            max_val = torch.max(image_tensor)
            
            if max_val == min_val:
                # Handle constant images (e.g., all black or all white)
                # Set to 0 or any other constant value in [0, 255]
                # Here, we set it to a tensor of zeros with the same shape
                normalized_image = torch.zeros_like(image_tensor)
            else:
                # Perform min-max normalization to [0, 1] and then scale to [0, 255]
                normalized_image = (image_tensor - min_val) / (max_val - min_val)# * 255.0
            
            # Ensure values are strictly within [0, 255] due to potential float precision issues
            # Though for min-max this shouldn't be strictly necessary if formula is correct
            # normalized_image = torch.clamp(normalized_image, 0.0, 255.0) 
            
            normalized_images_in_seq.append(normalized_image)
            
        # Stack the normalized images back into a sequence tensor
        if normalized_images_in_seq: # Make sure the list is not empty
            normalized_seq_tensor = torch.stack(normalized_images_in_seq, dim=0)
            normalized_sequences.append(normalized_seq_tensor)
        elif seq_tensor.numel() == 0: # Handle empty input tensor
             normalized_sequences.append(torch.empty_like(seq_tensor)) # Append an empty tensor of the same properties
        # else: seq_tensor had shape[0] == 0, which torch.stack handles gracefully by returning an empty tensor if list is empty.

    return normalized_sequences