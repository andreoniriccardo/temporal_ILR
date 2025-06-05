import torch

def normalize_image_sequences_pytorch(list_of_seq_tensors):
    normalized_sequences = []
    for seq_tensor in list_of_seq_tensors:
        normalized_images_in_seq = []
        
        for i in range(seq_tensor.shape[0]):
            image_tensor = seq_tensor[i]
            
            min_val = torch.min(image_tensor)
            max_val = torch.max(image_tensor)
            
            if max_val == min_val:
                normalized_image = torch.zeros_like(image_tensor)
            else:
                normalized_image = (image_tensor - min_val) / (max_val - min_val)
            
            normalized_images_in_seq.append(normalized_image)
            
        if normalized_images_in_seq:
            normalized_seq_tensor = torch.stack(normalized_images_in_seq, dim=0)
            normalized_sequences.append(normalized_seq_tensor)
        elif seq_tensor.numel() == 0: 
             normalized_sequences.append(torch.empty_like(seq_tensor)) 

    return normalized_sequences