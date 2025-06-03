import matplotlib.pyplot as plt
import json

# Convert tensors to numpy arrays
def _convert_tensor_list(tensor_list):
    return [t.detach().cpu().numpy() if hasattr(t, 'detach') else t 
            for t in tensor_list]

def plot_metrics(loss_list, 
                 train_acc_list, 
                 test_acc_list, 
                 experiment_name,
                 save_path):
    """
    Creates a dual-axis training metrics plot
    Args:
        loss_list: List of loss values per epoch
        train_acc_list: List of training accuracies
        test_acc_list: List of test accuracies
        save_path: Path to save the figure
    """

    loss = _convert_tensor_list(loss_list)
    epochs = list(range(1, len(loss_list)+1))
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot accuracies on left axis
    color = 'tab:blue'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)', color=color)
    l1, = ax1.plot(epochs, train_acc_list, 'b-', label='Train Accuracy')
    l2, = ax1.plot(epochs, test_acc_list, 'g-', label='Test Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create second y-axis for loss
    ax2 = ax1.twinx()  
    color = 'tab:red'
    ax2.set_ylabel('Loss', color=color)
    l3, = ax2.plot(epochs, loss, 'r--', label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    # Combined legend
    lines = [l1, l2, l3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='upper center')

    plt.title(f"Training Metrics: {experiment_name}")
    fig.tight_layout()

    # Save and close
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_results(loss_list, train_acc_list, test_acc_list, time_list, save_path):
    loss = _convert_tensor_list(loss_list)
    loss = [l.item() for l in loss]
    result_dict = {
        "loss": loss,
        "train_image_classification_accuracy": train_acc_list,
        "test_image_classification_accuracy": test_acc_list,
        "time": time_list
    }

    with open(save_path, "w") as f:
        json.dump(result_dict, f)