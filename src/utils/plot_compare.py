import matplotlib.pyplot as plt
import glob
import os
import re
import json
import yaml
import argparse

# -----------------------------------------------------------------------------
flg_loss = False
# -----------------------------------------------------------------------------

experiments_path = os.path.abspath(os.path.join(os.path.dirname("__file__"), "../../experiments"))
pattern = "experiment_*"
folders = [d for d in os.listdir(experiments_path) if os.path.isdir(os.path.join(experiments_path, d)) and re.match(pattern, d)]

print(folders)

for folder in folders:
    with open(os.path.join(experiments_path, folder, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
        config = argparse.Namespace(**config)
    formula_1 = config.formula1
    formula_2 = config.formula2
    formula_3 = config.formula3
    formula_1n = config.formula1_name
    formula_2n = config.formula2_name
    formula_3n = config.formula3_name
    
    metrics = {}
    for v in ["old", "lrl"]:
        try:
            with open(os.path.join(experiments_path, folder,"results", f"metrics_{v}.json"), "r") as f:
                metrics[v] = json.load(f)
                print(folder)
        except:
            pass
    try:
        epochs = list(range(1, len(metrics["lrl"]["loss"])+1))
    
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color = 'tab:blue'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy (%)', color=color)
        l1, = ax1.plot(epochs, metrics["lrl"]["train_image_classification_accuracy"], 'b-', label='LRL Train Accuracy')
        l2, = ax1.plot(epochs, metrics["lrl"]["test_image_classification_accuracy"], 'b--', label='LRL Test Accuracy')
        l3, = ax1.plot(epochs, metrics["old"]["train_image_classification_accuracy"], '#FFA500', label='DFA Train Accuracy')
        l4, = ax1.plot(epochs, metrics["old"]["test_image_classification_accuracy"], '#FFA500', linestyle='dashed', label='DFA Test Accuracy')
        ticks = list(range(2, len(metrics["lrl"]["train_image_classification_accuracy"])+1, 2))        
        plt.xticks(ticks)
        ax1.tick_params(axis='y', labelcolor=color)
        
        if flg_loss == True:
            # Create second y-axis for loss
            ax2 = ax1.twinx()  
            color = 'tab:red'
            ax2.set_ylabel('Loss', color=color)
            l5, = ax2.plot(epochs, metrics["lrl"]["loss"], 'r--', label='LRL Loss')
            l6, = ax2.plot(epochs, metrics["old"]["loss"], '#FFA500',  label='DFA Loss', linestyle='dotted')
            ax2.tick_params(axis='y', labelcolor=color)

        lines = [l1, l2, l3, l4]
        ax1.legend(lines, [l.get_label() for l in lines], loc='lower right')
        plt.suptitle(f"Training Metrics: {folder}")
        plt.title(f"{formula_1n}: {formula_1}\n{formula_2n}: {formula_2}\n{formula_3n}: {formula_3}", loc='left')
        fig.tight_layout()

        # Save and close
        plt.savefig(os.path.join(experiments_path, folder,"results", f"metrics_compare.png"), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(experiments_path, "_compare_plots", f"metrics_compare_{folder}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    except:
        pass