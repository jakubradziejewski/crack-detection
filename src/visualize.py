import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def visualize_results_stratified(results, save_path='visualization_stratified.png'):
    """
    Visualize 3 best (<1), 3 median, and 3 worst predictions (>0.01) based on Dice score.
    """

    # Filter: Dice > 0.01 and < 1.0 - not to include non-crack predictions
    filtered_results = [r for r in results if 0.01 < r['dice'] < 1.0]
    
    # Sort by Dice score
    sorted_results = sorted(filtered_results, key=lambda x: x['dice'])
    n_samples = len(sorted_results)
    
    # Select 3 worst, 3 median, 3 best
    worst_indices = list(range(0, min(3, n_samples)))
    mid = n_samples // 2
    median_indices = [max(0, mid - 1), mid, min(n_samples - 1, mid + 1)]
    best_indices = list(range(max(0, n_samples - 3), n_samples))
    
    selected_indices = worst_indices + median_indices + best_indices
    selected_results = [sorted_results[i] for i in selected_indices]

    # Create visualization
    num_rows = len(selected_results)
    fig, axes = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))
    if num_rows == 1: 
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(selected_results):
        # Load data
        original_img = cv2.cvtColor(cv2.imread(result['img_path']), cv2.COLOR_BGR2RGB)
        true_mask = cv2.imread(result['mask_path'], cv2.IMREAD_GRAYSCALE)
        pred_display = result['pred'] * 255
        
        # Determine category and color
        if i < 3:
            category, color = "WORST", 'red'
        elif i < 6:
            category, color = "MEDIAN", 'orange'
        else:
            category, color = "BEST", 'green'
        
        # Column 1: Original image
        axes[i, 0].imshow(original_img)
        axes[i, 0].set_title(
            f'{category}\n{os.path.basename(result["img_path"])}', 
            color=color, fontweight='bold', fontsize=9
        )
        axes[i, 0].axis('off')
        
        # Column 2: Predicted mask
        axes[i, 1].imshow(pred_display, cmap='gray')
        axes[i, 1].set_title(f'Predicted Mask\nDice: {result["dice"]:.3f}')
        axes[i, 1].axis('off')
        
        # Column 3: True mask
        axes[i, 2].imshow(true_mask, cmap='gray')
        axes[i, 2].set_title('True Mask')
        axes[i, 2].axis('off')
    
    plt.suptitle(
        f'3 Worst, Median, and Best Predictions', 
        fontsize=16, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\Visualization saved to {save_path}")

    all_dice = [r['dice'] for r in results]
    print(f"\nStatistics:")
    print(f"  Total images:              {len(results)}")
    print(f"  Dice=1:                    {sum(1 for d in all_dice if d >= 0.999)}")
    print(f"  0.01<Dice<1:               {len(filtered_results)}")
    print(f"  Dice≤0.01:                 {sum(1 for d in all_dice if d <= 0.01)}")
    print(f"  Mean Dice:                 {np.mean(all_dice):.3f}")
    print(f"  Median Dice:               {np.median(all_dice):.3f}")