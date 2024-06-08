import numpy as np
import matplotlib.pyplot as plt

def plot_images(test_data: np.ndarray, reconstructed: np.ndarray) -> None:
    plt.figure(figsize=(9, 2))
    for i in range(9):
        # original
        plt.subplot(2, 9, i + 1)
        plt.imshow(test_data[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        # plt.title('Original')
        
        # output
        plt.subplot(2, 9, i + 10)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.axis('off')
        # plt.title('Output')
    plt.show()