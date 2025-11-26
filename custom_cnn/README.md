# Custom CNN for Deepfake Detection

This folder contains resources and results related to the development and training of a custom Convolutional Neural Network (CNN) for deepfake detection.

## Contents

### Files
- **`cnn-assignment-part-1-v2.ipynb`**: Jupyter notebook for training and evaluating the custom CNN model.
- **`README.md`**: This file, providing an overview of the folder structure and its contents.

### Results
The `results/` folder contains the following:
- **Model Checkpoints**:
  - `best_model_adam.pth`: Best model trained using the Adam optimizer.
  - `best_model_sgd_momentum.pth`: Best model trained using SGD with momentum.
  - `best_model_sgd.pth`: Best model trained using standard SGD.
  - `checkpoint_adam_complete.pth`: Complete checkpoint for the Adam optimizer training.
  - `final_model_adam.pth`: Final model trained using the Adam optimizer.
  - `final_model_sgd_momentum.pth`: Final model trained using SGD with momentum.
  - `final_model_sgd.pth`: Final model trained using standard SGD.

- **Training Results**:
  - `optimizer_comparison.csv`: CSV file comparing the performance of different optimizers.
  - `results_summary.json`: Summary of the training results.
  - `training_history.json`: Detailed training history for the models.

- **Miscellaneous**:
  - `__results___files/`: Additional files generated during training or evaluation.

## Usage

1. Open the `cnn-assignment-part-1-v2.ipynb` notebook to view or run the training and evaluation process.
2. Use the `.pth` files in the `results/` folder to load pre-trained models for inference or further fine-tuning.
3. Refer to `optimizer_comparison.csv` and `results_summary.json` for insights into the training process and optimizer performance.

## Notes

- Ensure that all required dependencies are installed before running the notebook.
- The models and results are specific to the deepfake detection task and may require preprocessing steps as outlined in the notebook.

## License

This project is for educational and research purposes only.