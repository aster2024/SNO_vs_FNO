# Spectral vs. Fourier Neural Operators in Parametric PDE Modeling

<p align="center">
  📄 <a href="https://aster2024.github.io/assets/pdf/SNO_vs_FNO.pdf" target="_blank">Paper</a> &nbsp;
</p>

This repository contains the official source code for the paper **"Spectral vs. Fourier Neural Operators in Parametric PDE Modeling: Analysis and Experiments"**.

This project conducts a systematic comparison between Fourier Neural Operators (FNOs) and Spectral Neural Operators (SNOs) for learning solution operators of parametric PDEs.

## Code Structure

-   `aliasing.py`: A script to visually demonstrate the aliasing phenomenon. It shows how the frequency band of a band-limited function widens significantly after passing through common activation functions (e.g., ReLU, GeLU), which is a core issue in FNOs.

-   `models.py`: Contains the PyTorch implementation of the 1D **Fourier Neural Operator (FNO)** and the 1D **Spectral Neural Operator (SNO)** used in the experiments.

-   `generate_data.py`: Used to generate the training and testing datasets for the **Burgers' equation**, **Korteweg-de Vries (KdV) equation**, and **Kuramoto-Sivashinsky (KS) equation**. The data is generated using a high-fidelity IMEX-RK2 numerical solver to ensure accuracy.

-   `train_eval.py`: The main script for training and evaluating the FNO and SNO models on the three PDE datasets. It handles data loading, model initialization, training loops, and final performance evaluation.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/aster2024/SNO_vs_FNO.git
    cd SNO_vs_FNO
    ```

2.  **Create a virtual environment and install dependencies:**
    We recommend using a virtual environment. The required packages can be installed from `requirements.txt`.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    pip install -r requirements.txt
    ```

## Usage

The typical workflow is to first generate the data, and then train the models.

### 1. Generate Data

Use `generate_data.py` to create datasets for each equation.

### 2. Train and Evaluate Models

Use `train_eval.py` to train and evaluate FNO or SNO on a specific dataset.

### 3. Demonstrate Aliasing

To run the aliasing visualization script:
```bash
python aliasing.py
```
This will generate plots showing the band-widening effect of activation functions.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
