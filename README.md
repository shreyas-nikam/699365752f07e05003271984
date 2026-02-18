# QuLab: Lab 44: LIME for Explaining Stock Predictions

![QuLab Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title: QuLab: Lab 44: LIME for Explaining Stock Predictions

This project, "QuLab: Lab 44: LIME for Explaining Stock Predictions," is a Streamlit-powered interactive application designed to demonstrate the application of Explainable AI (XAI) techniques, specifically LIME (Local Interpretable Model-agnostic Explanations), in the context of quantitative equity investing. It addresses the critical challenge faced by financial professionals, such as CFA Charterholders and Portfolio Managers, in understanding and justifying "black-box" model recommendations, aligning with due diligence requirements and regulatory standards.

The application guides users through a comprehensive workflow, from setting up a simulated investment environment and training a stock scoring model, to generating and evaluating LIME-based investment rationales, assessing explanation stability, and comparing LIME with SHAP (SHapley Additive exPlanations) for a holistic understanding.

## Features

This application offers a multi-stage workflow to explore LIME and other XAI concepts:

*   **1. Introduction & Setup**:
    *   Introduction to the challenge of black-box models in finance and the role of XAI.
    *   Simulation of a factor-based stock dataset.
    *   Training of a proprietary "black-box" **XGBoost Regressor** for next-month return prediction.
    *   Identification of top-scoring and bottom-scoring stocks in the test set.
    *   Initialization of the LIME explainer.

*   **2. LIME Local Explanations**:
    *   Interactive selection of a specific stock from the test set.
    *   Generation and visualization of LIME explanations for individual stock predictions, showing feature contributions.
    *   Mathematical formulation of LIME's objective function.
    *   Translation of technical LIME outputs into a clear, concise, and defensible **Investment Rationale** suitable for an investment committee.

*   **3. LIME Stability Analysis**:
    *   Assessment of LIME's explanation reliability by running the explanation multiple times.
    *   Calculation of the **Coefficient of Variation (CV)** for feature weights to quantify stability.
    *   Visualization of feature weight distributions across multiple runs using box plots.
    *   Guidance on interpreting stability results for due diligence.

*   **4. SHAP vs. LIME Comparison**:
    *   Comparison of LIME explanations with SHAP (SHapley Additive exPlanations) for the same stock instance.
    *   Calculation of **Spearman Rank Correlation Coefficient** to quantify agreement in feature rankings between the two methods.
    *   Visualization of rank agreement using scatter plots.
    *   Insights into the strengths and weaknesses of each method based on their agreement or disagreement.

*   **5. XAI Method Selection Guide**:
    *   A strategic decision matrix outlining the criteria for choosing between SHAP and LIME.
    *   Practical heuristics for selecting the appropriate XAI tool based on model type, regulatory context, and explanation needs.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Ensure you have Python 3.8+ installed. The application relies on the following Python libraries:

*   `streamlit`
*   `matplotlib`
*   `pandas`
*   `numpy`
*   `seaborn`
*   `lime` (for LIME explanations)
*   `scipy`
*   `xgboost` (for the black-box model)
*   `scikit-learn` (for general machine learning utilities, often a dependency of others)
*   `shap` (for SHAP explanations in the comparison section)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/qualtraining-lab44-lime.git
    cd qualtraining-lab44-lime
    ```
    *(Note: Replace `your-username/qualtraining-lab44-lime` with the actual repository path if it's hosted elsewhere.)*

2.  **Create a virtual environment (recommended)**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies**:
    Create a `requirements.txt` file in the root directory of your project with the following content:
    ```
    streamlit
    matplotlib
    pandas
    numpy
    seaborn
    lime
    scipy
    xgboost
    scikit-learn
    shap
    ```
    Then, install them:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the Streamlit application:

1.  Navigate to the project root directory in your terminal (where `app.py` is located).
2.  Execute the Streamlit command:
    ```bash
    streamlit run app.py
    ```
3.  Your web browser should automatically open the application. If not, open your browser and go to `http://localhost:8000` or the address indicated in your terminal.

### Basic Usage Instructions:

*   **Navigate the Workflow**: Use the sidebar dropdown menu to switch between different sections of the lab project.
*   **Introduction & Setup**: Start here. Click the "Setup Environment and Train Model" button to initialize the data, train the XGBoost model, and prepare the LIME explainer. This must be completed before proceeding to other sections.
*   **LIME Local Explanations**: Select a stock index (e.g., the top-scoring stock) to generate its LIME explanation and an automated investment rationale.
*   **LIME Stability Analysis**: Choose a stock and the number of runs to assess how consistent LIME's explanations are.
*   **SHAP vs. LIME Comparison**: Select a stock to see a side-by-side comparison of feature contributions and rankings from both SHAP and LIME.
*   **XAI Method Selection Guide**: Consult this section for a strategic framework on choosing between LIME and SHAP in various scenarios.

## Project Structure

```
.
├── app.py                  # Main Streamlit application, handles UI and page navigation
├── source.py               # Contains all backend logic:
|                           #   - setup_stock_scoring_environment (data simulation, model training)
|                           #   - explain_stock_lime (LIME explanation generation)
|                           #   - generate_investment_rationale (text generation from LIME)
|                           #   - test_lime_stability (multiple LIME runs, CV calculation)
|                           #   - compare_shap_lime (SHAP explanation, comparison with LIME)
└── requirements.txt        # List of Python dependencies
```

## Technology Stack

*   **Frontend**: Streamlit
*   **Backend/Data Science**: Python 3.x
*   **Data Manipulation**: Pandas, NumPy
*   **Machine Learning**: XGBoost, Scikit-learn
*   **Explainable AI (XAI)**: LIME, SHAP
*   **Visualization**: Matplotlib, Seaborn
*   **Statistical Analysis**: SciPy

## Contributing

This project is primarily a lab exercise, but contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Make your changes and commit them (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(Note: A `LICENSE` file would need to be created in the repository root.)*

## Contact

For questions, feedback, or further information, please reach out to:

*   **QuantUniversity** - [https://www.quantuniversity.com](https://www.quantuniversity.com)
*   **Email**: info@quantuniversity.com
*   **LinkedIn**: [QuantUniversity](https://www.linkedin.com/company/quantuniversity/)