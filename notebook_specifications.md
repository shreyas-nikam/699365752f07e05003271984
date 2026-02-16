
# LIME for Explaining Stock Picks: Unveiling the "Why" Behind Quantitative Recommendations

## Introduction: Trusting the Black Box in Equity Investing

As a **CFA Charterholder and Portfolio Manager** at a leading asset management firm, my daily challenge involves navigating complex market dynamics and making informed investment decisions. Increasingly, these decisions are influenced by sophisticated quantitative models, often developed by third-party vendors or internal data science teams. While these models offer compelling alpha generation potential, their "black-box" nature presents a significant hurdle.

Consider a recent scenario: a high-conviction recommendation lands on my desk – "**OVERWEIGHT AAPL: top-quintile alpha expected**." The model, an **XGBoost Regressor**, is proprietary, meaning I can use its `predict` method but cannot inspect its internal workings directly. My responsibility, outlined by **CFA Standard V(A): Diligence and Reasonable Basis**, dictates that I must have a reasonable basis for all investment recommendations. Simply trusting a model's score without understanding its rationale is not sufficient for presenting to the investment committee or for my own due diligence.

This Jupyter Notebook guides me through a real-world workflow using **LIME (Local Interpretable Model-agnostic Explanations)**. LIME is a powerful XAI (Explainable AI) technique that helps illuminate the underlying drivers of black-box model predictions for individual instances. By applying LIME, I aim to translate the opaque model output into a clear, concise, and defensible investment rationale, fostering trust in the model's recommendations and ensuring compliance. We will also compare LIME with **SHAP** to gain a comprehensive understanding of feature contributions and assess the stability of explanations, ultimately equipping me with the tools to confidently integrate quantitative insights into my portfolio management strategy.

---

## 1. Setting Up the Investment Environment: Data Simulation and Model Training

### Markdown Cell — Story + Context + Real-World Relevance

To start, I need to prepare my analytical environment. This involves installing the necessary Python libraries, simulating a realistic factor-based stock dataset, and training a black-box quantitative stock scoring model. This simulated environment mirrors the reality where I receive model recommendations based on various financial factors. The model I'm training is a simple **XGBoost Regressor**, acting as our proprietary, high-performance "black-box" stock scorer that provides `next_month_return` predictions.

### Code cell (function definition + function execution)

```python
# Install required libraries
!pip install pandas numpy scikit-learn xgboost lime shap matplotlib seaborn

# Import required dependencies
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import lime
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr

# Set a random seed for reproducibility
np.random.seed(42)

def setup_stock_scoring_environment(n_stocks=500, n_months=60, test_size=0.2, random_state=42):
    """
    Simulates factor-based stock data and trains an XGBoost regressor as a black-box stock scoring model.

    Args:
        n_stocks (int): Number of simulated stocks.
        n_months (int): Number of simulated months of data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generators.

    Returns:
        tuple: (model, X_train, y_train, X_test, y_test, feature_cols, factors_df)
    """
    print("--- Setting up Stock Scoring Environment ---")

    # Generate synthetic factor data mimicking Barra-style factors
    n_datapoints = n_stocks * n_months
    factors_df = pd.DataFrame({
        'momentum_12m': np.random.randn(n_datapoints),
        'pe_ratio_z': np.random.randn(n_datapoints),
        'pb_ratio_z': np.random.randn(n_datapoints),
        'earnings_growth': np.random.randn(n_datapoints),
        'revenue_growth': np.random.randn(n_datapoints),
        'log_market_cap': np.random.randn(n_datapoints) + 10, # Log Market Cap usually larger values
        'analyst_sentiment': np.random.randn(n_datapoints),
        'volatility_60d': np.abs(np.random.randn(n_datapoints)) * 0.3 # Volatility is positive
    })

    feature_cols = factors_df.columns.tolist()

    # Simulate next-month returns (target variable) as a linear combination of features with noise
    # Coefficients are chosen to simulate weak signals, representing the challenge of alpha generation
    factors_df['next_month_return'] = (
        0.003 * factors_df['momentum_12m'] +
        -0.002 * factors_df['pe_ratio_z'] +
        0.001 * factors_df['earnings_growth'] +
        0.002 * factors_df['analyst_sentiment'] +
        np.random.randn(n_datapoints) * 0.05 # Add random noise
    )

    X = factors_df[feature_cols]
    y = factors_df['next_month_return']

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train XGBoost regressor as the black-box stock scorer
    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    print(f"Model trained on {len(X_train)} samples with {len(feature_cols)} features.")
    print(f"Test R-squared: {model.score(X_test, y_test):.4f}")
    print("-------------------------------------------\n")

    return model, X_train, y_train, X_test, y_test, feature_cols, factors_df

# Execute the setup function
model, X_train, y_train, X_test, y_test, feature_cols, factors_df = setup_stock_scoring_environment()

# Select an instance from the test set for explanation (e.g., the top-scoring stock)
predictions = model.predict(X_test)
top_stock_idx_in_test = np.argmax(predictions)
bottom_stock_idx_in_test = np.argmin(predictions)

print(f"Top scoring stock in test set (index in X_test): {top_stock_idx_in_test}")
print(f"Bottom scoring stock in test set (index in X_test): {bottom_stock_idx_in_test}")
```

### Markdown cell (explanation of execution)

The environment is now set up. We have a synthetic dataset that simulates factor exposures for a large number of stocks over several months, along with their next-month returns. Our black-box model, an XGBoost Regressor, has been trained on this data. Its performance (R-squared) gives us an initial indication of its predictive power. We've also identified the indices of the top-scoring and bottom-scoring stocks in the test set; these will be our primary candidates for explanation. The goal now is to understand *why* the model assigned these particular scores to these specific stocks.

---

## 2. Deciphering a High-Conviction Pick with LIME

### Markdown Cell — Story + Context + Real-World Relevance

The model has identified a stock (let's say, **Stock #`top_stock_idx_in_test`**) with a high expected return, recommending it as "OVERWEIGHT." My next step as a Portfolio Manager is to understand the specific drivers behind this recommendation. LIME (Local Interpretable Model-agnostic Explanations) is perfect for this task because it can explain *any* black-box model by approximating its behavior locally around the prediction point.

LIME works by perturbing the input features of a specific instance, feeding these perturbed instances to the black-box model to get predictions, and then fitting a simple, interpretable (e.g., linear) surrogate model to these perturbed data points and their corresponding predictions. The coefficients of this local linear model then serve as the "explanation," indicating how each feature contributes to the specific prediction.

The mathematical formulation for LIME's surrogate model is given by:

$$
g^{*} = \underset{g \in G}{\operatorname{argmin}} \sum_{i=1}^{N} \pi_x(z_i) \cdot (f(z_i) - g(z_i))^2 + \Omega(g)
$$

Where:
*   $g$ is an interpretable model (e.g., a linear model) from the class $G$.
*   $x$ is the instance being explained.
*   $f$ is the black-box model.
*   $z_i$ are perturbed samples of $x$.
*   $\pi_x(z_i)$ is the proximity weight of $z_i$ to $x$, measuring how close $z_i$ is to the original instance $x$. A common choice is an exponential kernel, $\pi_x(z_i) = \exp(-d(x, z_i)^2 / \sigma^2)$, where $d$ is a distance function and $\sigma$ is the kernel width.
*   $(f(z_i) - g(z_i))^2$ is the squared error between the black-box model's prediction and the surrogate model's prediction for the perturbed sample $z_i$.
*   $\Omega(g)$ is a complexity penalty (e.g., L1 regularization for sparsity), ensuring the surrogate model remains simple.

By minimizing this objective function, LIME finds the best local interpretable model $g^*$ whose coefficients reveal the local feature contributions.

### Code cell (function definition + function execution)

```python
def explain_stock_lime(model, X_data, idx_in_X_data, lime_explainer, feature_names, n_features=8, stock_name="Stock"):
    """
    Generates a LIME explanation for one stock's predicted return.

    Args:
        model: The black-box model with a .predict method.
        X_data (pd.DataFrame): The dataset containing the instances to explain.
        idx_in_X_data (int): The integer index of the instance in X_data to explain.
        lime_explainer: An initialized LimeTabularExplainer.
        feature_names (list): List of feature names.
        n_features (int): Number of features to display in the explanation.
        stock_name (str): Name or identifier for the stock being explained.

    Returns:
        tuple: (explanation, contrib_df, prediction)
            explanation: The LIME explanation object.
            contrib_df (pd.DataFrame): DataFrame of feature contributions.
            prediction (float): The model's prediction for the instance.
    """
    instance = X_data.iloc[idx_in_X_data].values
    prediction = model.predict(instance.reshape(1, -1))[0]

    # Generate the LIME explanation
    explanation = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict,
        num_features=n_features,
        num_samples=1000 # Number of perturbed samples to generate
    )

    # Extract feature contributions into a DataFrame
    contributions = []
    for feat_rule, weight in explanation.as_list():
        contributions.append({
            'feature_rule': feat_rule,
            'lime_weight': weight,
            'direction': 'POSITIVE' if weight > 0 else 'NEGATIVE',
            'abs_weight': abs(weight)
        })
    contrib_df = pd.DataFrame(contributions).sort_values(by='abs_weight', ascending=False)

    print(f"\nLIME EXPLANATION for {stock_name} (Index in X_data: {idx_in_X_data})")
    print(f"Predicted Return: {prediction:+.4f} ({prediction*100:+.2f}%/month)")
    
    recommendation = "OVERWEIGHT" if prediction > 0.005 else ("UNDERWEIGHT" if prediction < -0.005 else "NEUTRAL")
    print(f"Model Recommendation: {recommendation}")
    print("=" * 60)

    # Print a textual bar chart for contributions (similar to LIME's internal plot)
    print(f"{'Feature Rule':<35s} {'Weight':>+8.4f} {'Bar'}")
    print("-" * 60)
    for _, row in contrib_df.iterrows():
        bar_len = int(abs(row['lime_weight']) * 5000) # Scale for visualization
        bar_char = '+' if row['lime_weight'] > 0 else '-'
        bar = bar_char * bar_len
        print(f" {row['feature_rule']:<35s} {row['lime_weight']:>+8.4f} {bar}")

    # Generate and save LIME plot (matplotlib figure)
    fig = explanation.as_pyplot_figure()
    fig.set_size_inches(10, 6)
    fig.suptitle(f"LIME Explanation for {stock_name} (Predicted Return: {prediction*100:+.2f}%)", y=1.02)
    plt.tight_layout()
    plt.savefig(f'lime_explanation_{stock_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
    plt.close(fig)

    return explanation, contrib_df, prediction

# Initialize LIME explainer
# discretize_continuous=True is often helpful for tabular data, binning continuous features.
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_cols,
    class_names=['next_month_return'], # Even for regression, LIME expects this
    mode='regression',
    random_state=42,
    discretize_continuous=True # Essential for understanding how LIME interprets continuous features
)

# Explain the top-scoring stock
lime_exp_top, contrib_top_df, pred_top = explain_stock_lime(
    model, X_test, top_stock_idx_in_test, lime_explainer, feature_cols,
    stock_name=f'Top-Scored Stock (Idx {top_stock_idx_in_test})'
)

# Explain the bottom-scoring stock (for comparison/completeness)
lime_exp_bottom, contrib_bottom_df, pred_bottom = explain_stock_lime(
    model, X_test, bottom_stock_idx_in_test, lime_explainer, feature_cols,
    stock_name=f'Bottom-Scored Stock (Idx {bottom_stock_idx_in_test})'
)
```

### Markdown cell (explanation of execution)

The LIME explanation reveals the key factors driving the model's recommendation for our top-scoring stock. The bar chart (and textual representation) clearly shows which features contributed positively (green/positive bar) and negatively (red/negative bar) to the predicted next-month return. For instance, if 'momentum_12m > 0.5' has a high positive weight, it suggests strong past performance was a significant positive factor. Conversely, 'pe_ratio_z > 1.0' with a negative weight implies the stock's valuation was considered expensive. This granular, instance-specific insight directly addresses my "why" question, moving beyond a simple "OVERWEIGHT" label to a data-backed understanding. It also helps in explaining the "why" for the bottom-scored stock, which is equally important for holistic portfolio management.

---

## 3. Crafting the Investment Rationale for the Committee

### Markdown Cell — Story + Context + Real-World Relevance

While the LIME bar chart provides crucial technical insight, presenting raw feature rules and weights to the investment committee is not effective. As a Portfolio Manager, I need to translate these technical outputs into a clear, concise, and compelling narrative that resonates with financial stakeholders. This narrative forms the "investment rationale" for the model's recommendation, demonstrating my due diligence and facilitating better decision-making by the committee. This translation involves mapping technical feature names and their conditions (e.g., `pe_ratio_z <= -0.5`) to common financial concepts (e.g., "attractive valuation" or "strong earnings momentum").

### Code cell (function definition + function execution)

```python
def generate_investment_rationale(explanation_as_list, prediction, stock_name='Stock'):
    """
    Converts LIME technical output into a PM-readable investment rationale.

    Args:
        explanation_as_list (list): The LIME explanation as a list of (feature_rule, weight) tuples.
        prediction (float): The model's prediction for the stock.
        stock_name (str): The name or identifier for the stock.

    Returns:
        str: A PM-friendly investment rationale statement.
    """
    print(f"\n--- INVESTMENT RATIONALE for {stock_name} ---")

    # Determine overall recommendation based on prediction threshold
    rec = "OVERWEIGHT" if prediction > 0.005 else ("UNDERWEIGHT" if prediction < -0.005 else "NEUTRAL")
    print(f"Model Recommendation: {rec}")
    print(f"Expected Alpha: {prediction*100:+.2f}%/month")
    print("=" * 55)

    # Feature rule to investment language mapping
    language_map = {
        'momentum_12m': ('price momentum', 'trailing 12-month returns'),
        'pe_ratio_z': ('valuation (P/E)', 'price-to-earnings ratio'),
        'pb_ratio_z': ('valuation (P/B)', 'price-to-book ratio'),
        'earnings_growth': ('earnings momentum', 'earnings growth rate'),
        'revenue_growth': ('revenue trajectory', 'revenue growth rate'),
        'log_market_cap': ('size factor', 'market capitalization'),
        'analyst_sentiment': ('analyst consensus', 'sell-side sentiment'),
        'volatility_60d': ('risk profile', '60-day realized volatility'),
    }

    positives = []
    negatives = []

    # Process top 6 features from the explanation
    # LIME explanations are provided as (feature_rule, weight) tuples
    for feat_rule, weight in explanation_as_list[:6]: # Focus on top N features for rationale
        feat_name = None
        for key in language_map:
            if key in feat_rule: # Check if the feature key is part of the rule string
                feat_name = key
                break

        if feat_name:
            concept, detail = language_map[feat_name]
            # Refine the concept based on the rule condition (e.g., low P/E means attractive valuation)
            if 'pe_ratio_z' in feat_rule and '<=' in feat_rule and weight > 0:
                concept = 'attractive ' + concept
            elif 'pe_ratio_z' in feat_rule and '>=' in feat_rule and weight < 0:
                concept = 'expensive ' + concept
            elif 'earnings_growth' in feat_rule and '<=' in feat_rule and weight < 0:
                concept = 'weak ' + concept
            elif 'earnings_growth' in feat_rule and '>=' in feat_rule and weight > 0:
                concept = 'strong ' + concept
            elif 'momentum_12m' in feat_rule and '<=' in feat_rule and weight < 0:
                concept = 'weak ' + concept
            elif 'momentum_12m' in feat_rule and '>=' in feat_rule and weight > 0:
                concept = 'strong ' + concept
            elif 'analyst_sentiment' in feat_rule and '<=' in feat_rule and weight < 0:
                concept = 'negative ' + concept
            elif 'analyst_sentiment' in feat_rule and '>=' in feat_rule and weight > 0:
                concept = 'positive ' + concept
            elif 'volatility_60d' in feat_rule and '>=' in feat_rule and weight < 0:
                concept = 'elevated ' + concept


            if weight > 0:
                positives.append(f"{concept} ({detail})")
            else:
                negatives.append(f"{concept} ({detail})")

    if positives:
        print("\nFavorable factors:")
        for p in positives:
            print(f" + {p}")
    if negatives:
        print("\nDetracting factors:")
        for n in negatives:
            print(f" - {n}")

    summary = f"\nSummary: The model recommends {rec.lower()} {stock_name} based primarily "
    if positives:
        summary += f"on {positives[0]}"
        if len(positives) > 1:
            summary += f" and {positives[1]}"
        summary += "."
    else:
        summary += "on a combination of factors."

    if negatives:
        summary += f" The main risk is {negatives[0]}."
        
    print(summary)
    print("-------------------------------------------\n")
    return summary

# Generate rationale for the top-scored stock
rationale_top_stock = generate_investment_rationale(
    lime_exp_top.as_list(), pred_top, f'Top-Scored Stock (Idx {top_stock_idx_in_test})'
)

# Generate rationale for the bottom-scored stock
rationale_bottom_stock = generate_investment_rationale(
    lime_exp_bottom.as_list(), pred_bottom, f'Bottom-Scored Stock (Idx {bottom_stock_idx_in_test})'
)
```

### Markdown cell (explanation of execution)

The `generate_investment_rationale` function has successfully transformed the technical LIME outputs into a coherent narrative. Instead of cryptic feature rules like "pe_ratio_z <= -0.5", we now have statements such as "attractive valuation (price-to-earnings ratio)" and "strong earnings momentum (earnings growth rate)". This report is now suitable for presentation to the investment committee, clearly articulating *why* the model made its recommendation in language that is easily understood by financial professionals. This step is critical for gaining organizational buy-in and fulfilling my due diligence requirements.

---

## 4. Assessing the Reliability of LIME Explanations (Stability Analysis)

### Markdown Cell — Story + Context + Real-World Relevance

A key concern with LIME, as a perturbation-based method, is its **stochasticity**. Because it randomly samples around the instance being explained, running LIME multiple times on the *exact same stock* can sometimes yield slightly different explanations. As a Portfolio Manager, I need to understand the **stability** of these explanations. If the top contributing features change significantly across multiple runs, it undermines trust and makes it harder to present a consistent rationale for due diligence.

To assess this, I will perform an explanation stability analysis by running LIME multiple times (e.g., 10 runs) for the same stock instance. For each feature, I'll calculate the mean, standard deviation, and critically, the **Coefficient of Variation (CV)** of its LIME weight across these runs.

The Coefficient of Variation (CV) is a standardized measure of dispersion of a probability distribution or frequency distribution. It is defined as the ratio of the standard deviation to the mean:

$$
CV = \frac{\sigma}{\mu}
$$

Where:
*   $\sigma$ is the standard deviation of the LIME weights for a feature across multiple runs.
*   $\mu$ is the mean of the LIME weights for that feature across multiple runs.

A lower CV (e.g., below 0.3) indicates a more stable and reliable feature contribution, suggesting that the feature consistently influences the prediction in a similar way. A high CV points to instability, warranting further investigation.

### Code cell (function definition + function execution)

```python
def test_lime_stability(model, X_data, idx_in_X_data, lime_explainer, feature_names, n_runs=10):
    """
    Runs LIME multiple times on the same instance and analyzes the stability of explanations.

    Args:
        model: The black-box model with a .predict method.
        X_data (pd.DataFrame): The dataset containing the instances to explain.
        idx_in_X_data (int): The integer index of the instance in X_data to explain.
        lime_explainer: An initialized LimeTabularExplainer.
        feature_names (list): List of feature names.
        n_runs (int): Number of times to run LIME for stability analysis.

    Returns:
        pd.DataFrame: A DataFrame showing stability metrics (mean, std, cv, frequency) for each feature.
    """
    print(f"\n--- LIME EXPLANATION STABILITY ANALYSIS (Stock Index: {idx_in_X_data}, {n_runs} runs) ---")

    instance = X_data.iloc[idx_in_X_data].values
    all_explanations_weights = [] # Store dictionaries of {feature_rule: weight} for each run

    for run in range(n_runs):
        exp = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict,
            num_features=len(feature_names), # Get all features for comprehensive stability
            num_samples=1000,
            random_state=None # Allow different perturbations for each run
        )
        weights = dict(exp.as_list())
        all_explanations_weights.append(weights)

    # Consolidate all unique features that appeared across runs
    all_features = set()
    for exp_weights in all_explanations_weights:
        all_features.update(exp_weights.keys())

    stability_data = {}
    for feat_rule in all_features:
        # Collect weights for this feature rule across all runs, use 0 if not present in a run
        values = [exp_weights.get(feat_rule, 0) for exp_weights in all_explanations_weights]
        
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Calculate CV, handle cases where mean is zero to avoid division by zero
        cv_val = std_val / (abs(mean_val) + 1e-6) # Add small epsilon to avoid division by zero

        # Count how many runs the feature rule appeared with a non-zero weight
        appeared_in = sum(1 for v in values if v != 0)

        stability_data[feat_rule] = {
            'mean_weight': mean_val,
            'std_weight': std_val,
            'cv': cv_val,
            'appeared_in_runs': appeared_in
        }
    
    stab_df = pd.DataFrame(stability_data).T.sort_values(by='mean_weight', key=abs, ascending=False)
    stab_df['rank'] = stab_df['mean_weight'].abs().rank(ascending=False).astype(int)
    stab_df = stab_df.round(4)
    
    print("\nFeature Explanation Stability Table:")
    print(f"{'Rank':<5s} {'Feature Rule':<35s} {'Mean':>+8s} {'Std':>8s} {'CV':>8s} {'Freq':>6s} {'Status':<10s}")
    print("=" * 90)
    for _, row in stab_df.iterrows():
        status = 'STABLE' if row['cv'] < 0.3 else ('MODERATE' if row['cv'] < 0.6 else 'UNSTABLE')
        print(f"{row['rank']:<5d} {row.name[:35]:<35s} {row['mean_weight']:>+8.4f} {row['std_weight']:>8.4f} {row['cv']:>8.2f} {row['appeared_in_runs']:>6d} {status:<10s}")
    
    avg_cv_overall = stab_df['cv'].mean()
    print(f"\nAverage CV across all features: {avg_cv_overall:.3f}")
    if avg_cv_overall > 0.5:
        print("WARNING: LIME explanations show significant instability for this instance. Consider increasing num_samples or using SHAP for more deterministic explanations.")
    elif avg_cv_overall > 0.3:
        print("Note: LIME explanations show moderate instability. Review critical features carefully.")
    else:
        print("PASS: LIME explanations appear reasonably stable on average for this instance.")


    # Visualization: Box plot of feature weights distribution
    plot_data = []
    for feat_rule in all_features:
        values = [exp_weights.get(feat_rule, 0) for exp_weights in all_explanations_weights]
        plot_data.extend([{'feature_rule': feat_rule, 'weight': w} for w in values])
    
    plot_df = pd.DataFrame(plot_data)
    
    # Filter to top N features by mean absolute weight for cleaner plot
    top_features_for_plot = stab_df.head(min(10, len(stab_df))).index.tolist()
    plot_df_filtered = plot_df[plot_df['feature_rule'].isin(top_features_for_plot)]

    plt.figure(figsize=(12, 7))
    sns.boxplot(x='weight', y='feature_rule', data=plot_df_filtered.sort_values(by='feature_rule', key=lambda x: x.map(stab_df['mean_weight'].abs())), orient='h')
    plt.title(f'LIME Feature Weight Stability (Box Plot across {n_runs} Runs) for Stock Index {idx_in_X_data}')
    plt.xlabel('LIME Weight')
    plt.ylabel('Feature Rule')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f'lime_stability_boxplot_stock_{idx_in_X_data}.png', dpi=150)
    plt.show()
    plt.close()

    return stab_df

# Execute stability analysis for the top-scoring stock
stability_df = test_lime_stability(
    model, X_test, top_stock_idx_in_test, lime_explainer, feature_cols, n_runs=10
)
```

### Markdown cell (explanation of execution)

The stability analysis provides critical insight into the reliability of LIME's explanations. The table shows the mean weight, standard deviation, and Coefficient of Variation (CV) for each feature across 10 independent LIME runs. Features with a low CV (e.g., < 0.3) are generally stable, meaning their influence on the prediction is consistent. A higher CV or features appearing in fewer runs (low "Freq") suggest instability, which I need to note for my due diligence. The box plot further visualizes the distribution of weights for the most impactful features, allowing me to see the spread and potential outliers in feature contributions. If a key factor, like "earnings_growth," has a high CV, I would exercise caution and perhaps seek alternative explanations or increase `num_samples` in LIME for potentially more robust estimates. This helps me understand the "Achilles' heel" of LIME and manage my trust in its output.

---

## 5. Comparing LIME with SHAP - A Second Opinion

### Markdown Cell — Story + Context + Real-World Relevance

For comprehensive due diligence and to gain deeper confidence in XAI explanations, it's often beneficial to compare insights from different methods. While LIME provides local, perturbation-based explanations, SHAP (SHapley Additive exPlanations) offers a game-theoretic approach that assigns each feature an "importance value" for a particular prediction. SHAP values have a strong theoretical foundation, guaranteeing properties like local accuracy and consistency. For tree-based models like our XGBoost, `TreeExplainer` in SHAP is exact and very efficient.

Comparing LIME and SHAP can reveal areas of agreement, bolstering confidence in the identified drivers, or areas of disagreement, which can flag potential issues like LIME's local approximation limitations or model interactions not captured by LIME. I will compare the feature contributions and their rankings from both methods for the same top-scoring stock. The **Spearman rank correlation coefficient** will quantify the agreement between the feature rankings.

Spearman's rank correlation coefficient $\rho$ is a non-parametric measure of the monotonic relationship between two ranked variables. It is calculated as:

$$
\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
$$

Where:
*   $d_i$ is the difference between the ranks of the $i$-th observation for the two variables.
*   $n$ is the number of observations (features).

A $\rho$ value close to +1 indicates a strong positive monotonic relationship (high agreement in rankings), while a value close to -1 indicates a strong negative monotonic relationship (complete disagreement in rankings). A value near 0 suggests no monotonic relationship.

### Code cell (function definition + function execution)

```python
def compare_shap_lime(model, X_data, idx_in_X_data, lime_explainer, feature_names, stock_name="Stock"):
    """
    Runs both SHAP and LIME on the same instance, compares their explanations,
    and visualizes the agreement.

    Args:
        model: The black-box model (XGBoost for TreeExplainer compatibility).
        X_data (pd.DataFrame): The dataset containing the instances to explain.
        idx_in_X_data (int): The integer index of the instance in X_data to explain.
        lime_explainer: An initialized LimeTabularExplainer.
        feature_names (list): List of feature names.
        stock_name (str): Name or identifier for the stock being explained.

    Returns:
        tuple: (shap_values, lime_weights_mapped, rank_corr_coeff)
            shap_values (dict): SHAP values for each feature.
            lime_weights_mapped (dict): LIME weights mapped to feature names.
            rank_corr_coeff (float): Spearman rank correlation coefficient.
    """
    print(f"\n--- SHAP vs. LIME HEAD-TO-HEAD COMPARISON for {stock_name} (Index: {idx_in_X_data}) ---")

    instance = X_data.iloc[idx_in_X_data:idx_in_X_data+1] # SHAP expects 2D array

    # --- SHAP Explanation ---
    print("\nGenerating SHAP explanation...")
    shap_explainer = shap.TreeExplainer(model) # Optimized for tree-based models
    shap_values = shap_explainer.shap_values(instance)[0] # [0] for regression output
    
    # Map SHAP values to feature names for easier comparison
    shap_contrib_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values,
        'abs_shap_value': np.abs(shap_values)
    }).sort_values(by='abs_shap_value', ascending=False).reset_index(drop=True)
    shap_contrib_df['shap_rank'] = shap_contrib_df.index + 1
    
    # --- LIME Explanation ---
    print("Generating LIME explanation...")
    lime_exp, _, _ = explain_stock_lime(model, X_data, idx_in_X_data, lime_explainer, feature_names,
                                        stock_name=f"{stock_name} (for comparison)")
    
    # Map LIME rules back to feature names and consolidate weights
    # LIME often gives rules like 'feature > X', need to extract base feature name
    lime_weights_raw = lime_exp.as_list()
    lime_weights_mapped = {}
    for rule, weight in lime_weights_raw:
        found_feat = False
        for feat in feature_names:
            if feat in rule:
                lime_weights_mapped[feat] = lime_weights_mapped.get(feat, 0) + weight # Sum weights if multiple rules for same feature
                found_feat = True
                break
        if not found_feat: # For rules not directly matching a feature (e.g., intercept, if any)
            print(f"Warning: LIME rule '{rule}' not directly mapped to a feature name.")

    lime_contrib_df = pd.DataFrame({
        'feature': list(lime_weights_mapped.keys()),
        'lime_weight': list(lime_weights_mapped.values())
    })
    lime_contrib_df['abs_lime_weight'] = np.abs(lime_contrib_df['lime_weight'])
    lime_contrib_df = lime_contrib_df.sort_values(by='abs_lime_weight', ascending=False).reset_index(drop=True)
    lime_contrib_df['lime_rank'] = lime_contrib_df.index + 1

    # --- Merge and Compare ---
    comparison_df = pd.merge(shap_contrib_df[['feature', 'shap_value', 'shap_rank']],
                             lime_contrib_df[['feature', 'lime_weight', 'lime_rank']],
                             on='feature', how='outer')
    comparison_df = comparison_df.fillna({'shap_value': 0, 'lime_weight': 0, 'shap_rank': len(feature_names)+1, 'lime_rank': len(feature_names)+1})
    comparison_df['shap_rank'] = comparison_df['shap_rank'].astype(int)
    comparison_df['lime_rank'] = comparison_df['lime_rank'].astype(int)
    
    # Sort by SHAP rank for consistent display
    comparison_df = comparison_df.sort_values(by='shap_rank').reset_index(drop=True)

    print("\n--- Feature Contribution & Rank Comparison ---")
    print(f"{'Feature':<22s} {'SHAP Value':>+10s} {'SHAP Rank':>10s} {'LIME Weight':>+12s} {'LIME Rank':>10s}")
    print("=" * 74)
    for _, row in comparison_df.iterrows():
        print(f"{row['feature']:<22s} {row['shap_value']:>+10.4f} {row['shap_rank']:>10d} {row['lime_weight']:>+12.4f} {row['lime_rank']:>10d}")
    print("----------------------------------------------")

    # Calculate Spearman Rank Correlation
    # Exclude features that did not appear in both or were zero in both for rank correlation
    # For Spearman, we need two series of ranks of equal length, padding missing features with a max_rank+1
    
    # Create aligned rank lists for correlation
    shap_ranks_aligned = comparison_df['shap_rank'].tolist()
    lime_ranks_aligned = comparison_df['lime_rank'].tolist()

    rank_corr_coeff, p_value = spearmanr(shap_ranks_aligned, lime_ranks_aligned)

    print(f"\nSpearman Rank Correlation (SHAP vs. LIME features): {rank_corr_coeff:.3f}")

    if rank_corr_coeff > 0.7:
        print("Conclusion: GOOD. SHAP and LIME largely agree on feature ranking. This increases confidence in the explanation.")
    elif rank_corr_coeff > 0.4:
        print("Conclusion: MODERATE. Some disagreement in feature ranking. Investigation may be needed for specific discrepancies.")
    else:
        print("Conclusion: WARNING. Substantial disagreement between SHAP and LIME rankings. This warrants further investigation into why explanations diverge.")

    # Visualization: Scatter plot of SHAP rank vs. LIME rank
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x='shap_rank', y='lime_rank', data=comparison_df, hue='feature', s=100)
    
    # Add diagonal line for perfect agreement
    max_rank = max(comparison_df['shap_rank'].max(), comparison_df['lime_rank'].max())
    plt.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.6, label='Perfect Agreement')
    
    plt.title(f'SHAP Rank vs. LIME Rank for {stock_name} (Spearman: {rank_corr_coeff:.2f})')
    plt.xlabel('SHAP Feature Rank (1 = most important)')
    plt.ylabel('LIME Feature Rank (1 = most important)')
    plt.xticks(range(1, max_rank + 1))
    plt.yticks(range(1, max_rank + 1))
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'shap_lime_rank_comparison_stock_{idx_in_X_data}.png', dpi=150)
    plt.show()
    plt.close()
    
    return shap_contrib_df, lime_contrib_df, rank_corr_coeff

# Execute SHAP vs. LIME comparison for the top-scoring stock
shap_contrib_df_top, lime_contrib_df_top, rank_corr = compare_shap_lime(
    model, X_test, top_stock_idx_in_test, lime_explainer, feature_cols,
    stock_name=f'Top-Scored Stock (Idx {top_stock_idx_in_test})'
)
```

### Markdown cell (explanation of execution)

The comparison between SHAP and LIME provides invaluable context. The side-by-side table allows for direct scrutiny of feature contributions and their relative importance (ranks) from both methods. The Spearman rank correlation coefficient quantifies their agreement: a high positive correlation (e.g., > 0.7) means both methods generally agree on which features are most important, significantly increasing my confidence in the explanation. If the correlation is low, the scatter plot helps visualize the disagreement. For example, if "momentum_12m" is ranked very high by SHAP but low by LIME, it suggests that either LIME's local approximation struggled, or the model has strong feature interactions that SHAP can capture but LIME's linear surrogate cannot. This diagnostic information is crucial for deciding how much weight to place on the explanation and whether further investigation into the model's behavior is necessary, a key part of risk management in quantitative investing.

---

## 6. Strategic XAI Selection: When to Use LIME, When to Use SHAP

### Markdown Cell — Story + Context + Real-World Relevance

As a Portfolio Manager, I encounter diverse models—from simple linear regressions to complex neural networks or proprietary vendor solutions. Each XAI method, like LIME and SHAP, has its strengths and weaknesses, making no single method universally superior. Having a clear decision framework for selecting the appropriate XAI tool is essential for efficiency, accuracy, and compliance. This guide, based on various criteria, helps me choose between LIME and SHAP based on the specific model type, explanation needs, and computational constraints. This strategic selection ensures I apply the most suitable tool for each unique explanatory challenge.

| Criterion               | SHAP                                                                   | LIME                                                                         |
| :---------------------- | :--------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| **Theoretical Basis**   | Game theory (Shapley values) - exact for tree models                   | Perturbation + local surrogate model - approximate                           |
| **Determinism**         | Yes (TreeExplainer); KernelSHAP can be stochastic if `num_samples` varies | No (random perturbations around instance)                                    |
| **Additivity**          | Exact: $\sum \text{SHAP}_j = f(x) - E[f]$                              | Approximate: local fidelity, but not globally additive                       |
| **Model Types**         | Best for trees (TreeExplainer); KernelSHAP for any black box (slow)      | Model-agnostic (any `predict` function); practical for many features         |
| **Global + Local Scope** | Both (summary plots for global, waterfall for local)                   | Local only (per instance)                                                    |
| **Feature Interactions**| Yes (interaction values available with SHAP)                           | No (linear surrogate struggles to capture interactions)                      |
| **Explanation Stability**| Deterministic for tree models; more stable generally                     | Stochastic (may vary across runs), sensitive to perturbation kernel width    |
| **Speed (100 samples)** | ~1-5s (trees); KernelSHAP is slower (O($2^P$))                         | ~5-30s (for tabular data)                                                    |
| **Regulatory Suitability**| High (deterministic, additive for trees)                               | Moderate (instability can be a concern for highly regulated decisions)      |

<br/>

**Decision Heuristic for XAI Method Selection:**

1.  **Is the model a tree-based ensemble (e.g., XGBoost, LightGBM, Random Forest)?**
    *   **Yes $\rightarrow$ SHAP `TreeExplainer`:** It's fast, exact, and deterministic, making it ideal.
2.  **Is the model a true black-box (e.g., proprietary vendor model, deep neural network, LLM accessed via API) where internals are inaccessible?**
    *   **Yes $\rightarrow$ LIME:** It only needs the `predict` method, making it highly model-agnostic. SHAP's KernelSHAP can also work but is often too slow for many features.
3.  **Is the decision highly regulated (e.g., credit denials, hiring, medical diagnosis)?**
    *   **Yes $\rightarrow$ SHAP:** Its deterministic nature and strong theoretical guarantees are often preferred for compliance and auditability. Instability in explanations (as seen with LIME) can be problematic.
4.  **Is the goal exploratory analysis or investment research (less regulated)?**
    *   **Yes $\rightarrow$ LIME is acceptable:** Its speed and model-agnosticism make it practical for quickly generating hypotheses, even with some instability.
5.  **Do I need insights into both global feature importance AND local predictions?**
    *   **Yes $\rightarrow$ SHAP:** Offers both global summary plots and detailed local explanations. LIME is strictly local.
6.  **Do I suspect strong feature interactions in the model's logic?**
    *   **Yes $\rightarrow$ SHAP:** SHAP interaction values can explicitly quantify how features interact to influence a prediction. LIME's linear surrogate struggles with this.
7.  **Are both SHAP and LIME available and computationally feasible?**
    *   **Yes $\rightarrow$ Use both and compare:** Agreement between methods increases confidence. Disagreement (as measured by rank correlation) is diagnostic, prompting further investigation into model behavior or explanation robustness.

### Code cell (function definition + function execution)

```python
# No code is executed in this section as it is a pure markdown-based decision guide.
# The previous sections have already demonstrated the practical application of both LIME and SHAP,
# and the metrics derived (stability, rank correlation) directly feed into this decision framework.

print("--- XAI Method Selection Guide: Markdown Table Provided Above ---")
print("This section provides a strategic framework for choosing between LIME and SHAP")
print("based on model characteristics and specific explanatory needs for financial professionals.")
```

### Markdown cell (explanation of execution)

This decision framework empowers me to make an informed choice about which XAI tool to deploy in different scenarios. For instance, if I'm analyzing a vendor's black-box model (Scenario 2), LIME is the go-to. If I'm using an internal, tree-based model for a highly regulated credit scoring task (Scenarios 1 & 3), SHAP's determinism and exactness would be preferred. The comparison conducted in the previous section (Scenario 7) directly feeds into this guide, allowing me to interpret discrepancies as diagnostic signals. By understanding the nuances of each method, I can apply XAI more effectively, enhancing my model governance, due diligence, and overall decision-making process.
