import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import lime
import lime.lime_tabular
from scipy.stats import spearmanr
from source import (
    setup_stock_scoring_environment,
    explain_stock_lime,
    generate_investment_rationale,
    test_lime_stability,
    compare_shap_lime
)

st.set_page_config(page_title="QuLab: Lab 44: LIME for Explaining Stock Predictions", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Lab 44: LIME for Explaining Stock Predictions")
st.divider()

# Initialize Session State
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.model = None
    st.session_state.X_train = None
    st.session_state.y_train = None
    st.session_state.X_test = None
    st.session_state.y_test = None
    st.session_state.feature_cols = None
    st.session_state.factors_df = None
    st.session_state.lime_explainer = None
    st.session_state.top_stock_idx_in_test = None
    st.session_state.bottom_stock_idx_in_test = None
    st.session_state.selected_stock_idx = None 
    st.session_state.stock_indices_options = [] 
    st.session_state.lime_explanation_data = {}
    st.session_state.rationale_data = {}
    st.session_state.stability_data = {}
    st.session_state.comparison_data = {}
    st.session_state.n_stability_runs = 10

if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction & Setup"

# Sidebar Navigation
with st.sidebar:
    st.title("XAI for Stock Picks")
    st.session_state.current_page = st.selectbox(
        "Navigate Workflow",
        [
            "Introduction & Setup",
            "LIME Local Explanations",
            "LIME Stability Analysis",
            "SHAP vs. LIME Comparison",
            "XAI Method Selection Guide"
        ]
    )

# Page 1: Introduction & Setup
if st.session_state.current_page == "Introduction & Setup":
    st.title("Introduction: Trusting the Black Box in Equity Investing")
    st.markdown(f"As a **CFA Charterholder and Portfolio Manager** at a leading asset management firm, my daily challenge involves navigating complex market dynamics and making informed investment decisions. Increasingly, these decisions are influenced by sophisticated quantitative models, often developed by third-party vendors or internal data science teams. While these models offer compelling alpha generation potential, their 'black-box' nature presents a significant hurdle.")
    st.markdown(f"Consider a recent scenario: a high-conviction recommendation lands on my desk – '**OVERWEIGHT AAPL: top-quintile alpha expected**.' The model, an **XGBoost Regressor**, is proprietary, meaning I can use its `predict` method but cannot inspect its internal workings directly. My responsibility, outlined by **CFA Standard V(A): Diligence and Reasonable Basis**, dictates that I must have a reasonable basis for all investment recommendations. Simply trusting a model's score without understanding its rationale is not sufficient for presenting to the investment committee or for my own due diligence.")
    st.markdown(f"This application guides me through a real-world workflow using **LIME (Local Interpretable Model-agnostic Explanations)**. LIME is a powerful XAI (Explainable AI) technique that helps illuminate the underlying drivers of black-box model predictions for individual instances. By applying LIME, I aim to translate the opaque model output into a clear, concise, and defensible investment rationale, fostering trust in the model's recommendations and ensuring compliance. We will also compare LIME with **SHAP** to gain a comprehensive understanding of feature contributions and assess the stability of explanations, ultimately equipping me with the tools to confidently integrate quantitative insights into my portfolio management strategy.")

    st.header("1. Setting Up the Investment Environment: Data Simulation and Model Training")
    st.markdown(f"To start, I need to prepare my analytical environment. This involves installing the necessary Python libraries, simulating a realistic factor-based stock dataset, and training a black-box quantitative stock scoring model. This simulated environment mirrors the reality where I receive model recommendations based on various financial factors. The model I'm training is a simple **XGBoost Regressor**, acting as our proprietary, high-performance 'black-box' stock scorer that provides `next_month_return` predictions.")

    if not st.session_state.initialized:
        st.markdown(f"**Click the button below to set up the environment and train the stock scoring model.**")
        if st.button("Setup Environment and Train Model"):
            with st.spinner("Setting up environment and training model..."):
                (model, X_train, y_train, X_test, y_test, feature_cols, factors_df_full) = setup_stock_scoring_environment()

                # Initialize LIME explainer once
                lime_explainer_init = lime.lime_tabular.LimeTabularExplainer(
                    training_data=X_train.values,
                    feature_names=feature_cols,
                    class_names=['next_month_return'],
                    mode='regression',
                    random_state=42,
                    discretize_continuous=True
                )

                predictions = model.predict(X_test)
                top_stock_idx_in_test = np.argmax(predictions)
                bottom_stock_idx_in_test = np.argmin(predictions)

                st.session_state.model = model
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.feature_cols = feature_cols
                st.session_state.factors_df = factors_df_full 
                st.session_state.lime_explainer = lime_explainer_init
                st.session_state.top_stock_idx_in_test = top_stock_idx_in_test
                st.session_state.bottom_stock_idx_in_test = bottom_stock_idx_in_test
                st.session_state.selected_stock_idx = top_stock_idx_in_test 
                st.session_state.stock_indices_options = sorted(list(X_test.index.values)) 
                st.session_state.initialized = True
            st.success("Environment setup complete! Model trained and explainer initialized.")
            st.rerun()

    if st.session_state.initialized:
        st.markdown(f"The environment is now set up. We have a synthetic dataset that simulates factor exposures for a large number of stocks over several months, along with their next-month returns. Our black-box model, an XGBoost Regressor, has been trained on this data. Its performance (R-squared) gives us an initial indication of its predictive power. We've also identified the indices of the top-scoring and bottom-scoring stocks in the test set; these will be our primary candidates for explanation. The goal now is to understand *why* the model assigned these particular scores to these specific stocks.")
        st.markdown(f"---")
        st.info(f"**Top scoring stock in test set (index in X_test):** `{st.session_state.top_stock_idx_in_test}`")
        st.info(f"**Bottom scoring stock in test set (index in X_test):** `{st.session_state.bottom_stock_idx_in_test}`")

# Page 2: LIME Local Explanations
elif st.session_state.current_page == "LIME Local Explanations":
    if st.session_state.initialized:
        st.header("2. Deciphering a High-Conviction Pick with LIME")
        st.markdown(f"The model has identified a stock (let's say, **Stock #{st.session_state.selected_stock_idx}**) with a high expected return, recommending it as 'OVERWEIGHT.' My next step as a Portfolio Manager is to understand the specific drivers behind this recommendation. LIME (Local Interpretable Model-agnostic Explanations) is perfect for this task because it can explain *any* black-box model by approximating its behavior locally around the prediction point.")
        st.markdown(f"LIME works by perturbing the input features of a specific instance, feeding these perturbed instances to the black-box model to get predictions, and then fitting a simple, interpretable (e.g., linear) surrogate model to these perturbed data points and their corresponding predictions. The coefficients of this local linear model then serve as the 'explanation,' indicating how each feature contributes to the specific prediction.")

        st.markdown(r"The mathematical formulation for LIME's surrogate model is given by:")
        st.markdown(r"$$g^{*} = \underset{g \in G}{\operatorname{argmin}} \sum_{i=1}^{N} \pi_x(z_i) \cdot (f(z_i) - g(z_i))^2 + \Omega(g)$$")
        st.markdown(r"where $g$ is an interpretable model (e.g., a linear model) from the class $G$.")
        st.markdown(r"where $x$ is the instance being explained.")
        st.markdown(r"where $f$ is the black-box model.")
        st.markdown(r"where $z_i$ are perturbed samples of $x$.")
        st.markdown(r"where $\pi_x(z_i)$ is the proximity weight of $z_i$ to $x$, measuring how close $z_i$ is to the original instance $x$. A common choice is an exponential kernel, $\pi_x(z_i) = \exp(-d(x, z_i)^2 / \sigma^2)$, where $d$ is a distance function and $\sigma$ is the kernel width.")
        st.markdown(r"where $(f(z_i) - g(z_i))^2$ is the squared error between the black-box model's prediction and the surrogate model's prediction for the perturbed sample $z_i$.")
        st.markdown(r"where $\Omega(g)$ is a complexity penalty (e.g., L1 regularization for sparsity), ensuring the surrogate model remains simple.")
        st.markdown(f"By minimizing this objective function, LIME finds the best local interpretable model $g^*$ whose coefficients reveal the local feature contributions.")

        st.subheader("Select Stock for LIME Explanation")
        
        # Ensure selected index is valid
        current_index = 0
        if st.session_state.selected_stock_idx in st.session_state.stock_indices_options:
            current_index = st.session_state.stock_indices_options.index(st.session_state.selected_stock_idx)
            
        st.session_state.selected_stock_idx = st.selectbox(
            "Choose a stock index from the test set:",
            options=st.session_state.stock_indices_options,
            index=current_index
        )
        
        stock_key = st.session_state.selected_stock_idx
        
        if stock_key not in st.session_state.lime_explanation_data:
            with st.spinner(f"Generating LIME explanation for Stock {stock_key}..."):
                explanation_obj, contrib_df, prediction = explain_stock_lime(
                    st.session_state.model,
                    st.session_state.X_test,
                    stock_key,
                    st.session_state.lime_explainer,
                    st.session_state.feature_cols,
                    stock_name=f'Stock (Idx {stock_key})'
                )
                st.session_state.lime_explanation_data[stock_key] = (explanation_obj, contrib_df, prediction)
        else:
            explanation_obj, contrib_df, prediction = st.session_state.lime_explanation_data[stock_key]

        st.subheader(f"LIME Explanation for Stock {stock_key}")
        st.write(f"**Predicted Return:** {prediction:+.4f} ({prediction*100:+.2f}%/month)")
        recommendation = "OVERWEIGHT" if prediction > 0.005 else ("UNDERWEIGHT" if prediction < -0.005 else "NEUTRAL")
        st.write(f"**Model Recommendation:** {recommendation}")

        # Plot LIME explanation
        fig, ax = plt.subplots(figsize=(10, 6))
        contrib_df_sorted = contrib_df.sort_values(by='lime_weight', ascending=True)
        colors = ['#ff6666' if x < 0 else '#66ff66' for x in contrib_df_sorted['lime_weight']]
        ax.barh(contrib_df_sorted['feature_rule'], contrib_df_sorted['lime_weight'], color=colors)
        ax.set_xlabel("LIME Weight (Contribution to Prediction)")
        ax.set_ylabel("Feature Rule")
        ax.set_title(f"LIME Explanation for Stock {stock_key} (Predicted Return: {prediction*100:+.2f}%)")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"The LIME explanation reveals the key factors driving the model's recommendation for our selected stock. The bar chart clearly shows which features contributed positively (green) and negatively (red) to the predicted next-month return. For instance, if 'momentum_12m > 0.5' has a high positive weight, it suggests strong past performance was a significant positive factor. Conversely, 'pe_ratio_z > 1.0' with a negative weight implies the stock's valuation was considered expensive. This granular, instance-specific insight directly addresses my 'why' question, moving beyond a simple 'OVERWEIGHT' label to a data-backed understanding.")

        st.header("3. Crafting the Investment Rationale for the Committee")
        st.markdown(f"While the LIME bar chart provides crucial technical insight, presenting raw feature rules and weights to the investment committee is not effective. As a Portfolio Manager, I need to translate these technical outputs into a clear, concise, and compelling narrative that resonates with financial stakeholders. This narrative forms the 'investment rationale' for the model's recommendation, demonstrating my due diligence and facilitating better decision-making by the committee. This translation involves mapping technical feature names and their conditions (e.g., `pe_ratio_z <= -0.5`) to common financial concepts (e.g., 'attractive valuation' or 'strong earnings momentum').")

        if stock_key not in st.session_state.rationale_data:
            with st.spinner(f"Generating investment rationale for Stock {stock_key}..."):
                rationale_summary = generate_investment_rationale(
                    explanation_obj.as_list(), prediction, f'Stock (Idx {stock_key})'
                )
                st.session_state.rationale_data[stock_key] = rationale_summary
        else:
            rationale_summary = st.session_state.rationale_data[stock_key]
        
        st.markdown(f"---")
        st.markdown(f"### Investment Rationale for Stock {stock_key}")
        st.markdown(rationale_summary)
        st.markdown(f"---")

        st.markdown(f"The `generate_investment_rationale` function has successfully transformed the technical LIME outputs into a coherent narrative. Instead of cryptic feature rules like 'pe_ratio_z <= -0.5', we now have statements such as 'attractive valuation (price-to-earnings ratio)' and 'strong earnings momentum (earnings growth rate)'. This report is now suitable for presentation to the investment committee, clearly articulating *why* the model made its recommendation in language that is easily understood by financial professionals. This step is critical for gaining organizational buy-in and fulfilling my due diligence requirements.")

    else:
        st.info("Please complete the 'Introduction & Setup' page first.")

# Page 3: LIME Stability Analysis
elif st.session_state.current_page == "LIME Stability Analysis":
    if st.session_state.initialized:
        st.header("4. Assessing the Reliability of LIME Explanations (Stability Analysis)")
        st.markdown(f"A key concern with LIME, as a perturbation-based method, is its **stochasticity**. Because it randomly samples around the instance being explained, running LIME multiple times on the *exact same stock* can sometimes yield slightly different explanations. As a Portfolio Manager, I need to understand the **stability** of these explanations. If the top contributing features change significantly across multiple runs, it undermines trust and makes it harder to present a consistent rationale for due diligence.")
        st.markdown(f"To assess this, I will perform an explanation stability analysis by running LIME multiple times (e.g., {st.session_state.n_stability_runs} runs) for the same stock instance. For each feature, I'll calculate the mean, standard deviation, and critically, the **Coefficient of Variation (CV)** of its LIME weight across these runs.")
        st.markdown(r"The Coefficient of Variation (CV) is a standardized measure of dispersion of a probability distribution or frequency distribution. It is defined as the ratio of the standard deviation to the mean:")
        st.markdown(r"$$CV = \frac{\sigma}{\mu}$$")
        st.markdown(r"where $\sigma$ is the standard deviation of the LIME weights for a feature across multiple runs.")
        st.markdown(r"where $\mu$ is the mean of the LIME weights for that feature across multiple runs.")
        st.markdown(f"A lower CV (e.g., below 0.3) indicates a more stable and reliable feature contribution, suggesting that the feature consistently influences the prediction in a similar way. A high CV points to instability, warranting further investigation.")

        st.subheader("Select Stock for Stability Analysis")
        
        current_index = 0
        if st.session_state.selected_stock_idx in st.session_state.stock_indices_options:
            current_index = st.session_state.stock_indices_options.index(st.session_state.selected_stock_idx)
            
        st.session_state.selected_stock_idx = st.selectbox(
            "Choose a stock index for stability analysis:",
            options=st.session_state.stock_indices_options,
            index=current_index,
            key="stability_stock_select"
        )

        st.session_state.n_stability_runs = st.number_input(
            "Number of LIME runs for stability analysis:",
            min_value=5, max_value=50, value=st.session_state.n_stability_runs, step=5
        )

        stock_key = st.session_state.selected_stock_idx

        if st.button(f"Run Stability Analysis for Stock {stock_key} (using {st.session_state.n_stability_runs} runs)"):
            with st.spinner(f"Running LIME stability analysis for Stock {stock_key} across {st.session_state.n_stability_runs} runs..."):
                stab_df, plot_df_filtered = test_lime_stability(
                    st.session_state.model,
                    st.session_state.X_test,
                    stock_key,
                    st.session_state.lime_explainer,
                    st.session_state.feature_cols,
                    n_runs=st.session_state.n_stability_runs
                )
                st.session_state.stability_data[stock_key] = (stab_df, plot_df_filtered)
            st.success("Stability analysis complete!")

        if stock_key in st.session_state.stability_data:
            stab_df, plot_df_filtered = st.session_state.stability_data[stock_key]

            st.subheader(f"LIME Explanation Stability for Stock {stock_key}")
            st.markdown(f"**Feature Explanation Stability Table:**")
            
            st.dataframe(stab_df.style.format({
                'mean_weight': '{:+.4f}',
                'std_weight': '{:.4f}',
                'cv': '{:.2f}'
            }))
            
            avg_cv_overall = stab_df['cv'].mean()
            st.markdown(f"\nAverage CV across all features: {avg_cv_overall:.3f}")
            if avg_cv_overall > 0.5:
                st.warning("WARNING: LIME explanations show significant instability for this instance. Consider increasing num_samples or using SHAP for more deterministic explanations.")
            elif avg_cv_overall > 0.3:
                st.info("Note: LIME explanations show moderate instability. Review critical features carefully.")
            else:
                st.success("PASS: LIME explanations appear reasonably stable on average for this instance.")

            if not plot_df_filtered.empty:
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.boxplot(x='weight', y='feature_rule', data=plot_df_filtered.sort_values(
                    by='feature_rule', key=lambda x: x.map(stab_df['mean_weight'].abs().to_dict()), ascending=False), orient='h', ax=ax)
                ax.set_title(f'LIME Feature Weight Stability (Box Plot across {st.session_state.n_stability_runs} Runs) for Stock Index {stock_key}')
                ax.set_xlabel('LIME Weight')
                ax.set_ylabel('Feature Rule')
                ax.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Not enough data to plot stability for this stock.")
            
        st.markdown(f"The stability analysis provides critical insight into the reliability of LIME's explanations. The table shows the mean weight, standard deviation, and Coefficient of Variation (CV) for each feature across {st.session_state.n_stability_runs} independent LIME runs. Features with a low CV (e.g., < 0.3) are generally stable, meaning their influence on the prediction is consistent. A higher CV or features appearing in fewer runs (low 'Freq') suggest instability, which I need to note for my due diligence. The box plot further visualizes the distribution of weights for the most impactful features, allowing me to see the spread and potential outliers in feature contributions. If a key factor, like 'earnings_growth,' has a high CV, I would exercise caution and perhaps seek alternative explanations or increase `num_samples` in LIME for potentially more robust estimates. This helps me understand the 'Achilles' heel' of LIME and manage my trust in its output.")

    else:
        st.info("Please complete the 'Introduction & Setup' page first.")

# Page 4: SHAP vs. LIME Comparison
elif st.session_state.current_page == "SHAP vs. LIME Comparison":
    if st.session_state.initialized:
        st.header("5. Comparing LIME with SHAP - A Second Opinion")
        st.markdown(f"For comprehensive due diligence and to gain deeper confidence in XAI explanations, it's often beneficial to compare insights from different methods. While LIME provides local, perturbation-based explanations, SHAP (SHapley Additive exPlanations) offers a game-theoretic approach that assigns each feature an 'importance value' for a particular prediction. SHAP values have a strong theoretical foundation, guaranteeing properties like local accuracy and consistency. For tree-based models like our XGBoost, `TreeExplainer` in SHAP is exact and very efficient.")
        st.markdown(f"Comparing LIME and SHAP can reveal areas of agreement, bolstering confidence in the identified drivers, or areas of disagreement, which can flag potential issues like LIME's local approximation limitations or model interactions not captured by LIME. I will compare the feature contributions and their rankings from both methods for the same top-scoring stock. The **Spearman rank correlation coefficient** will quantify the agreement between the feature rankings.")

        st.markdown(r"Spearman's rank correlation coefficient $\rho$ is a non-parametric measure of the monotonic relationship between two ranked variables. It is calculated as:")
        st.markdown(r"$$ \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)} $$")
        st.markdown(r"where $d_i$ is the difference between the ranks of the $i$-th observation for the two variables.")
        st.markdown(r"where $n$ is the number of observations (features).")
        st.markdown(f"A $\rho$ value close to +1 indicates a strong positive monotonic relationship (high agreement in rankings), while a value close to -1 indicates a strong negative monotonic relationship (complete disagreement in rankings). A value near 0 suggests no monotonic relationship.")

        st.subheader("Select Stock for SHAP vs. LIME Comparison")
        
        current_index = 0
        if st.session_state.selected_stock_idx in st.session_state.stock_indices_options:
            current_index = st.session_state.stock_indices_options.index(st.session_state.selected_stock_idx)
            
        st.session_state.selected_stock_idx = st.selectbox(
            "Choose a stock index for comparison:",
            options=st.session_state.stock_indices_options,
            index=current_index,
            key="comparison_stock_select"
        )
        
        stock_key = st.session_state.selected_stock_idx

        if st.button(f"Compare SHAP and LIME for Stock {stock_key}"):
            with st.spinner(f"Running SHAP and LIME comparison for Stock {stock_key}..."):
                shap_contrib_df, lime_contrib_df, rank_corr_coeff, comparison_df = compare_shap_lime(
                    st.session_state.model,
                    st.session_state.X_test,
                    stock_key,
                    st.session_state.lime_explainer,
                    st.session_state.feature_cols,
                    stock_name=f'Stock (Idx {stock_key})'
                )
                st.session_state.comparison_data[stock_key] = (shap_contrib_df, lime_contrib_df, rank_corr_coeff, comparison_df)
            st.success("Comparison complete!")

        if stock_key in st.session_state.comparison_data:
            shap_contrib_df, lime_contrib_df, rank_corr_coeff, comparison_df = st.session_state.comparison_data[stock_key]

            st.subheader(f"SHAP vs. LIME Comparison for Stock {stock_key}")

            st.markdown(f"**Feature Contribution & Rank Comparison:**")
            st.dataframe(comparison_df.style.format({
                'shap_value': '{:+.4f}',
                'lime_weight': '{:+.4f}',
            }))

            st.write(f"\n**Spearman Rank Correlation (SHAP vs. LIME features):** `{rank_corr_coeff:.3f}`")

            if rank_corr_coeff > 0.7:
                st.success("Conclusion: GOOD. SHAP and LIME largely agree on feature ranking. This increases confidence in the explanation.")
            elif rank_corr_coeff > 0.4:
                st.info("Conclusion: MODERATE. Some disagreement in feature ranking. Investigation may be needed for specific discrepancies.")
            else:
                st.warning("Conclusion: WARNING. Substantial disagreement between SHAP and LIME rankings. This warrants further investigation into why explanations diverge.")

            # Plot SHAP vs. LIME Rank Comparison
            fig, ax = plt.subplots(figsize=(8, 8))
            sns.scatterplot(x='shap_rank', y='lime_rank', data=comparison_df, hue='feature', s=100, ax=ax)
            
            max_rank = max(comparison_df['shap_rank'].max(), comparison_df['lime_rank'].max()) if not comparison_df.empty else 1
            ax.plot([1, max_rank], [1, max_rank], 'k--', alpha=0.6, label='Perfect Agreement')
            
            ax.set_title(f'SHAP Rank vs. LIME Rank for Stock {stock_key} (Spearman: {rank_corr_coeff:.2f})')
            ax.set_xlabel('SHAP Feature Rank (1 = most important)')
            ax.set_ylabel('LIME Feature Rank (1 = most important)')
            ax.set_xticks(range(1, int(max_rank) + 1))
            ax.set_yticks(range(1, int(max_rank) + 1))
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        st.markdown(f"The comparison between SHAP and LIME provides invaluable context. The side-by-side table allows for direct scrutiny of feature contributions and their relative importance (ranks) from both methods. The Spearman rank correlation coefficient quantifies their agreement: a high positive correlation (e.g., > 0.7) means both methods generally agree on which features are most important, significantly increasing my confidence in the explanation. If the correlation is low, the scatter plot helps visualize the disagreement. For example, if 'momentum_12m' is ranked very high by SHAP but low by LIME, it suggests that either LIME's local approximation struggled, or the model has strong feature interactions that SHAP can capture but LIME's linear surrogate cannot. This diagnostic information is crucial for deciding how much weight to place on the explanation and whether further investigation into the model's behavior is necessary, a key part of risk management in quantitative investing.")

    else:
        st.info("Please complete the 'Introduction & Setup' page first.")

# Page 5: XAI Method Selection Guide
elif st.session_state.current_page == "XAI Method Selection Guide":
    st.header("6. Strategic XAI Selection: When to Use LIME, When to Use SHAP")
    st.markdown(f"As a Portfolio Manager, I encounter diverse models—from simple linear regressions to complex neural networks or proprietary vendor solutions. Each XAI method, like LIME and SHAP, has its strengths and weaknesses, making no single method universally superior. Having a clear decision framework for selecting the appropriate XAI tool is essential for efficiency, accuracy, and compliance. This guide, based on various criteria, helps me choose between LIME and SHAP based on the specific model type, explanation needs, and computational constraints. This strategic selection ensures I apply the most suitable tool for each unique explanatory challenge.")

    st.subheader("SHAP vs. LIME Decision Matrix")
    st.markdown(
        """
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
        """
    )

    st.subheader("Decision Heuristic for XAI Method Selection:")
    st.markdown(
        """
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
        """
    )
    st.markdown(f"This decision framework empowers me to make an informed choice about which XAI tool to deploy in different scenarios. For instance, if I'm analyzing a vendor's black-box model (Scenario 2), LIME is the go-to. If I'm using an internal, tree-based model for a highly regulated credit scoring task (Scenarios 1 & 3), SHAP's determinism and exactness would be preferred. The comparison conducted in the previous section (Scenario 7) directly feeds into this guide, allowing me to interpret discrepancies as diagnostic signals. By understanding the nuances of each method, I can apply XAI more effectively, enhancing my model governance, due diligence, and overall decision-making process.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2026  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
