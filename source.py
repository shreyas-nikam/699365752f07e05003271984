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
import os

# Set Matplotlib backend to 'Agg' to prevent display issues in non-GUI environments
plt.switch_backend('Agg')
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

    n_datapoints = n_stocks * n_months
    factors_df = pd.DataFrame({
        'momentum_12m': np.random.randn(n_datapoints),
        'pe_ratio_z': np.random.randn(n_datapoints),
        'pb_ratio_z': np.random.randn(n_datapoints),
        'earnings_growth': np.random.randn(n_datapoints),
        'revenue_growth': np.random.randn(n_datapoints),
        'log_market_cap': np.random.randn(n_datapoints) + 10,
        'analyst_sentiment': np.random.randn(n_datapoints),
        'volatility_60d': np.abs(np.random.randn(n_datapoints)) * 0.3
    })

    feature_cols = factors_df.columns.tolist()

    factors_df['next_month_return'] = (
        0.003 * factors_df['momentum_12m'] +
        -0.002 * factors_df['pe_ratio_z'] +
        0.001 * factors_df['earnings_growth'] +
        0.002 * factors_df['analyst_sentiment'] +
        np.random.randn(n_datapoints) * 0.05
    )

    X = factors_df[feature_cols]
    y = factors_df['next_month_return']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

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

def explain_stock_lime(model, X_data, idx_in_X_data, lime_explainer, feature_names, n_features=8, stock_name="Stock", output_dir="explanation_outputs", save_plot=True):
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
        output_dir (str): Directory to save output plots.
        save_plot (bool): Whether to save the LIME explanation plot.

    Returns:
        tuple: (explanation, contrib_df, prediction)
            explanation: The LIME explanation object.
            contrib_df (pd.DataFrame): DataFrame of feature contributions.
            prediction (float): The model's prediction for the instance.
    """
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)

    instance = X_data.iloc[idx_in_X_data].values
    prediction = model.predict(instance.reshape(1, -1))[0]

    explanation = lime_explainer.explain_instance(
        data_row=instance,
        predict_fn=model.predict,
        num_features=n_features,
        num_samples=1000
    )

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

    print(f"{'Feature Rule':<35s} {'Weight':>8s} {'Bar'}")
    print("-" * 60)
    for _, row in contrib_df.iterrows():
        bar_len = int(abs(row['lime_weight']) * 200) # Adjusted scaling
        bar_char = '+' if row['lime_weight'] > 0 else '-'
        bar = bar_char * min(bar_len, 30) # Cap bar length to avoid excessively long bars
        print(f" {row['feature_rule']:<35s} {row['lime_weight']:>+8.4f} {bar}")

    if save_plot:
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        fig.suptitle(f"LIME Explanation for {stock_name} (Predicted Return: {prediction*100:+.2f}%)", y=1.02)
        plt.tight_layout()
        filename = f'lime_explanation_{stock_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace(".", "")}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close(fig)

    return explanation, contrib_df, prediction

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

    rec = "OVERWEIGHT" if prediction > 0.005 else ("UNDERWEIGHT" if prediction < -0.005 else "NEUTRAL")
    print(f"Model Recommendation: {rec}")
    print(f"Expected Alpha: {prediction*100:+.2f}%/month")
    print("=" * 55)

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

    for feat_rule, weight in explanation_as_list[:6]: # Consider top 6 features for narrative
        feat_name = None
        for key in language_map:
            if key in feat_rule:
                feat_name = key
                break

        if feat_name:
            concept, detail = language_map[feat_name]
            # Refine concept based on rule for better narrative
            if 'pe_ratio_z' in feat_rule:
                if weight > 0 and '<=' in feat_rule: concept = 'attractive ' + concept
                elif weight < 0 and '>=' in feat_rule: concept = 'expensive ' + concept
            elif 'earnings_growth' in feat_rule:
                if weight > 0 and '>=' in feat_rule: concept = 'strong ' + concept
                elif weight < 0 and '<=' in feat_rule: concept = 'weak ' + concept
            elif 'momentum_12m' in feat_rule:
                if weight > 0 and '>=' in feat_rule: concept = 'strong ' + concept
                elif weight < 0 and '<=' in feat_rule: concept = 'weak ' + concept
            elif 'analyst_sentiment' in feat_rule:
                if weight > 0 and '>=' in feat_rule: concept = 'positive ' + concept
                elif weight < 0 and '<=' in feat_rule: concept = 'negative ' + concept
            elif 'volatility_60d' in feat_rule:
                if weight < 0 and '>=' in feat_rule: concept = 'elevated ' + concept
                elif weight > 0 and '<=' in feat_rule: concept = 'low ' + concept

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
        if positives: summary += f" The main risk is {negatives[0]}."
        else: summary += f" The primary negative factor is {negatives[0]}."

    print(summary)
    print("-------------------------------------------\n")
    return summary

def test_lime_stability(model, X_data, idx_in_X_data, lime_explainer, feature_names, n_runs=10, output_dir="explanation_outputs", save_plot=True):
    """
    Runs LIME multiple times on the same instance and analyzes the stability of explanations.

    Args:
        model: The black-box model with a .predict method.
        X_data (pd.DataFrame): The dataset containing the instances to explain.
        idx_in_X_data (int): The integer index of the instance in X_data to explain.
        lime_explainer: An initialized LimeTabularExplainer.
        feature_names (list): List of feature names.
        n_runs (int): Number of times to run LIME for stability analysis.
        output_dir (str): Directory to save output plots.
        save_plot (bool): Whether to save the stability box plot.

    Returns:
        pd.DataFrame: A DataFrame showing stability metrics (mean, std, cv, frequency) for each feature.
    """
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- LIME EXPLANATION STABILITY ANALYSIS (Stock Index: {idx_in_X_data}, {n_runs} runs) ---")

    instance = X_data.iloc[idx_in_X_data].values
    all_explanations_weights = []

    for run in range(n_runs):
        exp = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict,
            num_features=len(feature_names),
            num_samples=1000,
        )
        weights = dict(exp.as_list())
        all_explanations_weights.append(weights)

    all_features = set()
    for exp_weights in all_explanations_weights:
        all_features.update(exp_weights.keys())

    stability_data = {}
    for feat_rule in all_features:
        values = [exp_weights.get(feat_rule, 0) for exp_weights in all_explanations_weights]

        mean_val = np.mean(values)
        std_val = np.std(values)

        cv_val = std_val / (abs(mean_val) + 1e-6) # Add small epsilon to avoid division by zero

        appeared_in = sum(1 for v in values if v != 0)

        stability_data[feat_rule] = {
            'mean_weight': mean_val,
            'std_weight': std_val,
            'cv': cv_val,
            'appeared_in_runs': appeared_in
        }

    stab_df = pd.DataFrame(stability_data).T.sort_values(by='mean_weight', key=abs, ascending=False)
    stab_df['rank'] = stab_df['mean_weight'].abs().rank(ascending=False, method='min').astype(int)
    stab_df = stab_df.round(4)

    print("\nFeature Explanation Stability Table:")
    print(f"{'Rank':<5s} {'Feature Rule':<35s} {'Mean':>8s} {'Std':>8s} {'CV':>8s} {'Freq':>6s} {'Status':<10s}")
    print("=" * 90)
    for _, row in stab_df.iterrows():
        status = 'STABLE' if row['cv'] < 0.3 else ('MODERATE' if row['cv'] < 0.6 else 'UNSTABLE')
        print(f"{int(row['rank']):<5d} {row.name[:35]:<35s} {row['mean_weight']:>+8.4f} {row['std_weight']:>8.4f} {row['cv']:>8.2f} {int(row['appeared_in_runs']):>6d} {status:<10s}")

    avg_cv_overall = stab_df['cv'].mean()
    print(f"\nAverage CV across all features: {avg_cv_overall:.3f}")
    if avg_cv_overall > 0.5:
        print("WARNING: LIME explanations show significant instability for this instance. Consider increasing num_samples or using SHAP for more deterministic explanations.")
    elif avg_cv_overall > 0.3:
        print("Note: LIME explanations show moderate instability. Review critical features carefully.")
    else:
        print("PASS: LIME explanations appear reasonably stable on average for this instance.")

    if save_plot:
        plot_data = []
        features_to_plot = stab_df[stab_df['appeared_in_runs'] > 0].index.tolist()
        if not features_to_plot:
            print("No features with non-zero weights found for plotting stability.")
        else:
            for feat_rule in features_to_plot:
                values = [exp_weights.get(feat_rule, 0) for exp_weights in all_explanations_weights]
                plot_data.extend([{'feature_rule': feat_rule, 'weight': w} for w in values])

            plot_df = pd.DataFrame(plot_data)

            # Plot top features by absolute mean weight for clarity
            top_features_for_plot = stab_df.head(min(10, len(stab_df))).index.tolist()
            plot_df_filtered = plot_df[plot_df['feature_rule'].isin(top_features_for_plot)]

            if not plot_df_filtered.empty:
                plt.figure(figsize=(12, 7))
                ordered_features = stab_df.loc[top_features_for_plot, 'mean_weight'].abs().sort_values(ascending=False).index
                sns.boxplot(x='weight', y='feature_rule', data=plot_df_filtered, orient='h', order=ordered_features)
                plt.title(f'LIME Feature Weight Stability (Box Plot across {n_runs} Runs) for Stock Index {idx_in_X_data}')
                plt.xlabel('LIME Weight')
                plt.ylabel('Feature Rule')
                plt.grid(axis='x', linestyle='--', alpha=0.7)
                plt.tight_layout()
                filename = f'lime_stability_boxplot_stock_{idx_in_X_data}.png'
                plt.savefig(os.path.join(output_dir, filename), dpi=150)
                plt.close()
            else:
                print("No data to plot for LIME stability after filtering.")

    return stab_df

def compare_shap_lime(model, X_data, idx_in_X_data, lime_explainer, feature_names, stock_name="Stock", output_dir="explanation_outputs", save_plot=True):
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
        output_dir (str): Directory to save output plots.
        save_plot (bool): Whether to save the SHAP vs LIME rank comparison plot.

    Returns:
        tuple: (shap_contrib_df, lime_contrib_df, rank_corr_coeff)
            shap_contrib_df (pd.DataFrame): SHAP values for each feature.
            lime_contrib_df (pd.DataFrame): LIME weights mapped to feature names.
            rank_corr_coeff (float): Spearman rank correlation coefficient.
    """
    if save_plot:
        os.makedirs(output_dir, exist_ok=True)

    print(f"\n--- SHAP vs. LIME HEAD-TO-HEAD COMPARISON for {stock_name} (Index: {idx_in_X_data}) ---")

    instance = X_data.iloc[idx_in_X_data:idx_in_X_data+1]

    print("\nGenerating SHAP explanation...")
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(instance)[0]

    shap_contrib_df = pd.DataFrame({
        'feature': feature_names,
        'shap_value': shap_values,
        'abs_shap_value': np.abs(shap_values)
    }).sort_values(by='abs_shap_value', ascending=False).reset_index(drop=True)
    shap_contrib_df['shap_rank'] = shap_contrib_df.index + 1

    print("Generating LIME explanation (for comparison)...")
    instance_lime_data = X_data.iloc[idx_in_X_data].values
    lime_exp_comparison = lime_explainer.explain_instance(
        data_row=instance_lime_data,
        predict_fn=model.predict,
        num_features=len(feature_names),
        num_samples=1000
    )
    lime_weights_raw = lime_exp_comparison.as_list()

    lime_weights_mapped = {}
    for rule, weight in lime_weights_raw:
        best_match_len = -1
        best_match_feat = None
        for feat in feature_names:
            if feat in rule and len(feat) > best_match_len:
                best_match_len = len(feat)
                best_match_feat = feat

        if best_match_feat:
            lime_weights_mapped[best_match_feat] = lime_weights_mapped.get(best_match_feat, 0) + weight

    lime_contrib_df = pd.DataFrame({
        'feature': list(lime_weights_mapped.keys()),
        'lime_weight': list(lime_weights_mapped.values())
    })
    lime_contrib_df['abs_lime_weight'] = np.abs(lime_contrib_df['lime_weight'])
    lime_contrib_df = lime_contrib_df.sort_values(by='abs_lime_weight', ascending=False).reset_index(drop=True)
    lime_contrib_df['lime_rank'] = lime_contrib_df.index + 1

    comparison_df = pd.merge(shap_contrib_df[['feature', 'shap_value', 'shap_rank']],
                             lime_contrib_df[['feature', 'lime_weight', 'lime_rank']],
                             on='feature', how='outer')
    comparison_df = comparison_df.fillna({
        'shap_value': 0,
        'lime_weight': 0,
        'shap_rank': len(feature_names) + 1,
        'lime_rank': len(feature_names) + 1
    })
    comparison_df['shap_rank'] = comparison_df['shap_rank'].astype(int)
    comparison_df['lime_rank'] = comparison_df['lime_rank'].astype(int)

    comparison_df = comparison_df.sort_values(by='shap_rank').reset_index(drop=True)

    print("\n--- Feature Contribution & Rank Comparison ---")
    print(f"{'Feature':<22s} {'SHAP Value':>10s} {'SHAP Rank':>10s} {'LIME Weight':>12s} {'LIME Rank':>10s}")
    print("=" * 74)
    for _, row in comparison_df.iterrows():
        print(f"{row['feature']:<22s} {row['shap_value']:>+10.4f} {row['shap_rank']:>10d} {row['lime_weight']:>+12.4f} {row['lime_rank']:>10d}")
    print("----------------------------------------------")

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

    if save_plot:
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='shap_rank', y='lime_rank', data=comparison_df, hue='feature', s=100)

        max_rank = max(comparison_df['shap_rank'].max(), comparison_df['lime_rank'].max())
        plt.plot([0, max_rank + 1], [0, max_rank + 1], 'k--', alpha=0.6, label='Perfect Agreement')

        plt.title(f'SHAP Rank vs. LIME Rank for {stock_name} (Spearman: {rank_corr_coeff:.2f})')
        plt.xlabel('SHAP Feature Rank (1 = most important)')
        plt.ylabel('LIME Feature Rank (1 = most important)')
        plt.xticks(range(1, max_rank + 2))
        plt.yticks(range(1, max_rank + 2))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        filename = f'shap_lime_rank_comparison_stock_{idx_in_X_data}.png'
        plt.savefig(os.path.join(output_dir, filename), dpi=150)
        plt.close()

    return shap_contrib_df, lime_contrib_df, rank_corr_coeff

def run_explanation_workflow(
    n_stocks=500, n_months=60, test_size=0.2, random_state=42, output_dir="explanation_outputs"
):
    """
    Orchestrates the entire stock scoring and explanation workflow.

    Args:
        n_stocks (int): Number of simulated stocks.
        n_months (int): Number of simulated months of data.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Seed for random number generators.
        output_dir (str): Directory to save output plots and reports.

    Returns:
        dict: A dictionary containing key results and explanation data.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Explanation outputs will be saved to: {os.path.abspath(output_dir)}")

    # 1. Setup the environment and train the model
    model, X_train, y_train, X_test, y_test, feature_cols, factors_df = \
        setup_stock_scoring_environment(n_stocks, n_months, test_size, random_state)

    # 2. Identify top and bottom scoring stocks in the test set
    predictions = model.predict(X_test)
    top_stock_idx_in_test = np.argmax(predictions)
    bottom_stock_idx_in_test = np.argmin(predictions)

    print(f"Top scoring stock in test set (index in X_test): {top_stock_idx_in_test}")
    print(f"Bottom scoring stock in test set (index in X_test): {bottom_stock_idx_in_test}")

    # 3. Initialize LIME Explainer
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=feature_cols,
        class_names=['next_month_return'],
        mode='regression',
        random_state=random_state,
        discretize_continuous=True
    )

    results = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'feature_cols': feature_cols,
        'top_stock_idx': top_stock_idx_in_test,
        'bottom_stock_idx': bottom_stock_idx_in_test,
        'explanations': {}
    }

    # 4. Explain the top-scoring stock using LIME
    print("\n--- Explaining Top-Scored Stock ---")
    lime_exp_top, contrib_top_df, pred_top = explain_stock_lime(
        model, X_test, top_stock_idx_in_test, lime_explainer, feature_cols,
        stock_name=f'Top-Scored Stock (Idx {top_stock_idx_in_test})',
        output_dir=output_dir
    )
    rationale_top_stock = generate_investment_rationale(
        lime_exp_top.as_list(), pred_top, f'Top-Scored Stock (Idx {top_stock_idx_in_test})'
    )
    results['explanations']['top_stock_lime'] = {
        'explanation_obj': lime_exp_top,
        'contributions_df': contrib_top_df,
        'prediction': pred_top,
        'rationale': rationale_top_stock
    }

    # 5. Explain the bottom-scoring stock using LIME
    print("\n--- Explaining Bottom-Scored Stock ---")
    lime_exp_bottom, contrib_bottom_df, pred_bottom = explain_stock_lime(
        model, X_test, bottom_stock_idx_in_test, lime_explainer, feature_cols,
        stock_name=f'Bottom-Scored Stock (Idx {bottom_stock_idx_in_test})',
        output_dir=output_dir
    )
    rationale_bottom_stock = generate_investment_rationale(
        lime_exp_bottom.as_list(), pred_bottom, f'Bottom-Scored Stock (Idx {bottom_stock_idx_in_test})'
    )
    results['explanations']['bottom_stock_lime'] = {
        'explanation_obj': lime_exp_bottom,
        'contributions_df': contrib_bottom_df,
        'prediction': pred_bottom,
        'rationale': rationale_bottom_stock
    }

    # 6. Test LIME stability for the top-scoring stock
    print("\n--- Testing LIME Stability for Top-Scored Stock ---")
    stability_df_top = test_lime_stability(
        model, X_test, top_stock_idx_in_test, lime_explainer, feature_cols, n_runs=10,
        output_dir=output_dir
    )
    results['explanations']['top_stock_lime_stability'] = stability_df_top

    # 7. Compare SHAP and LIME for the top-scoring stock
    print("\n--- Comparing SHAP and LIME for Top-Scored Stock ---")
    shap_contrib_df_top, lime_contrib_df_top_comp, rank_corr_top = compare_shap_lime(
        model, X_test, top_stock_idx_in_test, lime_explainer, feature_cols,
        stock_name=f'Top-Scored Stock (Idx {top_stock_idx_in_test})',
        output_dir=output_dir
    )
    results['explanations']['top_stock_shap_lime_comparison'] = {
        'shap_contributions_df': shap_contrib_df_top,
        'lime_contributions_df_for_comp': lime_contrib_df_top_comp,
        'rank_correlation': rank_corr_top
    }

    print("\n--- XAI Method Selection Guide: Markdown Table Provided Above ---")
    print("This section provides a strategic framework for choosing between LIME and SHAP")
    print("based on model characteristics and specific explanatory needs for financial professionals.")
    print("\n--- Explanation Workflow Completed ---")

    return results

if __name__ == "__main__":
    # This block will only execute when the script is run directly,
    # not when it's imported into another file like app.py.
    # In your app.py, you would import this module and call run_explanation_workflow().

    print("Running the full explanation workflow directly...")
    workflow_results = run_explanation_workflow(output_dir="app_explanation_outputs")

    # Example of accessing results:
    # print(f"\nTop stock predicted return: {workflow_results['explanations']['top_stock_lime']['prediction']:.4f}")
    # print(f"Top stock rationale summary: {workflow_results['explanations']['top_stock_lime']['rationale']}")
    # print(f"\nLIME stability for top stock (first few rows):\n{workflow_results['explanations']['top_stock_lime_stability'].head()}")
    # print(f"\nSHAP-LIME rank correlation for top stock: {workflow_results['explanations']['top_stock_shap_lime_comparison']['rank_correlation']:.3f}")

    print("\nWorkflow execution finished. Check 'app_explanation_outputs' directory for plots and detailed results.")
