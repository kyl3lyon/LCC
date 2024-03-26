def generate_evaluation_table(results_df):
    """Generates a Markdown table from a DataFrame of model evaluation results."""

    markdown_table = "| Model                          | Accuracy  | Balanced Accuracy | ROC AUC   | F1 Score  | Time Taken |\n"
    markdown_table += "|:-------------------------------|----------:|------------------:|----------:|----------:|-----------:|\n"

    for index, row in results_df.iterrows():
        model_name = row['Model'].ljust(30)  # Adjust the number to fit your longest model name
        accuracy = f"{row['Accuracy']:.6f}".rjust(9)
        bal_accuracy = f"{row['Balanced Accuracy']:.6f}".rjust(17)
        roc_auc = f"{row['ROC AUC']:.6f}".rjust(9)
        f1_score = f"{row['F1 Score']:.6f}".rjust(9)
        time_taken = f"{row['Time Taken']:.6f}".rjust(10)

        markdown_table += f"| {model_name} | {accuracy} | {bal_accuracy} | {roc_auc} | {f1_score} | {time_taken} |\n"

    return markdown_table