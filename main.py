from src.experiments.experiment_05 import run_experiment_05_comprehensive

def main():
    print("\nRunning Experiment 05 – Comprehensive Multimodal Analysis")
    print("=========================================================\n")
    results, best_name, best_metrics, shap_results = run_experiment_05_comprehensive()

    print("\nFinished Experiment 05")
    print(f"Best Model: {best_name}")
    print(f"F1 Score: {best_metrics['f1']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall: {best_metrics['recall']:.4f}")
    print(f"Accuracy: {best_metrics['accuracy']:.4f}")
    print(f"Threshold: {best_metrics['threshold']:.2f}")

if __name__ == "__main__":
    main()
