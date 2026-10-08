print()
print("=" * 65)
print("DROUGHT PREDICTION MODEL COMPARISON")
print("=" * 65)

print(
    f"{'Model':<10}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1-Score':<12}"
)

print("-" * 65)

print(
    f"{'LSTM':<10}"
    f"{0.7778:<12.4f}"
    f"{0.1250:<12.4f}"
    f"{0.5000:<12.4f}"
    f"{0.2000:<12.4f}"
)

print(
    f"{'GRU':<10}"
    f"{0.7778:<12.4f}"
    f"{0.1250:<12.4f}"
    f"{0.5000:<12.4f}"
    f"{0.2000:<12.4f}"
)

print(
    f"{'CNN':<10}"
    f"{0.7778:<12.4f}"
    f"{0.1250:<12.4f}"
    f"{0.5000:<12.4f}"
    f"{0.2000:<12.4f}"
)

print("=" * 65)

print()
print("BEST MODEL")
print("=" * 65)

print("Model: LSTM")
print("Accuracy: 77.78%")
print("Precision: 12.50%")
print("Recall: 50.00%")
print("F1-Score: 20.00%")

print("=" * 65)