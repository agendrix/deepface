import json

import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open("scripts/data.json", "r") as file:
    data = json.load(file)

# Parse the data to extract distance and rekognition_matches values
records = [{"distance": entry["@message"]["distance"], "rekognition_matches": entry["@message"]["rekognition_matches"]} for entry in data]
print(f"Loaded {len(records)} entries from data.json")
print("Min Distance:", min(record["distance"] for record in records))
print("Max Distance:", max(record["distance"] for record in records))

# Set thresholds to test
thresholds = np.arange(0.0, 1.0, 0.01)

# Initialize lists to store metrics
weighted_success_rates = []
false_positives_list = []
false_negatives_list = []

# Calculate the weights as fractions
weight_false = sum(1 for record in records if not record["rekognition_matches"])  # Number of false cases
weight_true = len(records) - weight_false  # Number of true cases

print("Weight False:", weight_false)
print("Weight True:", weight_true)

# Set normalized weights
false_weight = len(records) / (2 * weight_false) if weight_false else 1.0
true_weight = len(records) / (2 * weight_true) if weight_true else 1.0

print("False Weight:", false_weight)
print("True Weight:", true_weight)


# Function to calculate weighted success, false positives, and false negatives
def evaluate_threshold(threshold):
    weighted_success = 0
    false_positives = 0
    false_negatives = 0

    for record in records:
        matches = record["distance"] <= threshold
        rekognition_matches = record["rekognition_matches"]

        # Weighted success calculation
        if matches == rekognition_matches:
            if rekognition_matches:
                weighted_success += true_weight
            else:
                weighted_success += false_weight
        elif matches and not rekognition_matches:
            false_positives += 1
        elif not matches and rekognition_matches:
            false_negatives += 1

    weighted_success_rate = weighted_success / len(records)
    return weighted_success_rate, false_positives, false_negatives


# Calculate metrics for each threshold
for threshold in thresholds:
    weighted_success_rate, false_positives, false_negatives = evaluate_threshold(threshold)
    weighted_success_rates.append(weighted_success_rate)
    false_positives_list.append(false_positives)
    false_negatives_list.append(false_negatives)

# Find the threshold that maximizes weighted success rate with minimal false positives
optimal_index = np.argmax(weighted_success_rates)
optimal_threshold = thresholds[optimal_index]
optimal_weighted_success_rate = weighted_success_rates[optimal_index]

# Plot Weighted Success Rate vs. Threshold
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.plot(thresholds, weighted_success_rates, label="Weighted Success Rate", color="blue")
plt.axvline(x=optimal_threshold, color="red", linestyle="--", label=f"Optimal Threshold = {optimal_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Weighted Success Rate")
plt.title("Weighted Success Rate vs. Threshold")
plt.legend()
plt.grid()

# Plot False Positives and False Negatives vs. Threshold
plt.subplot(1, 2, 2)
plt.plot(thresholds, false_positives_list, label="False Positives", color="orange")
plt.plot(thresholds, false_negatives_list, label="False Negatives", color="purple")
plt.axvline(x=optimal_threshold, color="red", linestyle="--", label=f"Optimal Threshold = {optimal_threshold:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Count")
plt.title("False Positives and False Negatives vs. Threshold")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

# Display optimal threshold and associated metrics
print("Optimal Threshold:", optimal_threshold)
print("Optimal Weighted Success Rate:", optimal_weighted_success_rate)
print("At Optimal Threshold - False Positives:", false_positives_list[optimal_index])
print("At Optimal Threshold - False Negatives:", false_negatives_list[optimal_index])
