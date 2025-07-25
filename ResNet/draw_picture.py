import matplotlib.pyplot as plt
import numpy as np

# Attack types
attacks = ["badnet", "blend", "TaCT", "Trojan"]

# ResNet and VGG data (updated based on your table)
resnet_data = [2390, 4811, 6548, 4177]
vgg_data = [5437, 5143, 4171, 3873]
resnet_triggers = [145, 148, 150, 148]
vgg_triggers = [149, 145, 150, 148]

# Plot the line chart
plt.figure(figsize=(10, 6))
plt.plot(attacks, resnet_data, marker="o", linestyle="-", label="ResNet Isolated Data Volume")
plt.plot(attacks, vgg_data, marker="s", linestyle="-", label="VGG Isolated Data Volume")
plt.plot(attacks, resnet_triggers, marker="^", linestyle="--", label="ResNet Trigger Count")
plt.plot(attacks, vgg_triggers, marker="d", linestyle="--", label="VGG Trigger Count")

# Set legend, title, and axis labels
plt.xlabel("Attack Types")
plt.ylabel("Quantity")
plt.title("Isolated Data and Trigger Counts of ResNet & VGG")
plt.legend()
plt.grid(True)

# Show the chart
plt.show()