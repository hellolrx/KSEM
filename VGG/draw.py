 #图的源代码
import matplotlib.pyplot as plt

attacks = ['BadNet', 'Blend', 'TaCT', 'Trojan']
prune_suspicious = [11122, 9919, 9646, 9082]
full_suspicious = [5437, 5143, 4171, 3873]
prune_trigger = [150, 146, 139, 145]
full_trigger = [149, 145, 150, 148]


colors = ['#C8D7EB', '#FAEBC7', '#9FC9DF', '#F1E1C7']


fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
bar_width = 0.35
index = range(len(attacks))


axes[0].bar([i - bar_width/2 for i in index], prune_suspicious, bar_width, label='Pruning-only', color=colors[0])
axes[0].bar([i + bar_width/2 for i in index], full_suspicious, bar_width, label='Full', color=colors[2])
axes[0].set_title('Suspicious Data', fontsize=16)
axes[0].set_xlabel('Attack Type', fontsize=14)
axes[0].set_ylabel('Data Count', fontsize=14)
axes[0].set_xticks(index)
axes[0].set_xticklabels(attacks, fontsize=12)
axes[0].legend(fontsize=12)
axes[0].tick_params(axis='y', labelsize=12)


axes[1].bar([i - bar_width/2 for i in index], prune_trigger, bar_width, label='Pruning-only', color=colors[1])
axes[1].bar([i + bar_width/2 for i in index], full_trigger, bar_width, label='Full', color=colors[3])
axes[1].set_title('Trigger Data', fontsize=16)
axes[1].set_xlabel('Attack Type', fontsize=14)
axes[1].set_xticks(index)
axes[1].set_xticklabels(attacks, fontsize=12)
axes[1].legend(fontsize=12)
axes[1].tick_params(axis='y', labelsize=12)

plt.tight_layout()
plt.show()