import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# This script is meant to be used to create custom graphs for articles.
# All data series from CustomParameterData.csv are ploted on one graph.

vectorOutput_flag = False
customFontsize = 17

data = pd.read_csv("CustomParameterData.csv")

vector = data.iloc[0, 1:].values.astype(np.float)
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

plt.errorbar(x, data.iloc[0, 1:].values.astype(np.float), data.iloc[1, 1:].values.astype(np.float), label="Shure SM57", color='crimson', marker='D', linestyle='', ecolor='crimson', elinewidth=1.5, capsize=15)
plt.errorbar(x, data.iloc[2, 1:].values.astype(np.float), data.iloc[3, 1:].values.astype(np.float), label="Rode NT2", color='limegreen', marker='D', linestyle='', ecolor='limegreen', elinewidth=1.5, capsize=15)
plt.errorbar(x, data.iloc[4, 1:].values.astype(np.float), data.iloc[5, 1:].values.astype(np.float), label="Piezo", color='steelblue', marker='D', linestyle='', ecolor='steelblue', elinewidth=1.5, capsize=15)

plt.xlabel('Głębokość [mm]', fontsize=customFontsize)
plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], ['25', '30', '35', '40', '45', '50', '55', '60', '65', '70'])
plt.ylabel('', fontsize=customFontsize)
plt.subplots_adjust(bottom=0.2)
plt.legend()
plt.grid(True)
plt.yticks(fontsize=12)

# Saving graph
outputFile = "CustomGraph"
figure = plt.gcf()
figure.set_size_inches(9, 5.5)

if vectorOutput_flag:
        plt.savefig(outputFile + '.pdf', dpi=1200, format="pdf")
else:
        plt.savefig(outputFile, dpi=100)

#plt.show()
plt.clf()

print('Figure saved as: ' + outputFile)