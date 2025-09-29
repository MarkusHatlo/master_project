from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


csv_path = Path('post_process_data.csv')
csv_df = pd.read_csv(csv_path)
ER = csv_df['ER']
U = csv_df['velocity']


fig, ax = plt.subplots(1,1, figsize=(11,5))
ax.plot(ER,U)
ax.set_xlabel('Equivalence ratio [-]')
ax.set_ylabel('Velocity [m/s]')
plt.tight_layout()
plt.show()
