import pandas as pd
import numpy as np
import os

#If error occurs might be because of different operating systems
df = pd.read_csv(os.path.abspath("data/training_data_fall2024.csv"))

#Changes labeling to 0 and 1 (instead of the strings)
df['increase_stock'] = np.where(df['increase_stock'] == 'high_bike_demand', 1, 0)

#MinMax normalization
df = (df - df.min()) / (df.max() - df.min()).replace(0, 1)

df.to_csv(os.path.abspath("data/normalized_labeled_training_data.csv"), index=False)
