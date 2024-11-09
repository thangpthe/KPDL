import pandas as pd
from imblearn.under_sampling import RandomUnderSampler

# Đọc dataset
data = pd.read_csv('./dataset/obesity_with_bmi.csv')


X = data.iloc[:, :-1]  # Thuộc tính
y = data.iloc[:, -1]   # Nhãn(Obesity)

rus = RandomUnderSampler(random_state=42)

# Cân bằng dữ liệu
X_resampled, y_resampled = rus.fit_resample(X, y)

