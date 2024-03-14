import warnings
import numpy as np 
import pandas as pd 
from sklearn import metrics
import category_encoders as ce
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor  # Change here
from sklearn.metrics import mean_squared_error, r2_score

class NPKEstimator:
    def __init__(self, data='Nutrient_recommendation.csv'):
        self.df = pd.read_csv(data, header=None)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def renameCol(self):
        self.df.columns = ['Crop', 'Temperature', 'Humidity', 'Rainfall', 'Label_N', 'Label_P', 'Label_K']
        self.df = self.df.iloc[1:]  # Drop the first row

    def cropMapper(self):
        # Create mapping of crop (string) to int type
        mapping = dict()
        with open("mapped_crops.csv", "w") as fh:
            fh.write("Crops,Key\n")
            for i, crop in enumerate(np.unique(self.df['Crop']), 1):
                mapping[crop] = i
                fh.write("%s,%d\n" % (crop, i))
            mapping['NA'] = np.nan
            fh.write("NA,nan")

        ordinal_cols_mapping = [{"col": "Crop", "mapping": mapping}]
        encoder = ce.OrdinalEncoder(cols='Crop', mapping=ordinal_cols_mapping, return_df=True)
        return mapping, encoder

    def estimator(self, crop, temp, humidity, rainfall, y_label):
        X = self.df.drop(['Label_N', 'Label_P', 'Label_K'], axis=1)
        y = self.df[y_label]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.20, random_state=42)

        mapping, encoder = self.cropMapper()
        self.X_train = encoder.fit_transform(self.X_train)
        self.X_test = encoder.transform(self.X_test)

        regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)  # Change here
        regressor.fit(self.X_train, self.y_train)

        query = [mapping[crop.strip().lower()], temp, humidity, rainfall]
        y_pred = regressor.predict([query])
        return y_pred

    def accuracyCalculator(self):
        model = GradientBoostingRegressor()  # Change here
        n_estimators_values = [50, 100, 200, 500]
        learning_rates = [0.01, 0.1, 0.2, 0.5]
        max_depths = [3, 5, 7, 10]

        best_accuracy = -1

        for n_estimators in n_estimators_values:
            for lr in learning_rates:
                for depth in max_depths:
                    regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=lr, max_depth=depth, random_state=42)  # Change here
                    regressor.fit(self.X_train, self.y_train)
                    y_pred = regressor.predict(self.X_test)
                    accuracy = r2_score(self.y_test, y_pred)

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy

        return best_accuracy
if __name__ == '__main__':
    obj = NPKEstimator()
    obj.renameCol()
    # 'Label_N', 'Label_P', 'Label_K'
    # rice,21.94766735,80.97384195,213.3560921,67,59,41
    crop, temp, humidity, rainfall, y_label = 'rice',21.94766735,80.97384195,213.3560921,'Label_N'
    res = obj.estimator(crop, temp, humidity, rainfall, y_label)
    print(y_label, ":", res[0])
    
