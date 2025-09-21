from training import train, final_test, preprocessing
import pandas as pd
from isc301.config import maisons_raw_path

if __name__ == "__main__":
    df = pd.read_csv(maisons_raw_path)

    df_train = preprocessing(df.copy())
    df_final_test = preprocessing(df.copy())
    model = train(df_train)

    final_test(model, df_final_test)

    # you can also predict on new data with model.predict(new_X)