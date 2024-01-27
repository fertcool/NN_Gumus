import numpy as np
from keras.models import Sequential
from keras import layers, regularizers
from keras.layers import Dense, Dropout
from matplotlib import pyplot as plt
from sklearn import preprocessing
import pandas as pd
from keras.callbacks import EarlyStopping

dfs = pd.read_excel("./i_kanalov-1.xlsx")
dfs = dfs[["Р2О5", "К2О", "Гидролитическая кислотность", "рН водный", "рН солевой", "Гумус", "1 канал", "2 канал",
           "3 канал", "4 канал", "5 канал", "6 канал", "7 канал", "8 канал", "red color", "green color", "blue color"]]
dfs = dfs/dfs.max()

train_dfs = dfs.sample(frac=0.9, random_state=0)
predict_dfs = dfs.drop(train_dfs.index)

train_features = np.array(train_dfs[["Р2О5", "К2О", "Гидролитическая кислотность", "рН водный", "рН солевой"]])
train_target = np.array(train_dfs[["Гумус"]])
predict_features = np.array(predict_dfs[["Р2О5", "К2О", "Гидролитическая кислотность", "рН водный", "рН солевой"]])
predict_target = np.array(predict_dfs[["Гумус"]])



# normalizerf = layers.Normalization(axis=-1)
# normalizerf.adapt(np.array(dfs.drop(columns=["Гумус"], axis=1)))


model = Sequential([
                    Dense(240, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
                    Dropout(0.2),
                    Dense(400, activation='relu'),
                    Dropout(0.3),
                    Dense(100, activation='relu'),
                    Dense(1, activation='sigmoid')]
                   )

model.compile(loss='MAE', optimizer='adam', metrics=['MeanSquaredError'])

esc = EarlyStopping(monitor="loss", patience=5)
history = model.fit(train_features, train_target, epochs=1000000, callbacks=[esc])

# loss, accuracy = model.evaluate(train_features, train_target)
# print(f"Точность модели: {accuracy * 100:.2f}%")


train_pr = model.predict(train_features)
predict_pr = model.predict(predict_features)
df_pr_train = pd.DataFrame({"prediction": train_pr.T[0], "target": train_target.T[0], "abs": abs(train_pr.T[0]-train_target.T[0])})
df_pr_pr = pd.DataFrame({"prediction": predict_pr.T[0], "target": predict_target.T[0], "abs": abs(predict_pr.T[0]-predict_target.T[0])})

plt.plot(history.history["loss"], label="loss")
plt.xlabel("Эпоха обучения")
plt.ylabel("Ошибка (MAE)")
plt.legend()
plt.show()

plt.plot(df_pr_pr["abs"], label="abs")
plt.show()

plt.plot(df_pr_train["abs"], label="abs")
plt.show()
# model.save('gumus_model')
# model_loaded = keras.models.load_model('16_model')
# predictions = model_loaded.predict(InputArr)
#
#
# for i in range(count):
#     print((predictions*max(data.humus))[i]-data.humus[i])
