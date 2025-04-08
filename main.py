import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

st.header("Клиенты авиакомпании")

st.subheader("Загрузка и описание данных")
st.markdown("""
Целевая переменная

    satisfaction: удовлетворенность клиента полетом, бинарная (satisfied или neutral or dissatisfied)

Признаки

    Gender (categorical: Male или Female): пол клиента
    Age (numeric, int): количество полных лет
    Customer Type (categorical: Loyal Customer или disloyal Customer): лоялен ли клиент авиакомпании?
    Type of Travel (categorical: Business travel или Personal Travel): тип поездки
    Class (categorical: Business или Eco, или Eco Plus): класс обслуживания в самолете
    Flight Distance (numeric, int): дальность перелета (в милях)
    Departure Delay in Minutes (numeric, int): задержка отправления (неотрицательная)
    Arrival Delay in Minutes (numeric, int): задержка прибытия (неотрицательная)
    Inflight wifi service (categorical, int): оценка клиентом интернета на борту
    Departure/Arrival time convenient (categorical, int): оценка клиентом удобство времени прилета и вылета
    Ease of Online booking (categorical, int): оценка клиентом удобства онлайн-бронирования
    Gate location (categorical, int): оценка клиентом расположения выхода на посадку в аэропорту
    Food and drink (categorical, int): оценка клиентом еды и напитков на борту
    Online boarding (categorical, int): оценка клиентом выбора места в самолете
    Seat comfort (categorical, int): оценка клиентом удобства сиденья
    Inflight entertainment (categorical, int): оценка клиентом развлечений на борту
    On-board service (categorical, int): оценка клиентом обслуживания на борту
    Leg room service (categorical, int): оценка клиентом места в ногах на борту
    Baggage handling (categorical, int): оценка клиентом обращения с багажом
    Checkin service (categorical, int): оценка клиентом регистрации на рейс
    Inflight service (categorical, int): оценка клиентом обслуживания на борту
    Cleanliness (categorical, int): оценка клиентом чистоты на борту
""")

df = pd.read_csv("https://raw.githubusercontent.com/evgpat/stepik_from_idea_to_mvp/main/datasets/clients.csv")

st.image("Airbus_A310-308-ET,_Aeroflot_AN1625054.jpg", caption = "Самолёт")

st.dataframe(df[0:40])

st.subheader("Однофакторный анализ")
st.write("Определим зависимость задержки прибытия от задержки отправления.")

st.write("На первой диаграмме есть выбросы в виде сверхдальних рейсов. На второй диаграмме есть выбросы в виде многосуточных задержек. Удалим их.")

df["Departure Delay in Minutes"] = np.where(df["Departure Delay in Minutes"] > 90, 90, df["Departure Delay in Minutes"])
df["Arrival Delay in Minutes"] = np.where(df["Arrival Delay in Minutes"] > 90, 90, df["Arrival Delay in Minutes"])

plt.figure(figsize = (8, 5))
dep_del = sns.histplot(df["Departure Delay in Minutes"], kde = True)
st.pyplot(dep_del.get_figure())

plt.figure(figsize = (8, 5))
arr_del = sns.histplot(df["Arrival Delay in Minutes"], kde = True)
st.pyplot(arr_del.get_figure())

plt.figure(figsize = (8, 5))
dep_arr = sns.histplot(df[["Departure Delay in Minutes", "Arrival Delay in Minutes"]], kde = True)
st.pyplot(dep_arr.get_figure())

st.subheader("Класс обслуживания")
st.write("Определим связь класса обслуживания и задержки отправления рейса.")

st.write("Построим гистограмму классов обсуживания.")
plt.figure(figsize = (8, 5))
cl = sns.countplot(x = "Class", data = df, palette = "husl")
st.pyplot(cl.get_figure())

st.write("Визуализируем связь класса обслуживания и задержки отправления.")
plt.figure(figsize = (8, 5))
cl_del = sns.barplot(x = "Class", y = "Departure Delay in Minutes", data = df, palette = "husl")
st.pyplot(cl_del.get_figure())

st.write("Средняя задержка отправления для пассажиров бизнес-класса:", df[df.Class == "Business"]["Departure Delay in Minutes"].mean())
st.write("Средняя задержка отправления для пассажиров эконом-класса:", df[df.Class == "Eco"]["Departure Delay in Minutes"].mean())
st.write("Средняя задержка отправления для пассажиров комфорт-класса:", df[df.Class == "Eco Plus"]["Departure Delay in Minutes"].mean())

st.subheader("Тип поездки")
st.write("Определим связь типа поездки и задержки отправления рейса.")

st.write("Построим гистограмму типов поездок.")
plt.figure(figsize = (8, 5))
typ = sns.countplot(x = "Type of Travel", data = df, palette = "hls")
st.pyplot(typ.get_figure())

st.write("Визуализируем связь типа поездки и задержки отправления.")
plt.figure(figsize = (8, 5))
typ_del = sns.barplot(x = "Type of Travel", y = "Departure Delay in Minutes", data = df, palette = "husl")
st.pyplot(typ_del.get_figure())

st.write("Средняя задержка отправления для деловых поездок", df[df["Type of Travel"] == "Business travel"]["Departure Delay in Minutes"].mean())
st.write("Средняя задержка отправления для личных поездок", df[df["Type of Travel"] == "Personal Travel"]["Departure Delay in Minutes"].mean())

st.write("Посмотрим, как влияют оба фактора на зедержку отправления")
plt.figure(figsize = (8, 5))
cl_typ_del = sns.barplot(x = "Class", y = "Departure Delay in Minutes", hue = "Type of Travel", data = df, palette = "tab10")
st.pyplot(cl_typ_del.get_figure())

st.subheader("Возраст")
st.write("Определим связь возраста пассажира и задержки отправления рейса.")

st.write("Построим гистограмму возраста пассажира.")
plt.figure(figsize = (8, 5))
age = sns.histplot(df["Age"], kde = True)
st.pyplot(age.get_figure())

st.write("Визуализируем связь возраста пассажира и задержки отправления.")
plt.figure(figsize = (8, 5))
age_del = sns.histplot(df[["Departure Delay in Minutes", "Age"]], kde = True)
st.pyplot(age_del.get_figure())

st.subheader("Пол")
st.write("Определим связь пола пассажира и задержки отправления рейса.")

st.write("Построим гистограмму пола пассажира.")
plt.figure(figsize = (8, 5))
gender = sns.countplot(x = "Gender", data = df, palette = "mako")
st.pyplot(gender.get_figure())

st.write("Визуализируем связь пола пассажира и задержки отправления.")
plt.figure(figsize = (8, 5))
gender_del = sns.barplot(x = "Gender", y = "Departure Delay in Minutes", data = df, palette = "mako")
st.pyplot(gender_del.get_figure())

st.write("Средняя задержка отправления для мужчин:", df[df.Gender == "Male"]["Departure Delay in Minutes"].mean())
st.write("Средняя задержка отправления для женщин:", df[df.Gender == "Female"]["Departure Delay in Minutes"].mean())
st.write("Средняя задержка отправления для лиц неопределённого пола:", df[df.Gender == "Non-binary"]["Departure Delay in Minutes"].mean())

st.subheader("Предсказание лояльности")
st.write("Представьтесь:")
age = st.number_input("Возраст: ", 0, 123)
gender = st.selectbox("Пол:", ["Мужской", "Женский"])
loyal = st.checkbox("Лояльный клиент")
trip_type = st.selectbox("Тип поездки:", ["Деловой", "Личный"])
air_class = st.selectbox("Класс обслуживания:", ["Эконом", "Эконом плюс", "Бизнес"])
grade_depart = st.slider("Оцените время вылета: ", 0, 5)
grade_wifi = st.slider("Оцените WiFi: ", 0, 5)
grade_gate = st.slider("Оцените расположение выхода на посадку: ", 0, 5)
grade_booking = st.slider("Оцените удобство покупки билета: ", 0, 5)
grade_food = st.slider("Оцените бортовое питание: ", 0, 5)
grade_seat = st.slider("Оцените удобство сидений: ", 0, 5)
grade_legroom = st.slider("Оцените место для ног: ", 0, 5)
grade_ife = st.slider("Оцените развлечения на борту: ", 0, 5)
grade_registr = st.slider("Оцените регистрацию на рейс: ", 0, 5)
grade_service = st.slider("Оцените обслуживание на борту: ", 0, 5)
grade_luggage = st.slider("Оцените обработку багажа: ", 0, 5)
grade_clean = st.slider("Оцените чистоту на борту: ", 0, 5)

grades = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness"
]

X = df[grades]
X.fillna(X.median(), inplace = True)
y = df["satisfaction"]

model = LogisticRegression()
model.fit(X, y)

if st.button("Предсказать"):
	row = [age, gender, loyal, trip_type, air_class, grade_depart, grade_wifi, grade_gate, grade_booking, grade_food, grade_seat, grade_legroom, grade_ife, grade_registr, grade_service, grade_luggage, grade_clean]
	row = pd.DataFrame(row)
	grade_total = model.predict(row)
	print("Общая оценка:", grade_total)

