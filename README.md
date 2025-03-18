# eda-airline
Разведочный анализ клиентов вымышленной авиакомпании.

В данной задаче используется таблица, состоящая из 3 частей
- Сегментационные:
  - `Gender` (categorical: _Male_ или _Female_): пол клиента
  - `Age` (numeric, int): количество полных лет
  - `Customer Type` (categorical: _Loyal Customer_ или _disloyal Customer_): лоялен ли клиент авиакомпании?
  - `Type of Travel` (categorical: _Business travel_ или _Personal Travel_): тип поездки
  - `Class` (categorical: _Business_ или _Eco_, или _Eco Plus_): класс обслуживания в самолете
  - `Flight Distance` (numeric, int): дальность перелета (в милях)
  - `Departure Delay in Minutes` (numeric, int): задержка отправления (неотрицательная)
  - `Arrival Delay in Minutes` (numeric, int): задержка прибытия (неотрицательная)
- Оценки:
  - `Inflight wifi service` (categorical, int): оценка клиентом интернета на борту
  - `Departure/Arrival time convenient` (categorical, int): оценка клиентом удобство времени прилета и вылета
  - `Ease of Online booking` (categorical, int): оценка клиентом удобства онлайн-бронирования
  - `Gate location` (categorical, int): оценка клиентом расположения выхода на посадку в аэропорту
  - `Food and drink` (categorical, int): оценка клиентом еды и напитков на борту
  - `Online boarding` (categorical, int): оценка клиентом выбора места в самолете
  - `Seat comfort` (categorical, int): оценка клиентом удобства сиденья
  - `Inflight entertainment` (categorical, int): оценка клиентом развлечений на борту
  - `On-board service` (categorical, int): оценка клиентом обслуживания на борту
  - `Leg room service` (categorical, int): оценка клиентом места в ногах на борту
  - `Baggage handling` (categorical, int): оценка клиентом обращения с багажом
  - `Checkin service` (categorical, int): оценка клиентом регистрации на рейс
  - `Inflight service` (categorical, int): оценка клиентом обслуживания на борту
  - `Cleanliness` (categorical, int): оценка клиентом чистоты на борту
- Целевая переменная:
  - `satisfaction`: удовлетворенность клиента полетом, бинарная (*satisfied* или *neutral or dissatisfied*)

В блокноте мы анализируем взаимосвязь между сегментационными переменными, а также обучаем логическую регрессию на основе оценов и удовлетворения.
