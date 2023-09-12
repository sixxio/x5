import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import random

from tensorflow.keras.models import load_model
import pickle, numpy as np

st.set_page_config(layout = 'wide', page_title = 'AI Talent Hub | X5', page_icon = '🛒')
px.set_mapbox_access_token(st.secrets['mapbox'])


data = pd.read_parquet('all_with_avg.parquet.gz')

st.title('Статистика по магазинам сети Пятёрочка в городе Санкт-Петербург')

# a1,b1,c1 = st.tabs(['Карта', 'Прогнозирование недостающей оценки', 'Проводник по отзывам'])

a1,b1 = st.tabs(['Карта', 'Прогнозирование недостающей оценки'])

with a1:

    a,b = st.columns((9,1))

    with b:
        regions = ['Все'] + data['breadcrumbs_1'].dropna().unique().tolist()
        depth = st.selectbox('Выберите регион', regions, index = 0)
        if depth == 'Все':
            subregions = ['Все'] + data['breadcrumbs_2'].dropna().unique().tolist()
        else:
            subregions = ['Все'] + data[data["breadcrumbs_1"]==depth]['breadcrumbs_2'].dropna().unique().tolist()
        depth1 = st.selectbox('Выберите субрегион', subregions, index = 0)

        metric = st.selectbox('Выберите метрику', [f'Среднее за {i*7} {"день" if i%3==0 else "дней"}' for i in range(1, 5)] + ['Общая оценка'])
        metric = {'Среднее за 7 дней':'7d_avg','Среднее за 14 дней':'14d_avg','Среднее за 21 день':'21d_avg','Среднее за 28 дней':'28d_avg','Общая оценка': 'rating'}[metric]
        st.write(f'#### Средняя оценка ___{round(data[metric].mean(), 2) if depth1 == "Все" else round(data[data["breadcrumbs_1"]==depth].rating.mean(), 2) }___')

    if depth != 'Все':
        if depth1 != 'Все':
            data = data[(data['breadcrumbs_1']==depth)&(data['breadcrumbs_2']==depth1)]
        else:
            data = data[data['breadcrumbs_1']==depth]
    else:
        if depth1 != 'Все':
            data = data[(data['breadcrumbs_2']==depth1)]

    data[metric] = data[metric].round(2)
    # Display the map
    with a:
        fig = px.scatter_mapbox(
            data, lat="lat", lon="long", hover_name="address",
            zoom=10, hover_data = {'lat' : False, 'long' : False},
            color_continuous_scale = [(0, "red"), (0.5, 'yellow'), (1, "green")], 
            color = metric, range_color=(3, 5))
        fig.update_traces(marker={'opacity': 0.9, 'size':10})
        fig.update_layout(autosize=False, height=800)
        fig.update_layout(mapbox_style="dark")
        st.plotly_chart(fig, use_container_width=True)

with b1:
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    model = load_model('model.h5')

    current_review = st.text_area('Введите текст отзыва')
    if st.button('Спрогнозировать оценку'):
        tokenized_review = tokenizer.texts_to_sequences([current_review])
        tokenized_review = [tokenized_review[0] + [0 for i in range(len(tokenized_review[0]), 20000)]]
        rate = np.argmax(model.predict(tokenized_review)) + 1
        st.write('Вероятнее всего, пользователь, оставивший этот отзыв, оценил магазин Пятёрочка в ' + ':star:'*rate + ' звёзд.')

# with c1:
#     a2,b2 = st.columns([4,1])
#     with b2:
#         rate_1 = st.selectbox('Показать только отзывы', ['Все'] + [f'С оценкой ниже {i}' for i in range(5,2,-1)])
#         rate_1 = 6 if rate_1 == 'Все' else int(rate_1[-1]) 

#     with a2:
#         reviews = pd.read_parquet('all_reviews.parquet.gz').sort_values(by='date', ascending=False)
#         reviews = reviews[reviews['rate'] < rate_1]
#         for i in range(25):
#             cur_review = reviews.iloc[[i]]
#             with st.chat_message('human'):
#                 color = {1: ':red[', 2: ':red[', 3: ':red[', 4: ':orange[', 5:':green['}
#                 st.write(cur_review.iloc[0].date.strftime('%H:%M %d.%m.%Y'), '|', f"{data[data['id'] == cur_review.iloc[0].id].iloc[0]['address']}", '|', ':star:'*cur_review.iloc[0].rate)
#                 st.write(color[cur_review.iloc[0].rate] + cur_review.iloc[0].text.replace('[','').replace(']','') + ']')

