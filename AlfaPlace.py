from telebot import TeleBot
from telebot import types
import telebot
import sqlite3
import pandas as pd
import numpy as np
from sklearn.neighbors import KDTree
import folium
from sklearn.model_selection import KFold
from  sklearn.metrics import mean_absolute_error
from sklearn import linear_model

bot = TeleBot("5453557136:AAFLEp-FOJ13OTOsTFHbOKjfRZfJi-4M6Ys")

def algorithm(_city, _num):
    def drop_dups(df):
        return df.drop_duplicates(subset=[geo_id])
    column_mapper = {

        'Автозапчасти для иномарок': 'autoparts',
        'Авторемонт и техобслуживание (СТО)': 'autoremont',
        'Алкогольные напитки': 'alcohols',
        'Аптеки': 'pharmacies',
        'Банки': 'banks',
        'Быстрое питание': 'fastfood',
        'Доставка готовых блюд': 'delivery',
        'Женская одежда': 'female_clothes',
        'Кафе': 'cafe',
        'Косметика / Парфюмерия': 'cosmetics',
        'Ногтевые студии': 'nails',
        'Овощи / Фрукты': 'vegetables',
        'Парикмахерские': 'hairs',
        'Платёжные терминалы': 'pay_terminals',
        'Постаматы': 'mails',
        'Продуктовые магазины': 'products',
        'Пункты выдачи интернет-заказов': 'internet_orders',
        'Рестораны': 'restaurants',
        'Страхование': 'insurance',
        'Супермаркеты': 'supermarkets',
        'Цветы': 'flowers',
        'Шиномонтаж': 'tires'
    }
    cat_features = [] 
    num_features = ['population'] + list(column_mapper.values())
    target = 'target'
    features = cat_features + num_features
    geo_id = 'geo_h3_10'
    df_population = drop_dups(pd.read_csv("rosstat_population_all_cities.csv"))
    df_isochrones = drop_dups(pd.read_csv("isochrones_walk_dataset.csv"))
    df_companies = drop_dups(pd.read_csv("osm_amenity.csv"))
    df_target = drop_dups(pd.read_csv('target_hakaton_spb.csv', sep=';', encoding='Windows-1251'))
    geo_id_mapper = dict(df_isochrones.apply(lambda row: (row[geo_id], (row['lat'], row['lon'])), axis=1).tolist())
    def prepare_df(df):
        _df = df.merge(df_population, on=geo_id, suffixes=(None, '_y'), how='left').drop(['lat', 'lon', 'city'], axis=1)
        _df = _df.merge(df_companies, on=geo_id, suffixes=(None, '_y'), how='left').drop(['city', 'lat', 'lon'], axis=1).fillna(0)

        _df = _df.rename(columns=column_mapper)
        for feature in num_features:
            _df[feature] = (_df[feature] - _df[feature].mean())/ _df[feature].std()

        for feature in cat_features:
            _df[f'{feature}_ohe'] = _df[feature]
        _df = pd.get_dummies(_df, columns=[f'{col}_ohe' for col in cat_features], prefix=cat_features)

        return _df

    def prepare_df2(df):
        _df = df.merge(df_companies, on=geo_id, suffixes=(None, '_y'), how='left').drop(['city', 'lat', 'lon'], axis=1).fillna(0)

        _df = _df.rename(columns=column_mapper)
        for feature in num_features:
            _df[feature] = (_df[feature] - _df[feature].mean())/ _df[feature].std()

        for feature in cat_features:
            _df[f'{feature}_ohe'] = _df[feature]
        _df = pd.get_dummies(_df, columns=[f'{col}_ohe' for col in cat_features], prefix=cat_features)

        ohe_cols = [col for col in _df.columns if col.startswith(tuple([item + '_' for item in cat_features]))]
        X = _df[num_features + ohe_cols].values
        
        return X

    def prepare_target(_df):
        _df[target] = _df[target] / _df['atm_cnt']
        _df[target] = (_df[target] - _df[target].mean())/ _df[target].std()

        _df = _df[_df[target] - _df[target].mean() < 3 * _df[target].std()]
        return _df

    def get_regression(df_target):

        df_target = prepare_df(df_target)
        df_target = prepare_target(df_target)

        ohe_cols = [col for col in df_target.columns if col.startswith(tuple([item + '_' for item in cat_features]))]
        X = df_target[num_features + ohe_cols].values
        y = df_target[target].values
        y = (y - y.min()) / (y.max() - y.min())
        
        k_fold = KFold(5)
        scores = []
        lr = linear_model.LassoCV(max_iter=1000)
        for k, (train, test) in enumerate(k_fold.split(X, y)):
            lr.fit(X[train], y[train])
            scores.append(mean_absolute_error(y[test], lr.predict(X[test])))
        print(f'Mean linear regression MAE: {float(np.mean(scores))}')
        
        return lr
    lr = get_regression(df_target)




    
    def get_mean_radius(df_isochrones):
        def get_radius(row):
            lat = float(row['lat'])
            lon = float(row['lon'])
            poly_str = row['walk_15min'].replace('POLYGON ((', '').replace('))', '')
            points = poly_str.split(',')
            dist = 0
            for point_str in points:
                items = point_str.strip().split(' ')
                _lon = float(items[0])
                _lat = float(items[1])
                dist += np.sqrt((lat - _lat) ** 2 + (lon - _lon) ** 2)
            return float(dist) / len(points)

        mean_radius = (df_isochrones.apply(get_radius, axis=1)).mean() # 5 mins
        
        return mean_radius

    mean_radius = get_mean_radius(df_isochrones)
    print(f'Mean isochrone radius: {mean_radius}')

    cities = df_isochrones['city'].drop_duplicates().tolist()
    print('Available cities: ', cities)

    filled_hexs = set(df_target[geo_id].drop_duplicates().tolist())
    filled_trees = {}
    uncovered = {}
    covered = {}
    total_peoples = {}
    uncovered_df = {}
    uncovered_trees = {} 


    for city in cities:
        city_isochrones = df_isochrones[df_isochrones['city'].apply(lambda val: val == city)]
        filled_trees[city] = KDTree(city_isochrones[city_isochrones[geo_id].apply(lambda id: id in filled_hexs)][['lon', 'lat']].values)
        dist, ind = filled_trees[city].query(city_isochrones[['lon', 'lat']].values, k=1)
        uncovered[city] = set(city_isochrones[dist > mean_radius][geo_id].drop_duplicates().tolist())
        covered[city] = set(city_isochrones[dist < mean_radius][geo_id].drop_duplicates().tolist())
        peoples = df_population[df_population['city'].apply(lambda val: val == city)]['population'].sum()
        total_peoples[city] = df_population[df_population['city'].apply(lambda val: val == city) & df_population[geo_id].apply(lambda id: id in covered[city])]['population'].sum()
        print(f'City - {city}, peoples  - {peoples}, peoples covered - {total_peoples[city]}')
        
        df = df_population[df_population['city'].apply(lambda val: val == city) & df_population[geo_id].apply(lambda id: id in uncovered[city])]
        df['score'] = lr.predict(prepare_df2(df))
        uncovered_df[city] = df
        uncovered_trees[city] = KDTree(uncovered_df[city][['lon', 'lat']].values)
    if _city not in uncovered_df:
        return "Wrong name", 0   

    def get_predictions(city, uncovered_df, total_peoples, uncovered_trees, mean_radius, geo_id='geo_h3_10', n=20):
        uncovered_pops = uncovered_df[city]['population'].tolist()
        uncovered_hexs = uncovered_df[city][geo_id].tolist()
        uncovered_scores = uncovered_df[city]['score'].tolist()

        indices = uncovered_trees[city].query_radius(uncovered_df[city][['lon', 'lat']].values, r=mean_radius)

        tmp_covered = set()
        selected = []
        total_lift = 0
        lifts = []
        for current_terminal in range(n):
            best_lift = 0
            best_item = 0
            for item_id, items in enumerate(indices):
                added = sum([uncovered_pops[item] for item in set(items) if item not in tmp_covered])
                lift = added / total_peoples[city] * 100
                score = uncovered_scores[item_id]
                if lift * score > best_lift:
                    best_lift = lift * score
                    best_item = item_id

            total_lift += best_lift
            lifts.append(total_lift)
            selected.append(uncovered_hexs[best_item])
            tmp_covered.update(set(indices[best_item]))

        return lifts, selected

    total_lift, selected = get_predictions(_city, uncovered_df, total_peoples, uncovered_trees, mean_radius, geo_id = geo_id, n = _num)
    print(f'Total lift: {total_lift[-1]}')
    print('Hexagon indices: ', selected)
    hexagon = selected
    coords = [geo_id_mapper[id] for id in selected]
    print(coords)
    answer = list([hexagon, coords, f'Total lift: {total_lift[-1]}'])
    return answer


@bot.message_handler(commands=["start"])
def start(message):
    bot.send_message(message.chat.id,  "Welcome!\nWe hope we can help you. Enter the city you are interested in and we will find everything for you")
    keyboard = types.ReplyKeyboardMarkup(resize_keyboard=True)
    button_1 = types.KeyboardButton(text="/search")
    button_2 = types.KeyboardButton(text="/help")
    keyboard.add(button_1, button_2)
    bot.send_message(message.chat.id, "What would you do?", reply_markup=keyboard)

@bot.message_handler(commands=["help"])
def search(message):
    bot.send_message(message.chat.id, "You can do it!)\nAt the moment, the bot can find you the best place only within the Russian Federation.\n\nYou can choose many major cities to find the best location for your ATM.\nBy the way, you can send your complaint and suggestion")


@bot.message_handler(commands=["search"])
def search(message):
    bot.send_message(message.chat.id, "Please, enter your city and the desired number.\nFor example: Санкт-Петербург, 5")
@bot.message_handler(content_types = ["text"])
def analisys(message):
    bot.send_message(message.chat.id, "Great, I'll find it now")
    lst = message.text.split(",")
    if len(lst) == 1: lst.append(1)
    hexagon, coords, total = algorithm(lst[0], int(lst[1]))
    hexagon = "\n".join(hexagon)
    bot.send_message(message.chat.id, f"ID of hexagon: {hexagon}")
    print(coords)
    bot.send_message(message.chat.id, str(coords))
    for i in range(len(coords)):
        map = folium.Map(location = coords[i], title = "Your best place", zoom_start = 14)
        folium.Marker(location = coords[i]).add_to(map)
        map.save("map.html")
        from html2image import Html2Image
        hti = Html2Image()
        with open('map.html') as f:
            hti.screenshot(f.read(), save_as='out.png')
        photo = open('out.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        bot.send_message(message.chat.id, str(coords[i]))

bot.polling()
