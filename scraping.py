from urllib.request import urlopen
from bs4 import BeautifulSoup as BeautifulSoup
import json
import ssl
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

ssl._create_default_https_context = ssl._create_unverified_context

def download_and_save_page(url, file):
    html_code = urlopen(url).read() #.decode('utf-8')
    f = open(file, 'wb')
    f.write(html_code)
    f.close()
    return html_code

def update_dict(the_dict, the_key):
    the_dict[the_key] = True

def labelEncode(categories, car_df) :
    # creating instance of labelencoder
    labelencoder = LabelEncoder()

    for cat in categories :
        # Assigning numerical values and storing in another column
        car_df[cat] = labelencoder.fit_transform(car_df[cat])

    # # creating instance of one-hot-encoder
    # enc = OneHotEncoder(handle_unknown='ignore')
    # # passing bridge-types-cat column (label encoded values of bridge_types)
    # enc_df = pd.DataFrame(enc.fit_transform(car_df[[cat]]).toarray())
    # # merge with main df bridge_df on key values
    # car_df = car_df.join(enc_df)

    return car_df


# Using readlines()
file1 = open('extra.txt', 'r')
Lines = file1.readlines()

urls = []
brands = []
models = []
transmission = [] # Automatic = 1, Manual = 0
colour = []
mileage = []
year = []
seats = []
doors = []
prices = []

count = 0
# Loops through the urls
for line in Lines:
    # print url
    print(str(line))
    urls.append(line)
    # get data
    download_and_save_page(str(line), 'car.html')

    with open('car.html', 'r') as f:
        contents = f.read()

    #Set up soup
    soup = BeautifulSoup(contents, 'html.parser')
    # Finds
    res = soup.find('script', type='application/ld+json')
    id_list = soup.find_all('li', class_='fpa-features__item')
    list = soup.find_all('span', class_='fpa-features__item__text')

    value_added = {'transmission': False, 'colour': False, 'mileage': False, 'year': False}#, 'seats': False, 'doors': False}

    #Printing ids to compare
    for i in range(len(list)):
        id_name = (id_list[i]).get('id')
        if id_name == 'transmission':   # Note There maybe an electric transmission
            if ((list[i]).get_text() == 'Manual') :
               transmission.append(0)
            elif ((list[i]).get_text() == 'Automatic') :
                transmission.append(1)
            update_dict(value_added, 'transmission')
        elif id_name == 'colour':
            colour.append((list[i]).get_text())
            update_dict(value_added, 'colour')
        elif id_name == 'mileage':
            formatted = ""
            str_kms = (list[i]).get_text()
            for i in range(len(str_kms)) :
                if str_kms[i].isdigit() :
                    formatted = formatted + str_kms[i]
            mileage.append(formatted)
            update_dict(value_added, 'mileage')
        elif id_name == 'year': # Note With new data set make sure year is before 161 etc
            newyear = int(list[i].get_text()[0:4])
            age = 2020 - newyear
            # print(age)
            year.append(age)
            update_dict(value_added, 'year')
        # elif id_name == 'seats': # Note Also here check that number of seats & doors are first
        #     # print(list[i].get_text()[0])
        #     seats.append(list[i].get_text()[0])
        #     update_dict(value_added, 'seats')
        # elif id_name == 'doors':
        #     # print(list[i].get_text()[0])
        #     doors.append((list[i].get_text()[0]))
        #     update_dict(value_added, 'doors')
        #print((list[i]).get_text())

    if value_added['transmission'] == False:
        transmission.append('$')
    if value_added['colour'] == False:
        colour.append('$')
    if value_added['mileage'] == False:
        mileage.append('$')
    if value_added['year'] == False:
        year.append('$')
    # if value_added['seats'] == False:
    #     seats.append('$')
    # if value_added['doors'] == False:
    #     doors.append('$')

    # print(res)
    json_object = json.loads(res.contents[0])
    # print(json_object)
    # print(json_object["brand"])
    # print(json_object["model"])
    # print(json_object["offers"]["price"])

    brands.append(json_object["brand"])
    models.append(json_object["model"])
    prices.append(json_object["offers"]["price"])


#rows = [brands, models, prices, transmission, colour, mileage, year, seats, doors]
rows = [brands, models, prices, transmission, colour, mileage, year]#, seats, doors]
rows = np.array(rows)

#d = {'Brands': brands, 'Models': models, 'Transmission': transmission, 'Colour': colour, 'Mileage': mileage, 'Year': year, 'Seats': seats, 'Doors': doors, 'Prices': prices }
d = {'Brands': brands, 'Models': models, 'Transmission': transmission, 'Colour': colour, 'Mileage': mileage, 'Year': year, 'Prices': prices }
car_df = pd.DataFrame(data=d)

# Label Encoding
car_df = labelEncode(['Colour', 'Models','Brands'], car_df)

car_df.to_csv('car2.csv')

#
#     id_list = soup.find_all('li', class_='fpa-features__item')
#     list = soup.find_all('span', class_='fpa-features__item__text')
# # print(id_list)
# # print(list)
#
#     for i in range(len(list)):
#         print((id_list[i]).get('id'))
#         print((list[i]).get_text())

#order of list
# engine
# bodytype
# transmission
# colour
# mileage
# year
# owners
# doors
# seats
# nct
# tax-band
