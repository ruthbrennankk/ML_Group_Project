from urllib.request import urlopen
from bs4 import BeautifulSoup as BeautifulSoup
import json
import ssl
import pandas as pd
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context

def download_and_save_page(url, file):
    html_code = urlopen(url).read() #.decode('utf-8')
    f = open(file, 'wb')
    f.write(html_code)
    f.close()
    return html_code


# Using readlines()
file1 = open('links.txt', 'r')
Lines = file1.readlines()

urls = []
brands = []
models = []
transmission = []
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
    # print(res)
    json_object = json.loads(res.contents[0])
    # print(json_object)
    # print(json_object["brand"])
    # print(json_object["model"])
    # print(json_object["offers"]["price"])

    brands.append(json_object["brand"])
    models.append(json_object["model"])
    prices.append(json_object["offers"]["price"])


rows = [brands, models, prices]
rows = np.array(rows)

d = {'Brands': brands, 'Models': models, 'Prices': prices}
df = pd.DataFrame(data=d)
df.to_csv('car.csv')

#
#     id_list = soup.find_all('li', class_='fpa-features__item')
#     list = soup.find_all('span', class_='fpa-features__item__text')
# # print(id_list)
# # print(list)
#
#     for i in range(len(list)):
#         print((id_list[i]).get('id'))
#         print((list[i]).get_text())


#download_and_save_page('https://www.carzone.ie/used-cars/audi/a4/fpa/202003078142361?journey=Search', 'car_test.html')
