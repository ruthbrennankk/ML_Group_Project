from urllib.request import urlopen
from bs4 import BeautifulSoup as BeautifulSoup
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def download_and_save_page(url, file):
    html_code = urlopen(url).read() #.decode('utf-8')
    f = open(file, 'wb')
    f.write(html_code)
    f.close()
    return html_code


# # Using readlines()
# file1 = open('links.txt', 'r')
# Lines = file1.readlines()
#
# count = 0
# # Strips the newline character
# for line in Lines:
#     print("Line{}: {}".format(count, line.strip()))

# Notes for gettings data
#
#   links = soup.find_all('a', href=True, class_='car-link')
#    for link in links:
#     print(link['href'])
#

with open('car_test.html', 'r') as f:
    contents = f.read()

soup = BeautifulSoup(contents, 'html.parser')
# #print(soup.find('span', class_='fpa-features__item__text').get_text())
# id_list = soup.find_all('li', class_='fpa-features__item')
# list = soup.find_all('span', class_='fpa-features__item__text')
# print(id_list)
# print(list)
#
# for i in range(len(list)):
#     print((id_list[i]).get('id'))
#     print((list[i]).get_text())

print(soup.find('script', type='application/ld+json'))

#download_and_save_page('https://www.carzone.ie/used-cars/audi/a4/fpa/202003078142361?journey=Search', 'car_test.html')
