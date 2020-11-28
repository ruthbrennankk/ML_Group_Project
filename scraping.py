from urllib.request import urlopen
from bs4 import BeautifulSoup as soup
import ssl
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

count = 0
# Strips the newline character
for line in Lines:
    print("Line{}: {}".format(count, line.strip()))

#download_and_save_page('https://www.carzone.ie/used-cars/audi/a4/fpa/202003078142361?journey=Search', 'car_test.html')
