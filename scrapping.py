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



# https://www.carzone.ie/used-cars/ireland/leitrim
#

download_and_save_page('https://www.carzone.ie/used-cars/ireland/leitrim', 'test.txt')