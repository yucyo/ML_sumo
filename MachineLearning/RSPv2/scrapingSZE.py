import requests
from bs4 import BeautifulSoup

r = requests.get("http://park11.wakwak.com/~hkn/result1996.htm")

soup = BeautifulSoup(r.content,"html.parser")

#リスト数チェック
print(len(soup.select("td")))

for i in range(len(soup.select("td"))):
    if i % 6 == 0:
        print("1996")
        print(soup.select("td")[i].text)
    elif i % 6 == 1:
        print(soup.select("td")[i].text)
    elif i % 6 == 4:
        if soup.select("td")[i].text == 'グー':
            print("1 0 0")
        elif soup.select("td")[i].text == 'チョキ':
            print("0 1 0")
        elif soup.select("td")[i].text == 'パー':
            print("0 0 1")
    else:
        pass
