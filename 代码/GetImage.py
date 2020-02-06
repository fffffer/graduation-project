import datetime
import string
import time

import requests
from bs4 import BeautifulSoup



def get_html(n):
    url = 'https://search.jd.com/Search?keyword=%E7%9F%AD%E9%9D%B4&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&page=' + str(n*2-1)
    headers = {
        'authority': 'search.jd.com',
        'method': 'GET',
        'path': '/im.php?r=1770779415&t=1565315379.7375&cs=6121bcfdb623cb194bcc80ae9037a1c8',
        'scheme': 'https',
        'referer': url,
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }
    html = requests.get(url,headers=headers)
    html.encoding = 'utf-8'
    return html.text

def get_last_html(n):
    # 获取当前的Unix时间戳，并且保留小数点后5位
    a = time.time()
    b = '%.5f' % a
    url = 'https://search.jd.com/Search?keyword=%E7%9F%AD%E9%9D%B4&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&page=' + str(n*2-1)
    headers = {
        'authority': 'search.jd.com',
        'method': 'GET',
        'path': '/s_new.php?keyword=%E6%89%8B%E6%9C%BA&enc=utf-8&qrst=1&rt=1&stop=1&vt=2&wq=%E6%89%8B%E6%9C%BA&cid2=653&cid3=655&page=4&s=86&scrolling=y&log_id=1565327127.45950&tpl=3_M&show_items=6176077,8058010,7652089,100004323348,100003258297,100005087998,7438288,50633925625,7652013,100000827661,100000822969,100002293114,100005945610,100000766433,100004050001,8485229,100006841262,100000084109,100001467225,100002493099,100005819880,8636676,100000773875,100003429677,33155679178,7437564,100003332220,100001548579,100005150846,7293054',
        'scheme': 'https',
        'referer': url,
        'user-agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.131 Safari/537.36',
        'x-requested-with': 'XMLHttpRequest'
    }
    html = requests.get(url,headers=headers)
    html.encoding = 'utf-8'
    return html.text



def parse_html(html):
    soup = BeautifulSoup(html,'html.parser')
    lis = soup.find_all('li',attrs={'class':'gl-item'})
    p_price = []
    ps_items = []
    p_name = []
    p_detail = []
    p_sales = []
    for li in range(len(lis)):
        p_price.append(lis[li].select('div div.p-price strong i')[0].text)
        ps_items.append(lis[li].select('div div.p-img a img')[0].attrs['source-data-lazy-img'])
        p_detail.append(lis[li].select('div div.p-name em')[0].text)
        p_name.append(p_detail[li][:20])
        p_sales.append(lis[li].select('div div.p-commit strong a')[0].text)
    
    for i in range(len(ps_items)):    
        get_img(ps_items[i])
    # print(p_sales)
    # insert_date(p_price,ps_items,p_name,p_sales,p_detail)

def get_img(imgs_url):
    
    imgs_url = "https:" + imgs_url
    print(imgs_url)
    html = requests.get(imgs_url)

    if html.status_code == 200:
        with open('C:/Users/djasl/Desktop/images/Ankle boot/'+ imgs_url[-20:-3] + '.jpg', 'wb') as f:
            f.write(html.content)

    f.close()


def main():
     for i in range(9,10):
        html = get_html(i)
        parse_html(html)
        #
        html = get_last_html(i)
        parse_html(html)






if __name__ == '__main__':
    main()