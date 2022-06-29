import api_key

flickr_api_key = api_key.flickr_api_key()
flickr_api_secret = api_key.flickr_api_secret()



import flickrapi
import urllib

flickr=flickrapi.FlickrAPI(flickr_api_key, flickr_api_secret, cache=True)

keyword = 'beach'

photos = flickr.walk(text=keyword,
                     tag_mode='all',
                     tags=keyword,
                     extras='url_c',
                     per_page=100,           # may be you can try different numbers..
                     sort='relevance')


urls = []
for i, photo in enumerate(photos):
    print (i)

    url = photo.get('url_c')
    urls.append(url)

    # get 50 urls
    if i > 50:
        break

print (urls)
