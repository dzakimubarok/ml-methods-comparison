from __future__ import absolute_import, print_function

# Import modules
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import dataset
from sqlalchemy.exc import ProgrammingError

# Your credentials go here

consumer_key = "Hgjk5odnm7GNLkNBB6MrrEAs6"
consumer_secret = "JNz9Qq89Uapz02crdTwQaBJiCTTYVwBzaW7P2Eaz0Tf5GA4hjX"
access_token = "575355711-rfELPXGM6J7Zzx7XjrE070wgzgf8Vk4sKejIEI2H"
access_token_secret = "uzQK9dqPFQZK0q1xpKfUH9ZZvusA7YVXN5JnNuTFMjuUm"



class StdOutListener(StreamListener):
    def on_status(self, status):
        print(status.text)
        if status.retweeted:
            return

        id_str = status.id_str
        created = status.created_at
        text = status.text
        fav = status.favorite_count
        name = status.user.screen_name
        description = status.user.description
        loc = status.user.location
        user_created = status.user.created_at
        followers = status.user.followers_count

        table = db['myTable']

        try:
            table.insert(dict(
                id_str=id_str,
                created=created,
                text=text,
                fav_count=fav,
                user_name=name,
                user_description=description,
                user_location=loc,
                user_created=user_created,
                user_followers=followers,
            ))
        except ProgrammingError as err:
            print(err)

    def on_error(self, status_code):
        if status_code == 420:
            return False


if __name__ == '__main__':
    db = dataset.connect("sqlite:///tweets.db")
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    stream.filter(track=['Donald Trump'])