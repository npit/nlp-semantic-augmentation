from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.datasets import fetch_20newsgroups
from nltk.corpus import reuters

import getpass
import pandas
import smtplib
# build vocabulary and train model


def mail(email, passw, msg, title="wordvec-training"):
    # email me
    TO = mail
    SUBJECT = title
    TEXT = msg
    # Gmail Sign In
    gmail_sender = mail
    recipient= mail
    gmail_passwd = passw

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.ehlo()
    server.starttls()
    server.login(gmail_sender, gmail_passwd)

    BODY = '\r\n'.join(['To: %s' % TO, 'From: %s' % gmail_sender, 'Subject: %s' % SUBJECT, '', TEXT])
    try:
        server.sendmail(gmail_sender, [TO], BODY)
        print('Email sent to [%s]' % recipient)
    except Exception as x:
        print('Error sending mail to [%s]' % recipient)
        print(x)





def get_reuters():
    print("Getting reuters")
    categories = reuters.categories()
    # get content
    train=[]
    for cat_index, cat in enumerate(categories):
        # get all docs in that category
        for doc in reuters.fileids(cat):
            # get its content
            content = reuters.raw(doc)
            # assign content
            if doc.startswith("training"):
                train.append(content)
    return train

def get_20ng():
    print("Getting 20ng")
    train = fetch_20newsgroups(subset='train', shuffle=True, random_state=114)
    data = train.data




num_epochs = 50
window=10
size=300
word_threshold=2

email="pittarasnikif@gmail.com"
print("mail password:")
passw=getpass.getpass()


data = get_reuters()
data=[simple_preprocess(d) for d in data]


print("Epochs {}, window {}, size {}, word-freq-thresh {}".format(num_epochs, window, size, word_threshold))
print("Defining w2v model")
model = Word2Vec(
    data,
    size=size,
    window=window,
    min_count=word_threshold,
    workers=2)


print("Training model")
model.train(data, total_examples=len(data), epochs=num_epochs)

model_name = "word2vec_ep{}_window{}_dim{}_wthresh{}.pickle".format(num_epochs, window, size, word_threshold)
vocabulary = list(model.wv.vocab)
df = pandas.DataFrame(model.wv[vocabulary], index=vocabulary)
print("Vocaulary size:", len(vocabulary))
print("Saving to ", model_name)
df.to_pickle(model_name)


mail(email,passw,"word2vec training done")
