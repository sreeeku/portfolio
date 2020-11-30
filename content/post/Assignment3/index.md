---
title: Naive Bayes Classifier
subtitle: Sentiment Analysis using Naive Bayes Classifier from scratch
url_code: "https://www.dropbox.com/s/5z4dvlu2d3psdyq/Uppuluri_03.ipynb?dl=0"
summary:  
authors:
- admin
tags: []
categories: []
date: "2020-29-11T00:00:00Z"
lastMod: "2020-29-11T00:00:00Z"
#featured: true
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image: 
  caption: ""
  focal_point: ""

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references 
#   `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---



<h1 style ='text-align:center;'>Naive Bayes Classifier<h1>

### Extracting the DataSet


```python
import tarfile
tar = tarfile.open("aclImdb_v1.tar.gz")
tar.extractall()
tar.close()
```

### Libraries Used


```python
import glob
import numpy as np
import pickle
import pandas as pd
import nltk
nltk.download('punkt')
from os.path import basename, splitext
from sklearn.model_selection import train_test_split
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import KFold
import itertools
import re
import time
from bs4 import BeautifulSoup
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
from collections import Counter
from nltk.stem import WordNetLemmatizer
import decimal
```

    [nltk_data] Downloading package punkt to
    [nltk_data]     C:\Users\sreekupc\AppData\Roaming\nltk_data...
    [nltk_data]   Package punkt is already up-to-date!


### As the Folders of Train are divided in to Pos and Neg, keeping them in Different dataframes for the time being


```python
pos_rev_path = 'aclImdb/train/pos'
neg_rev_path = 'aclImdb/train/neg'
pos_files = glob.glob(pos_rev_path + "/*.txt")
neg_files = glob.glob(neg_rev_path + "/*.txt")

df_movie_pos = pd.DataFrame()
df_movie_neg = pd.DataFrame()

reviews_pos_list = []
filenames_pos_list = []

reviews_neg_list = []
filenames_neg_list = []

for file in pos_files:
    with open(file, encoding="utf8") as fp:
        review = " ".join(line.strip() for line in fp)
        reviews_pos_list.append(review)
        filenames_pos_list.append(splitext(basename(file))[0])

for file in neg_files:
    with open(file, encoding="utf8") as fp:
        review = " ".join(line.strip() for line in fp)
        reviews_neg_list.append(review)
        filenames_neg_list.append(splitext(basename(file))[0])


df_movie_pos["Filenames"] = filenames_pos_list
df_movie_pos["Reviews"] = reviews_pos_list
df_movie_pos["Label"] = "pos"

df_movie_neg["Filenames"] = filenames_neg_list
df_movie_neg["Reviews"] = reviews_neg_list
df_movie_neg["Label"] = "neg"    
```

### As the Folders of test are divided in to Pos and Neg, keeping them in Different dataframes for the time being



```python
test_pos_rev_path = 'aclImdb/test/pos'
test_neg_rev_path = 'aclImdb/test/neg'
test_pos_files = glob.glob(test_pos_rev_path + "/*.txt")
test_neg_files = glob.glob(test_neg_rev_path + "/*.txt")

test_df_movie_pos = pd.DataFrame()
test_df_movie_neg = pd.DataFrame()

test_reviews_pos_list = []
test_filenames_pos_list = []

test_reviews_neg_list = []
test_filenames_neg_list = []

for file in test_pos_files:
    with open(file, encoding="utf8") as fp:
        review = " ".join(line.strip() for line in fp)
        test_reviews_pos_list.append(review)
        test_filenames_pos_list.append(splitext(basename(file))[0])

for file in test_neg_files:
    with open(file, encoding="utf8") as fp:
        review = " ".join(line.strip() for line in fp)
        test_reviews_neg_list.append(review)
        test_filenames_neg_list.append(splitext(basename(file))[0])


test_df_movie_pos["Filenames"] = test_filenames_pos_list
test_df_movie_pos["Reviews"] = test_reviews_pos_list
test_df_movie_pos["Label"] = "pos"

test_df_movie_neg["Filenames"] = test_filenames_neg_list
test_df_movie_neg["Reviews"] = test_reviews_neg_list
test_df_movie_neg["Label"] = "neg"    
```

### Positive Dataframes of Test set


```python
test_df_movie_pos.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0_10</td>
      <td>I went and saw this movie last night after bei...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000_7</td>
      <td>Actor turned director Bill Paxton follows up h...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10001_9</td>
      <td>As a recreational golfer with some knowledge o...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10002_8</td>
      <td>I saw this film in a sneak preview, and it is ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10003_8</td>
      <td>Bill Paxton has taken the true story of the 19...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10004_9</td>
      <td>I saw this film on September 1st, 2005 in Indi...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10005_8</td>
      <td>Maybe I'm reading into this too much, but I wo...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10006_7</td>
      <td>I felt this film did have many good qualities....</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
</div>



### Negative Dataframes of Test set


```python
test_df_movie_neg.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0_2</td>
      <td>Once again Mr. Costner has dragged out a movie...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000_4</td>
      <td>This is an example of why the majority of acti...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10001_1</td>
      <td>First of all I hate those moronic rappers, who...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10002_3</td>
      <td>Not even the Beatles could write songs everyon...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10003_3</td>
      <td>Brass pictures (movies is not a fitting word f...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10004_2</td>
      <td>A funny thing happened to me while watching "M...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10005_2</td>
      <td>This German horror film has to be one of the w...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10006_2</td>
      <td>Being a long-time fan of Japanese film, I expe...</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
</div>



### Positive Dataframe of Train Set


```python
df_movie_pos.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0_9</td>
      <td>Bromwell High is a cartoon comedy. It ran at t...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000_8</td>
      <td>Homelessness (or Houselessness as George Carli...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10001_10</td>
      <td>Brilliant over-acting by Lesley Ann Warren. Be...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10002_7</td>
      <td>This is easily the most underrated film inn th...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10003_8</td>
      <td>This is not the typical Mel Brooks film. It wa...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10004_8</td>
      <td>This isn't the comedic Robin Williams, nor is ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10005_7</td>
      <td>Yes its an art... to successfully make a slow ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10006_7</td>
      <td>In this "critically acclaimed psychological th...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
</div>



### Negative DataFrame of Train Set


```python
df_movie_neg.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0_3</td>
      <td>Story of a man who has unnatural feelings for ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10000_4</td>
      <td>Airport '77 starts as a brand new luxury 747 p...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10001_4</td>
      <td>This film lacked something I couldn't put my f...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10002_1</td>
      <td>Sorry everyone,,, I know this is supposed to b...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10003_1</td>
      <td>When I was little my parents took me along to ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>5</th>
      <td>10004_3</td>
      <td>"It appears that many critics find the idea of...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>10005_3</td>
      <td>The second attempt by a New York intellectual ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10006_4</td>
      <td>I don't know who to blame, the timid writers o...</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
</div>



### Merging both the Negative and Positive Dataframes of Test and Sampling them by fraction 1


```python
test_frames = [test_df_movie_pos, test_df_movie_neg]
test_df_movie_reviews = pd.concat(test_frames)
test_df_movie_reviews = test_df_movie_reviews.sample(frac = 1)
test_df_movie_reviews.reset_index(drop = True,inplace = True)
test_df_movie_reviews.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1333_7</td>
      <td>I felt cheated out of knowing the whole story....</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1226_1</td>
      <td>Was really looking forward to seeing a continu...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4242_1</td>
      <td>Playmania is extremely boring. This is the bas...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2804_3</td>
      <td>A young man, who never knew his birth parents,...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7051_4</td>
      <td>This is a case of taking a fairy tale too far....</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5878_3</td>
      <td>Technically speaking, this movie sucks...lol. ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>12431_4</td>
      <td>Pick a stereotype, any stereotype (whether rac...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>2498_7</td>
      <td>The Love Letter is one of my all-time favorite...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
</div>



### Merging both the Negative and Positive Dataframes of Train and Sampling them by fraction 1


```python
frames = [df_movie_pos, df_movie_neg]
df_movie_reviews = pd.concat(frames)
df_movie_reviews = df_movie_reviews.sample(frac = 1)
df_movie_reviews.reset_index(drop = True,inplace = True)
df_movie_reviews.head(8)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985_10</td>
      <td>Im not usually a lover of musicals,but if i ha...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8834_9</td>
      <td>A very well made film set in early '60s commun...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5948_8</td>
      <td>Some of the posters seem less than gruntled be...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6489_10</td>
      <td>i watched this series when it first came out i...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8596_3</td>
      <td>Basically this is a pale shadow of High Fideli...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>5</th>
      <td>112_1</td>
      <td>This film is just plain horrible. John Ritter ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1498_2</td>
      <td>Justifications for what happened to his movie ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10569_1</td>
      <td>I tried to watch this movie in a military camp...</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
</div>



### Train Data Set Shape


```python
df_movie_reviews.shape
```




    (25000, 3)



### Here is the Train Data set with reviews


```python
df_movie_reviews
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985_10</td>
      <td>Im not usually a lover of musicals,but if i ha...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8834_9</td>
      <td>A very well made film set in early '60s commun...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5948_8</td>
      <td>Some of the posters seem less than gruntled be...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6489_10</td>
      <td>i watched this series when it first came out i...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8596_3</td>
      <td>Basically this is a pale shadow of High Fideli...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>3613_8</td>
      <td>When "Girlfight" came out, the reviews praised...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>964_8</td>
      <td>"Yokai Daisenso" is a children's film by Takas...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>6461_4</td>
      <td>I think it's a great movie!! It's fun, maybe a...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>8801_9</td>
      <td>This is the greatest film I saw in 2002, where...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>7504_7</td>
      <td>Hey guys I'm actually in this movie! I didn't ...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 3 columns</p>
</div>



## Text Preprocessing

### Below method removes any html tags from the Review


```python
def html_remove(df_row):
    tag_removed_text = BeautifulSoup(df_row, "html.parser").get_text()
    clean_text = re.sub("\[[^]]*\]", " ", tag_removed_text, flags=re.IGNORECASE)
    return re.sub("(\s+)", " ", clean_text)
```

### Below method removes any digits or special characters from the review


```python
def special_characters_remove(df_row):
    regex = r'[^a-zA-z0-9\s]'
    clean_text = re.sub(regex, '', df_row)
    return clean_text
```

### Below method tokenizes the sentence


```python
def wrd_tknize(df_row):
    words_list = word_tokenize(df_row)
    tokens = [word for word in words_list if word.isalpha()]
    return tokens
```

### Below method removes the stop words from the review


```python
def stop_words_remove(df_row):
    stp_words = set(stopwords.words('english'))
    stop_words_removed = [word for word in df_row if word not in stp_words]
    return stop_words_removed
    
```

### Below method lemmatizes the word in the review. For example Cats ===> cat, is ===> be, better ===> good



```python
def lemmatize_words(df_row):
    lemmatizer = WordNetLemmatizer()
    stem_words_list = [lemmatizer.lemmatize(word) for word in df_row]
    return stem_words_list
```

### Below method applies the above mentioned text preprocessing techniques for all the rows in the dataframe


```python
%%time
def text_preprocessing(df):
    df["Reviews"] = df["Reviews"].apply(html_remove)    
    df['Reviews'] = df['Reviews'].apply(special_characters_remove)
    df['Reviews'] = df['Reviews'].str.lower()    
    df['Reviews'] = df['Reviews'].apply(wrd_tknize)
    df['Reviews'] = df['Reviews'].apply(stop_words_remove)
    df["Reviews"] = df["Reviews"].apply(lemmatize_words)
    return df
```

    Wall time: 0 ns


### Below is the DataFrame after Text preprocessing


```python
%%time
df_movie_reviews
```

    Wall time: 0 ns





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985_10</td>
      <td>[im, usually, lover, choose, would, favourite,...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8834_9</td>
      <td>[well, made, film, set, early, communist, yugo...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5948_8</td>
      <td>[poster, seem, le, neither, mark, twain, rodge...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6489_10</td>
      <td>[watched, series, first, came, year, old, watc...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8596_3</td>
      <td>[basically, pale, shadow, high, fidelity, witt...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>3613_8</td>
      <td>[girlfight, came, review, praised, didnt, get,...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>964_8</td>
      <td>[yokai, childrens, film, takashi, miike, might...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>6461_4</td>
      <td>[think, great, movie, fun, maybe, little, unre...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>8801_9</td>
      <td>[greatest, film, saw, whereas, im, used, mains...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>7504_7</td>
      <td>[hey, guy, im, actually, movie, didnt, even, k...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 3 columns</p>
</div>



total_train_df = text_preprocessing(df_movie_reviews)

### Here we perform 5-Fold Cross Validation on the Dataset


```python
%%time
k_fold_val = KFold(5, True, 1)

k_fold_val_train = []
k_fold_val_dev = []

for X_train, X_dev in k_fold_val.split(df_movie_reviews):
    print(X_train,X_dev)
    print()
    k_fold_val_train.append(X_train)
    k_fold_val_dev.append(X_dev)
```

    [    1     3     5 ... 24997 24998 24999] [    0     2     4 ... 24984 24986 24995]
    
    [    0     1     2 ... 24997 24998 24999] [    5     6     7 ... 24991 24992 24994]
    
    [    0     1     2 ... 24995 24998 24999] [    3     8    17 ... 24989 24996 24997]
    
    [    0     2     3 ... 24996 24997 24998] [    1     9    10 ... 24987 24993 24999]
    
    [    0     1     2 ... 24996 24997 24999] [   14    15    18 ... 24982 24990 24998]
    
    Wall time: 38.9 ms


### We add the respective folds as tuples to k_fold_val_list. So the Length of that list is 5 as we are performing 5-fold validation


```python
%%time
k_fold_val_list = []
col_names = ["Filenames","Reviews","Label"]

for train_index_info, dev_index_info in zip(k_fold_val_train, k_fold_val_dev):
    df_kval_train = pd.DataFrame(columns=col_names)
    df_kval_dev = pd.DataFrame(columns=col_names)
    for i in train_index_info:
        df_kval_train = df_kval_train.append(df_movie_reviews.iloc[i])

    for j in dev_index_info:
        df_kval_dev = df_kval_dev.append(df_movie_reviews.iloc[j])
    
    k_fold_val_list.append((df_kval_train,df_kval_dev)) 

```

    Wall time: 4min 9s


### Pickling our 5-folds to save time


```python
pickle.dump( k_fold_val_list, open( "5_fold_cv.p", "wb" ) )
```

### Loading our Dataset


```python
k_fold = pickle.load( open( "5_fold_cv.p", "rb" ) )
```

### Below is the Dev Set of First Fold


```python
k_fold[0][1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985_10</td>
      <td>[im, usually, lover, choose, would, favourite,...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5948_8</td>
      <td>[poster, seem, le, neither, mark, twain, rodge...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8596_3</td>
      <td>[basically, pale, shadow, high, fidelity, witt...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>11</th>
      <td>2095_9</td>
      <td>[let, eliminate, discussion, use, actor, playi...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>13</th>
      <td>11267_10</td>
      <td>[robert, wuhl, teaching, class, film, student,...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24978</th>
      <td>4585_2</td>
      <td>[extremely, suspicious, idea, presented, movie...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24981</th>
      <td>5872_9</td>
      <td>[vote, rating, jacknife, beautifully, acted, b...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24984</th>
      <td>9500_2</td>
      <td>[movie, really, bad, acting, plain, awful, exc...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24986</th>
      <td>5212_3</td>
      <td>[unknowingly, movie, shelf, mill, creek, colle...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>3613_8</td>
      <td>[girlfight, came, review, praised, didnt, get,...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 3 columns</p>
</div>




```python
len(k_fold[0][1].iloc[0]['Reviews'])
```




    149



# Omitting rare words for example if the occurrence is less than five times


```python
def less_than_five_occ(lst):
    word_counts = Counter(lst)
    less_than_five_list = [word for word in lst if word_counts[word] >= 5]
    return less_than_five_list
```

# Building a vocabulary as list


```python
def build_vocabulary(set_type):
    all_words = set_type['Reviews'].values.tolist()
    all_words_list = list(itertools.chain.from_iterable(all_words))
    less_than_five_list = less_than_five_occ(all_words_list)
    unique_word_list = list(set(less_than_five_list))
    unique_dict = {word: index for index,word in enumerate(unique_word_list)}
    return unique_dict, unique_word_list
```


```python
%%time
train_vocab_list_lists = []
train_vocab_list_dicts = []
for i in range(len(k_fold)):
    words_dict, words_list = build_vocabulary(k_fold[i][0]) 
    train_vocab_list_dicts.append(words_dict)
    train_vocab_list_lists.append(words_list)
```

    Wall time: 4.59 s



```python
%%time
dev_vocab_list_lists = []
dev_vocab_list_dicts = []
for i in range(len(k_fold)):
    words_dict, words_list = build_vocabulary(k_fold[i][1]) 
    dev_vocab_list_dicts.append(words_dict)
    dev_vocab_list_lists.append(words_list)
```

    Wall time: 1.08 s



```python
def check_presence(df_row, words_dict):
    return [word for word in df_row if word in words_dict]
    
```


```python
def modify_dataframe(set_type, words_dict):
    set_type['Reviews'] = set_type['Reviews'].apply(check_presence, args=(words_dict,))
    return set_type
```


```python
%%time
train_dfs = []
for i in range(len(k_fold)):
    train_dfs.append(modify_dataframe(k_fold[i][0],train_vocab_list_dicts[i])) 
```

    Wall time: 1.82 s



```python
train_dfs[0].shape[0]
```




    20000




```python
dict_train_rows = []

for df in train_dfs:
    dummy_list = []
    for i in range(df.shape[0]):
        dummy_list.append(dict.fromkeys(df.iloc[i]['Reviews']))
    dict_train_rows.append(dummy_list)
```


```python
train_word_probabilities_list = []
for i in range(len(train_vocab_list_dicts)):
    word_probability = {}
    for word in train_vocab_list_dicts[i]:
        count = 0
        for row in dict_train_rows[i]:
            if word in row:
                count += 1
        word_probability[word] = round(count / len(dict_train_rows[i]), 5)               
    train_word_probabilities_list.append(word_probability)   
    
```

# Probability of the Occurence
> P[word] = num of documents containing the word / num of all documents


```python
len(train_word_probabilities_list[0])
dict(itertools.islice(train_word_probabilities_list[0].items(), 10))
```




    {'bloodthirsty': 0.00115,
     'filmim': 0.00025,
     'tod': 0.0003,
     'mindboggling': 0.0006,
     'halle': 0.0004,
     'rolethis': 0.00025,
     'seemed': 0.0482,
     'wanting': 0.01065,
     'sickness': 0.00145,
     'exhusband': 0.00055}




```python
dev_dfs = []
for i in range(len(k_fold)):
    dev_dfs.append(modify_dataframe(k_fold[i][1],dev_vocab_list_dicts[i]))
```


```python
train_pos_vocab_lists = []
for i in range(len(train_dfs)):
    pos_words_list = train_dfs[i].loc[train_dfs[i]['Label'] == 'pos', 'Reviews'].values.tolist()
    train_pos_vocab_lists.append(dict.fromkeys(list(itertools.chain.from_iterable(pos_words_list))))
```


```python
 len(train_pos_vocab_lists[0])

```




    23371




```python
train_neg_vocab_lists = []
for i in range(len(train_dfs)):
    neg_words_list = train_dfs[i].loc[train_dfs[i]['Label'] == 'neg', 'Reviews'].values.tolist()
    train_neg_vocab_lists.append(dict.fromkeys(list(itertools.chain.from_iterable(neg_words_list))))
```


```python
len(train_neg_vocab_lists[0])
```




    23010




```python
pos_dfs = []
neg_dfs= []
for i in range(len(train_dfs)):
    pos_dfs.append(train_dfs[i].loc[train_dfs[i]['Label'] == 'pos'])
    neg_dfs.append(train_dfs[i].loc[train_dfs[i]['Label'] == 'neg'])

```


```python
pos_dfs[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8834_9</td>
      <td>[well, made, film, set, early, communist, yugo...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6489_10</td>
      <td>[watched, series, first, came, year, old, watc...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7263_10</td>
      <td>[heart, warming, uplifting, movie, outstanding...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12102_8</td>
      <td>[begin, wager, edgar, allen, poe, bet, man, sp...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4347_8</td>
      <td>[wont, go, much, detail, plot, movie, reviewer...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24985</th>
      <td>10968_7</td>
      <td>[grey, garden, enthralling, crazy, couldnt, re...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24993</th>
      <td>2586_10</td>
      <td>[character, alive, interesting, plot, excellen...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>964_8</td>
      <td>[yokai, childrens, film, takashi, miike, might...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>8801_9</td>
      <td>[greatest, film, saw, whereas, im, used, mains...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>7504_7</td>
      <td>[hey, guy, im, actually, movie, didnt, even, k...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>9973 rows × 3 columns</p>
</div>




```python
neg_dfs[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>112_1</td>
      <td>[film, plain, horrible, john, ritter, pratt, f...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1498_2</td>
      <td>[justification, happened, movie, term, distrib...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10569_1</td>
      <td>[tried, watch, movie, military, camp, overseas...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9583_4</td>
      <td>[proximity, tell, convict, lowe, think, prison...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>12</th>
      <td>641_4</td>
      <td>[knowledge, largo, winch, famous, belgium, com...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24990</th>
      <td>11229_1</td>
      <td>[made, watch, school, terrible, movie, outdate...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24991</th>
      <td>6021_1</td>
      <td>[believe, managed, spend, film, spectacularly,...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24992</th>
      <td>5893_1</td>
      <td>[movie, fantastic, great, movie, scary, hell, ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24994</th>
      <td>1430_1</td>
      <td>[flipping, channel, late, saturday, night, fri...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>6461_4</td>
      <td>[think, great, movie, fun, maybe, little, unre...</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
<p>10027 rows × 3 columns</p>
</div>




```python
for i in range(len(pos_dfs)):
    pos_dfs[i]['Reviews'] = pos_dfs[i]['Reviews'].apply(lambda x: dict.fromkeys(x))
```

    C:\Users\sreekupc\anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy


​    


```python
pos_dfs[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>8834_9</td>
      <td>{'well': None, 'made': None, 'film': None, 'se...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6489_10</td>
      <td>{'watched': None, 'series': None, 'first': Non...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7263_10</td>
      <td>{'heart': None, 'warming': None, 'uplifting': ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>9</th>
      <td>12102_8</td>
      <td>{'begin': None, 'wager': None, 'edgar': None, ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>14</th>
      <td>4347_8</td>
      <td>{'wont': None, 'go': None, 'much': None, 'deta...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24985</th>
      <td>10968_7</td>
      <td>{'grey': None, 'garden': None, 'enthralling': ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24993</th>
      <td>2586_10</td>
      <td>{'character': None, 'alive': None, 'interestin...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>964_8</td>
      <td>{'yokai': None, 'childrens': None, 'film': Non...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>8801_9</td>
      <td>{'greatest': None, 'film': None, 'saw': None, ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>7504_7</td>
      <td>{'hey': None, 'guy': None, 'im': None, 'actual...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>9973 rows × 3 columns</p>
</div>




```python
for i in range(len(neg_dfs)):
    neg_dfs[i]['Reviews'] = neg_dfs[i]['Reviews'].apply(lambda x: dict.fromkeys(x))
```

    C:\Users\sreekupc\anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy


​    


```python
neg_dfs[0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>112_1</td>
      <td>{'film': None, 'plain': None, 'horrible': None...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1498_2</td>
      <td>{'justification': None, 'happened': None, 'mov...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10569_1</td>
      <td>{'tried': None, 'watch': None, 'movie': None, ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9583_4</td>
      <td>{'proximity': None, 'tell': None, 'convict': N...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>12</th>
      <td>641_4</td>
      <td>{'knowledge': None, 'largo': None, 'winch': No...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24990</th>
      <td>11229_1</td>
      <td>{'made': None, 'watch': None, 'school': None, ...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24991</th>
      <td>6021_1</td>
      <td>{'believe': None, 'managed': None, 'spend': No...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24992</th>
      <td>5893_1</td>
      <td>{'movie': None, 'fantastic': None, 'great': No...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24994</th>
      <td>1430_1</td>
      <td>{'flipping': None, 'channel': None, 'late': No...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>6461_4</td>
      <td>{'think': None, 'great': None, 'movie': None, ...</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
<p>10027 rows × 3 columns</p>
</div>



# Conditional probability based on the sentiment
> P[“the” | Positive]  = # of positive documents containing “the” / num of all positive review documents


```python
train_word_given_pos_probabilities_list = []
for i in range(len(train_pos_vocab_lists)):
    word_given_pos_probs = {}
    for word in train_pos_vocab_lists[i]:
        count = 0
        for row in pos_dfs[i]['Reviews']:
            if word in row:
                count += 1
        word_given_pos_probs[word] = count / pos_dfs[i].shape[0]
    train_word_given_pos_probabilities_list.append(word_given_pos_probs)  
        
```


```python
train_word_given_pos_probabilities_list[0]
```




    {'well': 0.31144089040409106,
     'made': 0.22380427153313948,
     'film': 0.5890905444700691,
     'set': 0.11029780407099168,
     'early': 0.05865837761957285,
     'communist': 0.004211370700892409,
     'yugoslavia': 0.0007018951168154016,
     'five': 0.024666599819512684,
     'young': 0.13315953073297904,
     'actor': 0.18590193522510778,
     'teenager': 0.01494033891507069,
     'center': 0.01413817306728166,
     'story': 0.35004512182893815,
     'give': 0.16865536949764365,
     'strong': 0.05183996791336609,
     'sincere': 0.003509475584077008,
     'emotionally': 0.012333299909756342,
     'deep': 0.028075804672616063,
     'performance': 0.18179083525518902,
     'clear': 0.02817607540358969,
     'depiction': 0.01012734382833651,
     'natural': 0.022861726661987367,
     'trust': 0.01333600721949263,
     'naivete': 0.00030081219292088637,
     'inherent': 0.002907851198235235,
     'teen': 0.011430863330993683,
     'easily': 0.03439286072395468,
     'manipulated': 0.0027073097362879774,
     'impacted': 0.000501353654868144,
     'rest': 0.05805675323373107,
     'life': 0.2395467762959992,
     'highly': 0.06106487516293994,
     'recommended': 0.02747418028677429,
     'watched': 0.0745011531134062,
     'series': 0.08472876767271634,
     'first': 0.2626090444199338,
     'came': 0.06517597513285872,
     'year': 0.2193923593702998,
     'old': 0.12754436979845582,
     'best': 0.24957384939336208,
     'friend': 0.10538453825328387,
     'house': 0.056151609345232126,
     'dad': 0.015241151107991577,
     'didnt': 0.10929509676125539,
     'want': 0.14017848190113305,
     'watch': 0.20976636919683145,
     'itit': 0.0027073097362879774,
     'became': 0.02937932417527324,
     'weekly': 0.0017046024265516895,
     'ritual': 0.0022059560814198334,
     'every': 0.12904843076306027,
     'sunday': 0.008723553594705705,
     'anyone': 0.08432768474882182,
     'tell': 0.08653364083024165,
     'two': 0.20385039606938735,
     'fourteen': 0.0012032487716835455,
     'documentary': 0.031284468063772185,
     'almost': 0.10588589190815201,
     'silence': 0.00671813897523313,
     'must': 0.10728968214178282,
     'mean': 0.06387245563020154,
     'something': 0.1347638624285571,
     'specialthe': 0.0001002707309736288,
     'broad': 0.004111099969918781,
     'sweep': 0.0026070390053143486,
     'event': 0.04572345332397473,
     'world': 0.1375714428958187,
     'war': 0.06086433370099268,
     'make': 0.31665496841471974,
     'difficult': 0.026972826631906147,
     'subject': 0.030382031485009526,
     'maker': 0.012132758447809085,
     'broke': 0.006417326782312243,
     'considered': 0.01905143888498947,
     'significant': 0.007119221899127645,
     'key': 0.01664494134162238,
     'happening': 0.013937631605334403,
     'devoted': 0.005214078010628697,
     'one': 0.5662288178080819,
     'episode': 0.06176677027975534,
     'covered': 0.007620575553995789,
     'long': 0.10718941141080919,
     'period': 0.03258798756642936,
     'wolf': 0.003008121929208864,
     'pack': 0.007620575553995789,
     'nearly': 0.031083926601824928,
     'six': 0.012934924295598114,
     'battle': 0.027975533941642435,
     'stalingrad': 0.0004010829238945152,
     'itselfthis': 0.000501353654868144,
     'could': 0.1963300912463652,
     'today': 0.06106487516293994,
     'quite': 0.13747117216484508,
     'simply': 0.05695377519302116,
     'interviewed': 0.0013035195026571743,
     'deadthe': 0.0006016243858417727,
     'list': 0.01734683645843778,
     'player': 0.02025468765667302,
     'appearing': 0.004512182893813296,
     'amazing': 0.06898626290985661,
     'insight': 0.01092950967612554,
     'thinking': 0.0330893412212975,
     'eden': 0.0008021658477890304,
     'foreign': 0.009224907249573849,
     'architect': 0.0009024365787626592,
     'confident': 0.0036097463150506367,
     'later': 0.08292389451519101,
     'minister': 0.002907851198235235,
     'see': 0.33991777800060163,
     'traudl': 0.0004010829238945152,
     'junge': 0.0004010829238945152,
     'hitler': 0.005615160934523212,
     'bunker': 0.0007018951168154016,
     'dictated': 0.0006016243858417727,
     'last': 0.10037100170460243,
     'left': 0.0654767873257796,
     'suicide': 0.011430863330993683,
     'escaped': 0.0038102877769978943,
     'russian': 0.00912463651860022,
     'many': 0.2220996691065878,
     'others': 0.06898626290985661,
     'play': 0.1515090745011531,
     'major': 0.030582572946956783,
     'role': 0.1479995989170761,
     'realism': 0.011531134061967312,
     'criticism': 0.009224907249573849,
     'park': 0.012934924295598114,
     'included': 0.010428156021257394,
     'revelation': 0.007018951168154016,
     'part': 0.16173668906046326,
     'emerged': 0.0016043316955780607,
     'blame': 0.006417326782312243,
     'programme': 0.002807580467261606,
     'opening': 0.03499448510979645,
     'title': 0.04742805575052642,
     'music': 0.09966910658778702,
     'lawrence': 0.004512182893813296,
     'olivier': 0.0033089341221297502,
     'narration': 0.007821117015943046,
     'lends': 0.003008121929208864,
     'gravity': 0.001504060964604432,
     'scriptthe': 0.0004010829238945152,
     'ever': 0.1750726962799559,
     'without': 0.11360673819312143,
     'heart': 0.06076406297001905,
     'warming': 0.0026070390053143486,
     'uplifting': 0.0050135365486814396,
     'movie': 0.5926000200541462,
     'outstanding': 0.02657174370801163,
     'alisan': 0.0004010829238945152,
     'porter': 0.0026070390053143486,
     'curly': 0.0016043316955780607,
     'sue': 0.002807580467261606,
     'saw': 0.11460944550285772,
     'released': 0.04030883385139878,
     'enjoyed': 0.06327083124435977,
     'immensely': 0.005615160934523212,
     'caught': 0.024466058357565428,
     'channel': 0.012734382833650857,
     'touched': 0.010027073097362879,
     'begin': 0.051238343527524315,
     'wager': 0.0008021658477890304,
     'edgar': 0.0038102877769978943,
     'allen': 0.01012734382833651,
     'poe': 0.0021056853504462045,
     'bet': 0.008723553594705705,
     'man': 0.17076105484808984,
     'spend': 0.01504060964604432,
     'entire': 0.042314248470871355,
     'night': 0.07008924095056653,
     'creepy': 0.0198536047327785,
     'castle': 0.010227614559310138,
     'course': 0.09716233831344631,
     'come': 0.17988569136669005,
     'unscathed': 0.00030081219292088637,
     'hard': 0.08282362378421738,
     'say': 0.17767973528527023,
     'strange': 0.03208663391156122,
     'people': 0.23934623483405193,
     'arent': 0.026371202246064374,
     'supposed': 0.02266118520004011,
     'wandering': 0.0025067682743407198,
     'around': 0.10809184798957185,
     'including': 0.04542264113105385,
     'icy': 0.0021056853504462045,
     'barbara': 0.01082923894515191,
     'steele': 0.003409204853103379,
     'fairly': 0.021959290083224708,
     'odd': 0.0198536047327785,
     'presentation': 0.006517597513285872,
     'french': 0.02687255590093252,
     'english': 0.032888799759350246,
     'switch': 0.004512182893813296,
     'back': 0.15501855008523013,
     'forth': 0.006818409706206758,
     'time': 0.395467762959992,
     'perhaps': 0.05926000200541462,
     'done': 0.10227614559310137,
     'bit': 0.12142785520906448,
     'dialog': 0.02236037300711922,
     'lost': 0.05264213376115512,
     'also': 0.2986062368394666,
     'rather': 0.08553093352050536,
     'dark': 0.04973428256291988,
     'claustrophobic': 0.003409204853103379,
     'doesnt': 0.13025167953474381,
     'much': 0.2625087736889602,
     'beyond': 0.027173368093853404,
     'small': 0.06828436779304121,
     'circle': 0.005715431665496842,
     'light': 0.044520204552291186,
     'candle': 0.002406497543367091,
     'generate': 0.001504060964604432,
     'plus': 0.01995387546375213,
     'there': 0.07079113606738194,
     'feel': 0.12413516494535246,
     'dread': 0.0027073097362879774,
     'impending': 0.002005414619472576,
     'doom': 0.0033089341221297502,
     'pretty': 0.09866639927805074,
     'version': 0.06256893612754437,
     'synapse': 0.000501353654868144,
     'uncensored': 0.000501353654868144,
     'wondered': 0.005113807279655069,
     'might': 0.0811190213576657,
     'censored': 0.0009024365787626592,
     'topless': 0.001403790233630803,
     'scene': 0.2440589591898125,
     'guess': 0.0309836558708513,
     'overall': 0.04802968013636819,
     'good': 0.37511280457234536,
     'gloomy': 0.002306226812393462,
     'black': 0.056251880076205754,
     'white': 0.04602426551689562,
     'definitely': 0.07229519703198636,
     'wont': 0.04823022159831545,
     'go': 0.21357665697382933,
     'detail': 0.032888799759350246,
     'plot': 0.14739797453123432,
     'reviewer': 0.015541963300912464,
     'wanted': 0.038604231424847084,
     'really': 0.2854707710819212,
     'peter': 0.02937932417527324,
     'falks': 0.000501353654868144,
     'alone': 0.03479394364784919,
     'reason': 0.07470169457535346,
     'enough': 0.0915471773789231,
     'filma': 0.0008021658477890304,
     'scale': 0.007420034092048531,
     'road': 0.01574250476285972,
     'trip': 0.02015441692569939,
     'falk': 0.0033089341221297502,
     'paul': 0.0316855509876667,
     'reiser': 0.0019051438884989471,
     'upstate': 0.0004010829238945152,
     'ny': 0.002306226812393462,
     'fall': 0.05434673618770681,
     'setting': 0.03379123633811291,
     'action': 0.09836558708512985,
     'filmvery': 0.0004010829238945152,
     'written': 0.05093753133460343,
     'adult': 0.03479394364784919,
     'target': 0.00752030482302216,
     'audience': 0.08152010428156022,
     'mind': 0.07239546776295999,
     'plenty': 0.027173368093853404,
     'reality': 0.03820314850095257,
     'based': 0.0515391557204452,
     'humor': 0.04863130452220997,
     'played': 0.09836558708512985,
     'drama': 0.059460543467361876,
     'feeling': 0.06016243858417728,
     'cant': 0.10227614559310137,
     'except': 0.02988067783014138,
     'damn': 0.01092950967612554,
     'shame': 0.019452521808883988,
     'lovely': 0.020655770580567532,
     'like': 0.4285571041812895,
     'get': 0.3115411611350647,
     'exposure': 0.003509475584077008,
     'trashy': 0.002406497543367091,
     'junk': 0.0022059560814198334,
     'great': 0.32989070490323874,
     'big': 0.11029780407099168,
     'leading': 0.025268224205354458,
     'prof': 0.01604331695578061,
     'famous': 0.03369096560713927,
     'writerdirector': 0.006116514589391357,
     'john': 0.07500250676827434,
     'right': 0.11621377719843577,
     'casting': 0.021959290083224708,
     'ground': 0.012834653564624486,
     'breaking': 0.008422741401784818,
     'lately': 0.003910558507971523,
     'got': 0.11019753334001806,
     'habit': 0.0032086633911561214,
     'purchasing': 0.0011029780407099167,
     'interesting': 0.09926802366389251,
     'dvd': 0.08342524817005915,
     'criterion': 0.001504060964604432,
     'company': 0.020354958387646647,
     'release': 0.032989070490323874,
     'figure': 0.03318961195227113,
     'even': 0.2712323272836659,
     'dislike': 0.006016243858417728,
     'usually': 0.03278852902837662,
     'supply': 0.002907851198235235,
     'extra': 0.0191517096159631,
     'material': 0.02426551689561817,
     'compensate': 0.001002707309736288,
     'shortcoming': 0.003409204853103379,
     'actual': 0.024766870550486313,
     'read': 0.06106487516293994,
     'buy': 0.030582572946956783,
     'million': 0.01504060964604432,
     'latest': 0.007821117015943046,
     'purchase': 0.0050135365486814396,
     'disappointed': 0.02426551689561817,
     'cheery': 0.0007018951168154016,
     'funny': 0.11501052842675223,
     'romantic': 0.035596109495638226,
     'everything': 0.07700792138774691,
     'excellent': 0.11230321869046425,
     'song': 0.05775594104081019,
     'wonderful': 0.09254988468865938,
     'understood': 0.008121929208863933,
     'would': 0.2837661686553695,
     'probably': 0.09676125538955178,
     'hum': 0.001504060964604432,
     'sing': 0.009625990173468364,
     'day': 0.13255790634713727,
     'acting': 0.15802667201443898,
     'kind': 0.08753634813997795,
     'american': 0.07640629700190514,
     'musical': 0.03469367291687556,
     'classic': 0.08472876767271634,
     'hollywood': 0.06497543367091146,
     'era': 0.025268224205354458,
     'relied': 0.0007018951168154016,
     'dance': 0.026270931515090745,
     'character': 0.3301915170961596,
     'le': 0.06657976536648952,
     'developed': 0.015341421838965206,
     'anything': 0.07700792138774691,
     'extremely': 0.03609746315050637,
     'impressive': 0.02326280958588188,
     'lacking': 0.006818409706206758,
     'loved': 0.07249573849393362,
     'development': 0.019251980346936728,
     'relationship': 0.05705404592399479,
     'especially': 0.11079915772585983,
     'michel': 0.0016043316955780607,
     'prosper': 0.000501353654868144,
     'moment': 0.0964604431966309,
     'direction': 0.04692670209565828,
     'perfect': 0.08101875062669207,
     'several': 0.05464754838062769,
     'memorable': 0.033891507069086536,
     'single': 0.02927905344429961,
     'occurs': 0.006216785320364985,
     'lead': 0.0716935726461446,
     'couple': 0.058858919081520106,
     'argument': 0.005514890203549584,
     'hide': 0.009726260904441994,
     'stage': 0.02897824125137872,
     'opera': 0.011130051138072796,
     'singer': 0.011831946254888199,
     'line': 0.09555800661786824,
     'beatrice': 0.0007018951168154016,
     'interpret': 0.0009024365787626592,
     'situation': 0.04251478993281861,
     'high': 0.06246866539657074,
     'point': 0.10708914067983556,
     'cinema': 0.05494836057354858,
     'history': 0.05073698987265617,
     'managed': 0.014338714529228918,
     'laugh': 0.05665296300010027,
     'win': 0.0243657876265918,
     'sweet': 0.02827634613456332,
     'romance': 0.02998094856111501,
     'smirk': 0.0006016243858417727,
     'clever': 0.020956582773488417,
     'director': 0.1360673819312143,
     'information': 0.014539255991176175,
     'faith': 0.009625990173468364,
     'planning': 0.004512182893813296,
     'amongst': 0.005414619472575955,
     'disc': 0.004512182893813296,
     'contains': 0.018048731575253184,
     'feature': 0.04903238744610448,
     'photo': 0.00581570239647047,
     'gallery': 0.0026070390053143486,
     'useful': 0.003409204853103379,
     'flip': 0.001403790233630803,
     'rare': 0.02356362177880277,
     'television': 0.032888799759350246,
     'interview': 0.011130051138072796,
     'rene': 0.0017046024265516895,
     'clair': 0.0008021658477890304,
     'piece': 0.05855810688859922,
     'interest': 0.036899628998295395,
     'started': 0.02927905344429961,
     'silent': 0.015140880377017949,
     'talkie': 0.0026070390053143486,
     'said': 0.06607841171162138,
     'represented': 0.004311641431866038,
     'death': 0.06286974832046525,
     'think': 0.22029479594906246,
     'understand': 0.05765567030983656,
     'meant': 0.020555499849593904,
     'hear': 0.02757445101774792,
     'explain': 0.012533841371703599,
     'description': 0.008021658477890305,
     'amazoncom': 0.001002707309736288,
     'please': 0.02165847789030382,
     'note': 0.03078311440890404,
     'wrong': 0.04752832648150005,
     'important': 0.03980748019653063,
     'respect': 0.02155820715933019,
     'sung': 0.003509475584077008,
     'fact': 0.1208262308232227,
     'regular': 0.013235736488519001,
     'actually': 0.11661486012233029,
     'lot': 0.1570239647047027,
     'chaplin': 0.0050135365486814396,
     'buster': 0.0021056853504462045,
     'keaton': 0.005113807279655069,
     'marx': 0.001504060964604432,
     'brother': 0.052241050837260605,
     'heck': 0.0063170560513386145,
     'smack': 0.0012032487716835455,
     'making': 0.07901333600721949,
     'complaint': 0.010628697483204653,
     'spent': 0.013536548681439888,
     'hundred': 0.010027073097362879,
     'hour': 0.0515391557204452,
     'fixing': 0.0007018951168154016,
     'voted': 0.0026070390053143486,
     'imdb': 0.01494033891507069,
     'look': 0.1698586182693272,
     'web': 0.0050135365486814396,
     'site': 0.008723553594705705,
     'nowhere': 0.009625990173468364,
     'found': 0.09054447006918681,
     'clue': 0.009224907249573849,
     'proud': 0.008422741401784818,
     'surprisingly': 0.02356362177880277,
     'horrible': 0.011430863330993683,
     'fine': 0.05835756542665196,
     'job': 0.09896721147097162,
     'bravo': 0.003409204853103379,
     'deserve': 0.007720846284969418,
     'money': 0.04592399478592199,
     'stand': 0.0449212874761857,
     'endearing': 0.007219492630101273,
     'tale': 0.037902336308031685,
     'ten': 0.02496741201243357,
     'average': 0.022260102276145593,
     'age': 0.0560513386142585,
     'sex': 0.03639827534342725,
     'comedy': 0.10367993582673218,
     'compared': 0.020655770580567532,
     'wet': 0.003409204853103379,
     'although': 0.09305123834352752,
     'fan': 0.11160132357364885,
     'little': 0.19612954978441793,
     'grab': 0.00752030482302216,
     'drink': 0.005514890203549584,
     'soda': 0.0004010829238945152,
     'missing': 0.020555499849593904,
     'maybe': 0.062268123934623484,
     'fifty': 0.005113807279655069,
     'harry': 0.011831946254888199,
     'met': 0.012032487716835455,
     'sally': 0.003910558507971523,
     'ahab': 0.0002005414619472576,
     'top': 0.06397272636117517,
     'mine': 0.011531134061967312,
     'leaf': 0.028075804672616063,
     'warm': 0.01092950967612554,
     'fuzzy': 0.0016043316955780607,
     'reminding': 0.0022059560814198334,
     'shoot': 0.013135465757545373,
     'summary': 0.005314348741602326,
     'describes': 0.004913265817707811,
     'word': 0.05665296300010027,
     'charming': 0.024666599819512684,
     'girlfight': 0.0009024365787626592,
     'using': 0.026270931515090745,
     'wellknown': 0.0038102877769978943,
     'formula': 0.009525719442494735,
     'someone': 0.05795648250275744,
     'pointed': 0.0031083926601824926,
     'however': 0.11029780407099168,
     'seen': 0.2134763862428557,
     'dont': 0.19973929609946856,
     'credible': 0.005514890203549584,
     'believe': 0.07480196530632709,
     'end': 0.19272034493131454,
     'easier': 0.005414619472575955,
     'woman': 0.12092650155419633,
     'empathize': 0.001403790233630803,
     'way': 0.2563922590995688,
     'moviethe': 0.004612453624786924,
     'encouraging': 0.001504060964604432,
     'mentally': 0.005715431665496842,
     'physically': 0.00581570239647047,
     'environment': 0.00832247067081119,
     'completely': 0.051138072796550686,
     'different': 0.0974631505063672,
     'viewer': 0.07600521407801063,
     'still': 0.20044119121628395,
     'gain': 0.007821117015943046,
     'seeing': 0.07610548480898426,
     'took': 0.03990775092750426,
     'start': 0.08362578963200641,
     'boxing': 0.0033089341221297502,
     'sport': 0.011330592600020055,
     'general': 0.02677228516995889,
     'gave': 0.04030883385139878,
     'thought': 0.11029780407099168,
     'spark': 0.0031083926601824926,
     'opinion': 0.040709916775293294,
     'impression': 0.015341421838965206,
     'know': 0.20074200340920487,
     'im': 0.11761756743206658,
     'brilliant': 0.05905946054346736,
     'execution': 0.006116514589391357,
     'displaying': 0.0019051438884989471,
     'venue': 0.0011029780407099167,
     'politics': 0.008121929208863933,
     'intention': 0.008121929208863933,
     'pave': 0.00030081219292088637,
     'hell': 0.029680136368194124,
     'fox': 0.012032487716835455,
     'pulled': 0.010027073097362879,
     'plug': 0.0017046024265516895,
     'midway': 0.0013035195026571743,
     'lame': 0.00581570239647047,
     'expect': 0.0421137070089241,
     'u': 0.13205655269226912,
     'invest': 0.0007018951168154016,
     'new': 0.14438985260202547,
     'show': 0.1992379424446004,
     'realistic': 0.033991777800060165,
     'risk': 0.0066178682442595005,
     'never': 0.20214579364283566,
     'finding': 0.016444399879675123,
     'happens': 0.03178582171864033,
     'werent': 0.012333299909756342,
     'remaining': 0.004111099969918781,
     'already': 0.0442193923593703,
     'filmed': 0.027173368093853404,
     'aired': 0.006918680437180387,
     'broadcast': 0.005214078010628697,
     'listening': 0.007319763361074903,
     'hanging': 0.007018951168154016,
     'youre': 0.052341321568234234,
     'going': 0.1163140479294094,
     'introduce': 0.003509475584077008,
     'least': 0.07981550185500852,
     'air': 0.02075604131154116,
     'conclusion': 0.01824927303720044,
     'abruptly': 0.0017046024265516895,
     'ending': 0.0766068384638524,
     'seems': 0.10367993582673218,
     'happen': 0.032387446104482104,
     'ie': 0.01012734382833651,
     'wait': 0.02496741201243357,
     'investing': 0.0004010829238945152,
     'artificially': 0.0008021658477890304,
     'low': 0.03018149002306227,
     'share': 0.02105685350446205,
     'likely': 0.01604331695578061,
     'vicious': 0.00581570239647047,
     'cycle': 0.0025067682743407198,
     'let': 0.07540358969216886,
     'starstruck': 0.0002005414619472576,
     'box': 0.01754737792038504,
     'nothing': 0.08583174571342625,
     'jamie': 0.004111099969918781,
     'kennedy': 0.003910558507971523,
     'favorite': 0.0766068384638524,
     'darker': 0.007119221899127645,
     'side': 0.05194023864433972,
     'forced': 0.02085631204251479,
     'agreed': 0.0021056853504462045,
     'flick': 0.03449313145492831,
     'tired': 0.010528426752231024,
     'daughter': 0.038905043617767976,
     'constantly': 0.017046024265516897,
     'subjected': 0.0019051438884989471,
     'nickelodeon': 0.0004010829238945152,
     'disney': 0.01855008523012133,
     'cartoon': 0.017046024265516897,
     'rehashed': 0.0004010829238945152,
     'couldnt': 0.03539556803369097,
     'fair': 0.015241151107991577,
     'afternoon': 0.008222199939837561,
     'sick': 0.013536548681439888,
     'relax': 0.004111099969918781,
     'flipped': 0.000501353654868144,
     'tv': 0.07690765065677328,
     'searching': 0.006918680437180387,
     'finger': 0.005314348741602326,
     'stopped': 0.00752030482302216,
     'surfing': 0.0018048731575253183,
     'heard': 0.04111099969918781,
     'harvey': 0.0027073097362879774,
     'voice': 0.04081018750626692,
     'adore': 0.0026070390053143486,
     'solitary': 0.001002707309736288,
     'thing': 0.21127043016143587,
     'voiceover': 0.002005414619472576,
     'work': 0.18149002306226814,
     'duck': 0.0021056853504462045,
     'change': 0.05424646545673318,
     'instantly': 0.00581570239647047,
     'mesmerized': 0.0011029780407099167,
     'together': 0.08533039205855811,
     'grew': 0.012132758447809085,
     'love': 0.23102376416324075,
     'along': 0.07039005314348741,
     'message': 0.03379123633811291,
     'portrayed': 0.025468765667301715,
     'necessarily': 0.007921387746916675,
     'proponent': 0.0004010829238945152,
     'gay': 0.01173167552391457,
     'child': 0.07770981650456232,
     'picked': 0.010628697483204653,
     'another': 0.12965005514890204,
     'fat': 0.006417326782312243,
     'skinny': 0.001504060964604432,
     'feminine': 0.0022059560814198334,
     'bully': 0.004111099969918781,
     'smart': 0.015943046224806978,
     'parent': 0.03208663391156122,
     'name': 0.05795648250275744,
     'kid': 0.07369898726561716,
     'rule': 0.01664494134162238,
     'cruel': 0.006016243858417728,
     'happy': 0.03940639727263612,
     'entertaining': 0.06046325077709817,
     'conveyed': 0.0027073097362879774,
     'girl': 0.098866940739998,
     'accept': 0.011230321869046425,
     'pick': 0.02155820715933019,
     'older': 0.027975533941642435,
     'theyre': 0.032387446104482104,
     'mom': 0.011531134061967312,
     'often': 0.06397272636117517,
     'warms': 0.0007018951168154016,
     'condemning': 0.0008021658477890304,
     'advocate': 0.0013035195026571743,
     'certain': 0.03318961195227113,
     'view': 0.044520204552291186,
     'clearly': 0.02917878271332598,
     'missed': 0.022159831545171964,
     'poignant': 0.009726260904441994,
     'need': 0.0877368896019252,
     'explained': 0.006517597513285872,
     'help': 0.0825228115912965,
     'offer': 0.02827634613456332,
     'frankly': 0.007018951168154016,
     'gone': 0.026972826631906147,
     'head': 0.052942945954076004,
     'opposite': 0.013035195026571744,
     'perdition': 0.0008021658477890304,
     'salvation': 0.002306226812393462,
     'saved': 0.006918680437180387,
     'deal': 0.042314248470871355,
     'concept': 0.01754737792038504,
     'explores': 0.0047127243557605536,
     'disappointment': 0.007620575553995789,
     'attend': 0.004211370700892409,
     'fatherson': 0.0017046024265516895,
     'noted': 0.006818409706206758,
     'outset': 0.0019051438884989471,
     'none': 0.02326280958588188,
     'currently': 0.005113807279655069,
     'fashionable': 0.0007018951168154016,
     'premise': 0.016143587686754238,
     'father': 0.06778301413817307,
     'move': 0.050636719141682544,
     'hitman': 0.0047127243557605536,
     'michael': 0.040609646044319665,
     'sullivan': 0.004010829238945152,
     'tom': 0.022560914469066478,
     'hank': 0.006116514589391357,
     'crime': 0.031284468063772185,
     'bos': 0.01343627795046626,
     'rooney': 0.0021056853504462045,
     'newman': 0.0026070390053143486,
     'son': 0.044620475283264814,
     'protect': 0.007018951168154016,
     'rooneys': 0.0002005414619472576,
     'connor': 0.0006016243858417727,
     'evil': 0.03710017046024266,
     'kill': 0.04431966309034393,
     'loyal': 0.0050135365486814396,
     'soldier': 0.017647648651358667,
     'cover': 0.0191517096159631,
     'stealing': 0.005414619472575955,
     'learns': 0.012533841371703599,
     'witnessed': 0.002807580467261606,
     'mistakenly': 0.0007018951168154016,
     'wife': 0.07309736287977539,
     'attempt': 0.038002607039005314,
     'decides': 0.019452521808883988,
     'revenge': 0.014639526722149805,
     'price': 0.010628697483204653,
     'terribly': 0.006517597513285872,
     'curse': 0.004913265817707811,
     'born': 0.014739797453123434,
     'refuse': 0.009525719442494735,
     'hire': 0.006016243858417728,
     'contract': 0.004812995086734182,
     'killer': 0.035195026571743705,
     'named': 0.028777699789431465,
     'maguire': 0.0018048731575253183,
     'jude': 0.0012032487716835455,
     'law': 0.016444399879675123,
     'join': 0.01012734382833651,
     'accompanied': 0.0036097463150506367,
     'surviving': 0.004010829238945152,
     'pursues': 0.0013035195026571743,
     'confronts': 0.002306226812393462,
     'church': 0.010628697483204653,
     'basement': 0.004010829238945152,
     'demand': 0.009224907249573849,
     'murdered': 0.008121929208863933,
     'family': 0.11160132357364885,
     'murderer': 0.007921387746916675,
     'room': 0.032888799759350246,
     'guarantee': 0.004211370700892409,
     'heaven': 0.012233029178782714,
     'somewhat': 0.03980748019653063,
     'predictably': 0.0018048731575253183,
     'step': 0.017948460844279555,
     'convincingly': 0.005113807279655069,
     'giving': 0.02907851198235235,
     'subtle': 0.02426551689561817,
     'laconic': 0.0008021658477890304,
     'irish': 0.00671813897523313,
     'gangster': 0.01343627795046626,
     'showing': 0.032888799759350246,
     'edge': 0.021959290083224708,
     'face': 0.06377218489922791,
     'manner': 0.017647648651358667,
     'eye': 0.06968815802667201,
     'haunted': 0.008823824325679333,
     'connors': 0.0007018951168154016,
     'suitably': 0.0036097463150506367,
     'tyler': 0.003008121929208864,
     'hoechlin': 0.0007018951168154016,
     'naturally': 0.009726260904441994,
     'cinematography': 0.04201343627795047,
     'darkness': 0.0063170560513386145,
     'echoing': 0.0007018951168154016,
     'theme': 0.04993482402486714,
     'camera': 0.045222099669106584,
     'drew': 0.0066178682442595005,
     'palette': 0.002306226812393462,
     'green': 0.012032487716835455,
     'grey': 0.005214078010628697,
     'belonged': 0.0013035195026571743,
     'urban': 0.008121929208863933,
     'landscape': 0.008523012132758448,
     'depression': 0.00671813897523313,
     'illinois': 0.0012032487716835455,
     'younger': 0.022159831545171964,
     'state': 0.03138473879474581,
     'rural': 0.006016243858417728,
     'thomas': 0.008021658477890305,
     'lush': 0.0036097463150506367,
     'haunting': 0.012132758447809085,
     'faint': 0.0022059560814198334,
     'overtone': 0.0025067682743407198,
     'arrangement': 0.002306226812393462,
     'authentic': 0.007119221899127645,
     'midwestern': 0.000501353654868144,
     'factory': 0.005615160934523212,
     'home': 0.06868545071693573,
     'shone': 0.0007018951168154016,
     'gleaming': 0.0007018951168154016,
     'excellence': 0.002406497543367091,
     'lie': 0.01744710718941141,
     'generation': 0.013636819412413516,
     'unique': 0.03278852902837662,
     'profound': 0.006918680437180387,
     'distinctive': 0.002306226812393462,
     'enveloping': 0.0004010829238945152,
     'negative': 0.013636819412413516,
     'slight': 0.006517597513285872,
     'slipped': 0.0018048731575253183,
     'screenplay': 0.025468765667301715,
     'nod': 0.0047127243557605536,
     'political': 0.020655770580567532,
     'acknowledged': 0.0018048731575253183,
     'farce': 0.004111099969918781,
     'intended': 0.012333299909756342,
     'thankfully': 0.008021658477890305,
     'neither': 0.013135465757545373,
     'social': 0.02246064373809285,
     'slightest': 0.002406497543367091,
     'take': 0.1837962498746616,
     'taken': 0.037902336308031685,
     'place': 0.09956883585681339,
     'particular': 0.03318961195227113,
     'location': 0.02266118520004011,
     'unlikely': 0.008222199939837561,
     'merely': 0.01173167552391457,
     'spring': 0.0066178682442595005,
     'imagination': 0.015541963300912464,
     'goal': 0.007319763361074903,
     'entertain': 0.006517597513285872,
     'certainly': 0.057254587385942045,
     'educate': 0.0011029780407099167,
     'delightfully': 0.0038102877769978943,
     'tossed': 0.0011029780407099167,
     'wind': 0.014539255991176175,
     'agree': 0.020655770580567532,
     'enlightenment': 0.0007018951168154016,
     'authenticity': 0.003008121929208864,
     'entertainment': 0.03379123633811291,
     'nowadays': 0.008523012132758448,
     'described': 0.009726260904441994,
     'nobrainer': 0.000501353654868144,
     'mating': 0.0006016243858417727,
     'game': 0.035295297302717334,
     'chuckle': 0.003008121929208864,
     'outright': 0.002306226812393462,
     'fare': 0.008723553594705705,
     'fantasy': 0.025569036398275343,
     'budget': 0.03419231926200742,
     'enterprise': 0.002807580467261606,
     'filmmaker': 0.026972826631906147,
     'manufacturing': 0.000501353654868144,
     'distributing': 0.000501353654868144,
     'shouldnt': 0.011531134061967312,
     'broken': 0.009625990173468364,
     'form': 0.03810287776997894,
     'yet': 0.09585881881078913,
     'whats': 0.022761455931013738,
     'remarkable': 0.01754737792038504,
     'whole': 0.0898425749523714,
     'achievement': 0.009425448711521107,
     'james': 0.0376015241151108,
     'cameron': 0.0037100170460242655,
     'decidedly': 0.002306226812393462,
     'regard': 0.009224907249573849,
     'latter': 0.01413817306728166,
     'flaw': 0.02165847789030382,
     'transfer': 0.003910558507971523,
     'otherwise': 0.019653063270831245,
     'original': 0.08382633109395367,
     'ratio': 0.0012032487716835455,
     'demonstrating': 0.001002707309736288,
     'technical': 0.01092950967612554,
     'looking': 0.06968815802667201,
     'expected': 0.02266118520004011,
     'indeed': 0.027774992479695178,
     'given': 0.05905946054346736,
     'ferraris': 0.00030081219292088637,
     'hand': 0.05685350446204753,
     'approach': 0.018951168154015843,
     'putting': 0.014037902336308031,
     'true': 0.09565827734884187,
     'soundtrack': 0.0330893412212975,
     'offered': 0.007921387746916675,
     'mix': 0.015943046224806978,
     'whilst': 0.007720846284969418,
     'uncertain': 0.0021056853504462045,
     'deemed': 0.0019051438884989471,
     'ferrari': 0.0008021658477890304,
     'involvement': 0.004612453624786924,
     'inferior': 0.0031083926601824926,
     'though': 0.1462949964905244,
     'may': 0.1215281259400381,
     'atmosphere': 0.032888799759350246,
     'viewing': 0.037902336308031685,
     'experience': 0.050536448410708916,
     'owing': 0.0009024365787626592,
     'score': 0.044018850897423044,
     'equally': 0.02015441692569939,
     'free': 0.021157124235435677,
     'positively': 0.002807580467261606,
     'overwhelmed': 0.0018048731575253183,
     'sidebar': 0.00030081219292088637,
     'screen': 0.08081820916474482,
     'youll': 0.04371803870450216,
     'notice': 0.01504060964604432,
     'numerous': 0.01012734382833651,
     'commentary': 0.01333600721949263,
     'load': 0.007420034092048531,
     'featurettes': 0.000501353654868144,
     'various': 0.024466058357565428,
     'minute': 0.07059059460543467,
     'chunk': 0.0012032487716835455,
     'compiled': 0.0004010829238945152,
     'lengthy': 0.004211370700892409,
     'discus': 0.004812995086734182,
     'anatomy': 0.001002707309736288,
     'stunt': 0.006818409706206758,
     'featurette': 0.0013035195026571743,
     'example': 0.052241050837260605,
     'exactly': 0.03369096560713927,
     'claim': 0.013235736488519001,
     'coverage': 0.001504060964604432,
     'aspect': 0.033490424145192016,
     'preproduction': 0.000501353654868144,
     'production': 0.05324375814699689,
     'postproduction': 0.0006016243858417727,
     'preferable': 0.0004010829238945152,
     'find': 0.1750726962799559,
     'digestible': 0.0004010829238945152,
     'easy': 0.03720044119121629,
     'access': 0.0032086633911561214,
     'whatever': 0.01905143888498947,
     'special': 0.06718138975233129,
     'wish': 0.04141181189210869,
     'discussion': 0.005113807279655069,
     'predictable': 0.01664494134162238,
     'chat': 0.0016043316955780607,
     'track': 0.017647648651358667,
     'involving': 0.017246565727464153,
     'overly': 0.008623282863732077,
     'jokey': 0.00030081219292088637,
     'seriously': 0.022560914469066478,
     'incredibly': 0.018149002306226812,
     'enthusiastic': 0.0025067682743407198,
     'crossover': 0.0004010829238945152,
     'elsewhere': 0.003409204853103379,
     'welcome': 0.011029780407099168,
     'package': 0.0033089341221297502,
     'beginning': 0.055650255690363984,
     'stargate': 0.002306226812393462,
     'wasnt': 0.05875864835054648,
     'blown': 0.007921387746916675,
     'away': 0.08883986764263511,
     'sci': 0.0013035195026571743,
     'fi': 0.0016043316955780607,
     'potential': 0.016745212072596008,
     'erm': 0.0002005414619472576,
     'star': 0.10738995287275645,
     'alien': 0.010628697483204653,
     'fanatic': 0.0036097463150506367,
     'admit': 0.022159831545171964,
     'hardcore': 0.0032086633911561214,
     'remember': 0.066178682442595,
     'either': 0.04642534844079013,
     'wearing': 0.008823824325679333,
     'shirt': 0.002807580467261606,
     'person': 0.052341321568234234,
     'ah': 0.002907851198235235,
     'getting': 0.04983455329389351,
     'slightly': 0.022260102276145593,
     'core': 0.006818409706206758,
     'unfortunately': 0.02817607540358969,
     'ended': 0.017246565727464153,
     'season': 0.02817607540358969,
     'drop': 0.01012734382833651,
     'quality': 0.04642534844079013,
     'nearer': 0.0004010829238945152,
     'forward': 0.02105685350446205,
     'effect': 0.07470169457535346,
     'chemistry': 0.02075604131154116,
     'blew': 0.004612453624786924,
     'affect': 0.007921387746916675,
     'better': 0.14940338915070692,
     'running': 0.02827634613456332,
     'glad': 0.020655770580567532,
     'wise': 0.011029780407099168,
     'wouldnt': 0.02837661686553695,
     'benchmark': 0.001002707309736288,
     'basically': 0.025268224205354458,
     'isnt': 0.08683445302316253,
     'worth': 0.08954176275945051,
     'watching': 0.12995086734182293,
     'date': 0.0191517096159631,
     'idea': 0.06507570440188509,
     'cry': 0.025468765667301715,
     'freedom': 0.012934924295598114,
     'primer': 0.0006016243858417727,
     'wanting': 0.01092950967612554,
     'overview': 0.0006016243858417727,
     'apartheid': 0.0018048731575253183,
     'cruelty': 0.0036097463150506367,
     'famed': 0.0026070390053143486,
     'richard': 0.02917878271332598,
     'attenborough': 0.002306226812393462,
     'gandhi': 0.0025067682743407198,
     'stranger': 0.009325177980547479,
     'genre': 0.047227514288579166,
     'collaboration': 0.0032086633911561214,
     'reallife': 0.004913265817707811,
     'mr': 0.04662588990273739,
     'wood': 0.013235736488519001,
     'main': 0.0716935726461446,
     'book': 0.06647949463551589,
     ...}




```python
%%time
top_pos_words = []
for i in range(len(train_word_given_pos_probabilities_list)):
    pos_wrds_dict = {}
    pos_prob = len(train_word_given_pos_probabilities_list[i])/len(train_word_probabilities_list[i])
    for word in train_word_given_pos_probabilities_list[i]:
        pos_wrds_dict[word] = (train_word_given_pos_probabilities_list[i].get(word)* pos_prob) 
    top_pos_words.append(pos_wrds_dict)
```

    Wall time: 48.8 ms


# Deriving Top 10 words that predicts Positive Class 


```python
top_pos_words[0]
x = dict(sorted(top_pos_words[0].items(), key=lambda item: item[1], reverse=True))
dict(itertools.islice(x.items(), 20))
```




    {'movie': 0.5727257906163863,
     'film': 0.5693340135145971,
     'one': 0.5472390083943711,
     'like': 0.4141844380870448,
     'time': 0.38220482541303336,
     'good': 0.3625325182226567,
     'story': 0.33830553892416315,
     'see': 0.3285178392875718,
     'character': 0.3191177713197563,
     'great': 0.31882704756817437,
     'make': 0.3060352024985698,
     'get': 0.30109289872167716,
     'well': 0.3009959908044832,
     'also': 0.28859177740365455,
     'really': 0.27589684025124395,
     'would': 0.2742494056589464,
     'even': 0.26213591600969965,
     'first': 0.25380183513101784,
     'much': 0.25370492721382387,
     'way': 0.24779354426499142}




```python
%%time
top_neg_words = []
for i in range(len(train_word_given_neg_probabilities_list)):
    neg_wrds_dict = {}
    neg_prob = len(train_word_given_neg_probabilities_list[i])/len(train_word_probabilities_list[i])
    for word in train_word_given_neg_probabilities_list[i]:
        neg_wrds_dict[word] = (train_word_given_neg_probabilities_list[i].get(word)*neg_prob)
    top_neg_words.append(neg_wrds_dict)
```

    Wall time: 45.9 ms


# Deriving Top 10 words that predicts Negative Class 


```python
top_neg_words[1]
y = dict(sorted(top_neg_words[3].items(), key=lambda item: item[1], reverse=True))
dict(itertools.islice(y.items(), 20))
```




    {'movie': 0.6529992131576644,
     'film': 0.542282755030307,
     'one': 0.5234818470464162,
     'like': 0.47999287807357766,
     'time': 0.36794706281604617,
     'even': 0.3677571546545928,
     'good': 0.361870001649536,
     'would': 0.3375617569834953,
     'make': 0.3363273539340479,
     'get': 0.33015533868681096,
     'bad': 0.324363139762481,
     'character': 0.30964525724983916,
     'really': 0.30746131339312455,
     'see': 0.30385305832550913,
     'much': 0.2830581146463571,
     'dont': 0.28077921670891576,
     'story': 0.2629278495322921,
     'could': 0.25998427302976373,
     'scene': 0.2543819822668872,
     'thing': 0.2527677628945329}



> P[“the” | Negative]  = # of negative documents containing “the” / num of all negative review documents


```python
train_word_given_neg_probabilities_list = []
for i in range(len(train_neg_vocab_lists)):
    word_given_neg_probs = {}
    for word in train_neg_vocab_lists[i]:
        count = 0
        for row in neg_dfs[i]['Reviews']:
            if word in row:
                count += 1
        word_given_neg_probs[word] = count / neg_dfs[i].shape[0]
    train_word_given_neg_probabilities_list.append(word_given_neg_probs)
```


```python
%%time
prob_pos_given_review_dev_list = []
for i in range(len(dev_dfs)):
    prob_pos_given_review = []
    for review in dev_dfs[i]['Reviews']:
        prob_of_each_word = 1
        for word in review:
            if word in train_word_given_pos_probabilities_list[i]:
                prob_of_each_word *= train_word_given_pos_probabilities_list[i].get(word)
            else:
                prob_of_each_word = 0
        prob_pos_given_review.append(prob_of_each_word*(pos_dfs[i].shape[0]/ train_dfs[i].shape[0]))  
    prob_pos_given_review_dev_list.append(prob_pos_given_review)
    
```

    Wall time: 1 s



```python
%%time
prob_neg_given_review_dev_list = []
for i in range(len(dev_dfs)):
    prob_neg_given_review = []
    for review in dev_dfs[i]['Reviews']:
        prob_of_each_word = 1
        for word in review:
            if word in train_word_given_neg_probabilities_list[i]:
                prob_of_each_word *= train_word_given_neg_probabilities_list[i].get(word)
            else:
                prob_of_each_word = 0
        prob_neg_given_review.append(prob_of_each_word*(neg_dfs[i].shape[0]/ train_dfs[i].shape[0]))  
    prob_neg_given_review_dev_list.append(prob_neg_given_review)
```

    Wall time: 896 ms



```python
for i in range(len(dev_dfs)):
    dev_dfs[i]=dev_dfs[i].drop(['Predicted_Label','Positive_Prob_Smooth','Negative_Prob_Smooth','Predicted_Label_Smooth'], axis=1)
    dev_dfs[i]['Positive_Prob'] = prob_pos_given_review_dev_list[i]
    dev_dfs[i]['Negative_Prob'] = prob_neg_given_review_dev_list[i]
```


```python
dev_dfs[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
      <th>Positive_Prob</th>
      <th>Negative_Prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>112_1</td>
      <td>[film, plain, horrible, john, ritter, fall, ac...</td>
      <td>neg</td>
      <td>6.816118e-151</td>
      <td>4.932634e-141</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1498_2</td>
      <td>[justification, happened, movie, term, distrib...</td>
      <td>neg</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10569_1</td>
      <td>[tried, watch, movie, military, camp, mission,...</td>
      <td>neg</td>
      <td>3.266468e-75</td>
      <td>4.838720e-69</td>
    </tr>
    <tr>
      <th>12</th>
      <td>641_4</td>
      <td>[knowledge, largo, famous, comic, never, read,...</td>
      <td>neg</td>
      <td>9.304063e-107</td>
      <td>5.564924e-112</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5914_1</td>
      <td>[movie, truly, boring, banned, chinese, cinema...</td>
      <td>neg</td>
      <td>1.487580e-94</td>
      <td>2.758977e-88</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24976</th>
      <td>2439_8</td>
      <td>[oh, dear, yet, another, example, religion, do...</td>
      <td>pos</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>24988</th>
      <td>4916_4</td>
      <td>[hit, film, aged, well, frankly, good, release...</td>
      <td>neg</td>
      <td>2.536866e-295</td>
      <td>6.809330e-300</td>
    </tr>
    <tr>
      <th>24991</th>
      <td>6021_1</td>
      <td>[believe, managed, spend, film, bad, acting, s...</td>
      <td>neg</td>
      <td>1.252909e-92</td>
      <td>3.967790e-85</td>
    </tr>
    <tr>
      <th>24992</th>
      <td>5893_1</td>
      <td>[movie, fantastic, great, movie, scary, hell, ...</td>
      <td>neg</td>
      <td>9.521111e-103</td>
      <td>2.131386e-96</td>
    </tr>
    <tr>
      <th>24994</th>
      <td>1430_1</td>
      <td>[flipping, channel, late, saturday, night, fri...</td>
      <td>neg</td>
      <td>1.307142e-129</td>
      <td>5.720444e-131</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 5 columns</p>
</div>




```python
for i in range(len(dev_dfs)):
    conditions = [(dev_dfs[i]['Positive_Prob'] > dev_dfs[i]['Negative_Prob']),  (dev_dfs[i]['Positive_Prob'] < dev_dfs[i]['Negative_Prob']), 
    (dev_dfs[i]['Positive_Prob'] == dev_dfs[i]['Negative_Prob'])]
    choices = ['pos', 'neg', "Can't Decide"]
    dev_dfs[i]['Predicted_Label'] = np.select(conditions, choices)
```


```python
dev_dfs[1]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
      <th>Positive_Prob</th>
      <th>Negative_Prob</th>
      <th>Predicted_Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>112_1</td>
      <td>[film, plain, horrible, john, ritter, fall, ac...</td>
      <td>neg</td>
      <td>6.816118e-151</td>
      <td>4.932634e-141</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1498_2</td>
      <td>[justification, happened, movie, term, distrib...</td>
      <td>neg</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Can't Decide</td>
    </tr>
    <tr>
      <th>7</th>
      <td>10569_1</td>
      <td>[tried, watch, movie, military, camp, mission,...</td>
      <td>neg</td>
      <td>3.266468e-75</td>
      <td>4.838720e-69</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>12</th>
      <td>641_4</td>
      <td>[knowledge, largo, famous, comic, never, read,...</td>
      <td>neg</td>
      <td>9.304063e-107</td>
      <td>5.564924e-112</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>22</th>
      <td>5914_1</td>
      <td>[movie, truly, boring, banned, chinese, cinema...</td>
      <td>neg</td>
      <td>1.487580e-94</td>
      <td>2.758977e-88</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24976</th>
      <td>2439_8</td>
      <td>[oh, dear, yet, another, example, religion, do...</td>
      <td>pos</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Can't Decide</td>
    </tr>
    <tr>
      <th>24988</th>
      <td>4916_4</td>
      <td>[hit, film, aged, well, frankly, good, release...</td>
      <td>neg</td>
      <td>2.536866e-295</td>
      <td>6.809330e-300</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24991</th>
      <td>6021_1</td>
      <td>[believe, managed, spend, film, bad, acting, s...</td>
      <td>neg</td>
      <td>1.252909e-92</td>
      <td>3.967790e-85</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24992</th>
      <td>5893_1</td>
      <td>[movie, fantastic, great, movie, scary, hell, ...</td>
      <td>neg</td>
      <td>9.521111e-103</td>
      <td>2.131386e-96</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24994</th>
      <td>1430_1</td>
      <td>[flipping, channel, late, saturday, night, fri...</td>
      <td>neg</td>
      <td>1.307142e-129</td>
      <td>5.720444e-131</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>5000 rows × 6 columns</p>
</div>




```python
accuracy_list = []
for i in range(len(dev_dfs)):
    count = sum(dev_dfs[i]['Label'] == dev_dfs[i]['Predicted_Label'])
    accuracy = round((count/dev_dfs[i].shape[0])*100, 5)
    accuracy_list.append(accuracy)
```

# Accuracy Using Dev Set 


```python
accuracy_list
```




    [69.9, 69.3, 69.42, 69.72, 70.16]



# Average Accuracy on Dev Set


```python
average_accuracy = sum(accuracy_list) / len(accuracy_list)
average_accuracy
```




    69.7



# Comparing the effect of Smoothing


```python
%%time
prob_pos_given_review_dev_list_smooth = []
for i in range(len(dev_dfs)):
    prob_pos_given_review_smooth = []
    for review in dev_dfs[i]['Reviews']:
        prob_of_each_word = 1
        for word in review:
            if word in train_word_given_pos_probabilities_list[i]:
                prob_of_each_word *= train_word_given_pos_probabilities_list[i].get(word)
            else:
                prob_of_each_word *= (1 / (pos_dfs[i].shape[0] + len(train_pos_vocab_lists[i])))
        prob_pos_given_review_smooth.append(prob_of_each_word*(pos_dfs[i].shape[0]/ train_dfs[i].shape[0]))  
    prob_pos_given_review_dev_list_smooth.append(prob_pos_given_review_smooth)
    
```

    Wall time: 851 ms



```python
%%time
prob_neg_given_review_dev_list_smooth = []
for i in range(len(dev_dfs)):
    prob_neg_given_review_smooth = []
    for review in dev_dfs[i]['Reviews']:
        prob_of_each_word = 1
        for word in review:
            if word in train_word_given_neg_probabilities_list[i]:
                prob_of_each_word *= train_word_given_neg_probabilities_list[i].get(word)
            else:
                prob_of_each_word *= (1 / (neg_dfs[i].shape[0] + len(train_neg_vocab_lists[i])))
        prob_neg_given_review_smooth.append(prob_of_each_word*(neg_dfs[i].shape[0]/ train_dfs[i].shape[0]))  
    prob_neg_given_review_dev_list_smooth.append(prob_neg_given_review_smooth)
```

    Wall time: 844 ms



```python
for i in range(len(dev_dfs)):
    dev_dfs[i]['Positive_Prob_Smooth'] = prob_pos_given_review_dev_list_smooth[i]
    dev_dfs[i]['Negative_Prob_Smooth'] = prob_neg_given_review_dev_list_smooth[i]
```


```python
for i in range(len(dev_dfs)):
    conditions = [(dev_dfs[i]['Positive_Prob_Smooth'] > dev_dfs[i]['Negative_Prob_Smooth']),  (dev_dfs[i]['Positive_Prob_Smooth'] < dev_dfs[i]['Negative_Prob_Smooth']), 
    (dev_dfs[i]['Positive_Prob_Smooth'] == dev_dfs[i]['Negative_Prob_Smooth'])]
    choices = ['pos', 'neg', "Can't Decide"]
    dev_dfs[i]['Predicted_Label_Smooth'] = np.select(conditions, choices)
```


```python
accuracy_list_smooth = []
for i in range(len(dev_dfs)):
    count = sum(dev_dfs[i]['Label'] == dev_dfs[i]['Predicted_Label_Smooth'])
    accuracy = round((count/dev_dfs[i].shape[0])*100, 5)
    accuracy_list_smooth.append(accuracy)
```

# Accuracy on Dev Set After Smoothing


```python
accuracy_list_smooth
```




    [76.36, 75.2, 76.14, 76.82, 76.16]



# Average accuracy on Dev Set after Smoothing


```python
average = sum(accuracy_list_smooth) / len(accuracy_list_smooth)
average
```




    76.136



# Using optimal hyperparameters on the test dataset



```python
processed_test_df = text_preprocessing(test_df_movie_reviews)
```


```python
u_dict, u_list = build_vocabulary(processed_test_df)
```


```python
processed_test_df = modify_dataframe(processed_test_df, u_dict)
```


```python
tr_u_dict, tr_u_list = build_vocabulary(total_train_df)
total_train_df = modify_dataframe(total_train_df, tr_u_dict)
total_train_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985_10</td>
      <td>[im, usually, lover, choose, would, favourite,...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8834_9</td>
      <td>[well, made, film, set, early, communist, yugo...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5948_8</td>
      <td>[poster, seem, le, neither, mark, twain, rodge...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6489_10</td>
      <td>[watched, series, first, came, year, old, watc...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8596_3</td>
      <td>[basically, pale, shadow, high, fidelity, witt...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>3613_8</td>
      <td>[girlfight, came, review, praised, didnt, get,...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>964_8</td>
      <td>[yokai, childrens, film, takashi, miike, might...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>6461_4</td>
      <td>[think, great, movie, fun, maybe, little, unre...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>8801_9</td>
      <td>[greatest, film, saw, whereas, im, used, mains...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>7504_7</td>
      <td>[hey, guy, im, actually, movie, didnt, even, k...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 3 columns</p>
</div>




```python
total_dict_train_rows = []
for i in range(total_train_df.shape[0]):
    total_dict_train_rows.append(dict.fromkeys(total_train_df.iloc[i]['Reviews']))
```


```python
total_train_pos_vocab_lists = []
total_pos_words_list = total_train_df.loc[total_train_df['Label'] == 'pos', 'Reviews'].values.tolist()
total_train_pos_vocab_lists = dict.fromkeys(list(itertools.chain.from_iterable(total_pos_words_list)))
```


```python
total_train_neg_vocab_lists = []
total_neg_words_list = total_train_df.loc[total_train_df['Label'] == 'neg', 'Reviews'].values.tolist()
total_train_neg_vocab_lists.append(dict.fromkeys(list(itertools.chain.from_iterable(total_neg_words_list))))
```


```python
total_pos_dfs = total_train_df.loc[total_train_df['Label'] == 'pos']
total_neg_dfs= total_train_df.loc[total_train_df['Label'] == 'neg']
```


```python
total_pos_dfs.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1985_10</td>
      <td>[im, usually, lover, choose, would, favourite,...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8834_9</td>
      <td>[well, made, film, set, early, communist, yugo...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5948_8</td>
      <td>[poster, seem, le, neither, mark, twain, rodge...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6489_10</td>
      <td>[watched, series, first, came, year, old, watc...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>8</th>
      <td>7263_10</td>
      <td>[heart, warming, uplifting, movie, outstanding...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
</div>




```python
total_pos_dfs['Reviews'] = total_pos_dfs['Reviews'].apply(lambda x: dict.fromkeys(x))
total_neg_dfs['Reviews'] = total_neg_dfs['Reviews'].apply(lambda x: dict.fromkeys(x))
```

    C:\Users\sreekupc\anaconda3\lib\site-packages\ipykernel_launcher.py:1: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      """Entry point for launching an IPython kernel.
    C:\Users\sreekupc\anaconda3\lib\site-packages\ipykernel_launcher.py:2: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy


​    


```python
total_train_word_given_pos_probabilities_list = []
total_word_given_pos_probs = {}
for word in total_train_pos_vocab_lists:
    count = 0
    for row in total_pos_dfs['Reviews']:
        if word in row:
            count += 1
    total_word_given_pos_probs[word] = count / total_pos_dfs.shape[0]
total_train_word_given_pos_probabilities_list.append(total_word_given_pos_probs)  

```


```python
total_train_word_given_neg_probabilities_list = []
total_word_given_neg_probs = {}
for word in total_train_neg_vocab_lists[0]:
    count = 0
    for row in total_neg_dfs['Reviews']:
        if word in row:
            count += 1
    total_word_given_neg_probs[word] = count / total_neg_dfs.shape[0]
total_train_word_given_neg_probabilities_list.append(total_word_given_neg_probs)  

```


```python
total_train_word_given_neg_probabilities_list
contains_1 = 0.0000 in total_train_word_given_neg_probabilities_list[0].values()
print(contains_1)
```

    False



```python
%%time
prob_pos_given_review_test_list_smooth = []

for review in processed_test_df['Reviews']:
    prob_of_each_word = 1
    for word in review:
        if word in total_train_word_given_pos_probabilities_list[0]:
            prob_of_each_word *= total_train_word_given_pos_probabilities_list[0].get(word)
            
        else:
            prob_of_each_word *= (1 / (total_pos_dfs.shape[0] + len(total_train_pos_vocab_lists)))
    prob_pos_given_review_test_list_smooth.append(prob_of_each_word*(total_pos_dfs.shape[0]/ total_train_df.shape[0]))  

```

    Wall time: 901 ms



```python
prob_neg_given_review_test_list_smooth = []

for review in processed_test_df['Reviews']:
    prob_of_each_word = 1
    for word in review:
        if word in total_train_word_given_neg_probabilities_list[0]:
            prob_of_each_word *= total_train_word_given_neg_probabilities_list[0].get(word)
        else:
            prob_of_each_word *= (1 / (total_neg_dfs.shape[0] + len(total_train_neg_vocab_lists)))
    prob_neg_given_review_test_list_smooth.append(prob_of_each_word*(total_neg_dfs.shape[0]/ total_train_df.shape[0]))  

```


```python
processed_test_df['Positive_Prob_Smooth'] = prob_pos_given_review_test_list_smooth
processed_test_df['Negative_Prob_Smooth'] = prob_neg_given_review_test_list_smooth
```


```python
processed_test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
      <th>Positive_Prob_Smooth</th>
      <th>Negative_Prob_Smooth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8940_10</td>
      <td>[betty, boris, eye, junior, dance, theyre, sti...</td>
      <td>pos</td>
      <td>5.483435e-152</td>
      <td>6.872855e-157</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11282_2</td>
      <td>[walk, movie, screening, movie, managed, becom...</td>
      <td>neg</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8612_8</td>
      <td>[particularly, hard, director, capture, filmma...</td>
      <td>pos</td>
      <td>7.265876e-168</td>
      <td>8.011279e-170</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4266_10</td>
      <td>[film, everyone, life, sweden, watch, film, sh...</td>
      <td>pos</td>
      <td>3.249323e-70</td>
      <td>9.019348e-74</td>
    </tr>
    <tr>
      <th>4</th>
      <td>155_8</td>
      <td>[vampire, craze, opinion, actually, proved, in...</td>
      <td>pos</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>3541_8</td>
      <td>[dvd, release, superior, made, tv, bbc, drama,...</td>
      <td>pos</td>
      <td>1.087612e-28</td>
      <td>3.102179e-32</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>2751_10</td>
      <td>[ive, seen, riverdance, person, nothing, compa...</td>
      <td>pos</td>
      <td>2.257800e-33</td>
      <td>2.474213e-35</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>7180_7</td>
      <td>[seeing, movie, didnt, grow, france, spent, le...</td>
      <td>pos</td>
      <td>3.813200e-64</td>
      <td>4.255711e-64</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>7941_1</td>
      <td>[original, assault, precinct, gritty, witty, p...</td>
      <td>neg</td>
      <td>5.351927e-131</td>
      <td>3.673533e-128</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>8133_4</td>
      <td>[julien, hernandez, certainly, attractive, lik...</td>
      <td>neg</td>
      <td>9.671321e-287</td>
      <td>2.181559e-282</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 5 columns</p>
</div>




```python
conditions = [(processed_test_df['Positive_Prob_Smooth'] > processed_test_df['Negative_Prob_Smooth']),  (processed_test_df['Positive_Prob_Smooth'] < processed_test_df['Negative_Prob_Smooth']), 
    (processed_test_df['Positive_Prob_Smooth'] == processed_test_df['Negative_Prob_Smooth'])]
choices = ['pos', 'neg', "Can't Decide"]
processed_test_df['Predicted_Label_Smooth'] = np.select(conditions, choices)
```


```python
processed_test_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
      <th>Positive_Prob_Smooth</th>
      <th>Negative_Prob_Smooth</th>
      <th>Predicted_Label_Smooth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8940_10</td>
      <td>[betty, boris, eye, junior, dance, theyre, sti...</td>
      <td>pos</td>
      <td>5.483435e-152</td>
      <td>6.872855e-157</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>11282_2</td>
      <td>[walk, movie, screening, movie, managed, becom...</td>
      <td>neg</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Can't Decide</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8612_8</td>
      <td>[particularly, hard, director, capture, filmma...</td>
      <td>pos</td>
      <td>7.265876e-168</td>
      <td>8.011279e-170</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4266_10</td>
      <td>[film, everyone, life, sweden, watch, film, sh...</td>
      <td>pos</td>
      <td>3.249323e-70</td>
      <td>9.019348e-74</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>4</th>
      <td>155_8</td>
      <td>[vampire, craze, opinion, actually, proved, in...</td>
      <td>pos</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>Can't Decide</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>3541_8</td>
      <td>[dvd, release, superior, made, tv, bbc, drama,...</td>
      <td>pos</td>
      <td>1.087612e-28</td>
      <td>3.102179e-32</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>2751_10</td>
      <td>[ive, seen, riverdance, person, nothing, compa...</td>
      <td>pos</td>
      <td>2.257800e-33</td>
      <td>2.474213e-35</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>7180_7</td>
      <td>[seeing, movie, didnt, grow, france, spent, le...</td>
      <td>pos</td>
      <td>3.813200e-64</td>
      <td>4.255711e-64</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>7941_1</td>
      <td>[original, assault, precinct, gritty, witty, p...</td>
      <td>neg</td>
      <td>5.351927e-131</td>
      <td>3.673533e-128</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>8133_4</td>
      <td>[julien, hernandez, certainly, attractive, lik...</td>
      <td>neg</td>
      <td>9.671321e-287</td>
      <td>2.181559e-282</td>
      <td>neg</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 6 columns</p>
</div>




```python
count = sum(processed_test_df['Label'] == processed_test_df['Predicted_Label_Smooth'])
test_accuracy = round((count/processed_test_df.shape[0])*100, 5)
```

# Accuracy 


```python
test_accuracy
```




    71.832



# Using five fold cross validation for final accuracy with Smoothing



```python
test_df_movie_reviews
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Filenames</th>
      <th>Reviews</th>
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1333_7</td>
      <td>[felt, cheated, knowing, whole, story, could, ...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1226_1</td>
      <td>[really, looking, forward, seeing, continuatio...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4242_1</td>
      <td>[playmania, extremely, boring, basis, show, me...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2804_3</td>
      <td>[young, man, never, knew, birth, parent, recei...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7051_4</td>
      <td>[case, taking, fairy, tale, far, enchanted, co...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>5958_1</td>
      <td>[one, latest, disaster, movie, york, entertain...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>10891_1</td>
      <td>[movie, bad, theyre, good, there, movie, like,...</td>
      <td>neg</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>11807_8</td>
      <td>[nothing, else, tv, yesterday, afternoon, thou...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>3548_8</td>
      <td>[respected, director, shot, short, film, opera...</td>
      <td>pos</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>12269_9</td>
      <td>[wonderful, rollercoaster, film, bear, repeate...</td>
      <td>pos</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 3 columns</p>
</div>




```python
test_df = text_preprocessing(test_df_movie_reviews)
test_k_fold_val = KFold(5, True, 1)

test_k_fold_val_train = []
test_k_fold_val_dev = []

for test_X_train, test_X_dev in test_k_fold_val.split(test_df):
    print(test_X_train,test_X_dev)
    test_k_fold_val_train.append(test_X_train)
    test_k_fold_val_dev.append(test_X_dev)
```

    [    1     3     5 ... 24997 24998 24999] [    0     2     4 ... 24984 24986 24995]
    [    0     1     2 ... 24997 24998 24999] [    5     6     7 ... 24991 24992 24994]
    [    0     1     2 ... 24995 24998 24999] [    3     8    17 ... 24989 24996 24997]
    [    0     2     3 ... 24996 24997 24998] [    1     9    10 ... 24987 24993 24999]
    [    0     1     2 ... 24996 24997 24999] [   14    15    18 ... 24982 24990 24998]



```python
%%time
test_k_fold_val_list = []
col_names = ["Filenames","Reviews","Label"]

for test_train_index_info, test_dev_index_info in zip(test_k_fold_val_train, test_k_fold_val_dev):
    test_df_kval_train = pd.DataFrame(columns=col_names)
    test_df_kval_dev = pd.DataFrame(columns=col_names)
    for i in test_train_index_info:
        test_df_kval_train = test_df_kval_train.append(test_df.iloc[i])

    for j in test_dev_index_info:
        test_df_kval_dev = test_df_kval_dev.append(test_df.iloc[j])
    
    test_k_fold_val_list.append((test_df_kval_train,test_df_kval_dev)) 

```

    Wall time: 12min 50s



```python
pickle.dump( test_k_fold_val_list, open( "test_5_fold_cv.p", "wb" ) )
```


```python
test_k_fold = pickle.load( open( "test_5_fold_cv.p", "rb" ) )
```


```python
test_train_vocab_list_lists = []
test_train_vocab_list_dicts = []
for i in range(len(test_k_fold)):
    test_words_dict, test_words_list = build_vocabulary(test_k_fold[i][0]) 
    test_train_vocab_list_dicts.append(words_dict)
    test_train_vocab_list_lists.append(words_list)

test_dev_vocab_list_lists = []
test_dev_vocab_list_dicts = []
for i in range(len(test_k_fold)):
    words_dict, words_list = build_vocabulary(test_k_fold[i][1]) 
    test_dev_vocab_list_dicts.append(words_dict)
    test_dev_vocab_list_lists.append(words_list)
```


```python
test_train_dfs = []
for i in range(len(test_k_fold)):
    test_train_dfs.append(modify_dataframe(test_k_fold[i][0],test_train_vocab_list_dicts[i])) 

test_dict_train_rows = []

for df in test_train_dfs:
    dummy_list = []
    for i in range(df.shape[0]):
        dummy_list.append(dict.fromkeys(df.iloc[i]['Reviews']))
    test_dict_train_rows.append(dummy_list)
    
test_train_word_probabilities_list = []
for i in range(len(test_train_vocab_list_dicts)):
    test_word_probability = {}
    for word in test_train_vocab_list_dicts[i]:
        count = 0
        for row in test_dict_train_rows[i]:
            if word in row:
                count += 1
        test_word_probability[word] = round(count / len(test_dict_train_rows[i]), 5)               
    test_train_word_probabilities_list.append(test_word_probability)
```


```python
test_dev_dfs = []
for i in range(len(test_k_fold)):
    test_dev_dfs.append(modify_dataframe(test_k_fold[i][1],test_dev_vocab_list_dicts[i]))

test_train_pos_vocab_lists = []
for i in range(len(test_train_dfs)):
    test_pos_words_list = test_train_dfs[i].loc[test_train_dfs[i]['Label'] == 'pos', 'Reviews'].values.tolist()
    test_train_pos_vocab_lists.append(dict.fromkeys(list(itertools.chain.from_iterable(test_pos_words_list))))

test_train_neg_vocab_lists = []
for i in range(len(test_train_dfs)):
    test_neg_words_list = test_train_dfs[i].loc[test_train_dfs[i]['Label'] == 'neg', 'Reviews'].values.tolist()
    test_train_neg_vocab_lists.append(dict.fromkeys(list(itertools.chain.from_iterable(test_neg_words_list))))


test_pos_dfs = []
test_neg_dfs= []
for i in range(len(test_train_dfs)):
    test_pos_dfs.append(test_train_dfs[i].loc[test_train_dfs[i]['Label'] == 'pos'])
    test_neg_dfs.append(test_train_dfs[i].loc[test_train_dfs[i]['Label'] == 'neg'])


for i in range(len(pos_dfs)):
    test_pos_dfs[i]['Reviews'] = test_pos_dfs[i]['Reviews'].apply(lambda x: dict.fromkeys(x))

for i in range(len(neg_dfs)):
    test_neg_dfs[i]['Reviews'] = test_neg_dfs[i]['Reviews'].apply(lambda x: dict.fromkeys(x))
```

    C:\Users\sreekupc\anaconda3\lib\site-packages\ipykernel_launcher.py:24: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
    C:\Users\sreekupc\anaconda3\lib\site-packages\ipykernel_launcher.py:27: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy



```python
%%time
test_train_word_given_pos_probabilities_list = []
for i in range(len(test_train_pos_vocab_lists)):
    test_word_given_pos_probs = {}
    for word in test_train_pos_vocab_lists[i]:
        count = 0
        for row in test_pos_dfs[i]['Reviews']:
            if word in row:
                count += 1
        test_word_given_pos_probs[word] = count / test_pos_dfs[i].shape[0]
    test_train_word_given_pos_probabilities_list.append(test_word_given_pos_probs) 
```

    Wall time: 4min 43s



```python
test_train_word_given_neg_probabilities_list = []
for i in range(len(test_train_neg_vocab_lists)):
    test_word_given_neg_probs = {}
    for word in test_train_neg_vocab_lists[i]:
        count = 0
        for row in test_neg_dfs[i]['Reviews']:
            if word in row:
                count += 1
        test_word_given_neg_probs[word] = count / test_neg_dfs[i].shape[0]
    test_train_word_given_neg_probabilities_list.append(test_word_given_neg_probs)
```


```python
test_prob_pos_given_review_dev_list = []
for i in range(len(test_dev_dfs)):
    test_prob_pos_given_review = []
    for review in test_dev_dfs[i]['Reviews']:
        test_prob_of_each_word = 1
        for word in review:
            if word in test_train_word_given_pos_probabilities_list[i]:
                test_prob_of_each_word *= test_train_word_given_pos_probabilities_list[i].get(word)
            else:
                test_prob_of_each_word *= (1 / (test_pos_dfs[i].shape[0] + len(test_train_pos_vocab_lists[i])))
        test_prob_pos_given_review.append(test_prob_of_each_word*(test_pos_dfs[i].shape[0]/ test_train_dfs[i].shape[0]))  
    test_prob_pos_given_review_dev_list.append(test_prob_pos_given_review)
```


```python
test_prob_neg_given_review_dev_list = []
for i in range(len(test_dev_dfs)):
    test_prob_neg_given_review = []
    for review in test_dev_dfs[i]['Reviews']:
        test_prob_of_each_word = 1
        for word in review:
            if word in test_train_word_given_neg_probabilities_list[i]:
                test_prob_of_each_word *= test_train_word_given_neg_probabilities_list[i].get(word)
            else:
                test_prob_of_each_word *= (1 / (test_neg_dfs[i].shape[0] + len(test_train_neg_vocab_lists[i])))
        test_prob_neg_given_review.append(test_prob_of_each_word*(test_neg_dfs[i].shape[0]/ test_train_dfs[i].shape[0]))  
    test_prob_neg_given_review_dev_list.append(test_prob_neg_given_review)
```


```python
for i in range(len(test_dev_dfs)):
    test_dev_dfs[i]['Positive_Prob'] = test_prob_pos_given_review_dev_list[i]
    test_dev_dfs[i]['Negative_Prob'] = test_prob_neg_given_review_dev_list[i]

for i in range(len(test_dev_dfs)):
    conditions = [(test_dev_dfs[i]['Positive_Prob'] > test_dev_dfs[i]['Negative_Prob']),  (test_dev_dfs[i]['Positive_Prob'] < test_dev_dfs[i]['Negative_Prob']), 
    (test_dev_dfs[i]['Positive_Prob'] == test_dev_dfs[i]['Negative_Prob'])]
    choices = ['pos', 'neg', "Can't Decide"]
    test_dev_dfs[i]['Predicted_Label'] = np.select(conditions, choices)

test_accuracy_list = []
for i in range(len(test_dev_dfs)):
    count = sum(test_dev_dfs[i]['Label'] == test_dev_dfs[i]['Predicted_Label'])
    test_accuracy = round((count/test_dev_dfs[i].shape[0])*100, 5)
    test_accuracy_list.append(test_accuracy)
```

## Accuracies list after 5-Fold Cross Validation


```python
test_accuracy_list
```




    [76.62, 76.4, 76.74, 77.06, 77.42]



# Average test Accuracy after 5-fold


```python
test_average = sum(test_accuracy_list) / len(test_accuracy_list)
test_average
```




    76.848



# Using the optimal hyperparameter smoothing with 5-fold cross validation on Test data, the average accuracy is 76.848

### References used:
> https://codereview.stackexchange.com/questions/196823/loading-txt-file-content-and-filename-into-pandas-dataframe

> https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908

> https://www.kaggle.com/sudalairajkumar/getting-started-with-text-preprocessing

> https://machinelearningmastery.com/k-fold-cross-validation/

> https://levelup.gitconnected.com/movie-review-sentiment-analysis-with-naive-bayes-machine-learning-from-scratch-part-v-7bb869391bab

> https://stackoverflow.com/questions/37443138/python-stemming-with-pandas-dataframe

> https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

> https://pythonhealthcare.org/2018/12/14/101-pre-processing-data-tokenization-stemming-and-removal-of-stop-words/

> https://stackoverflow.com/questions/29216889/slicing-a-dictionary


