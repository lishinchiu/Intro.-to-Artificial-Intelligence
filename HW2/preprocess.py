import nltk
import string
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import SnowballStemmer


def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text


def preprocessing_function(text: str) -> str:
    preprocessed_text = remove_stopwords(text)
    
    # TO-DO 0: Other preprocessing function attemption
    # Begin your code 
    text = text.lower()
    text = text.replace("<br />", " ")
    text = "".join([char for char in text if (char not in string.punctuation)])
    english_stemmer = SnowballStemmer(language='english')
    text = " ".join([english_stemmer.stem(i) for i in text.split()]) 
    preprocessed_text = text
    # End your code

    return preprocessed_text