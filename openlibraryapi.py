#%%
import requests
import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
#%%
# Goal is to access the summary of book using the Open Library API, organize the data, 
# and create a word cloud from the summary text.



# %%
# Link to the Open Library Search API: https://openlibrary.org/dev/docs/api/authors

# %%
# Explain what is happening in the code below, go line by line. 
url = "https://openlibrary.org/authors/OL19981A/works.json?offset=50&limit=50"
responsea = requests.get(url, timeout=10).json()
docs = responsea.get("entries", [])
print(f"\nTotal works found for author OL19981A: {len(docs)}")
print(f"\nFirst work:")
print(docs[0]) 

#%% Run the code below to get the summary of "The Gunslinger". Change the 
# offset to 0 and limit to 100 above, rerun the code, do you still get the summary? 
# Why or why not?

gunslinger_summary = None
for doc in docs:
    if doc.get("title") == "The Gunslinger":
        gunslinger_summary = doc.get("description", "No summary available")
        print(f"\nDescription:")
        print(gunslinger_summary)
        break


#%% 
# Create a word frequencie counter

def word_frequencies(s: str) -> Counter:
    """
    Returns a Counter of word -> frequency (lowercased).
    """
    words = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", s.lower())
    return Counter(words)

#%%
wf = word_frequencies(gunslinger_summary)

# %%
# Create and display a word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(wf)
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

# %%
