import pandas as pd
import re

df = pd.read_parquet("./data/measuring-hate-speech.parquet")
comment_attrs_df = df.groupby('comment_id')[['sentiment','respect','insult','humiliate','status','dehumanize','violence','genocide','attack_defend','hatespeech','hate_speech_score']].median().reset_index()
comment_text_df = df[['comment_id','text']].reset_index(drop=True).drop_duplicates()
final_df = pd.merge(comment_attrs_df,comment_text_df,on="comment_id")
#### keep comments in english only 
final_english_df = final_df[final_df['text'].apply(lambda x: bool(re.match('^[a-zA-Z0-9\s]*$', str(x))))].reset_index(drop=True)
final_english_df.to_csv("./data/englishcomments.csv")
