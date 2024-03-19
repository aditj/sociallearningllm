import pandas as pd
import json 

df = pd.read_csv("./data/outputcsv.csv")

def process_response_string(response_string):
    return json.loads(response_string.replace('\\n','\n').split('\n\n',1)[0].replace(" ","").replace("\n","").lower().replace("true",'1').replace("false",'0'))

df = pd.concat([df,df['response'].apply(process_response_string).apply(pd.Series)],axis=1)

df[['state', 'is_insulting', 'is_dehumanizing', 'is_humiliating','promotes_violence', 'promotes_genocide', 'is_respectful']].to_csv("./data/output_csv_cleaned.csv",index=False)
