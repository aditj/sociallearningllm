import numpy as np
import pandas as pd
### rest api data
import requests
import json

final_english_df = pd.read_csv("./data/englishcomments_and_state.csv")

endpoint = 'https://api.together.xyz/v1/chat/completions'

def generate_prompt(comment):
    return """[INST] Return a JSON with the following format for the given text, and objectively rate the text without any other assumptions: 
{
"is_insulting": Bool,
"is_dehumanizing":Bool,
"is_humiliating":Bool,
"promotes_violence":Bool,
"promotes_genocide":Bool,
"is_respectful":Bool,
}
Text:"""+comment+"""	 [/INST]"""
def return_obs(comment):
    print(comment)
    prompt = generate_prompt(comment)
    res = requests.post(endpoint, json={
        "model": "mistralai/Mistral-7B-Instruct-v0.1",
        "max_tokens": 512,
        "prompt": prompt,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1,
        "stop": [
            "[/INST]",
            "</s>"
        ],
        "repetitive_penalty": 1,
        "update_at": "2024-03-18T20:18:29.908Z",
        "prompt_format_string": "[INST]  Return a JSON with the following format for the given text: \n{\n\"is_insulting\": Bool,\n\"is_dehumanizing\":Bool,\n\"is_humiliating\":Bool,\n\"promotes_violence\":Bool,\n\n}\nText: \" \"\n [/INST] {\n\"is_insulting\": false,\n\"is_dehumanizing\": false,\n\"is_humiliating\": false,\n\"is_promotes_violence\": false\n}\n\nThe text provided is empty, so all the values in the JSON are set to false by default. If the text contained any insulting, dehumanizing, humiliating, or violent language, the corresponding value in the JSON would be set to true.\n[INST]  Return a JSON with the following format for the given text: \n{\n\"is_insulting\": Bool,\n\"is_dehumanizing\":Bool,\n\"is_humiliating\":Bool,\n\"promotes_violence\":Bool,\n\"promotes_genocide\":Bool,\n\"is_respectful\":Bool,\n}\nText: \"yay she deserves better\t\"\n [/INST] {\n\"is_insulting\": True,\n\"is_dehumanizing\": false,\n\"is_humiliating\": false,\n\"promotes_violence\": false,\n\"promotes_genocide\": false,\n\"is_respectful\": false\n}\n\nThe text \"yay she deserves better\" can be considered insulting as it implies that the person being referred to deserves something negative. The value of \"is_insulting\" is set to true. The other values are set to false as there is no dehumanizing, humiliating, violent, or genocidal language in the text. The value of \"is_respectful\" is also set to false as the text is not particularly respectful.\n[INST]  Return a JSON with the following format for the given text: \n{\n\"is_insulting\": Int between 0 and 1,\n\"is_dehumanizing\":Bool,\n\"is_humiliating\":Bool,\n\"promotes_violence\":Bool,\n\"promotes_genocide\":Bool,\n\"is_respectful\":Bool,\n}\nText: \"yay she deserves better\t\"\n [/INST] {\n\"is_insulting\": 1.0,\n\"is_dehumanizing\": false,\n\"is_humiliating\": false,\n\"promotes_violence\": false,\n\"promotes_genocide\": false,\n\"is_respectful\": false\n}\n\nThe text \"yay she deserves better\" can be considered insulting as it implies that the person being referred to deserves something negative. The value of \"is_insulting\" is set to 1.0 to indicate that it is a strong insult. The other values are set to false as there is no dehumanizing, humiliating, violent, or genocidal language in the text. The value of \"is_respectful\" is also set to false as the text is not particularly respectful.\n\nNote: In the JSON format, the value of \"is_insulting\" is represented as a decimal number between 0 and 1, with 1 indicating a strong insult and 0 indicating a mild insult or no insult at all. This representation allows for a more nuanced evaluation of the level of insult in the text.\n[INST]  {prompt}\n [/INST]"
    }, headers={
        "Authorization": "Bearer 7498540083c198b4080c6104cfeaee5cd0634e37e5dd434994898c6ec83a68fe",
    })
    response_string = res.json()['choices'][0]['message']['content']
    return response_string
def process_response_string(response_string):
    return json.loads(response_string.replace('\\n','\n').split('\n\n',1)[0].replace(" ","").replace("\n","").lower().replace("true",'"true"').replace("false",'"false"'))

final_english_df['response'] = final_english_df['text'].apply(return_obs)
final_english_df.to_csv("./data/outputcsv.csv")


