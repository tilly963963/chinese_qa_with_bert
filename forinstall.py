import pandas as pd 
import os 
df = pd.read_csv('nlp_tilly_qa_requirements.txt', delimiter = "\t")


for i in df.values:
    query="pip install "+str(i[0])
    os.system(query)