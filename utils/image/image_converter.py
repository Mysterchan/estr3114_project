import pyarrow.parquet as pq
import pandas as pd

table_train = pq.read_table('C:\\Users\\User\\Desktop\\estr3114\\project\\lsun-bedrooms\\data\\train-00000-of-00009-5beebd96eb33b02b.parquet')
df_train = table_train.to_pandas()

from io import BytesIO
from PIL import Image

for i in range(len(df_train['image'])):
    image = Image.open(BytesIO(df_train['image'][i]['bytes']))
    image.save(f"C:\\Users\\User\\Desktop\\estr3114\\project\\bedroom\\bedroom_train\\{i+15158}.jpg")

table_test = pq.read_table('C:\\Users\\User\\Desktop\\estr3114\\project\\lsun-bedrooms\\data\\test-00000-of-00001-7c2280a6897e2462.parquet')
df_test = table_test.to_pandas()

for i in range(len(df_test['image'])):
    image = Image.open(BytesIO(df_test['image'][i]['bytes']))
    image.save(f"C:\\Users\\User\\Desktop\\estr3114\\project\\bedroom\\bedroom_train\\{i+1}.jpg")