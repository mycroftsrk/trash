# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.ticker as ticker
import japanize_matplotlib
%matplotlib inline

import seaborn as sns
import seaborn.objects as so
sns.set()

# df_iris = sns.load_dataset("iris")
# print(df_iris)

# %%
import pandas as pd
index_list = ["row_"+str(i) for i in range(6)]
df = pd.DataFrame({"A":[2,4,3,6,1,50],
                   "B":[101,105,104,106,102,103],
                   "C":[99,100,114,106,122,113]},
                   index=index_list)
df

# %%
import matplotlib.pyplot as plt
import japanize_matplotlib
# %matplotlib inline

def add_value_label(x_list,y_list):
    for i in range(0, x_num):
        plt.text(x_list[i],y_list[i], y_list[i], ha='center') 
               #(x座標,y座標,表示するテキスト)

fig, ax1 = plt.subplots(1,1,figsize=(10,8))
ax2 = ax1.twinx()
ax1.plot(df["B"],linestyle="solid",color="k",marker="^",label="B")
ax1.plot(df["C"],linestyle="solid",color="b",marker="*",label="C")
ax2.bar(df.index,df["A"],color="lightblue",label="A")
ax1.set_ylim(min(df["B"].min(), df["C"].min() - 5),max(df["B"].max(), df["C"].max()) + 5)
ax2.set_ylim(0,df["A"].max()+5)
ax1.set_zorder(2)
ax2.set_zorder(1)
ax1.patch.set_alpha(0)
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1+handler2,label1+label2,borderaxespad=0)
ax1.grid(True)

x_num = len(df["C"])
add_value_label(df["A"].index,df["C"].values)

#グラフタイトルを付ける
plt.title("2020年各月の平均気温と降水量の推移", fontsize=15)
fig.show()
# %%
