import pandas as pd
from math import log2
from sklearn.metrics import classification_report

def ImportData():
    data_file_encode = "gb18030"  # the watermelon_3.csv is file codec type
    with open("3.0.csv", mode='r', encoding=data_file_encode) as data_file:
        data = pd.read_csv(data_file)
    return data


class Node:
    def __init__(self, attr=None, attr_down={}, label=None):
        self.attr = attr
        self.attr_down = attr_down
        self.label = label


def TreeGenerate(data):
    currNode = Node()
    label_count = CountLabel(data)

    if label_count:
        currNode.label = max(label_count, key=label_count.get)
        if len(label_count) == 1:  # ①或②?
            return currNode
        currNode.attr, div_value = OptAttr(data)  # 选择最优的划分属性，信息增益
        if div_value == 0:# 离散时
            attr_count = CountAttr(data, currNode.attr)
            for item in attr_count:
                datav = data[data[currNode.attr].isin([item])]
                if(len(datav) == 0): #③
                    currNode.attr_down[item] = Node(label=currNode.label)
                    return currNode
                else:
                    datav = datav.drop(currNode.attr, axis=1)
                    currNode.attr_down[item] = TreeGenerate(datav)

        else: #连续时
            value_l = "<=%.3f" % div_value
            value_r = ">%.3f" % div_value

            df_v_l = data[data[currNode.attr] <= div_value]  # get sub set
            df_v_r = data[data[currNode.attr] > div_value]

            currNode.attr_down[value_l] = TreeGenerate(df_v_l)
            currNode.attr_down[value_r] = TreeGenerate(df_v_r)

    return currNode


def CountLabel(data):
    s = {}
    for item in data[data.columns[-1]]:
        if item in s:
            s[item] += 1
        else:
            s[item] = 1
    return s


def CountAttr(data, attr_id):
    s = {}
    for item in data[attr_id]:
        if item in s:
            s[item] += 1
        else:
            s[item] = 1
    return s



def OptAttr(data):
    info_gain = 0
    for attr_id in data.columns[1:-1]:
        info_gian_tmp, div_value_tmp = InfoGain(data, attr_id)
        # print(attr_id,info_gian_tmp)
        if info_gian_tmp > info_gain:
            info_gain = info_gian_tmp
            opt_attr = attr_id
            div_value = div_value_tmp

    return opt_attr, div_value


def InfoGain(data, attr_id):  # 增益信息法
    gain = GetEnt(data)
    div_pos = 0
    n = len(data)

    if data[attr_id].dtype == float:  # 当数值为连续值时
        sub_info_ent = {}
        data = data.sort_values(by=[attr_id])  # 排序
        data = data.reset_index(drop=True)
        data_arr = data[attr_id]
        for i in range(n - 1):
            div = (data_arr[i] + data_arr[i + 1]) / 2
            # 前面的加后面的
            sub_info_ent[div] = ((i + 1) * GetEnt(data[0:i + 1]) / n) \
                                + ((n - i - 1) * GetEnt(data[i + 1:]) / n)
        div_pos, sub_info_ent_max = min(sub_info_ent.items(), key=lambda x: x[1])
        gain -= sub_info_ent_max

    else:  # 当数值为离散值时
        attr_count = CountAttr(data, attr_id)
        for item in attr_count:
            gain -= attr_count[item] * GetEnt(data[data[attr_id].isin([item])]) / n

    return gain, div_pos


def GetEnt(data):
    label_count = CountLabel(data)
    sze = len(data)
    ent = 0
    for item in label_count:
        x = label_count[item]
        ent -= (x / sze) * log2(x / sze)
    return ent


def Predict(root, X):
    try:
        import re  # using Regular Expression to get the number in string
    except ImportError:
        print("module re not found")

    while root.attr != None:
        # 下一个分支为连续时
        if type(X[root.attr]) == float:
            # get the div_value from root.attr_down
            for key in list(root.attr_down):
                num = re.findall(r"\d+\.?\d*", key)
                div_value = float(num[0])
                break
            if X[root.attr].values[0] <= div_value:
                key = "<=%.3f" % div_value
                root = root.attr_down[key]
            else:
                key = ">%.3f" % div_value
                root = root.attr_down[key]

        # 下一个分支为离散时
        else:
            key = X[root.attr]
            # check whether the attr_value in the child branch
            if key in root.attr_down:
                root = root.attr_down[key]
            else:
                break

    return root.label

if __name__ == '__main__':
    data = ImportData()
    prediction = []
    root = TreeGenerate(data)

    for item in range(len(data)):
        prediction.append([Predict(root, data.iloc[item])])
    print(classification_report(data['好瓜'],prediction))

