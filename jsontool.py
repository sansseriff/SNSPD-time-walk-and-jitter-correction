import orjson
import numpy as np
from datetime import datetime

'''
A custom class for serializing, and deserializing data with json. 
Uses features of orjson to serialize numpy arrays
'''
class JsonTool:
    def __init__(self, name = None, dic=None):
        if name is not None:
            self.load_file(name)
        if dic is not None:
            self.load_dic(dic)

    def export(self, name, include_time = False):
        dic = self.export_dic()
        strb = orjson.dumps(dic, option=orjson.OPT_SERIALIZE_NUMPY)
        if include_time:
            now = datetime.now()
            dt_string = now.strftime("%d.%m.%Y_%H.%M.%S")
            name = name + dt_string
        if name[-5:] != '.json':
            name = name + '.json'
        with open(name,'wb') as file:
            file.write(strb)

    def load_file(self, name):
        with open(name, 'rb') as file:
            strb = file.read()
        self.load_dic(orjson.loads(strb))
        print(self.__dict__.keys())

    def load_dic(self, dic):
        for key in dic.keys():
            # handle numpy arrays
            if type(dic[key]) is list:
                if (type(dic[key][0]) is float) or (type(dic[key][0]) is int):
                    dic[key] = np.array(dic[key])

            # handle lists of objects of type JsonTool
            if (type(dic[key]) is list) and (key[-3:] == "_jt"):
                ls = dic[key]
                del dic[key]
                for i in range(len(ls)):
                    ls[i] = JsonTool(dic=ls[i])
                dic[key[:-3]] = ls
                continue

            # handle keys with value of type JsonTool
            if key[-3:] == "_jt":
                sub_dic = dic[key]
                del dic[key]
                dic[key[:-3]] = JsonTool(dic=sub_dic)
        self.__dict__.update(dic)

    def export_dic(self):
        dic = self.__dict__
        for key in dic.keys():
            if type(dic[key]) is type(self):
                dic[key] = dic[key].export_dic()
                json_tool_dic = dic[key].export_dic()
                del dic[key]
                new_key = str(key) + "_jt"
                dic[new_key] = json_tool_dic

            if (type(dic[key]) is list) and (type(dic[key][0]) is type(self)):
                ls = []
                for i, item in enumerate(dic[key]):
                        ls.append(dic[key][i].export_dic())

                del dic[key]
                new_key = str(key) + "_jt"
                dic[new_key] = ls
        return dic

# would be cool if this could overload the list functionality if I only wanted to use this object as a list.
# right now it does not encode lists of numpy arrays correctly.


if __name__ == '__main__':
    parent = JsonTool()
    parent.items = []

    for i in range(20):
        struct = JsonTool()
        struct.item = "this is an item"
        struct.dB = "dB"
        struct.ls = [np.array([3, 34, 2, 43]),np.array([3, 34, 2, 43]),np.array([3, 34, 2, 43])]
        struct.arr = np.array([23, 2, 34, 2, 43, 3, 4.433])
        parent.items.append(struct)

    parent.export("try_this.json")
    parent = JsonTool("try_this.json")
    item = JsonTool(dic={"this":"is"})

    # del e
    # e = JsonTool("this_is_my_json.json")
    # print(e.this)
    # print(e.how)
    # print(e.array)
    # print(type(e.array))
    # print(type(e.array[0]))


