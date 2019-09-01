from torch import nn


class _BaseNetwork():

    def __repr__(self):
        ret = ""
        for p in ["inhibition_start", "inhibition_end", "freeze", "scopes", "width", "damp"]:
            if p in self.__dict__.keys():
                ret += str(self.__dict__[p])
            ret += ","
        return f"{self.__class__.__name__},{ret[:-1]}"
