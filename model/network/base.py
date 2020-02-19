class _BaseNetwork:
    """Superclass for uniform representation string for easier logging and debugging"""

    def __repr__(self):
        ret = ""
        for p in ["coverage", "freeze", "scopes", "width", "damp", "is_circular", "self_connection", "pad"]:
            if p in self.__dict__.keys():
                ret += str(self.__dict__[p])
            ret += ","
        return f"{self.__class__.__name__},{ret[:-1]}"
