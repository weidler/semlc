class A:
    def __init__(self):
        super().__init__()
        print("A")


class B:
    def __init__(self):
        super().__init__()
        print("B")


class C(B, A):
    def __init__(self):
        super().__init__()


if __name__ == '__main__':
    C()