import zhongrj.test.module123 as module123

print(module123.__doc__)
print(module123.__file__)
print(__file__)

print(__name__)
print(module123.__name__)

o = [2]
a = [1, o, 3]
b = a[1:]
b[0][0] = 100
print(b)
print(a)


class A:
    s = 0

    def test123(self):
        print('test')

    def __next__(self):
        A.s += 1
        if A.s == 10:
            raise StopIteration
        return A.s

    def __iter__(self):
        return self

    def __dir__(self):
        return [1, 2]

    def __call__(self, *args, **kwargs):
        print(args)
        print(kwargs)

    def __str__(self):
        return 'A object ..............'

    def __bool__(self):
        print('bool')
        return True


for i in A():
    print(i)

print(dir(A()))
print(dir([]))
print(dir())
print(callable(A()))
A()(1, a=1)

print(getattr([], '__add__'))
print(list.__add__.__doc__)

print(A())
print(bool(A()))
print(bool(0))

print(callable(getattr([], '__add__')))

print(getattr(A, 's'))

getattr(A(), 'test123')()  # .__call__()

print(100000 is 100000)
print(100000 == 100000)

a, b = lambda: 'a', lambda: 'b'
print((1 and a or b)())
# print(0 and a or b)

print(dir({
    1: 2
}))

print(repr({'1': 2}))
