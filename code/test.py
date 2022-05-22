def foo():
    ds = 100
    i = 0
    while True:
        print(ds)
        i += 1
        yield i


g = foo()
print(next(g))
print(next(g))
