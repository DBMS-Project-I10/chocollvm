a:int = 1
b:int = 2
#b:str = "Hello"
c:bool = True
d:bool = False

def sum(a:int, b:int) -> int:
    return a + b

c = not d

#printf("%d", sum(a, b))
printf("%d", c)
printf("%d", sum(a, b))

if c:
    printf("%d", 1)
else:
    printf("%d", 2)
