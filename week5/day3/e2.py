cache = []
BLOCK = 4  # tokens per block

def add_tokens(n):
    while n > 0:
        space = BLOCK - (len(cache[-1]) if cache else 0)
        take = min(space, n)
        if not cache or space == 0:
            cache.append([])  # new page
        cache[-1].extend([1]*take)
        n -= take

add_tokens(10)
print("Pages:", len(cache))