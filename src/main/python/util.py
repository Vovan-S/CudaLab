# Utilities

def clear_comments(code: str):
    res = code
    while True:
        i = res.find('//')
        j = res.find('/*')
        if i == -1 and j == -1:
            return res
        if j == -1 or (i != -1 and i < j):
            k = res.find('\n', i)
            if k == -1:
                return res[:i]
            res = res[:i] + res[k:]
        else:
            k = res.find('*/', j)
            if k == -1:
                raise ValueError('Unclosed comment')
            res = res[:j] + res[k+2:]
    return res

# Comments are cleared, passed part of string after {
def get_body(part, op='{', cl='}'):
    c = 1
    for i in range(len(part)):
        if part[i] == op:
            c += 1
        elif part[i] == cl:
            if c == 1:
                return part[:i]
            c -= 1
    raise ValueError('EOF while finding end of body')


if __name__ == "__main__":
    print(clear_comments("afafad/*fadfdaf\na*/dfafafa"))
