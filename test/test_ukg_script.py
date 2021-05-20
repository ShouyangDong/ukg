import tvm
from tvm.script import ty
from tvm import tir

@tvm.script.tir
def add_intrin(a: ty.handle, b: ty.handle, c:ty.handle)->None:
    A = tir.match_buffer(a, [128, 128])
    B = tir.match_buffer(b, [128, 128])
    C = tir.match_buffer(c, [128, 128])

    tir.add(C, B, A)
    tir.sub(C, B, A)

def test_add_intrin():
    func = add_intrin
    print("func: ", func)

if __name__ == "__main__":
    test_add_intrin()
