
import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='Value'):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        
        out._backward = _backward

        return out
    
    def softplus(self):
        try:
            beta = 10
            out_value = self.data + math.log1p(math.exp(-beta*self.data))/beta
            if out_value > 10:
                print(f"The following value seems high: {out_value}")
            out = Value(out_value, (self,), 'SoftPlus')

            def _backward():
                out_grad_value = 1. / (1.+math.exp(-beta*out.data))
                self.grad += out_grad_value * out.grad
                if abs(self.grad) > 0.25:
                    print(f"The following gradient seems high: {self.grad}")

            out._backward = _backward

            return out
        except Exception as e:
            print(f"The following value caused an exception: {self.data}")


    def tanh(self):
        out = Value(math.tanh(self.data), (self,), 'tanh')

        def _backward():
            self.grad += (1- out.data**2) * out.grad

        out._backward = _backward

        return out

    def sin(self):
        out = Value(math.sin(self.data), (self,), 'sin')

        def _backward():
            self.grad += math.cos(self.data) * out.grad

        out._backward = _backward

        return out


    def fact(self):
        if self <= Value(0):
            return Value(1)
        return self * (self - Value(1))


    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        return self * -1

    def __radd__(self, other): # other + self
        return self + other

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def __repr__(self):
        return f"{self._op}:(data={self.data:.5f}, grad={self.grad:.5f})"
