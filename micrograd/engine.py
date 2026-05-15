
import math
import cmath

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _children=(), _op='Value'):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def noop(self):
        return self
    
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
            self.grad += other.data.conjugate() * out.grad
            other.grad += self.data.conjugate() * out.grad
        out._backward = _backward

        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)).conjugate() * out.grad
        out._backward = _backward

        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        
        out._backward = _backward

        return out
    
    def signed_sqr(self):
        out_value = 0.5 * (self.data ** 2.)
        out_value *= 1.0 if self.data < 0. else -1.0

        out = Value(out_value, (self,), 'SignedSqr')

        def _backward():
            self.grad += (1.0 if self.data < 0. else -1.0) * out.grad

        out._backward = _backward

        return out
        
    def max(self, other):
        out_value = self.data if self.data > other.data else other.data

        out = Value(out_value, (self,other), 'Max')

        def _backward():
            if self.data >= other.data:
                self.grad += out.grad
            else:
                other.grad += out.grad

        out._backward = _backward

        return out

    def softplus(self,beta = 10.):
        out_value = self.data + math.log1p(math.exp(-beta*self.data)) / beta
        
        out = Value(out_value, (self,), 'SoftPlus')

        def _backward():
            out_grad_value = 1. / (1.+math.exp(-beta * self.data))
            self.grad += out_grad_value * out.grad

        out._backward = _backward

        return out

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

    def abs(self):
        out = Value(abs(self.data), (self,), 'abs')

        def _backward():
            self.grad += (-out.grad) if self.data < 0 else out.grad 

        out._backward = _backward

        return out

    def exp(self):
        out_value = cmath.exp(self.data) if isinstance(self.data, complex) else math.exp(self.data)
        out = Value(out_value, (self,), 'exp')

        def _backward():
            self.grad += out_value.conjugate() * out.grad
        out._backward = _backward

        return out

    def log(self):
        out_value = cmath.log(self.data) if isinstance(self.data, complex) else cmath.log(self.data)
        out = Value(out_value, (self,), 'log')

        def _backward():
            self.grad += (1. / self.data).conjugate() * out.grad
        out._backward = _backward

        return out

    def eml(self, other):
        # EML(x, y) = exp(x) - log(y), with principal-branch complex log.
        exp_x = cmath.exp(self.data) if isinstance(self.data, complex) else math.exp(self.data)
        log_y = cmath.log(other.data)
        out = Value(exp_x - log_y, (self, other), 'EML')

        def _backward():
            self.grad  += exp_x.conjugate()        * out.grad
            other.grad += (-1. / other.data).conjugate() * out.grad
        out._backward = _backward

        return out

    def real(self):
        # Bridge from a complex sub-graph back to a real-valued Value.
        d = self.data
        out_value = d.real if isinstance(d, complex) else d
        out = Value(out_value, (self,), 'Re')

        def _backward():
            # For L real, ∂L/∂Re(z) flows back as the real part of z.grad;
            # adding a real out.grad to a complex self.grad is the identity on Re,
            # leaving Im(self.grad) untouched (correct: Re(z) doesn't depend on Im(z)).
            self.grad += out.grad
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
