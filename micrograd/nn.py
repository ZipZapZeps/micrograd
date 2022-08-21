import random
from micrograd.engine import Value

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def init_learning_rate(self):
        for p in self.parameters():
            p.learning_rate = 1.0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, name, nonlin=True):
        self.w = [Value(random.uniform(-1,1),_op= f"{name}w{i}") for i in range(nin)]
        self.b = Value(0, _op = f"{name}b")
        self.s = Value(0., _op = f"{name}s")
        self.nonlin = nonlin
        self.name = name

    def __call__(self, x):
        t2 = self.s.tanh() * 0.5
        h = Value(0.5)
        b1 = h+t2
        b2 = h-t2
        act1 = self.b
        act2 = Value(0)
        for wi,xi in zip(self.w,x):
            act1 = act1 + wi*xi
            act2 = act2 + (wi-xi)**2
        return (act1*b1-act2*b2).tanh()
        """act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh() * b1 + act * b2"""
        """act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act"""
        """act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.tanh()"""

    def parameters(self):
        return self.w + [self.b,self.s]

    def __repr__(self):
        return f"{'Tanh' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, name, **kwargs):
        self.neurons = [Neuron(nin, f"{name}n{i}", **kwargs) for i in range(nout)]
        self.name = name

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer '{self.name}' of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], f"l{i}", nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
