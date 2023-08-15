import random
from micrograd.engine import Value

class Module:

    def __init__(self) -> None:
        self.active = True

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def init_learning_rate(self):
        for p in self.parameters():
            p.learning_rate = 1.

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, name, activation_func=Value.noop):
        super().__init__()
        self.w = [Value(random.uniform(-1,1),_op= f"{name}w{i}") for i in range(nin)]
        self.b = Value(0., _op = f"{name}b")
        self.activation_func = activation_func

    def __call__(self, x):
        if self.active:
            wx = sum(wi*xi for wi,xi in zip(self.w, x))
            act = wx + self.b
            return self.activation_func(act)
        
        return Value(0.)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation_func.__name__.title()}Neuron({len(self.w)})"

class RBFNeuron(Module):
    def __init__(self, nin, name, activation_func=Value.noop):
        super().__init__()
        self.w = [Value(random.uniform(-1,1),_op= f"{name}w{i}") for i in range(nin)]
        self.b = Value(random.uniform(1.,2.), _op = f"{name}b")
        self.activation_func = activation_func

    def __call__(self, x):
        if self.active:
            wx = sum((wi+xi) ** 2. for wi,xi in zip(self.w, x)) ** 0.5
            act = wx * -1. + self.b
            return self.activation_func(act)
        return Value(0.)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation_func.__name__.title()}RBFNeuron({len(self.w)})"
    
class Layer(Module):

    def __init__(self, name:str, neurons):
        super().__init__()
        self.neurons = neurons
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
        layers = []

        def generate_neurons(nin:int, nout:int, name:str, last_layer:bool, **kwargs):
            for i in range(nout):
                name = f".{name}n{i}"
                if last_layer:
                    yield Neuron(nin, name)
                elif i % 2:
                    yield Neuron(nin, name, **kwargs)
                else:
                    yield RBFNeuron(nin, name, **kwargs)

        for i in range(len(nouts)):
            name = f"l{i}"
            layer = Layer(name, list(generate_neurons(sz[i],sz[i+1],name,i+1==len(nouts),activation_func=Value.relu)))
            layers.append(layer)

        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
