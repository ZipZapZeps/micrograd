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
        self.w = [Value(random.uniform(-1,1),_op= f"{name}.w{i}") for i in range(nin)]
        self.b = Value(0., _op = f"{name}.b")
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
        self.w = [Value(random.uniform(-1,1),_op= f"{name}.w{i}") for i in range(nin)]
        self.b = Value(random.uniform(1.,2.), _op = f"{name}.b")
        self.activation_func = activation_func

    def __call__(self, x):
        if self.active:
            wx = sum((xi-wi) ** 2. for wi,xi in zip(self.w, x)) ** 0.5
            act = wx * -1. + self.b
            return self.activation_func(act)
        return Value(0.)

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{self.activation_func.__name__.title()}RBFNeuron({len(self.w)})"
    
class BlendedNeuron(Module):
    def __init__(self, nin, name, activation_func=Value.noop):
        super().__init__()
        self.w = [Value(random.uniform(-1,1),_op= f"{name}.w{i}") for i in range(nin)]
        self.b = Value(random.uniform(0.,2.), _op = f"{name}.b")
        self.a = Value(random.uniform(0.,1.), _op = f"{name}.a")
        self.activation_func = activation_func

    def __call__(self, x):
        if self.active:
            wx1 = sum((xi-wi) ** 2. for wi,xi in zip(self.w, x)) ** 0.5
            wx2 = sum(wi*xi for wi,xi in zip(self.w, x))
            act1 = self.a * (wx1 + self.b)
            act2 = (Value(1.) - self.a) * (wx2 * -1. + self.b)
            return self.activation_func(act1+act2)
        
        return Value(0.)

    def parameters(self):
        return self.w + [self.a, self.b]

    def __repr__(self):
        return f"{self.activation_func.__name__.title()}BlendedNeuron({len(self.w)})"
        
class EMLNeuron(Module):
    """ Binary EML node: eml(x, y) = exp(x) - log(y), with x, y two affine maps
        over the input. Weights are complex-valued; gradients flow through the
        principal-branch complex log so y is free to pass through zero. """

    def __init__(self, nin, name):
        super().__init__()
        def cw(i, tag):
            return Value(complex(random.uniform(-1, 1), random.uniform(-0.1, 0.1)),
                         _op=f"{name}.{tag}{i}")
        self.wx = [cw(i, 'wx') for i in range(nin)]
        self.wy = [cw(i, 'wy') for i in range(nin)]
        self.bx = Value(complex(0., 0.), _op=f"{name}.bx")
        # Bias y away from 0 so the initial log is well-conditioned.
        self.by = Value(complex(1., 0.), _op=f"{name}.by")

    def __call__(self, x):
        if self.active:
            xi = sum((wi * xj for wi, xj in zip(self.wx, x)), self.bx)
            yi = sum((wi * xj for wi, xj in zip(self.wy, x)), self.by)
            return xi.eml(yi)
        return Value(complex(0., 0.))

    def parameters(self):
        return self.wx + self.wy + [self.bx, self.by]

    def __repr__(self):
        return f"EMLNeuron({len(self.wx)})"

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
                    yield BlendedNeuron(nin, name, **kwargs)
                else:
                    yield BlendedNeuron(nin, name, **kwargs)

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

class EMLMLP(Module):
    """ MLP whose neurons are EMLNeurons. All intermediate Values are complex;
        the final scalar output is converted to a real Value via .real() so the
        existing SVM-margin loss in the demo notebooks works unchanged. """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        layers = []
        for i in range(len(nouts)):
            name = f"eml_l{i}"
            neurons = [EMLNeuron(sz[i], f"{name}.n{j}") for j in range(sz[i+1])]
            layers.append(Layer(name, neurons))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        # x is either a single Value or a list at the head; project to real.
        if isinstance(x, list):
            return [xi.real() for xi in x]
        return x.real()

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"EMLMLP of [{', '.join(str(layer) for layer in self.layers)}]"
