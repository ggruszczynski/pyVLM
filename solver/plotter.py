import matplotlib.pyplot as plt

def plot_gamma(wing_span, gamma):
    plt.plot(wing_span,gamma, 'ro')
    plt.ylabel('Gamma')
    plt.xlabel('wing span')
    plt.show()