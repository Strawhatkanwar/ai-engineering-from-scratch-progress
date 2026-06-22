'''
Here I build the intuition behind complex number from scratch and work on exercise related to complex numbers and how are are 
useful in DFT and FFT.
'''

import math
import cmath
import random

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag
    
    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)

    def __mul__(self, other):
        r = self.real * other.real - self.imag * other.imag
        i = self.real * other.imag + self.imag * other.real
        return Complex(r, i)
    
    def __truediv__(self, other):
        denom = other.real ** 2 + other.imag ** 2
        r = (self.real * other.real + self.imag * other.imag) / denom
        i = (self.imag * other.real - self.real * other.imag) / denom
        return Complex(r, i)
    

    def __repr__(self):
        return f"({self.real:.6f}, {self.imag:.6f})"
    
    def magnitude(self):
        return math.sqrt(self.real ** 2 + self.imag ** 2)
    
    def phase(self):
        return math.atan2(self.imag, self.real)
    
    def conjugate(self):
        return Complex(self.real, -self.imag)
    



## Exercise: Rotation sequence. Start with the point (1, 0). Multiply by e^(i*pi/6) twelve times. Verify that you 
# return to (1, 0) after 12 multiplications. Print the coordinates at each step and confirm they trace a regular 12-gon.

start = Complex(1, 0)
rot = Complex(                                 # e^(i * pi/6) = cos(theta) + i * sin(theta)
    math.cos(math.pi / 6),
    math.sin(math.pi / 6)
)

# for i in range(12):
#     point = start * rot
#     print(i, point)
# print(
#     abs(point.real - 1) < 1e-10 and abs(point.imag) < 1e-10
# )

## Exercise: DFT of a known signal. Create a signal that is the sum of sin(2pi3t) and 0.5sin(2pi7*t) sampled at 32 points. 
# Run your DFT. Verify that the magnitude spectrum has peaks at frequencies 3 and 7, with the peak at 7 
# being half the height of the peak at 3.

def euler(theta):
    return Complex(math.cos(theta), math.sin(theta))

def dft(signal):
    N = len(signal)

    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n / N
            total = total + Complex(signal[n], 0) * euler(angle)
        result.append(total)
    return result

def f_signal(t):
    return math.sin(2 * math.pi * 3 * t) + 0.5 * math.sin(2 * math.pi * 7 * t)

N = 32
signal = [f_signal(n / N) for n in range(N)]

spectrum = dft(signal)

# magnitude at each frequency 
for k, Xk in enumerate(spectrum):
    print(k, Xk.magnitude())

# i got 16.0 at k=3 and 8 at k=7 at k=25 i got 8 and at k=29 i got 16. which proves the peak at 3 is exactly half of peak at k=7

## Exercise:Roots of unity visualization. Compute the 8th roots of unity. 
# Verify that they sum to zero. Verify that multiplying any root by the primitive root e^(2pii/8) gives the next root.


roots = []
for k in range(8):
    root = euler(2 * math.pi * k / 8)
    roots.append(root)
    print(k, root)

## Exercise: Rotation matrix equivalence. For 10 random angles and 10 random points, verify that complex multiplication gives 
# the same result as matrix-vector multiplication with the 2x2 rotation matrix. Print the maximum numerical difference.

thetas = [random.randint(10, 60) for _ in range(10)]
points = [(random.randint(i, j), random.randint(i+1, j+2)) for i, j in zip(range(10), range(100, 200, 10))]
max_diff = 0
for i in range(10):
    x = points[i][0]
    y = points[i][1]
    z = Complex(x,y)
    theta = math.radians(thetas[i])
    rot = euler(theta)
    z_rot = z * rot

    x2 = x * math.cos(theta) - y * math.sin(theta)
    y2 = x * math.sin(theta) + y * math.cos(theta)

    dx = abs(z_rot.real - x2)
    dy = abs(z_rot.imag - y2)
    max_diff = max(max_diff, dx, dy)
print(f"The maximum numerical difference across all 10 trials is: {max_diff}")
# we got the max difference of 0. which is correct and proves the (x + iy)e^(itheta) = [[cos(theta), -sin(theta)],[sin(theta), cos(theta)]][x, y]

