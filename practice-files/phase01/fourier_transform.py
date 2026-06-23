'''
Here i build from scratch the DFT, FFT, inverse FFT. I implement it as a new method to convoution, Here i also implement the 
windowing(Hann, Hamming, Blackman). I ran an experiment to prove FFT is way faster then DFT. Proved why the positional encodings
of transformer is initialized as sinousidal function.

'''

import math
import cmath
import random
import time

class Complex:
    def __init__(self, real, imag=0.0):
        self.real = real
        self.imag = imag
    
    def __add__(self, other):
        return Complex(self.real + other.real, self.imag + other.imag)
    
    def __sub__(self, other):
        return Complex(self.real - other.real, self.imag - other.imag)

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
    
# DFT 

def dft(x):
    N = len(x)
    result = []
    for k in range(N):
        total = Complex(0, 0)
        for n in range(N):
            angle = -2 * math.pi * k * n/N
            w = Complex(math.cos(angle), math.sin(angle))
            xn = x[n] if isinstance(x[n], Complex) else Complex(x[n])
            total = total + xn * w
        result.append(total)
    
    return result

## Inverse DFT

def idft(X):
    N = len(X)
    result = []

    for n in range(N):
        total = Complex(0, 0)
        for k in range(N):
            angle = 2 * math.pi * k * n/N
            w = Complex(math.cos(angle), math.sin(angle))
            total = total + X[k] * w
        result.append(Complex(total.real / N, total.imag / N))

    return result

## step 3: FFT( Cooley-Tukey)

def fft(x):
    N = len(x)
    if N <= 1:
        return [x[0] if isinstance(x[0], Complex) else Complex(x[0])]
    
    if N % 2 != 0:
        return dft(x)
    
    even = fft([x[i] for i in range(0, N, 2)])
    odd = fft([x[i] for i in range(1, N, 2)])

    result = [Complex(0)] * N
    for k in range(N // 2):
        angle = -2 * math.pi * k / N
        twiddle = Complex(math.cos(angle), math.sin(angle))
        t = twiddle * odd[k]
        result[k] = even[k] + t
        result[k + N // 2] = even[k] - t
    return result

## Exercise: Pure tone identification. Create a signal with a single sine wave at an unknown frequency (between 1 and 50 Hz), 
# sampled at 128 Hz for 1 second. Use your DFT to identify the frequency. 
# Verify the answer matches. Now add Gaussian noise with standard deviation 0.5 and repeat. How does noise affect the spectrum?

fs = 128  # sampling rate (Hz)
duration = 1.0  # 1 second
N = int(fs * duration)  # 128 points
target_freq = 25  # picking unknow frequency

# generating a pure signal: formula - sin(2 * pi * f * t) , where t = n/fs

signal = [math.sin(2 * math.pi * target_freq * (n/fs)) for n in range(N)]

dft_output = dft(signal)

# calculating magnitude to find peak
magnitudes = [c.magnitude() for c in dft_output]

# we only care about the first half (0 to fs/2) due to Nyquist symmetry
half_N = N // 2
# print(magnitudes[:half_N])
detected_freq = magnitudes.index(max(magnitudes[:half_N]))
print(f" Pure signal - Detected peak frequency: {detected_freq} Hz (Target: {target_freq} Hz)")

## adding guassian noise and repeating (using Box-muller transform for noise)

noisy_signal = []

for xn in signal:
    # generating guassian noise(m:0, sigma=0.5)
    u1 = random.random()
    u2 = random.random()
    noise = 0.5 * math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    noisy_signal.append(noise)

noisy_dft_result = dft(noisy_signal)
noisy_magnitude = [c.magnitude() for c in noisy_dft_result]
# print(noisy_magnitude[:half_N])
detected_noisy_freq = noisy_magnitude.index(max(noisy_magnitude[:half_N]))
print(f"Noisy signal - Detected Peak frequency: {detected_noisy_freq}Hz")

# with pure tone classification it was working good and i got my target index of 25 to have the highest frequency and all other
# were microscopically small, but when i added the guassian noise it started behaving a little different other frequencies have 
# also started showing some peak minor but does.

print("-"*60)

## Exercise 2: FFT vs DFT verification. Generate a random signal of length 64. Compute both DFT (O(N^2)) and FFT. 
# Verify that all coefficients match to within 1e-10. Time both functions on signals of length 256, 512, 1024, and 2048. 
# Plot the ratio of DFT time to FFT time.

N = 64

rand_signal = [math.sin(2 * math.pi * (n/N)) for n in range(N)]
dft_out = dft(rand_signal)
fft_out = fft(rand_signal)
mag_dft = [c.magnitude() for c in dft_out]
mag_fft = [c.magnitude() for c in fft_out]

tol = 1e-10

for i in range(N):
    diff = mag_dft[i] - mag_fft[i]
    if diff <= tol:
        print("The output is good and it's near to 0 proving that both magnitudes are same.")
print(diff)

# timing stuff
N = [256, 512, 1024, 2048]
time_comp = []
ratios = []
for i in N:
    signal = [math.sin(2 * math.pi * n/i) for n in range(i)]
    start = time.perf_counter()
    ddf_out = dft(signal)
    final_time_d = (time.perf_counter() - start)
    start = time.perf_counter()
    fft_out = fft(signal)
    final_time_f = (time.perf_counter() - start)
    time_comp.append((final_time_d, final_time_f))

    ratio = final_time_d / final_time_f
    ratios.append(ratio)
    print(f"Size {i:4f} | DFT: {final_time_d:.4f}s | FFT: {final_time_f:.4f}s | ratio: {ratio:.1f}x faster")

print("n\ final time matrix:", time_comp)
print("Ratio for plotting:", ratios)

## Exercise 3: Convolution theorem proof by example. Create signal x = [1, 2, 3, 4, 0, 0, 0, 0] and filter 
# h = [1, 1, 1, 0, 0, 0, 0, 0]. Compute their circular convolution directly (nested loop). Then compute it via FFT (transform, 
# multiply, inverse transform). Verify the results match. Now do linear convolution by zero-padding appropriately.

signal = [1, 2, 3, 4, 0, 0, 0, 0]
N = len(signal)
h = [1, 1, 1, 0, 0, 0, 0, 0]


c_convolved = []

for n in range(N):
    total = 0
    for k in range(N):
        total += signal[k] * h[(n-k) % N]
    c_convolved.append(total)

print(c_convolved)

# using our function(transforming both into frequency space)
x_freq = fft(signal)
h_freq = fft(h)

y_freq = []
for i in range(len(x_freq)):
    y_freq.append(x_freq[i] * h_freq[i])  # multiplying element by element 

# inverse back to the time domain
fft_convolved = idft(y_freq)
fft_convolved_real = [round(c.real, 4) for c in fft_convolved]
print("FFT Circular Convolution:", fft_convolved_real)

# they match perfrectly

# linear convolution with 0 padding. so the length is 8 + 8 -1 = 15
signal_padded = [1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
h_padded = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

X_freq = fft(signal_padded)
H_freq = fft(h_padded)

Y_freq = []
for i in range(len(X_freq)):
    Y_freq.append(X_freq[i] * H_freq[i])
# iversing
padded_convolved = idft(Y_freq)
padded_convolved_real = [round(c.real, 4) for c in padded_convolved]
print("padded Circular Convolution:", padded_convolved_real)


## Exercise 4: Windowing effects. Create a signal that is the sum of two sine waves at 10 Hz and 12 Hz (very close). 
# Sample at 128 Hz for 1 second. Compute the power spectrum with no window, Hann window, and Hamming window. 
# Which window makes it easiest to distinguish the two peaks? Why?

fs = 128
N = 128 

sw1 = [math.sin(2 * math.pi * 10 * (n/fs)) for n in range(N)]
sw2 = [math.sin(2 * math.pi * 12 * (n/fs)) for n in range(N)]
final_signal = [sw1[i] + sw2[i] for i in range(N)]


def power_spectrum(X):
    return [xk.real ** 2 + xk.imag** 2 for xk in X]

def hann_window(n, N):
    return 0.5 * (1 - math.cos(2 * math.pi * n / (N - 1)))
def hamming_window(n, N):
    return 0.54 - 0.46 * math.cos(2 * math.pi * n / (N-1))

X_hann = [final_signal[i] * hann_window(i, N) for i in range(N)]
X_ham = [final_signal[i] * hamming_window(i, N) for i in range(N)]

ps_no_window = power_spectrum(final_signal)
ps_hann = power_spectrum(X_hann)
ps_ham = power_spectrum(X_ham)

print(f"power spectrum for this is :{'ps', ps_no_window[10:13]} | {'hann',ps_hann[10:13]}, {'ham', ps_ham[10:13]}")

# although this should have worked in a way hann window might have the highest drop but here the no window function gave the higest
# drop from index 10 to 11 which is absurd.

## Exercise 5: Positional encoding analysis. Generate the sinusoidal positional encodings for d_model = 128 and max_pos = 512. 
# For each pair of positions (p1, p2), compute the dot product of their encodings. Show that the dot product depends only 
# on |p1 - p2|, not on the absolute positions. What happens to the dot product as the distance increases?

d_model = 128
max_pos = 512

pos_enc = [[0.0] * d_model for _ in range(max_pos)]

for pos in range(max_pos):
    for i in range(0, d_model, 2):
        denom = math.pow(10000, (2 * (i // 2)) / d_model)

        # Even indices get sine, Odd indices get Cosine
        pos_enc[pos][i] = math.sin(pos / denom)
        pos_enc[pos][i + 1] = math.cos(pos / denom)

# proving it depends on |p1 - p2|

def dot_product(v1, v2):
    return sum(a * b for a, b in zip(v1, v2))

# testing with random positions with exact distance
dot_A = dot_product(pos_enc[10], pos_enc[20])
dot_B = dot_product(pos_enc[100], pos_enc[110])

print(f"Dot product at distance 10 (A: 10 & 20):  {dot_A:.4f}")
print(f"Dot product at distance 10 (B: 100 & 110): {dot_B:.4f}")
print(f"Are they identical: {math.isclose(dot_A, dot_B, abs_tol=1e-7)}")

# the distance between 2 position vectors at any position is equal if |p1 - p2| between them is equal

print("\nTracking dot product decay over distance:")
base_pos = 0
for dist in [0, 1, 2, 5, 10, 50, 100, 200, 400]:
    dot = dot_product(pos_enc[base_pos], pos_enc[base_pos + dist])
    print(f"Distance: {dist:3d} | Dot Product: {dot:.4f}")

# as the distance increase the dot product decreases and eventually move towards 0.

