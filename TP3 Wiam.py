import numpy as np
import matplotlib.pyplot as plt


# Exercice 1 : 
x = np.linspace(-10, 10, 100)
y = 2 * x**3 - 5 * x**2 + 3 * x - 7
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'b-')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Polynomial Function: y = 2x^3 - 5x^2 + 3x - 7')
plt.show()


# Exercice 2 : 
x = np.linspace(0.1, 10, 500)
y1 = np.exp(x)
y2 = np.log(x)
plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'r-', label='exp(x)')
plt.plot(x, y2, 'g--', label='log(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential and Logarithmic Functions')
plt.grid(True)
plt.legend()
plt.savefig('exponential_logarithmic_plot.png', dpi=100)
plt.show()


# Exercice 3 : 
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
y1 = np.tan(x)
y2 = np.arctan(x)
y3 = np.sinh(x)
y4 = np.cosh(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(x, y1, 'b-', label='tan(x)')
ax1.plot(x, y2, 'r--', label='arctan(x)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('tan(x) and arctan(x)')
ax1.legend()
ax1.grid(True)

x = np.linspace(-2, 2, 500)
ax2.plot(x, y3, 'g-', label='sinh(x)')
ax2.plot(x, y4, 'm--', label='cosh(x)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('sinh(x) and cosh(x)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()


# Exercice 4 : 
n = np.random.randn(1000)
plt.figure(figsize=(10, 6))
plt.hist(n, bins=30, color='purple')
plt.title('Histogram of Normally Distributed Data')
plt.xlim([n.min(), n.max()])
plt.show()


# Exercice 5 : 
x = np.random.uniform(0, 10, 500)
y = np.sin(x) + np.random.normal(0, 0.1, 500)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, c=y, s=50, alpha=0.5, cmap='viridis')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scatter Plot of y = sin(x) with Noise')
plt.grid(True)
plt.xticks([])
plt.yticks([])
plt.savefig('scatter_plot.pdf')
plt.show()


# Exercice 6 : 
x = np.linspace(-5, 5, 200)
y = np.linspace(-5, 5, 200)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))
plt.figure(figsize=(10, 6))
contour = plt.contour(X, Y, Z, cmap='plasma')
plt.clabel(contour, inline=True, fontsize=8)
plt.title('Contour Plot of f(x, y) = sin(sqrt(x^2 + y^2))')
plt.show()


# Exercice 7 :
from mpl_toolkits.mplot3d import Axes3D

x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)
Z = np.cos(np.sqrt(X**2 + Y**2))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='coolwarm')
ax.set_title('3D Surface Plot of Z = cos(sqrt(X^2 + Y^2))')
fig.colorbar(surf)
plt.show()


# Exercice 8 : 
x = np.linspace(-2, 2, 10)
y1 = x**2
y2 = x**3
y3 = x**4

plt.figure(figsize=(10, 6))
plt.plot(x, y1, 'ro-', label='x^2')
plt.plot(x, y2, 'bs--', label='x^3')
plt.plot(x, y3, 'g^:', label='x^4')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Different Line and Marker Styles')
plt.legend()
plt.show()


# Exercice 9 : Logarithmic Scale
x = np.linspace(1, 100, 50)
y1 = 2**x
y2 = np.log2(x)

plt.figure(figsize=(12, 6))
plt.plot(x, y1, 'r-', label='2^x')
plt.plot(x, y2, 'b--', label='log2(x)')
plt.yscale('log')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Logarithmic Scale Plot')
plt.grid(True)
plt.legend()
plt.show()

print("####################### Exercice 10 #######################")
# Exercice 10 : Changing Viewing Angle
fig = plt.figure(figsize=(14, 6))

ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='coolwarm')
ax1.view_init(elev=30, azim=30)
ax1.set_title('Viewing Angle 1')

ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='coolwarm')
ax2.view_init(elev=60, azim=60)
ax2.set_title('Viewing Angle 2')

plt.show()

print("####################### Exercice 11 #######################")
# Exercice 11 : 3D Wireframe Plot
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')
ax.set_title('3D Wireframe Plot of Z = sin(X) * cos(Y)')
plt.show()

print("####################### Exercice 12 #######################")
# Exercice 12 : 3D Contour Plot
x = np.arange(-5, 5, 0.25)
y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(x, y)
Z = np.exp(-0.1 * (X**2 + Y**2))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
contour = ax.contour3D(X, Y, Z, 50, cmap='viridis')
ax.set_title('3D Contour Plot of Z = exp(-0.1 * (X^2 + Y^2))')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print("####################### Exercice 13 #######################")
# Exercice 13 : 3D Parametric Plot
t = np.linspace(0, 2 * np.pi, 100)
X = np.sin(t)
Y = np.cos(t)
Z = t

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z, color='blue')
ax.set_title('3D Parametric Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print("####################### Exercice 14 #######################")
# Exercice 14 : 3D Bar Plot
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
X, Y = np.meshgrid(x, y)
Z = np.exp(-0.1 * (X**2 + Y**2))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(Z.ravel()), 0.5, 0.5, Z.ravel(), shade=True)
ax.set_title('3D Bar Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print("####################### Exercice 15 #######################")
# Exercice 15 : 3D Vector Field
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)
z = np.linspace(-5, 5, 10)
X, Y, Z = np.meshgrid(x, y, z)
U = -Y
V = X
W = Z

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.1, normalize=True)
ax.set_title('3D Vector Field')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print("####################### Exercice 16 #######################")
# Exercice 16 : 3D Scatter Plot
x = np.random.randn(100)
y = np.random.randn(100)
z = np.random.randn(100)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(x, y, z, c=z, cmap='viridis')
fig.colorbar(scatter)
ax.set_title('3D Scatter Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print("####################### Exercice 17 #######################")
# Exercice 17 : 3D Line Plot
t = np.linspace(0, 4 * np.pi, 100)
X = np.sin(t)
Y = np.cos(t)
Z = t

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot(X, Y, Z, color='red', linewidth=2)
ax.set_title('3D Line Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print("####################### Exercice 18 #######################")
# Exercice 18 : 3D Filled Contour Plot
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
filled_contour = ax.contourf3D(X, Y, Z, 50, cmap='plasma')
fig.colorbar(filled_contour)
ax.set_title('3D Filled Contour Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()

print("####################### Exercice 19 #######################")
# Exercice 19 : 3D Heatmap
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

plt.figure(figsize=(10, 6))
plt.imshow(Z, extent=[-5, 5, -5, 5], origin='lower', cmap='hot', aspect='auto')
plt.colorbar()
plt.title('3D Heatmap')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

print("####################### Exercice 20 #######################")
# Exercice 20 : 3D Density Plot
x = np.random.randn(1000)
y = np.random.randn(1000)
z = np.random.randn(1000)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
hist, edges = np.histogramdd((x, y, z), bins=(30, 30, 30))
xpos, ypos, zpos = np.meshgrid(edges[0][:-1] + 0.25, edges[1][:-1] + 0.25, edges[2][:-1] + 0.25)
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = zpos.flatten()
dx = dy = dz = 0.5 * np.ones_like(zpos)
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.6)
ax.set_title('3D Density Plot')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
