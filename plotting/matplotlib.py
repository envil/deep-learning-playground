# importing the required module
from matplotlib import pyplot

# x axis values
x = [1, 2, 3]
# corresponding y axis values
y = [2, 4, 1]

# plotting the points
pyplot.plot(x, y)

# naming the x axis
pyplot.xlabel('x - axis')
# naming the y axis
pyplot.ylabel('y - axis')

# giving a title to my graph
pyplot.title('My first graph!')

# function to show the plot
pyplot.show()