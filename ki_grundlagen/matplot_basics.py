import matplotlib.pyplot as plt

monate = [1, 2, 3, 4, 5, 6]
temperaturen = [2, 4, 10, 15, 20, 25]

plt.bar(monate, temperaturen, color = "green")
plt.xlabel("Monate")
plt.ylabel("Temperaturen")
plt.title("Mein erstes MatPlotlib Diagramm")
plt.show()



