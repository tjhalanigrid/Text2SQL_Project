import matplotlib.pyplot as plt

labels = ["Without", "With"]
constraint = [0, 88]

plt.figure()
plt.bar(labels, constraint)

plt.title("Constraint Satisfaction (Task 3)")
plt.ylabel("Percentage")

plt.savefig("task3_constraint.png")
plt.show()


