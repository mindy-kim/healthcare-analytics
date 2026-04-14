import numpy as np
from ortools.linear_solver.python import model_builder

#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)

class IPInstance:
    numTests: int  # number of tests
    numDiseases: int  # number of diseases
    costOfTest: np.ndarray  # [numTests] the cost of each test
    A: np.ndarray  # [numTests][numDiseases] 0/1 matrix if test is positive for disease

    def __init__(self, filename: str) -> None:
        self.load_from_file(filename)
        self.model = model_builder.Model()
        self.solver = model_builder.Solver("SCIP")
        self.solution = None
        self.objective_value = None

    def solve(self):
        """
        Healthcare Analytics Model
        """

        self.solution = ...
        self.objective_value = ...

        return self.solution, self.objective_value

    def load_from_file(self, filename: str):
        try:
            with open(filename, "r") as fl:
                self.numTests = int(fl.readline().strip())  # n
                self.numDiseases = int(fl.readline().strip())  # m

                self.costOfTest = np.array([float(i) for i in fl.readline().strip().split()])

                self.A = np.zeros((self.numTests, self.numDiseases))
                for i in range(0, self.numTests):
                    self.A[i, :] = np.array([int(i) for i in fl.readline().strip().split()])

        except Exception as e:
            print(f"Error reading instance file. File format may be incorrect.{e}")
            exit(1)

    def __str__(self):
        out = ""
        out = f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0, self.A.shape[0])])
        out += f"A:\n{A_str}"
        return out
