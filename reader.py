from sympy import Matrix
import csv


def getFromTable(path) -> Matrix:
    rows = []

    with open(path, "r") as csvfile:
        table = csv.reader(csvfile, delimiter=';')

        for row in table:
            thisRow = []
            for number in row:

                number = number.replace(",", ".")

                if number == "":
                    number = 0.0
                else:
                    number = float(number)
                thisRow.append(number)

            rows.append(thisRow)

    # for r in rows:
    #     print(r)

    res: Matrix = Matrix(rows)

    return res
