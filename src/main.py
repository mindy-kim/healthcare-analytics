import json
from argparse import ArgumentParser
from pathlib import Path
from ipinstance import IPInstance
from timer import Timer

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()

    input_file = args.input_file
    path = Path(input_file)
    filename = path.name

    timer = Timer()
    timer.start()

    instance = IPInstance(str(input_file))
    solution, objective_value = instance.solve()

    timer.stop()

    output_dict = {"Instance": filename,
                   "Time": f"{timer.getTime():.2f}",
                   "Result": objective_value if objective_value else "--",
                   "Solution": "OPT" if solution else "--"}

    print(json.dumps(output_dict))
