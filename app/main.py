import importlib
import sys

try:
    sys.path.append("app/scripts")
    script_name = sys.argv[1]

    importlib.import_module(script_name)
except ModuleNotFoundError as error:
    print(error)
    print(f"Script {script_name} not found.")
    sys.exit(1)
