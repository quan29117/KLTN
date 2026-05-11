from pathlib import Path
import importlib

def main():
    script_dir = Path("script")

    for file in script_dir.glob("*.py"):
        if file.name == "__init__.py":
            continue

        module_name = file.stem

        module = importlib.import_module(f"script.{module_name}")

        if hasattr(module, "run"):
            module.run()

if __name__ == "__main__":
    main()