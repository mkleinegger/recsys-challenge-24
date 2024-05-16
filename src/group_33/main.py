from pathlib import Path

touchpath = Path("main_touched").resolve()
print(f"touching {touchpath}")
touchpath.touch()