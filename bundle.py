import argparse
import zipfile
from pathlib import Path

BLACKLIST = ["__pycache__", ".pyc", ".ipynb"]
MAXSIZE_MB = 40


def bundle(source_dir: str, output_name: str):
    """
    Usage: python3 bundle.py <source_directory> <output_name>
    """
    source_dir = Path(source_dir).resolve()
    output_path = Path(__file__).parent / f"{output_name}.zip"

    # Get the files from the source directory
    files = []

    for f in source_dir.rglob("*"):
        if all(b not in str(f) for b in BLACKLIST):
            files.append(f)

    print("Bundling the following files:")
    print("\n".join(str(f.relative_to(source_dir)) for f in files))

    # Zip all files, keeping the directory structure
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, source_dir.stem / f.relative_to(source_dir))

    output_size_mb = output_path.stat().st_size / 1024 / 1024

    if output_size_mb > MAXSIZE_MB:
        print(f"Warning: The created zip file ({output_size_mb:.2f} MB) is larger than {MAXSIZE_MB} MB!")

    print(f"\nSubmission created: {output_path.resolve()!s} {output_size_mb:.2f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bundle a source directory into a zip file.")
    parser.add_argument("source_directory", help="The source directory to bundle (e.g., 'src').")
    parser.add_argument("output_name", help="The name for the output zip file (without extension).")

    args = parser.parse_args()

    bundle(args.source_directory, args.output_name)